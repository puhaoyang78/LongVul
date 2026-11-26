"""Historical optimizer state, matched short replay, and balanced overfit checks."""
from collections import defaultdict
from math import ceil, sqrt
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import get_linear_schedule_with_warmup, set_seed

from coarse_encoding import collate_fine
from paired import (ADAM_EPSILON, BATCH_SIZE, EPOCHS, LEARNING_RATE, LOSVERAdamW, MAX_GRAD_NORM, WARMUP_FRACTION,
                    predict, to_gpu, train_epoch)
from rgcl.artifacts import read_json, write_json

PROBE_STEPS = (1,2,8,32,64)


def parameter_group(name):
    if name.startswith("fine.encoder.encoder.layer."):
        return f"layer_{int(name.split('.')[4]):02d}"
    if name.startswith("fine.encoder.embeddings."):
        return "embeddings"
    if name.startswith("fine.classifier."):
        return "head"
    if name.startswith("fusion."):
        return "global_projection"
    raise ValueError(f"Unexpected fine parameter {name}")


def norms(tensors):
    groups = defaultdict(list)
    for name,value in tensors:
        groups[parameter_group(name)].append(value.detach().float().norm().square())
    return {group:float(torch.stack(values).sum().sqrt()) for group,values in groups.items()}


def relative_changes(start,end):
    base = norms(start.items())
    flat = norms((name,end[name]-old) for name,old in start.items())
    return {g:dict(parameter_norm=base[g],absolute_update=delta,
                   relative_update=delta/base[g] if base[g] else None) for g,delta in flat.items()}


def stored_optimizer(checkpoint,named):
    saved = checkpoint["optimizer"]
    groups = saved["param_groups"]
    if len(groups)!=1:
        raise ValueError("The historical fine optimizer used one parameter group")
    group = groups[0]
    states = [(name,saved["state"][i]) for i,name in zip(group["params"],named,strict=True)]
    values = defaultdict(list)
    for name,state in states:
        avg,v = state["exp_avg"],state["exp_avg_sq"]
        values[parameter_group(name)].append((avg.numel(),float(avg.norm().square()),float(v.sum()),int(state["step"])))
    stats = {}
    for name,summary in values.items():
        n,anorm,vsum = map(sum,zip(*(r[:3] for r in summary)))
        actual_steps = [r[3] for r in summary]
        stats[name] = dict(numel=n,first_moment_rms=sqrt(anorm/n),second_moment_sqrt_mean=sqrt(vsum/n),
                           min_steps=min(actual_steps),max_steps=max(actual_steps))
    return dict(groups=stats,last_epoch=checkpoint["scheduler"]["last_epoch"],
                current_lr=group["lr"],scheduler_last_lr=checkpoint["scheduler"]["_last_lr"],
                hyperparameters={k:group[k] for k in ("eps","betas","weight_decay","initial_lr")})


def reconstruct_order(count,valid_count):
    """Replay DataLoader’s CPU RNG-only sampling and evaluation draws."""
    with torch.random.fork_rng(devices=[]):
        loader = DataLoader(range(count),batch_size=BATCH_SIZE,shuffle=True)
        valid = DataLoader(range(valid_count),batch_size=BATCH_SIZE)
        first = []
        for epoch in range(EPOCHS):
            indices = [row.tolist() for row in loader]
            if epoch==0:
                first = indices[:PROBE_STEPS[-1]]
            for _ in valid:
                pass
        final = torch.get_rng_state()
    return first,final


def optimizer_for(model,n):
    optimizer = LOSVERAdamW(model.parameters(),lr=LEARNING_RATE,eps=ADAM_EPSILON)
    steps = ceil(n/BATCH_SIZE)*EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer,steps*WARMUP_FRACTION,steps)
    return optimizer,scheduler,steps


def short_replay(model,train,first,full_count):
    ordered = Subset(train,[i for batch in first for i in batch])
    loader = DataLoader(ordered,batch_size=BATCH_SIZE,collate_fn=collate_fine,
                        generator=torch.Generator().manual_seed(42))
    opt,scheduler,steps = optimizer_for(model,full_count)
    snapshots,losses = {},[]
    model.train()
    for step,batch in enumerate(loader,1):
        data = to_gpu(batch,labels=True)
        out = model(data)
        loss = out["loss"]
        loss.backward()
        lr = opt.param_groups[0]["lr"]
        record = step in PROBE_STEPS
        if record:
            params = dict(model.named_parameters())
            before = {n:p.detach().clone() for n,p in params.items()}
            pre = norms((n,p.grad) for n,p in params.items() if p.grad is not None)
        actual = torch.nn.utils.clip_grad_norm_(model.parameters(),MAX_GRAD_NORM,error_if_nonfinite=True)
        if record:
            post = norms((n,p.grad) for n,p in params.items() if p.grad is not None)
        opt.step()
        if record:
            snapshots[str(step)] = dict(lr=lr,uncut_norm=float(actual),pre_clip=pre,post_clip=post,
                update=relative_changes(before,params),
                sample_keys=[train.metadata[i]["sample_key"] for i in first[step-1]],
                probability=out["probability"].detach().tolist())
        opt.zero_grad(set_to_none=True)
        scheduler.step()
        losses.append(float(loss.detach()))
        if step==PROBE_STEPS[-1]:
            break
    return dict(steps=steps,warmup_updates=steps*WARMUP_FRACTION,losses=losses,snapshots=snapshots)


def subset_indices(sources):
    starts = [i for i in range(0,len(sources),2) if min(len(s.ids) for s in sources[i:i+2])>510]
    pairs = sorted(np.random.default_rng(42).choice(starts,64,replace=False).tolist())
    return [i for left in pairs for i in (left,left+1)]


def overfit(model,dataset,subset):
    selected = Subset(dataset,subset)
    train = DataLoader(selected,batch_size=BATCH_SIZE,shuffle=True,collate_fn=collate_fine)
    eval_ = DataLoader(selected,batch_size=BATCH_SIZE,collate_fn=collate_fine)
    opt,schedule,steps = optimizer_for(model,len(subset))
    y = np.array([dataset.metadata[i]["target"] for i in subset])
    if int(y.sum())*2!=len(y):
        raise ValueError("Overfit probe must be balanced")

    def measure():
        scores,_ = predict(model,eval_)
        p = np.clip(np.array(scores),1e-15,1.-1e-15)
        return dict(bce=float(-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))),accuracy=float(((p>.5)==y).mean()))

    started = perf_counter()
    initial = measure()
    history = []
    for epoch in range(1,EPOCHS+1):
        stats = train_epoch(model,train,opt,schedule)
        history.append(dict(epoch=epoch,training=stats,reevaluated=measure()))
    return dict(epochs=history,initial=initial,updates=steps,warmup_updates=steps*WARMUP_FRACTION,
                sample_keys=[dataset.metadata[i]["sample_key"] for i in subset],seconds=perf_counter()-started)


def run(seed,tag,model_factory,train,valid_count,results,weights,output,subset,first,full_count):
    file = output/f"optimizer_{tag}_{seed}.json"
    if file.is_file():
        return read_json(file)
    start = perf_counter()
    set_seed(seed)
    model = model_factory()
    named = dict(model.named_parameters())
    init = {n:p.detach().cpu() for n,p in named.items()}
    history = read_json(results/"epochs.json")
    checkpoint = torch.load(weights/"last.pt",map_location="cpu",weights_only=False,mmap=True)
    best = torch.load(weights/"best.pt",map_location="cpu",weights_only=True,mmap=True)
    if checkpoint["loop"]!=history:
        raise ValueError("Historical epoch log disagrees with the final optimizer checkpoint")
    state = stored_optimizer(checkpoint,named)
    deltas = dict(initial_to_best=relative_changes(init,best),initial_to_last=relative_changes(init,checkpoint["model"]),
                  best_to_last=relative_changes(best,checkpoint["model"]))
    _,reconstructed = reconstruct_order(full_count,valid_count)
    replay = short_replay(model,train,first,full_count)
    end = checkpoint["rng"]["torch"]
    exact = torch.equal(end,reconstructed)
    del model,named,init,checkpoint,best
    set_seed(seed)
    probe = overfit(model_factory(),train,subset)
    curves = [{k:e[k] for k in ("epoch","train_loss","validation")}|(
        {"positive_fraction":(e["validation"]["tp"]+e["validation"]["fp"])/valid_count}) for e in history]
    result = dict(tag=tag,seed=seed,historical_curves=curves,historical_optimizer=state,
                  historical_parameter_displacement_from_recreated_initialization=deltas,
                  replay_original_prefix=replay,reconstructed_final_sampler_rng_matches_available_state=exact,
                  balanced_matched_overfit=probe,seconds=perf_counter()-start)
    write_json(file,result)
    return result
