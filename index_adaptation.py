"""Task adaptation of the fixed-32 LongVul indexer under the existing fine pipeline."""
import argparse
from itertools import combinations
import json
from math import log
from statistics import mean, median
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import set_seed

import coarse_to_fine as pipeline
from coarse_encoding import BLOCK, QUERY_HEADS, BlockData, CoarseClassifier, collate_blocks, repair_block_mass, repair_kl
from coarse_retrieval import lengths
from compaction import BUDGET
from context_selection import visible_source, write_prediction_rows
from paired import BATCH_SIZE, EPOCHS, ROOT, SEED, UNIX_BASE, fit_classifier, load_tokenizer, to_gpu
from predicted_diff import paired_annotations
from repair_oracle import EDIT_TYPES, GAP_TYPES, grouped_edit_retention, overlap_bpe
from rgcl.artifacts import read_json, write_json
from rgcl.config import SEEDS
from window_inputs import GROUPS

RESULTS = ROOT/"results"/"index_adaptation_512"
WEIGHTS = ROOT/"checkpoints"/RESULTS.name
METHODS = ("A","B","C","D")
CONTEXT = 96
WARM_EPOCHS = 2
PROBE_BATCHES = 64
TARGET_RATIOS = (.5,1.)
HEAD_PAIRS = tuple(combinations(range(QUERY_HEADS),2))


def published_setup():
    old = read_json(pipeline.RESULTS/"seed42"/"global_fusion"/"config.json")
    fields = ("block_layout","fusion","context_budget","center_budget","length")
    expected = ("fixed32","global",CONTEXT,BUDGET-CONTEXT,512)
    if tuple(old[key] for key in fields)!=expected:
        raise ValueError("The experiment must use the published fixed-32 Global Fusion settings")
    return {k:old[k] for k in fields}


def historical_weights(seed,name):
    path = pipeline.ARCHIVE_WEIGHTS/f"seed{seed}"/"coarse"/name/"best.pt"
    return torch.load(path,weights_only=True,map_location="cpu")


def initialize(seed,method,width):
    set_seed(seed)
    if method == "D":
        model = CoarseClassifier(width,"joint",bypass=False)
        state = historical_weights(seed,pipeline.historical_index(seed)["name"])
        model.indexer.load_state_dict({k.removeprefix("indexer."):v for k,v in state.items()
                                       if k.startswith("indexer.")},strict=True)
        return model
    # Match the original seed and module construction order before narrowing the input.
    model = CoarseClassifier(width,"joint")
    missing,unexpected = model.load_state_dict(historical_weights(seed,"global"),strict=False)
    if set(missing)!={"indexer.query.weight","indexer.key.weight"} or unexpected:
        raise ValueError("Historical global warm start differs from the fixed-32 model")
    if method != "A":
        original = model.representation
        model.bypass = False
        constructor_rng = torch.get_rng_state()
        model.representation = torch.nn.Sequential(torch.nn.LayerNorm(width),torch.nn.Linear(width,width),torch.nn.Tanh())
        torch.set_rng_state(constructor_rng)
        with torch.no_grad():
            model.representation[0].weight.copy_(original[0].weight[-width:])
            model.representation[0].bias.copy_(original[0].bias[-width:])
            model.representation[1].weight.copy_(original[1].weight[:,-width:])
            model.representation[1].bias.copy_(original[1].bias)
    return model


def supervised_data(train):
    edits = paired_annotations(train.sources)
    mass = [repair_block_mass(s,e,a,(*EDIT_TYPES,*GAP_TYPES))
            for s,e,a in zip(train.sources,train.ends,edits,strict=True)]
    return BlockData(train.rows,train.ends,train.vectors,[q/q.sum().clamp_min(1) for q in mass])


def fixed_probe(train):
    generator = torch.Generator().manual_seed(SEED)
    ids = torch.randperm(len(train),generator=generator)[:PROBE_BATCHES*BATCH_SIZE].tolist()
    return ids,DataLoader(Subset(train,ids),batch_size=BATCH_SIZE,collate_fn=collate_blocks,generator=generator)


def gradient_probe(model,loader,weight,*,keep_batches=False):
    params = tuple(model.indexer.parameters())
    if not all(p.requires_grad for p in params):
        raise ValueError("Gradient probe requires unfrozen Query/Key")
    started = perf_counter()
    prior = model.training
    model.eval()
    rows = []
    for cpu in loader:
        batch = to_gpu(cpu,labels=True)
        out = model(batch,details=True)
        bce = out["loss_components"]["train_bce"]
        kl = repair_kl(out["importance"],batch["repair"])
        a = torch.autograd.grad(bce,params,retain_graph=True)
        b = torch.autograd.grad(kl,params)
        components = [torch.stack((x.norm(),y.norm(),(x*y).sum())) for x,y in zip(a,b,strict=True)]
        q,k = torch.stack(components).detach().cpu().tolist()
        joint = ((q[0]**2+k[0]**2)**.5,(q[1]**2+k[1]**2)**.5,q[2]+k[2])
        rows.append(dict(query=q,key=k,joint=joint))
    model.train(prior)
    summary = {}
    for kind in ("joint","query","key"):
        lengths_and_dot = [row[kind] for row in rows]
        usable = [(x,y,dot) for x,y,dot in lengths_and_dot if x>0 and y>0]
        ratio = median(y/x for x,y,_ in usable) if usable else None
        summary[kind] = dict(bce_norm=median(x for x,_,_ in lengths_and_dot),
                             kl_norm=median(y for _,y,_ in lengths_and_dot),
                             both_nonzero=len(usable),batches=len(rows),kl_to_bce=ratio,
                             weighted_kl_to_bce=ratio*weight if ratio is not None else None,
                             cosine=median(dot/(x*y) for x,y,dot in usable) if usable else None)
    result = dict(summary=summary,weight=weight,seconds=perf_counter()-started)
    if keep_batches:
        result["batches"] = rows
    return result


def jaccard(a,b):
    common = overlap_bpe(a,b)
    union = lengths(a)+lengths(b)-common
    return common/union if union else 1.


def index_view(model,split):
    model.eval()
    data = BlockData(split.rows,split.ends,split.vectors)
    features,ranks,rows = [],[],[]
    with torch.inference_mode():
        for cpu in DataLoader(data,batch_size=BATCH_SIZE,collate_fn=collate_blocks):
            out = model(to_gpu(cpu),details=True,inspect=True)
            features.append(out["global_state"].cpu())
            for n,p,heads,score,uniform in zip(cpu["valid"].sum(1).tolist(),out["importance"].cpu().numpy(),
                    out["head_importance"].cpu().numpy(),out["probability"].tolist(),
                    out["uniform_probability"].tolist(),strict=True):
                probabilities = np.vstack((p[:n],heads[:,:n])).astype(np.float64)
                entropy = -(probabilities*np.log(np.maximum(probabilities,1e-300))).sum(axis=1)
                normalized = entropy/log(n) if n>1 else entropy
                top = np.argsort(-heads[:,:n],axis=-1,kind="stable")[:,:min(10,n)]
                overlap = [len(set(top[i]) & set(top[j]))/len(set(top[i]) | set(top[j])) for i,j in HEAD_PAIRS]
                rows.append(dict(probability=score,uniform_probability=uniform,blocks=n,
                                 entropy=float(entropy[0]),normalized_entropy=float(normalized[0]),
                                 head_normalized_entropy=normalized[1:].tolist(),head_top10=top.tolist(),
                                 head_top10_jaccard=overlap,
                                 head_top1_agreement=mean(int(top[i,0]==top[j,0]) for i,j in HEAD_PAIRS),
                                 uniform_abs_delta=abs(score-uniform),
                                 uniform_flip=int((score>.5)!=(uniform>.5))))
                ranks.append(p[:n].copy())
    # These are frozen inputs and are read-only during layout construction.
    return torch.cat(features),ranks,rows


def reference_runs(seed,split,rows):
    method = "global_fusion" if seed==SEED else "final"
    path = pipeline.RESULTS/f"seed{seed}"/method/f"{split}.csv.gz"
    return [json.loads(row["source_runs"]) for row in pipeline.read_predictions(path,rows)]


def grouped_layout_change(before,after,sizes):
    change = [jaccard(a,b) for a,b in zip(before,after,strict=True)]
    return {group:dict(samples=len(values),mean=mean(values),median=median(values),
                       identical=sum(v==1. for v in values))
            for group,(lo,hi) in GROUPS.items()
            if (values := [v for v,n in zip(change,sizes,strict=True) if lo<=n<=hi])}


def inspect_valid(model,valid,tokenizer,edits,area,*,original=None,initial=None):
    started = perf_counter()
    coarse,ranks,rows = index_view(model,valid)
    raw = [len(s.raw_ids) for s in valid.sources]
    layout,cost = pipeline.fine_data(valid,coarse,ranks,tokenizer,CONTEXT)
    visible = [visible_source(example) for example in layout.examples]
    # Exact original BPE spans are the comparison unit, including all context and fill.
    for j,(row,meta,runs) in enumerate(zip(rows,layout.metadata,visible,strict=True)):
        row.update(meta,source_runs=runs,edit_audit_only=True)
        if original is not None:
            row["jaccard_historical"] = jaccard(original[j],runs)
        if initial is not None:
            row["jaccard_initial"] = jaccard(initial[j],runs)
    groups = {}
    for group,(lo,hi) in GROUPS.items():
        subset = [r for r in rows if lo<=r["original_bpe"]<=hi]
        groups[group] = dict(functions=len(subset),entropy=mean(r["entropy"] for r in subset),
                             normalized_entropy=mean(r["normalized_entropy"] for r in subset),
                             head_normalized_entropy=[mean(v) for v in zip(*(r["head_normalized_entropy"] for r in subset))],
                             mean_pairwise_head_top10_jaccard=[mean(v) for v in zip(*(r["head_top10_jaccard"] for r in subset))],
                             mean_head_top1_agreement=mean(r["head_top1_agreement"] for r in subset),
                             probability_abs_delta_median=median(r["uniform_abs_delta"] for r in subset),
                             probability_abs_delta_mean=mean(r["uniform_abs_delta"] for r in subset),
                             uniform_threshold_flips=sum(r["uniform_flip"] for r in subset))
    area.mkdir(parents=True,exist_ok=True)
    write_prediction_rows(area/"valid.csv.gz",rows)
    result = dict(auxiliary=pipeline.classification(valid.rows,[r["probability"] for r in rows],raw),
                  uniform_counterfactual=pipeline.classification(valid.rows,[r["uniform_probability"] for r in rows],raw),
                  indexer=pipeline.index_diagnostics(valid,ranks,edits,raw),score_mechanism=groups,
                  head_pairs=HEAD_PAIRS,layout_seconds=cost,
                  retention=grouped_edit_retention(zip(raw,edits,visible,strict=True)))
    if original is not None:
        result["layout_vs_historical"] = grouped_layout_change(original,visible,raw)
    if initial is not None:
        result["layout_vs_initial"] = grouped_layout_change(initial,visible,raw)
    result["seconds"] = perf_counter()-started
    write_json(area/"diagnostics.json",result)
    return result,visible


def index_costs(seed,method):
    path = pipeline.ARCHIVE/f"seed{seed}"/"coarse"
    selection = pipeline.historical_index(seed)["name"]
    inherited = {"global": read_json(path/"global"/"fit.json")["train_seconds"]}
    if method in ("reference","D"):
        inherited[selection] = read_json(path/selection/"fit.json")["train_seconds"]
    return inherited


def index_config(seed,method,weight,epochs,phase):
    return dict(pipeline.run_config(f"index_{method}_{phase}",seed),stage=phase,epochs=epochs,
                precision="FP32 on cached FP16 blocks",block_layout="fixed32",repair_weight=weight,
                auxiliary_bypass=method=="A",historical_initialization=pipeline.historical_index(seed)["name"]
                if method=="D" else "global head plus seeded Q/K",
                loss="function BCE + labeled function mean KL" if weight else "function BCE",
                inherited_training_seconds=index_costs(seed,method))


def validate_weights(model,reference):
    if any(not torch.equal(a,b) for a,b in zip(model.indexer.parameters(),reference,strict=True)):
        raise ValueError("Query/Key changed while the auxiliary head was warming")


def warm_head(model,seed,train,val,area,weights):
    model.indexer.requires_grad_(False)
    frozen = [p.detach().clone() for p in model.indexer.parameters()]
    config = index_config(seed,"D",0.,WARM_EPOCHS,"head_warmup")
    best,history = fit_classifier(model,train,val,weights,area,config,collator=collate_blocks,epochs=WARM_EPOCHS)
    # Transfer the terminal warmup weights, regardless of which warmup epoch had higher MCC.
    last = torch.load(weights/"last.pt",weights_only=False,map_location="cpu",mmap=True)
    model.load_state_dict(last["model"],strict=True)
    validate_weights(model,frozen)
    result = pipeline.fitted(config,best,history,len(val))
    result["transferred_epoch"] = WARM_EPOCHS
    result["terminal_validation"] = history[-1]["validation"]
    write_json(area/"fit.json",result)
    model.indexer.requires_grad_(True)
    return result


def train_method(tokenizer,seed,method,splits):
    path = RESULTS/f"seed{seed}"/method
    weights = WEIGHTS/f"seed{seed}"/method
    if (path/"metrics.json").is_file():
        return
    source,val_split = (splits[k] for k in ("train","valid"))
    train = supervised_data(source)
    val = BlockData(val_split.rows,val_split.ends,val_split.vectors)
    probe_ids,probe = fixed_probe(train)
    edits = paired_annotations(val_split.sources)
    original = reference_runs(seed,"valid",val_split.rows)
    model = initialize(seed,method,source.vectors.shape[-1]).cuda()
    # The old protocol initializes Q/K and then trains with the constructor's RNG state.
    # Probing uses evaluation mode and a private loader generator.
    warm = warm_head(model,seed,train,val,path/"warmup",weights/"warmup") if method=="D" else None
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state()
    initial_grad = gradient_probe(model,probe,1.,keep_batches=True)
    _,initial = inspect_valid(model,val_split,tokenizer,edits,path/"initial",original=original)
    write_json(path/"initial"/"gradients.json",dict(initial_grad,train_sample_keys=[source.rows[i]["sample_key"]
                                                                                 for i in probe_ids]))
    ratio = initial_grad["summary"]["joint"]["kl_to_bce"]
    if ratio is None or ratio<=0:
        raise ValueError("Cannot calibrate with zero Query/Key supervision")
    if method=="B":
        candidates = (("lambda_1",1.,None),)
    elif seed==SEED:
        candidates = tuple((f"ratio_{target:g}",target/ratio,target) for target in TARGET_RATIOS)
    else:
        selected = read_json(RESULTS/f"seed{SEED}"/method/"index_policy.json")["selected"]
        candidates = ((selected["candidate"],selected["weight"],selected["target_weighted_ratio"]),)
    initial_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    # Each candidate of a method receives the same initialization, shuffle and dropout stream.
    trials = []
    epochs = EPOCHS-(WARM_EPOCHS if warm else 0)
    for name,weight,target in candidates:
        print("Index",seed,method,name,"lambda",weight,flush=True)
        area = path/"index"/name
        wdir = weights/"index"/name
        config = dict(index_config(seed,method,weight,epochs,"index_adaptation" if warm else "index_direct"),
                      target_weighted_ratio=target,new_index_training_epochs=EPOCHS,
                      probe_batches=PROBE_BATCHES)
        model.load_state_dict(initial_state,strict=True)
        model.repair_weight = weight
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        if not (area/"fit.json").is_file():
            def observe(current,epoch):
                return {"index_gradients":gradient_probe(current,probe,weight)}
            best,history = fit_classifier(model,train,val,wdir,area,config,collator=collate_blocks,
                                          epochs=epochs,after_epoch=observe)
            observation,_ = inspect_valid(model,val_split,tokenizer,edits,area,original=original,initial=initial)
            if observation["auxiliary"]["groups"]["overall"] != best["validation"]:
                raise ValueError("Reloaded indexer auxiliary probabilities did not reproduce the selected Valid checkpoint")
            measured = gradient_probe(model,probe,weight,keep_batches=True)
            write_json(area/"gradients.json",measured)
            write_json(area/"fit.json",pipeline.fitted(config,best,history,len(val)))
        fit = read_json(area/"fit.json")
        observation = read_json(area/"diagnostics.json")
        trials.append(dict(candidate=name,weight=weight,target_weighted_ratio=target,
                           selected_epoch=fit["selected_epoch"],valid_mcc=fit["validation"]["mcc"],
                           valid_long_edit_retention=observation["retention"]["long"]["actual_edits"]["fraction"]))
    winner = max(trials,key=lambda x:(x["valid_mcc"],-x["weight"]))
    write_json(path/"index_policy.json",dict(seed=seed,method=method,initial_kl_to_bce=ratio,
        selection="full auxiliary Valid MCC, then smaller lambda" if seed==SEED else "frozen Seed 42 lambda",
        trials=trials,selected=winner))
    name = winner["candidate"]
    model.load_state_dict(torch.load(weights/"index"/name/"best.pt",weights_only=True,map_location="cpu"),strict=True)
    model.requires_grad_(False)
    config = dict(pipeline.run_config(method,seed),**published_setup(),stage="fine",coarse_frozen=True,
                  selection_policy="center-first then syntax then score fill",indexer_checkpoint=f"index/{name}/best.pt",
                  repair_weight=winner["weight"],auxiliary_bypass=method=="A",
                  inherited_index_training_seconds=index_costs(seed,method))
    pipeline.fit_frozen_fine(tokenizer,model,config,path,weights/"fine",cached=splits)


def references(tokenizer,splits):
    train = supervised_data(splits["train"])
    ids,probe = fixed_probe(train)
    val = splits["valid"]
    edits = paired_annotations(val.sources)
    for seed in SEEDS:
        path = RESULTS/f"seed{seed}"/"historical"
        if (path/"gradients.json").is_file():
            continue
        model = pipeline.load_index(seed,False)
        historic = pipeline.historical_index(seed)["weight"]
        model.repair_weight = historic
        model.indexer.requires_grad_(True)
        original = reference_runs(seed,"valid",val.rows)
        _,visible = inspect_valid(model,val,tokenizer,edits,path,original=original)
        if visible!=[list(map(tuple,x)) for x in original]:
            raise ValueError("Historical indexer did not reproduce published layouts")
        value = gradient_probe(model,probe,historic,keep_batches=True)
        write_json(path/"gradients.json",dict(value,train_sample_keys=[splits["train"].rows[i]["sample_key"]
                     for i in ids],inherited_training_seconds=index_costs(seed,"reference")))
        print("Historical index",seed,value["summary"]["joint"],flush=True)


def smoke(tokenizer):
    from coarse_encoding import encode_blocks, encoder_window_ends
    from locator_quality import load_context_encoder
    from paired import LOSVERAdamW
    from predicted_diff import load_sources

    all_rows = pipeline.prime_source_rows(pipeline.DATA,"train")
    pair = max(range(len(all_rows)),key=lambda i:len(all_rows[i]["func"]))//2*2
    rows = all_rows[pair:pair+2]
    sources = load_sources(rows,tokenizer)
    windows = [encoder_window_ends(s) for s in sources]
    ends = pipeline.cuts_for(sources,None,False)
    encoder = load_context_encoder().requires_grad_(False)
    vectors,encoded = encode_blocks(encoder,sources,tokenizer,windows,ends)
    del encoder
    split = pipeline.SourceSplit(rows,sources,windows,ends,vectors,encoded,0.)
    model = initialize(SEED,"D",vectors.shape[-1]).cuda()
    legacy = pipeline.load_index(SEED,False)
    qk = [p.detach().clone() for p in legacy.indexer.parameters()]
    validate_weights(model,qk)
    data = supervised_data(split)
    batch = to_gpu(collate_blocks([data[i] for i in range(len(data))]),labels=True)
    model.indexer.requires_grad_(False)
    model.train()
    head_before = model.head[-1].weight.detach().clone()
    opt = LOSVERAdamW(model.parameters(),lr=2e-5)
    opt.zero_grad(set_to_none=True)
    model(batch)["loss"].backward()
    opt.step()
    validate_weights(model,qk)
    if torch.equal(head_before,model.head[-1].weight) or any(p.grad is not None for p in model.indexer.parameters()):
        raise ValueError("Head warmup did not isolate Query/Key")
    model.indexer.requires_grad_(True)
    model.repair_weight = 1.
    measurement = gradient_probe(model,DataLoader(data,batch_size=2,collate_fn=collate_blocks),1.)
    if any(measurement["summary"][name][k]<=0 for name in ("query","key") for k in ("bce_norm","kl_norm")):
        raise ValueError("One supervision term did not reach both index projections")
    model.eval()
    with torch.no_grad():
        output = model(batch,details=True)
        orig = legacy(batch,details=True)
        torch.testing.assert_close(output["importance"],orig["importance"],rtol=0.,atol=0.)
        torch.testing.assert_close(output["global_state"],orig["global_state"],rtol=0.,atol=0.)
        _,lhs = model.auxiliary(output["global_state"],output["weighted"])
        _,rhs = model.auxiliary(output["global_state"]+1,output["weighted"])
        torch.testing.assert_close(lhs,rhs,rtol=0.,atol=0.)
    features,ranks,_ = pipeline.score_coarse(model,split,"global")
    layout,_ = pipeline.fine_data(split,features,ranks,tokenizer,CONTEXT)
    before = [p.detach().clone() for p in model.indexer.parameters()]
    opt.zero_grad(set_to_none=True)
    model(batch)["loss"].backward()
    opt.step()
    if any(torch.equal(p,value) for p,value in zip(model.indexer.parameters(),before,strict=True)):
        raise ValueError("Joint phase did not update Query/Key")
    if any(ex["length"]!=512 for ex in layout.examples) or encoded["body_bpe"]!=sum(len(s.ids) for s in sources):
        raise ValueError("Extreme function smoke violated full coverage or the fine budget")
    write_json(RESULTS/"smoke.json",dict(original_bpe=[len(s.raw_ids) for s in sources],encoding=encoded,
               fine_lengths=[ex["length"] for ex in layout.examples],gradients=measurement,
               warmup_qk_unchanged=True,joint_qk_updated=True,no_direct_global_path=True,historical_score_parity=True))
    print("Index adaptation GPU smoke passed",max(len(s.raw_ids) for s in sources),"original BPE",flush=True)


def report():
    winner = pipeline.valid_winner({m:RESULTS/f"seed{SEED}"/m for m in METHODS},RESULTS/"fine_policy.json")
    benchmark = read_json(pipeline.RESULTS/"summary.json")
    methods = {}
    rows = pipeline.prime_source_rows(pipeline.DATA,"test")
    codebert = pipeline.codebert_lengths("test",rows)
    reference = {s:reference_runs(s,"test",rows) for s in SEEDS}
    for seed in SEEDS:
        relative = f"seed{seed}"
        methods[relative] = {}
        for method in METHODS if seed==SEED else (winner,):
            path = RESULTS/relative/method
            fit = read_json(path/"metrics.json")
            policy = read_json(path/"index_policy.json")
            trial = path/"index"/policy["selected"]["candidate"]
            view = read_json(trial/"diagnostics.json")
            saved = pipeline.read_predictions(path/"test.csv.gz",rows)
            visible = [json.loads(row["source_runs"]) for row in saved]
            unix = [int(row["original_bpe"]) for row in saved]
            cost = {x["candidate"]:read_json(path/"index"/x["candidate"]/"fit.json")["train_seconds"]
                    for x in policy["trials"]}
            if method=="D":
                cost["head_warmup"] = read_json(path/"warmup"/"fit.json")["train_seconds"]
            methods[relative][method] = dict(selected_epoch=fit["selected_epoch"],validation=fit["validation"],
                training_seconds=fit["train_seconds"],peak_train_allocated_bytes=fit["peak_train_allocated_bytes"],
                test=fit["split"]["test"],policy=policy,valid_index_diagnostics=view,
                gradient_at_selected_index=read_json(trial/"gradients.json")["summary"],
                index_training_seconds=cost,inherited_index_training_seconds=fit["config"]["inherited_index_training_seconds"],
                test_layout_vs_historical={k:grouped_layout_change(reference[seed],visible,sizes)
                                          for k,sizes in (("unixcoder",unix),("codebert",codebert))})
    aggregate = {}
    runs = [methods[f"seed{s}"][winner]["test"] for s in SEEDS]
    for axis in ("unixcoder","codebert"):
        aggregate[axis] = {}
        for group in GROUPS:
            fields = {}
            for key in ("mcc","f1","precision","recall","auc","auprc"):
                fields[key] = pipeline.seed_stats([run[axis]["groups"][group][key] for run in runs])
            for key in ("p_c","pair_ranking_accuracy"):
                fields[key] = pipeline.seed_stats([run[axis]["pairs"][group][key] for run in runs])
            for kind in ("actual_edits",*GAP_TYPES):
                fields[f"retention_{kind}"] = pipeline.seed_stats(
                    [run[axis]["edit_retention"][group][kind]["fraction"] for run in runs])
            layouts = [run[axis]["layout"][group] for run in runs]
            if all("same_anchor_context" in layout for layout in layouts):
                fields["same_anchor_context"] = pipeline.seed_stats(
                    [layout["same_anchor_context"]["fraction"] for layout in layouts])
            aggregate[axis][group] = fields
    comparison = {name:benchmark["three_seed"][name] for name in ("simplified","matched_uniform","context_aware")}
    write_json(RESULTS/"summary.json",dict(seeds=list(SEEDS),selected_method=winner,methods=methods,
        selected_three_seed=aggregate,baselines=comparison,
        current_index_gradients={str(s):read_json(RESULTS/f"seed{s}"/"historical"/"gradients.json")["summary"]
                                 for s in SEEDS}))
    print("Completed index adaptation results for",winner,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",required=True,choices=("smoke","reference","fit","select","report"))
    parser.add_argument("--seed",type=int,choices=SEEDS,default=SEED)
    parser.add_argument("--methods",nargs="+",choices=METHODS)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage == "report":
        report()
        return
    if args.stage == "select":
        print(pipeline.valid_winner({m:RESULTS/f"seed{SEED}"/m for m in METHODS},RESULTS/"fine_policy.json"))
        return
    if torch.cuda.device_count()!=1 or "4090" not in torch.cuda.get_device_name():
        raise ValueError("Select one RTX 4090 via CUDA_VISIBLE_DEVICES")
    torch.backends.cudnn.deterministic = True
    tokenizer = load_tokenizer("unixcoder",UNIX_BASE)
    published_setup()
    if args.stage=="smoke":
        smoke(tokenizer)
        return
    methods = args.methods
    if args.stage=="fit":
        if args.seed!=SEED:
            method = read_json(RESULTS/"fine_policy.json")["selected"]
            if methods not in (None,[method]):
                parser.error("supplementary seeds must use the chosen Seed 42 method")
            methods = (method,)
        elif methods is None:
            parser.error("specify at least one Seed 42 method")
    splits = {name:pipeline.load_split(name,tokenizer) for name in ("train","valid")}
    if args.stage=="reference":
        references(tokenizer,splits)
    else:
        for method in methods:
            train_method(tokenizer,args.seed,method,splits)


if __name__=="__main__":
    main()
