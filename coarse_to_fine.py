"""Repair-guided coarse-to-fine classification on the original PrimeVul split."""
import argparse
import csv
from dataclasses import dataclass
import gc
import gzip
import math
import os
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import set_seed

from coarse_encoding import (BLOCK, INDEX_WIDTH, QUERY_HEADS, BlockData, CoarseClassifier, FusionClassifier,
                             collate_blocks, collate_fine, encode_blocks, encoder_window_ends, repair_block_mass,
                             retrieve)
from compaction import BUDGET, MAX_LENGTH, REGIONS, PackedData, pack_fixed
from context_selection import base_windows, merge, uniform_fill, visible_source, write_prediction_rows
from locator_quality import CONTEXT_BATCH, load_context_encoder
from paired import (BACKBONES, BATCH_SIZE, DATA, ROOT, SEED, UNIX_BASE, UNIX_RESULTS, classifier_config,
                    fit_classifier, load_tokenizer, LOSVERAdamW, predict, prime_source_rows, to_gpu)
from predicted_diff import HISTORICAL_REVISION, fp32_classifier, load_sources, paired_annotations
from region_locator import region_metrics
from repair_oracle import EDIT_TYPES, GAP_TYPES, grouped_edit_retention, packed_edit_retention, raw_runs
from rgcl.artifacts import read_json, write_json
from rgcl.config import SEEDS
from rgcl.metrics import paired_metrics
from window_inputs import GROUPS, grouped_metrics

RESULTS = ROOT / "results" / "coarse_to_fine_512"
WEIGHTS = ROOT / "checkpoints" / "coarse_to_fine_512"
METHODS = ("uniform", "task", "repair", "joint")
LAMBDAS = (.1, .5, 1.)


@dataclass(slots=True)
class SourceSplit:
    rows: list
    sources: list
    ends: list
    vectors: torch.Tensor
    encoding: dict
    source_seconds: float
    source_reload_seconds: float
    cache_read_seconds: float


def source_inputs(split,tokenizer):
    started = perf_counter()
    rows = prime_source_rows(DATA, split)
    sources = load_sources(rows, tokenizer)
    ends = [encoder_window_ends(s) for s in sources]
    return rows,sources,ends,perf_counter()-started


def load_split(split, tokenizer, *, build=False):
    rows,sources,ends,elapsed = source_inputs(split,tokenizer)
    source_reload = elapsed
    cache = WEIGHTS / "blocks" / f"{split}.pt"
    config = dict(dataset="PrimeVul_original", split=split, members=[r["sample_key"] for r in rows],
                  model=str(UNIX_BASE), revision=BACKBONES["unixcoder"][1], compression=HISTORICAL_REVISION,
                  block_size=BLOCK, windows=ends, batch=CONTEXT_BATCH, precision="BF16 SDPA / FP32 mean / FP16 cache")
    tokens = torch.from_numpy(np.concatenate([s.ids for s in sources]))
    cache_read = 0.
    if cache.is_file():
        started = perf_counter()
        saved = torch.load(cache, weights_only=True, mmap=True, map_location="cpu")
        if config != saved["config"] or not torch.equal(saved["tokens"], tokens):
            raise ValueError(f"Block cache does not match the current source inputs at {cache}")
        vectors, encoding = saved["vectors"], saved["encoding"]
        source_time = saved["source_seconds"]
        cache_read = perf_counter()-started
        elapsed = source_time
    else:
        if not build:
            raise FileNotFoundError(f"Build the shared {split} block cache in the coarse stage first")
        started = perf_counter()
        encoder = load_context_encoder().requires_grad_(False)
        torch.cuda.synchronize()
        initialization = perf_counter()-started
        vectors, encoding = encode_blocks(encoder, sources, tokenizer, ends)
        encoding["model_load_seconds"] = initialization
        del encoder
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_name(f".{cache.name}.{os.getpid()}.tmp")
        try:
            torch.save(dict(config=config, vectors=vectors, tokens=tokens, encoding=encoding, source_seconds=elapsed), temporary)
            temporary.replace(cache)
        finally:
            temporary.unlink(missing_ok=True)
        print(split, "cached", encoding["blocks"], "blocks in", round(encoding["seconds"],2), "s", flush=True)
    if vectors.dtype != torch.float16 or not torch.isfinite(vectors).all():
        raise ValueError(f"Invalid pooled values in {cache}")
    return SourceSplit(rows,sources,ends,vectors,encoding,elapsed,source_reload,cache_read)


def source_identity(split):
    return [a_end == b_end and np.array_equal(a.ids,b.ids)
            for a,b,a_end,b_end in zip(split.sources[::2],split.sources[1::2],split.ends[::2],split.ends[1::2],strict=True)]


def split_blocks(split, repair=None):
    return BlockData(split.rows, split.sources, split.vectors, repair_targets=repair)


@torch.inference_mode()
def score_coarse(model, split):
    model.eval()
    loader = DataLoader(split_blocks(split), batch_size=BATCH_SIZE, collate_fn=collate_blocks)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    probabilities, features, ranking = [], [], []
    for batch in loader:
        out = model(to_gpu(batch), details=True)
        probabilities.extend(out["probability"].tolist())
        features.append(out["coarse"].cpu())
        if "importance" in out:
            valid = batch["valid"].sum(1).tolist()
            ranking.extend(row[:n].numpy().copy() for n,row in zip(valid,out["importance"].cpu(),strict=True))
    return probabilities, torch.cat(features), ranking, dict(seconds=perf_counter()-started,
                              peak_allocated_bytes=torch.cuda.max_memory_allocated())


def coarse_fit(train, valid, seed, method, mode, weight, width):
    root, weights = RESULTS/f"seed{seed}"/"coarse"/method, WEIGHTS/f"seed{seed}"/"coarse"/method
    config = run_config(method, seed)
    config.update(stage="coarse", precision="FP32 on cached FP16 blocks", supervision=mode, repair_weight=weight,
                  loss="function BCE" if not weight else "function BCE and labeled-function mean KL",
                  repair_only_task_gradient="detached probability" if mode == "repair" else None)
    if (root/"fit.json").is_file():
        result = read_json(root/"fit.json")
        if result["config"] != config:
            raise ValueError(f"Existing coarse fit used a different configuration in {root}")
        return result
    set_seed(seed)
    model = CoarseClassifier(width, mode, weight)
    if mode != "global":
        location = weights.parent/"global"/"best.pt"
        previous = torch.load(location, weights_only=True,map_location="cpu")
        change = model.load_state_dict(previous, strict=False)
        if set(change.missing_keys) != {"indexer.query.weight","indexer.key.weight"} or change.unexpected_keys:
            raise ValueError("Coarse warm start did not match the global classifier")
    model = model.cuda()
    selected, history = fit_classifier(model, train, valid, weights, root, config, collator=collate_blocks)
    result = fitted(config, selected, history, len(valid))
    write_json(root/"fit.json", result)
    return result


def run_config(method, seed):
    config = classifier_config("uniform_sampling")
    config.update(method=method, seed=seed, model_id=BACKBONES["unixcoder"][0], model_revision=BACKBONES["unixcoder"][1],
                  block_size=BLOCK, query_heads=QUERY_HEADS, index_dimension=INDEX_WIDTH,
                  full_encoder_precision="BF16 SDPA", full_encoder_window_batch=CONTEXT_BATCH,
                  historical_input_revision=HISTORICAL_REVISION, split_functions=dict(train=8708,valid=1124,test=1128))
    return config


def fitted(config, selected, history, nvalid):
    return dict(config=config, selected_epoch=selected["epoch"], validation=selected["validation"],
                train_seconds=sum(row["training_seconds"] for row in history),
                validation_seconds=sum(row["validation_seconds"] for row in history),
                peak_train_allocated_bytes=max(row["peak_allocated_bytes"] for row in history),
                last_three_mean_train_loss=sum(row["train_loss"] for row in history[-3:])/3,
                one_class_valid_epochs=sum(row["validation"]["tp"]+row["validation"]["fp"] in (0,nvalid) for row in history))


def load_coarse(seed, name, width):
    root = RESULTS/f"seed{seed}"/"coarse"/name
    fit = read_json(root/"fit.json")
    config = fit["config"]
    weights = WEIGHTS/f"seed{seed}"/"coarse"/name/"best.pt"
    model = CoarseClassifier(width, config["supervision"], config["repair_weight"])
    model.load_state_dict(torch.load(weights, weights_only=True,map_location="cpu"), strict=True)
    return model.cuda().eval(), fit


def prepare(tokenizer, seed, fixed_lambda):
    root = RESULTS/f"seed{seed}"
    train, valid = [load_split(split,tokenizer,build=True) for split in ("train","valid")]
    train_data, valid_data = map(split_blocks,(train,valid))
    width = train.vectors.shape[-1]
    coarse_fit(train_data,valid_data,seed,"global","global",0.,width)
    edits = paired_annotations(train.sources)
    masses = [repair_block_mass(s,a,(*EDIT_TYPES,*GAP_TYPES)) for s,a in zip(train.sources,edits,strict=True)]
    target = [count/count.sum().clamp_min(1) for count in masses]
    supervised = split_blocks(train,target)
    if fixed_lambda is None:
        coarse_fit(train_data,valid_data,seed,"task","task",0.,width)
        coarse_fit(supervised,valid_data,seed,"repair","repair",1.,width)
    weights = LAMBDAS if fixed_lambda is None else (fixed_lambda,)
    trials = []
    for weight in weights:
        name = f"joint_{weight:.1f}"
        fit = coarse_fit(supervised,valid_data,seed,name,"joint",weight,width)
        trials.append(dict(name=name, weight=weight, epoch=fit["selected_epoch"], valid_mcc=fit["validation"]["mcc"]))
    selected = max(trials,key=lambda t:(t["valid_mcc"],-t["weight"]))
    policy = dict(seed=seed, trials=trials, selected=selected, selection="coarse Valid MCC then smaller lambda",
                  fixed_lambda=fixed_lambda, train_functions_without_supervision=sum(not t.any().item() for t in target))
    write_json(root/"policy.json",policy)
    print("selected lambda",selected,flush=True)
    # Freeze the policy before loading any Test source or paired annotations.
    test = load_split("test",tokenizer,build=True)
    write_json(root/"cache_costs.json", {k:dict(source_seconds=s.source_seconds,
              **{n:v for n,v in s.encoding.items() if n!="per_function_padded_positions"})
              for k,s in (("train",train),("valid",valid),("test",test))})
    path = root/"global"/"metrics.json"
    if not path.is_file():
        model, fit = load_coarse(seed,"global",width)
        result = dict(fit, split={})
        for name,split in (("valid",valid),("test",test)):
            scores, _, ranks, cost = score_coarse(model,split)
            result["split"][name] = evaluate(split,scores,path.parent,name,cost)
        verify_valid(result)
        write_json(path,result)


def fine_data(split, coarse, scores, tokenizer, uniform):
    def examples():
        blank = torch.full((REGIONS,4),-1,dtype=torch.int32)
        for source, f, ranking in zip(split.sources,coarse,scores,strict=True):
            length = len(source.ids)
            if uniform:
                if length <= BUDGET:
                    spans = [(0,length)]
                else:
                    base, limits = base_windows(source.boundaries)
                    spans = uniform_fill(base,source.boundaries,limits,contiguous=False)
            else:
                spans = retrieve(ranking,length)
            spans = merge(spans)
            fixed, n = pack_fixed(source.raw_ids,source.keep,spans,tokenizer)
            if n != min(MAX_LENGTH,length+2):
                raise ValueError("Fine encoder input does not fill the historical budget")
            yield dict(fixed=fixed,length=n,raw=len(source.raw_ids),compacted_bpe=length,coarse=f,
                       uniform=blank,bank=None,encoding_runs=spans)
    return PackedData(split.rows,examples())


def pair_results(rows, scores, lengths, global_same, fine_same=None):
    groups = {}
    for group,(lo,hi) in GROUPS.items():
        ids = [i for i in range(0,len(rows),2) if lo<=max(lengths[i:i+2])<=hi]
        subset = [i for start in ids for i in (start,start+1)]
        if subset:
            report = paired_metrics([rows[i] for i in subset],[scores[i] for i in subset])
            coarse_hits = sum(global_same[i//2] for i in ids)
            report["identical_global_inputs"] = coarse_hits
            report["identical_complete_inputs"] = coarse_hits if fine_same is None else sum(
                global_same[i//2] and fine_same[i//2] for i in ids)
            if fine_same is not None:
                report["identical_fine_inputs"] = sum(fine_same[i//2] for i in ids)
            groups[group] = report
    return groups


def index_diagnostics(split, scores, edits):
    result = {}
    actual = [repair_block_mass(s,a,EDIT_TYPES).numpy()>0 for s,a in zip(split.sources,edits,strict=True)]
    weak = {kind:[repair_block_mass(s,a,(kind,)).numpy()>0 for s,a in zip(split.sources,edits,strict=True)]
            for kind in GAP_TYPES}
    lengths = [len(s.raw_ids) for s in split.sources]
    for name,labels in dict(actual=actual,**weak).items():
        groups = {}
        for group,(lo,hi) in GROUPS.items():
            indices = [i for i,n in enumerate(lengths) if lo<=n<=hi]
            predicted, y = [scores[i] for i in indices], [labels[i] for i in indices]
            positives, total = sum(int(v.sum()) for v in y), sum(map(len,y))
            groups[group] = (region_metrics(predicted,y,ks=(1,5,10,15)) if 0<positives<total else
                             dict(functions=len(y), regions=total, positive_regions=positives, ap=None))
        result[name] = groups
    return result


def evaluate(split, scores, root, name, coarse_cost, *, data=None, ranks=None, retrieval=0., fine_cost=None):
    rows, sources = split.rows, split.sources
    lengths = [len(s.raw_ids) for s in sources]
    fine_same = None
    global_same = source_identity(split)
    if data is not None:
        ids = [ex["fixed"][0,:ex["length"]].tolist() for ex in data.examples]
        fine_same = [a==b for a,b in zip(ids[::2],ids[1::2],strict=True)]
    records = []
    for i,(row,source,p) in enumerate(zip(rows,sources,scores,strict=True)):
        record = dict(idx=row["idx"],sample_key=row["sample_key"],pair_key=row["pair_key"],target=row["target"],
                      original_bpe=lengths[i],compacted_bpe=len(source.ids),probability=p,prediction=int(p>.5),
                      global_blocks=math.ceil(len(source.ids)/BLOCK),global_windows=len(split.ends[i]),
                      global_input_positions=len(source.ids)+2*len(split.ends[i]),
                      global_padded_positions=split.encoding["per_function_padded_positions"][i],
                      identical_global_pair_input=global_same[i//2])
        if data is not None:
            ex = data.examples[i]
            spans = ex["encoding_runs"]
            record.update(input_length=ex["length"],encoding_runs=spans,source_runs=visible_source(ex),
                          identical_fine_pair_input=fine_same[i//2],
                          identical_complete_pair_input=global_same[i//2] and fine_same[i//2])
        records.append(record)
    root.mkdir(parents=True,exist_ok=True)
    write_prediction_rows(root/f"{name}.csv.gz",records)
    # Paired repair annotations enter only after this checkpoint's predictions are written.
    annotations = paired_annotations(sources)
    full_edit = grouped_edit_retention((n,a,raw_runs(s.keep,[(0,len(s.ids))]))
                                      for n,a,s in zip(lengths,annotations,sources,strict=True))
    fine = {} if fine_cost is None else fine_cost
    enc = {k:v for k,v in split.encoding.items() if k!="per_function_padded_positions"}
    costs = dict(source_seconds=split.source_seconds,global_encoding=enc,coarse_scoring=coarse_cost,
                 retrieval_seconds=retrieval,fine_encoding=fine,
                 uncached_pipeline_seconds=split.source_seconds+enc["seconds"]+coarse_cost["seconds"]+retrieval+fine.get("seconds",0.),
                 cached_inference_seconds=coarse_cost["seconds"]+retrieval+fine.get("seconds",0.),
                 peak_allocated_bytes=max(enc["peak_allocated_bytes"],coarse_cost["peak_allocated_bytes"],fine.get("peak_allocated_bytes",0)))
    for key in ("windows","batch_calls","unpadded_positions","padded_positions"):
        costs[key] = enc[key]+fine.get(key,0)
    group_data = SimpleNamespace(metadata=records, targets=torch.tensor([r["target"] for r in rows]))
    result = dict(groups=grouped_metrics(group_data,scores),
                  pairs=pair_results(rows,scores,lengths,global_same,fine_same),cost=costs,global_edit_coverage=full_edit)
    if data is not None:
        result["fine_edit_retention"] = packed_edit_retention(data,annotations)
    if ranks:
        result["indexer"] = index_diagnostics(split,ranks,annotations)
    return result


def predict_fine(model, data):
    loader = DataLoader(data,batch_size=BATCH_SIZE,collate_fn=collate_fine)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    scores,_ = predict(model,loader)
    return scores,dict(seconds=perf_counter()-started,windows=len(data),batch_calls=len(loader),
                       unpadded_positions=sum(x["length"] for x in data.examples),padded_positions=len(data)*MAX_LENGTH,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated())


def uncached_joint(model, tokenizer, seed, index_name, cached, expected):
    """Time the full Test path once with both PLMs resident and verify output parity."""
    encoder = load_context_encoder().requires_grad_(False)
    coarse,_ = load_coarse(seed,index_name,encoder.config.hidden_size)
    torch.cuda.synchronize()
    started = perf_counter()
    rows,sources,ends,source_time = source_inputs("test",tokenizer)
    vectors,encoding = encode_blocks(encoder,sources,tokenizer,ends)
    split = SourceSplit(rows,sources,ends,vectors,encoding,source_time,source_time,0.)
    _,features,ranks,index = score_coarse(coarse,split)
    retrieval = perf_counter()
    data = fine_data(split,features,ranks,tokenizer,False)
    retrieval = perf_counter()-retrieval
    scores,fine = predict_fine(model,data)
    total = perf_counter()-started
    changes = sum(not torch.equal(a["fixed"],b["fixed"]) for a,b in zip(data.examples,cached.examples,strict=True))
    largest_error = max(abs(a-b) for a,b in zip(scores,expected,strict=True))
    if changes or largest_error>1e-6:
        raise ValueError(f"Uncached Test changed {changes} fine inputs, max probability error {largest_error}")
    return dict(seconds=total,source_seconds=source_time,global_encoding={
                k:v for k,v in encoding.items() if k!="per_function_padded_positions"},coarse_scoring=index,
                retrieval_seconds=retrieval,fine_encoding=fine,max_abs_probability_difference=largest_error,
                peak_allocated_bytes=max(encoding["peak_allocated_bytes"],index["peak_allocated_bytes"],fine["peak_allocated_bytes"]))


def verify_valid(result):
    if result["validation"] != result["split"]["valid"]["groups"]["overall"]:
        raise ValueError("Reloaded model does not reproduce the selected validation checkpoint")


def fine_run(tokenizer, seed, methods):
    root = RESULTS/f"seed{seed}"
    policy = read_json(root/"policy.json")
    splits = {name:load_split(name,tokenizer) for name in ("train","valid","test")}
    width = splits["train"].vectors.shape[-1]
    index_names = {m:(policy["selected"]["name"] if m in ("uniform","joint") else m) for m in methods}
    representations = {}
    for method in methods:
        area = root/method
        path = area/"metrics.json"
        index_name = index_names[method]
        config = run_config(method,seed)
        config.update(stage="fine", index_name=index_name,
                      selection_policy="historical centered Uniform Sampling" if method=="uniform" else "ranked 32-BPE blocks",
                      coarse_frozen=True, fusion="CLS plus zero-initialized projection of frozen coarse representation")
        if path.is_file():
            if read_json(path)["config"] != config:
                raise ValueError(f"Existing fine model used a different configuration in {area}")
            continue
        if index_name not in representations:
            model, fit = load_coarse(seed,index_name,width)
            representations[index_name] = {name:score_coarse(model,s) for name,s in splits.items()}
            del model
        reps = representations[index_name]
        datasets, retrieval = {}, {}
        for name,split in splits.items():
            _, features, ranks, _ = reps[name]
            started = perf_counter()
            datasets[name] = fine_data(split,features,ranks,tokenizer,method=="uniform")
            retrieval[name] = perf_counter()-started
        set_seed(seed)
        model = FusionClassifier(fp32_classifier(tokenizer).detector).cuda()
        selected, history = fit_classifier(model,datasets["train"],datasets["valid"],
                    WEIGHTS/f"seed{seed}"/method,area,config,collator=collate_fine)
        result = fitted(config,selected,history,len(datasets["valid"]))
        result.update(split={},cached_preprocessing={name:dict(
             source_reload_seconds=split.source_reload_seconds, cache_read_seconds=split.cache_read_seconds,
             coarse_scoring=representations[index_name][name][3], retrieval_seconds=retrieval[name])
             for name,split in splits.items()})
        for name in ("valid","test"):
            _, features, ranks, timing = reps[name]
            scores, fine = predict_fine(model,datasets[name])
            if method == "joint":
                started = perf_counter()
                swapped = fine_data(splits[name],features,ranks,tokenizer,True)
                swap_time = perf_counter()-started
                swapped_p, swapped_cost = predict_fine(model,swapped)
                if name=="test":
                    result["uncached_test_measurement"] = uncached_joint(
                        model,tokenizer,seed,index_name,datasets[name],scores)
            result["split"][name] = evaluate(splits[name],scores,area,name,timing,data=datasets[name],ranks=ranks,
                                            retrieval=retrieval[name],fine_cost=fine)
            if method == "joint":
                result.setdefault("same_checkpoint_uniform",{})[name] = evaluate(splits[name],swapped_p,
                    area/"same_checkpoint_uniform",name,timing,data=swapped,retrieval=swap_time,fine_cost=swapped_cost)
        verify_valid(result)
        write_json(path,result)
        print(method,"complete",seed,result["split"]["test"]["groups"]["overall"]["mcc"],flush=True)
        del model, datasets
        gc.collect()


def historical():
    root = UNIX_RESULTS/"uniform_sampling"
    result = read_json(root/"metrics.json")
    for split in ("valid","test"):
        rows = prime_source_rows(DATA,split)
        with gzip.open(root/f"{split}.csv.gz","rt") as file:
            saved = list(csv.DictReader(file))
        if any(r["sample_key"]!=p["sample_key"] or r["target"]!=int(p["target"]) for r,p in zip(rows,saved,strict=True)):
            raise ValueError("Historical Uniform Sampling membership differs")
        section = result["split"][split]
        scores = [float(p["probability"]) for p in saved]
        lengths = [int(p["original_bpe"]) for p in saved]
        # Pair-classification metrics also apply to the historical saved predictions.
        section["pairs"] = {g: {k:v for k,v in report.items() if "identical_" not in k}
                            for g,report in pair_results(rows,scores,lengths,[False]*(len(rows)//2)).items()}
    return result


def summarize(seed):
    root = RESULTS/f"seed{seed}"
    methods = {m:read_json(root/m/"metrics.json") for m in ("global",*METHODS) if (root/m/"metrics.json").is_file()}
    write_json(root/"summary.json",dict(seed=seed, policy=read_json(root/"policy.json"),
                                      archived_uniform_sampling_seed42=historical(),methods=methods))


def smoke(tokenizer):
    set_seed(SEED)
    rows = prime_source_rows(DATA,"train")
    longest = max(range(len(rows)),key=lambda i:len(rows[i]["func"]))//2*2
    shortest = min(range(0,len(rows),2),key=lambda i:max(len(rows[i]["func"]),len(rows[i+1]["func"])))
    rows = [*rows[longest:longest+2],*rows[shortest:shortest+2]]
    sources = load_sources(rows,tokenizer)
    ends = [encoder_window_ends(s) for s in sources]
    encoder = load_context_encoder().requires_grad_(False)
    bank, encoding = encode_blocks(encoder,sources,tokenizer,ends)
    del encoder
    if max(map(len,ends))<=1 or encoding["body_bpe"]!=sum(len(s.ids) for s in sources):
        raise ValueError("GPU smoke did not cover the whole extreme source")
    split = SourceSplit(rows,sources,ends,bank,encoding,0.,0.,0.)
    del rows
    targets = [repair_block_mass(s,e,(*EDIT_TYPES,*GAP_TYPES))
               for s,e in zip(sources,paired_annotations(sources),strict=True)]
    data = split_blocks(split,[q/q.sum().clamp_min(1) for q in targets])
    gradient_stats = {}
    for mode, weight in (("task",0.),("repair",1.),("joint",.5)):
        model = CoarseClassifier(bank.shape[-1],mode,weight).cuda()
        parameters = list(model.indexer.parameters())
        batch = to_gpu(collate_blocks([data[i] for i in range(len(data))]),labels=True)
        report = model(batch,details=True)
        task_grad = torch.autograd.grad(report["loss_components"]["train_bce"],parameters,allow_unused=True,retain_graph=True)
        task_norm = sum(float(x.norm()) for x in task_grad if x is not None)
        before = parameters[0].detach().clone()
        report["loss"].backward()
        loss_norm = float(parameters[0].grad.norm())
        opt = LOSVERAdamW(model.parameters(),lr=2e-5)
        opt.step()
        if (task_norm == 0.) != (mode=="repair") or loss_norm<=0 or torch.equal(before,parameters[0]):
            raise ValueError(f"Unexpected indexer gradient or parameter update in {mode}")
        gradient_stats[mode] = dict(task_gradient=task_norm, training_gradient=loss_norm, loss=float(report["loss"].detach()))
    coarse_p, features, ranks, _ = score_coarse(model,split)
    layouts = fine_data(split,features,ranks,tokenizer,False)
    replay = fine_data(split,*score_coarse(model,split)[1:3],tokenizer,False)
    if any(not torch.equal(a["fixed"],b["fixed"]) for a,b in zip(layouts.examples,replay.examples,strict=True)):
        raise ValueError("Source-only inference changed hard retrieval")
    fine = FusionClassifier(fp32_classifier(tokenizer).detector).cuda()
    batch = to_gpu(collate_fine(layouts.examples),labels=True)
    out = fine(batch)
    out["loss"].backward()
    fusion_gradient = float(fine.fusion.weight.grad.norm())
    if not math.isfinite(fusion_gradient) or fusion_gradient<=0 or features.requires_grad:
        raise ValueError("Frozen coarse features or the fine fusion gradient are invalid")
    optimizer = LOSVERAdamW(fine.parameters(),lr=2e-5)
    optimizer.step()
    print("GPU smoke passed",encoding["body_bpe"],"source BPE",flush=True)
    write_json(RESULTS/"smoke.json",dict(original_bpe=[len(s.raw_ids) for s in sources],
              compacted_bpe=[len(s.ids) for s in sources],encoding=encoding,indexer_gradients=gradient_stats,
              fine_lengths=[x["length"] for x in layouts.examples],fine_loss=float(out["loss"].detach()),
              fusion_gradient=fusion_gradient,coarse_source_probabilities=coarse_p,
              cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed",type=int,choices=SEEDS,default=SEED)
    parser.add_argument("--stage",choices=("all","coarse","fine","report","smoke"),default="all")
    parser.add_argument("--methods",nargs="+",choices=METHODS,default=METHODS)
    parser.add_argument("--fixed-lambda",type=float,choices=LAMBDAS,help="Freeze the main-seed choice for supplementary seeds.")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage=="report":
        summarize(args.seed)
        return
    if torch.cuda.device_count()!=1 or "4090" not in torch.cuda.get_device_name():
        raise ValueError("Select one RTX 4090 via CUDA_VISIBLE_DEVICES")
    torch.backends.cudnn.deterministic = True
    tokenizer = load_tokenizer("unixcoder",UNIX_BASE)
    if args.stage=="smoke":
        smoke(tokenizer)
        return
    if args.stage in ("all","coarse"):
        prepare(tokenizer,args.seed,args.fixed_lambda)
    if args.stage in ("all","fine"):
        fine_run(tokenizer,args.seed,args.methods)
    if args.stage=="all":
        summarize(args.seed)


if __name__=="__main__":
    main()
