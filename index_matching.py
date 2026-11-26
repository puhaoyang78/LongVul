"""Compare block and Token matching with the published LongVul retrieval and fine encoder."""
import argparse
from collections import Counter
import gc
import json
from itertools import pairwise
from statistics import mean
from time import perf_counter

import numpy as np
import torch
from transformers import set_seed

import coarse_to_fine as pipeline
from coarse_encoding import (BLOCK, BlockData, CoarseClassifier, FusionClassifier, collate_blocks, collate_fine,
                             encode_blocks, repair_block_mass)
from coarse_retrieval import lengths
from compaction import BUDGET, MAX_LENGTH
from context_selection import merge
from index_adaptation import grouped_layout_change, published_setup, reference_runs
from locator_quality import context_encoder_batches, load_context_encoder
from paired import AdamW, DATA, ROOT, SEED, UNIX_BASE, fit_classifier, load_tokenizer, prime_source_rows, to_gpu
from predicted_diff import fp32_classifier, load_sources, paired_annotations
from repair_oracle import EDIT_TYPES, GAP_TYPES, overlap_bpe
from rgcl.artifacts import read_json, write_json
from rgcl.config import SEEDS
from rgcl.metrics import metrics, paired_metrics
from window_inputs import GROUPS

RESULTS = ROOT/"results"/"fine_indexing_512"
WEIGHTS = ROOT/"checkpoints"/RESULTS.name
MATCHING = dict(B0="block", B1="token_mean", B2="token_logmeanexp")
METHOD_BY_MATCHING = {matching: method for method,matching in MATCHING.items()}
REPAIR_WEIGHT = 1.


def token_cache(split, name, tokenizer, *, encoder=None):
    started = perf_counter()
    path = WEIGHTS/"tokens"/f"{name}.npy"
    metadata = path.with_suffix(".json")
    shape = (sum(len(source.ids) for source in split.sources),split.vectors.shape[-1])
    provenance = dict(split=name, members=[r["sample_key"] for r in split.rows], windows=split.windows,
                      model_revision=pipeline.BACKBONES["unixcoder"][1], simple=pipeline.HISTORICAL_REVISION,
                      window_batch=pipeline.CONTEXT_BATCH, precision="BF16 SDPA stored in FP16")
    if encoder is not None:
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary = path.with_suffix(".tmp.npy")
        mmap = np.lib.format.open_memmap(temporary,mode="w+",dtype=np.float16,shape=shape)
        stored = torch.from_numpy(mmap)
        pooled,encoded = encode_blocks(encoder,split.sources,tokenizer,split.windows,split.ends,token_features=stored)
        started_flush = perf_counter()
        mmap.flush()
        del stored,mmap
        temporary.replace(path)
        flush_seconds = perf_counter()-started_flush
        difference = (pooled.float()-split.vectors.float()).abs()
        timing = {k:v for k,v in encoded.items() if k!="per_function_padded_positions"}
        write_json(metadata,dict(config=provenance,timing=dict(timing,cache_flush_seconds=flush_seconds,
                         build_seconds=perf_counter()-started,cache_file_bytes=path.stat().st_size),
                         archived_block_max_abs_diff=difference.max().item(),archived_block_mean_abs_diff=difference.mean().item()))
        print(name,"Token cache",timing["body_bpe"],"BPE",path.stat().st_size,"bytes",flush=True)
        return
    config = read_json(metadata)
    if config["config"] != provenance:
        raise ValueError(f"Token cache provenance differs for {name}")
    values = np.load(path,mmap_mode="c",allow_pickle=False)
    if values.shape!=shape or values.dtype!=np.float16 or path.stat().st_size!=config["timing"]["cache_file_bytes"]:
        raise ValueError(f"Token cache size or dtype differs for {name}")
    split.token_features = torch.from_numpy(values)
    split.cache_read_seconds += perf_counter()-started


def cache(tokenizer):
    encoder = load_context_encoder().requires_grad_(False)
    costs = {}
    for name in ("train","valid"):
        split = pipeline.load_split(name,tokenizer)
        path = WEIGHTS/"tokens"/f"{name}.json"
        token_cache(split,name,tokenizer,encoder=None if path.is_file() else encoder)
        costs[name] = {k:v for k,v in read_json(path).items() if k!="config"}
    write_json(RESULTS/"token_cache.json",costs)


def fit_index(train,valid,seed,method):
    width = train.vectors.shape[-1]
    matching = MATCHING[method]
    root,weights = (pipeline.seed_root(parent,seed)/"coarse" for parent in (RESULTS,WEIGHTS))
    if (root/method/"fit.json").is_file():
        return read_json(root/method/"fit.json")
    labels = []
    for source,ends,edits in zip(train.sources,train.ends,paired_annotations(train.sources),strict=True):
        mass = repair_block_mass(source,ends,edits,(*EDIT_TYPES,*GAP_TYPES))
        labels.append(mass/mass.sum().clamp_min(1))
    for name,supervision,weight,token_match in (
        ("global","global",0.,"block"), (method,"joint",REPAIR_WEIGHT,matching)
    ):
        area,location = root/name,weights/name
        if (area/"fit.json").is_file():
            continue
        use_tokens = token_match!="block"
        validation = BlockData(valid.rows,valid.ends,valid.vectors,token_features=valid.token_features if use_tokens else None)
        data = BlockData(train.rows,train.ends,train.vectors,labels if weight else None,
                         token_features=train.token_features if use_tokens else None)
        config = dict(pipeline.run_config(name,seed),stage="coarse",block_layout="fixed32",
                      matching=token_match,supervision=supervision,repair_weight=weight,
                      precision="FP32 on cached FP16 states",initialization="fresh seeded weights",
                      loss="function BCE + labeled function mean KL" if weight else "function BCE")
        set_seed(seed)
        model = CoarseClassifier(width,supervision,weight,matching=token_match)
        if weight:
            config["initialization"] = "fresh Q/K; auxiliary head from newly trained same-seed global fit"
            update = model.load_state_dict(torch.load(weights/"global"/"best.pt",weights_only=True,map_location="cpu"),
                                           strict=False)
            if set(update.missing_keys)!={"indexer.query.weight","indexer.key.weight"} or update.unexpected_keys:
                raise ValueError("Newly trained auxiliary head did not match the joint model")
        best,history = fit_classifier(model.cuda(),data,validation,location,area,config,collator=collate_blocks)
        write_json(area/"fit.json",pipeline.fitted(config,best,history,len(validation)))
        del model
        gc.collect()
    return read_json(root/method/"fit.json")


def run(tokenizer,seed,method):
    area,weights = (pipeline.seed_root(parent,seed)/method for parent in (RESULTS,WEIGHTS))
    if (area/"metrics.json").is_file():
        return
    train,valid = [pipeline.load_split(name,tokenizer) for name in ("train","valid")]
    matching = MATCHING[method]
    if matching!="block":
        for name,split in (("train",train),("valid",valid)):
            token_cache(split,name,tokenizer)
    index_fit = fit_index(train,valid,seed,method)
    state = weights.parent/"coarse"/method/"best.pt"
    index = CoarseClassifier(train.vectors.shape[-1],"joint",matching=matching)
    index.load_state_dict(torch.load(state,weights_only=True,map_location="cpu"),strict=True)
    index.cuda().eval().requires_grad_(False)
    config = dict(pipeline.run_config(method,seed),**published_setup(),matching=matching,
                  index_selected_epoch=index_fit["selected_epoch"],repair_weight=REPAIR_WEIGHT,
                  stage="fine",precision="FP32 Eager",loss="function BCE",
                  initialization="original pretrained UniXcoder; fresh Global Fusion and classification head")
    pipeline.fit_frozen_fine(tokenizer,index,config,area,weights,cached=dict(train=train,valid=valid))


def choose():
    policy = pipeline.valid_winner(
        {name:pipeline.seed_root(RESULTS,SEED)/name for name in ("B1","B2")},
        RESULTS/"matching_policy.json")
    print(MATCHING[policy],flush=True)
    return policy


def detection_audit(rows,sources,ends,edits,predictions,reference,retrained_block=None):
    raw = [len(source.raw_ids) for source in sources]
    codebert = [int(p["codebert_bpe"]) for p in predictions]
    visible = [json.loads(p["source_runs"]) for p in predictions]
    centers = [json.loads(p["center_runs"]) for p in predictions]
    scores = [float(p["probability"]) for p in predictions]
    modified = [merge(span for kind in EDIT_TYPES for span in e[kind]) for e in edits]
    totals = [lengths(e) for e in modified]
    retained = [overlap_bpe(a,b) for a,b in zip(visible,modified,strict=True)]
    block_rows = []
    for source,cut,edit,runs,anchor in zip(sources,ends,edits,visible,centers,strict=True):
        counts = Counter()
        if len(source.ids)>BUDGET:
            mass = repair_block_mass(source,cut,edit,EDIT_TYPES).int().tolist()
            for (a,b),n in zip(pairwise([0,*cut]),mass,strict=True):
                if b-a!=BLOCK:
                    continue
                label = "0" if n==0 else "1-2" if n<=2 else "3-4" if n<=4 else "5-8" if n<=8 else "9-32"
                first,last = source.keep[a],source.keep[b-1]+1
                selected = overlap_bpe(runs,[(first,last)])
                counts[label,"blocks"] += 1
                counts[label,"centers"] += int(overlap_bpe(anchor,[(a,b)])==BLOCK)
                counts[label,"visible_any"] += int(selected>0)
                counts[label,"visible_full"] += int(selected==BLOCK)
                counts[label,"visible_bpe"] += selected
        block_rows.append(counts)
    result = {}
    for name,sizes in (("unixcoder",raw),("codebert",codebert)):
        omissions,blocks,small = {},{},{}
        for group,(a,b) in GROUPS.items():
            subset = [i for i,n in enumerate(sizes) if a<=n<=b]
            eligible = [i for i in subset if totals[i]]
            missed = sum(retained[i]==0 for i in eligible)
            omissions[group] = dict(with_actual_edits=len(eligible),missed=missed,
                                    fraction=missed/len(eligible) if eligible else None)
            counts = Counter()
            for i in subset:
                counts.update(block_rows[i])
            blocks[group] = {}
            for label in ("0","1-2","3-4","5-8","9-32"):
                entry = {kind:counts[label,kind] for kind in
                         ("blocks","centers","visible_any","visible_full","visible_bpe")}
                denominator = entry["blocks"]
                entry["center_rate"] = entry["centers"]/denominator if denominator else None
                entry["any_coverage_rate"] = entry["visible_any"]/denominator if denominator else None
                if label!="0":
                    matched = [block_rows[i] for i in subset
                               if block_rows[i][label,"blocks"] and block_rows[i]["0","blocks"]]
                    entry["same_function"] = dict(
                        functions=len(matched),
                        center_rate=mean(x[label,"centers"]/x[label,"blocks"] for x in matched) if matched else None,
                        unmodified_center_rate=mean(x["0","centers"]/x["0","blocks"] for x in matched) if matched else None)
                blocks[group][label] = entry
            small[group] = {}
            pairs = [i for i in range(0,len(sizes),2) if a<=max(sizes[i:i+2])<=b]
            for lower,upper in ((1,8),(9,16),(17,32),(33,float("inf"))):
                members = [i for start in pairs if lower<=sum(totals[start:start+2])<=upper for i in (start,start+1)]
                if not members:
                    continue
                available = sum(totals[i] for i in members)
                hits = sum(retained[i] for i in members)
                labels = [rows[i]["target"] for i in members]
                probabilities = [scores[i] for i in members]
                small[group][f"{lower}-{upper:g}"] = dict(
                    classification=metrics(labels,probabilities),
                    paired=paired_metrics([rows[i] for i in members],probabilities),
                    retained_bpe=hits,available_bpe=available,retention=hits/available,
                    identical_fine_inputs=sum(predictions[i]["identical_fine_pair_input"]=="True" for i in members[::2]))
        result[name] = dict(omissions=omissions,full_32_bpe_blocks=blocks,paired_edit_size=small,
                            layout_jaccard_vs_published=grouped_layout_change(reference,visible,sizes))
        if retrained_block is not None:
            result[name]["layout_jaccard_vs_B0"] = grouped_layout_change(retrained_block,visible,sizes)
    return result


def report(tokenizer):
    selected = read_json(RESULTS/"matching_policy.json")["selected"]
    runs = {seed:{name:read_json(pipeline.seed_root(RESULTS,seed)/name/"metrics.json")
                   for name in (MATCHING if seed==SEED else ("B0",selected))}
            for seed in SEEDS}
    for split in ("valid","test"):
        rows,sources,_,_ = pipeline.source_inputs(split,tokenizer)
        ends = pipeline.cuts_for(sources,None,False)
        edits = paired_annotations(sources)
        for seed,methods in runs.items():
            root = pipeline.seed_root(RESULTS,seed)
            historical = reference_runs(seed,split,rows)
            block = pipeline.read_predictions(root/"B0"/f"{split}.csv.gz",rows)
            b0_runs = [json.loads(p["source_runs"]) for p in block]
            for name,run in methods.items():
                preds = block if name=="B0" else pipeline.read_predictions(root/name/f"{split}.csv.gz",rows)
                result = detection_audit(rows,sources,ends,edits,preds,historical,b0_runs if name!="B0" else None)
                write_json(root/name/f"{split}_retrieval.json",result)
                labels = [r["target"] for r in rows]
                scores = [float(p["probability"]) for p in preds]
                if metrics(labels,scores)!=run["split"][split]["unixcoder"]["groups"]["overall"]:
                    raise ValueError("Saved predictions do not reproduce the recorded metrics")
    keys = ("mcc","f1","precision","recall","auc","auprc")
    pair_keys = ("p_c","pair_ranking_accuracy")
    average = {}
    for name in ("B0",selected):
        average[name] = {
            kind:{group:{key:pipeline.seed_stats([runs[seed][name]["split"]["test"][kind][section][group][key]
                                                for seed in SEEDS])
                         for section,fields in (("groups",keys),("pairs",pair_keys)) for key in fields}
                  for group in GROUPS}
            for kind in ("unixcoder","codebert")}
        for kind in ("unixcoder","codebert"):
            for group, entry in average[name][kind].items():
                results = [runs[seed][name]["split"]["test"][kind] for seed in SEEDS]
                for label in ("actual_edits",*GAP_TYPES):
                    entry[f"retention_{label}"] = pipeline.seed_stats(
                        [r["edit_retention"][group][label]["fraction"] for r in results])
                entry["fine_pair_collision_rate"] = pipeline.seed_stats(
                    [r["layout"][group]["fine_pair_collision_rate"] for r in results])
                entry["same_anchor_context"] = pipeline.seed_stats(
                    [r["layout"][group].get("same_anchor_context",{}).get("fraction") for r in results])
        average[name]["cost"] = {
            key:pipeline.seed_stats([getter(runs[seed][name]) for seed in SEEDS])
            for key,getter in (
                ("fine_train_seconds",lambda r:r["train_seconds"]),
                ("end_to_end_seconds",lambda r:r["split"]["test"]["cost"]["measured_end_to_end_seconds"]),
                ("peak_inference_allocated_bytes",lambda r:r["split"]["test"]["cost"]["peak_inference_allocated_bytes"]),
                ("index_scoring_seconds",lambda r:r["split"]["test"]["cost"]["index_scoring"]["seconds"]))}
        index_fits = [read_json(pipeline.seed_root(RESULTS,seed)/"coarse"/name/"fit.json") for seed in SEEDS]
        for key in ("train_seconds","validation_seconds","peak_train_allocated_bytes"):
            average[name]["cost"][f"index_{key}"] = pipeline.seed_stats([fit[key] for fit in index_fits])
    write_json(RESULTS/"summary.json",dict(selected=selected,seeds=list(SEEDS),three_seed=average,
                                          per_seed={seed:list(methods) for seed,methods in runs.items()}))


def smoke(tokenizer):
    rows = prime_source_rows(DATA,"train")
    longest = max(range(len(rows)),key=lambda i:len(rows[i]["func"]))//2*2
    rows = rows[longest:longest+2]
    sources = load_sources(rows,tokenizer)
    ends = pipeline.cuts_for(sources,None,False)
    windows = [pipeline.encoder_window_ends(s) for s in sources]
    encoder = load_context_encoder().requires_grad_(False)
    states = torch.empty((sum(len(s.ids) for s in sources),encoder.config.hidden_size),dtype=torch.float16)
    blocks,cost = encode_blocks(encoder,sources,tokenizer,windows,ends,token_features=states)
    if max(map(len,windows))<2 or cost["body_bpe"]!=len(states) or not torch.isfinite(states).all():
        raise ValueError("Entire extreme function must have finite frozen states")
    source = sources[0]
    torch.testing.assert_close(torch.from_numpy(source.raw_ids[source.keep]),torch.from_numpy(source.ids))
    requests = [(source,a,b,0) for a,b in zip([0,*windows[0][:-1]],windows[0])]
    batch,hidden,_ = next(context_encoder_batches(encoder,tokenizer,iter(requests),prefix_sum=False))
    for row,(_,a,b,_) in enumerate(batch):
        torch.testing.assert_close(states[a:b],hidden[row,1:1+b-a].cpu().half(),atol=0.,rtol=0.)
    del hidden,encoder
    bounds = [0,*ends[0]]
    i = next(i for i,(a,b) in enumerate(zip(bounds,bounds[1:])) if a<windows[0][0]<b)
    torch.testing.assert_close(blocks[i].float(),states[bounds[i]:bounds[i+1]].float().mean(0),atol=.002,rtol=.002)
    masses = [repair_block_mass(s,e,a,(*EDIT_TYPES,*GAP_TYPES))
              for s,e,a in zip(sources,ends,paired_annotations(sources),strict=True)]
    targets = [m/m.sum().clamp_min(1) for m in masses]
    report = {}
    for method,mode in MATCHING.items():
        set_seed(SEED)
        features = states if mode!="block" else None
        split = pipeline.SourceSplit(rows,sources,windows,ends,blocks,cost,0.,token_features=features)
        data = BlockData(rows,ends,blocks,targets,token_features=features)
        torch.cuda.reset_peak_memory_stats()
        inputs = to_gpu(collate_blocks([data[i] for i in range(len(data))]),labels=True)
        model = CoarseClassifier(blocks.shape[1],"joint",REPAIR_WEIGHT,matching=mode).cuda()
        result = model(inputs,details=True)
        params = tuple(model.indexer.parameters())
        norms = {name:[float(g.norm()) for g in torch.autograd.grad(loss,params,retain_graph=True)]
                 for name,loss in result["loss_components"].items()}
        if not all(np.isfinite(value) and value>0 for pair in norms.values() for value in pair):
            raise ValueError("BCE and KL must both reach Q/K")
        before = [p.detach().clone() for p in params]
        result["loss"].backward()
        AdamW(model.parameters(),lr=2e-5).step()
        if any(torch.equal(a,b) for a,b in zip(before,params,strict=True)):
            raise ValueError("Optimizer failed to update Q/K")
        peak = torch.cuda.max_memory_allocated()
        del inputs,result
        model.requires_grad_(False)
        g,ranks,_ = pipeline.score_coarse(model,split,"global")
        fine,_ = pipeline.fine_data(split,g,ranks,tokenizer,96)
        single = pipeline.SourceSplit(rows[:1],sources[:1],windows[:1],ends[:1],blocks[:len(ends[0])],{},0.,
                                     token_features=features[:len(source.ids)] if features is not None else None)
        one_g,one_rank,_ = pipeline.score_coarse(model,single,"global")
        one,_ = pipeline.fine_data(single,one_g,one_rank,tokenizer,96)
        torch.testing.assert_close(one.examples[0]["fixed"],fine.examples[0]["fixed"],atol=0.,rtol=0.)
        if max(ex["length"] for ex in fine.examples)!=MAX_LENGTH:
            raise ValueError("Extreme function did not use the fixed fine budget")
        report[method] = dict(gradient_norms_query_key=norms,peak_index_training_allocated_bytes=peak,
                              fine_lengths=[ex["length"] for ex in fine.examples])
    detector = FusionClassifier(fp32_classifier(tokenizer).detector,"global").cuda()
    inputs = to_gpu(collate_fine(fine.examples),labels=True)
    loss = detector(inputs)["loss"]
    loss.backward()
    if not float(detector.fusion.weight.grad.norm())>0:
        raise ValueError("Fine loss must reach Global Fusion")
    AdamW(detector.parameters(),lr=2e-5).step()
    write_json(RESULTS/"smoke.json",dict(matching=report,encoding={k:v for k,v in cost.items()
               if k!="per_function_padded_positions"},original_bpe=[len(s.raw_ids) for s in sources],
               window_counts=list(map(len,windows)),fine_loss=float(loss.detach())))
    print("Extreme GPU smoke passed",max(len(s.raw_ids) for s in sources),"original BPE",flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage",choices=("cache","smoke","run","choose","report"))
    cli.add_argument("--seed",type=int,choices=SEEDS,default=SEED)
    cli.add_argument("--matching",choices=METHOD_BY_MATCHING)
    args = cli.parse_args()
    torch.set_num_threads(4)
    if args.stage=="choose":
        choose()
        return
    if args.stage=="report":
        report(load_tokenizer("unixcoder",UNIX_BASE))
        return
    if torch.cuda.device_count()!=1 or "4090" not in torch.cuda.get_device_name():
        raise ValueError("Select one RTX 4090 via CUDA_VISIBLE_DEVICES")
    torch.backends.cudnn.deterministic = True
    tokenizer = load_tokenizer("unixcoder",UNIX_BASE)
    if args.stage=="cache":
        cache(tokenizer)
    elif args.stage=="smoke":
        smoke(tokenizer)
    else:
        if args.matching is None:
            cli.error("run requires --matching")
        run(tokenizer,args.seed,METHOD_BY_MATCHING[args.matching])


if __name__=="__main__":
    main()
