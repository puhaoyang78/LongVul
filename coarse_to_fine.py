"""Train and evaluate repair-guided, context-aware coarse-to-fine encoding."""
import argparse
import csv
from dataclasses import dataclass
import gc
import gzip
import json
import math
import os
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from transformers import set_seed

from coarse_encoding import (ATTENTION_HEADS, INDEX_WIDTH, LATENTS, QUERY_HEADS, WIDTH, BlockData,
                             CoarseClassifier, FusionClassifier, block_mass, collate_blocks, encode_blocks,
                             encoder_window_ends, repair_block_mass, repair_kl)
from coarse_retrieval import Candidates, LAYOUTS, REPLACEMENT_TRIALS, Layout, block_ends, lengths
from compaction import BUDGET, MAX_LENGTH, pack_fixed
from context_selection import (base_windows, contains, intersects, merge, occupied, related_statements,
                               source_parser, write_prediction_rows)
from locator_quality import CONTEXT_BATCH, load_context_encoder
from paired import (ADAM_EPSILON, BACKBONES, BATCH_SIZE, DATA, ROOT, SEED, UNIX_BASE, classifier_config,
                    fit_classifier, load_tokenizer, LOSVERAdamW, prime_source_rows, to_gpu)
from predicted_diff import HISTORICAL_REVISION, fp32_classifier, load_sources, paired_annotations
from region_locator import region_metrics
from repair_oracle import EDIT_TYPES, GAP_TYPES, grouped_edit_retention, overlap_bpe, raw_runs
from rgcl.artifacts import read_json, write_json
from rgcl.config import EPOCHS, LEARNING_RATE, MAX_GRAD_NORM, SEEDS
from rgcl.metrics import metrics, paired_metrics
from window_inputs import GROUPS, grouped_metrics

RESULTS = ROOT/"results"/"longvul_adaptive_512"
WEIGHTS = ROOT/"checkpoints"/"longvul_adaptive_512"
ARCHIVE = ROOT/"results"/"coarse_to_fine_512"
METHODS = ("block_context","balanced","context","full","uniform")
GLOBAL_EPOCHS, LAYOUT_INDEX_EPOCHS, INDEX_EPOCHS = 4,2,6
WARM_EPOCHS = 8
FEEDBACK_EPOCHS = (9,13)
TARGET_GRADIENT_RATIOS = (0.125,0.5)


@dataclass(slots=True)
class Corpus:
    rows: list
    sources: list
    windows: list
    syntax: list
    layouts: dict
    vectors: dict
    timing: dict


def source_inputs(split,tokenizer):
    started = perf_counter()
    rows = prime_source_rows(DATA,split)
    sources = load_sources(rows,tokenizer)
    windows = [encoder_window_ends(s) for s in sources]
    return rows,sources,windows,perf_counter()-started


def source_syntax(sources):
    parser = source_parser()
    output = []
    for s in sources:
        units = set()
        links,error = related_statements(s.code,s.offsets,s.keep,None,parser,units=units)
        output.append((sorted(units),sorted(links),error))
    return output


def corpus(split,tokenizer,layouts,*,build=False,fresh=False,encoder=None):
    rows,sources,windows,source_time = source_inputs(split,tokenizer)
    cached = WEIGHTS/"blocks"/f"{split}.pt"
    config = dict(dataset="PrimeVul_original",split=split,members=[r["sample_key"] for r in rows],
                  model=BACKBONES["unixcoder"][1],simple=HISTORICAL_REVISION,windows=windows,
                  batch=CONTEXT_BATCH,precision="BF16 SDPA / FP32 means / FP16 cache")
    tokens = torch.from_numpy(np.concatenate([s.ids for s in sources]))
    if cached.is_file() and not fresh:
        start = perf_counter()
        data = torch.load(cached,weights_only=True,mmap=True,map_location="cpu")
        if data["config"]!=config or not torch.equal(data["tokens"],tokens):
            raise ValueError(f"Source cache no longer matches {split}")
        if not set(layouts)<=data["vectors"].keys():
            raise ValueError(f"Missing requested layouts in {cached}")
        return Corpus(rows,sources,windows,data["syntax"],data["layouts"],data["vectors"],
                      dict(data["timing"],cache_load_seconds=perf_counter()-start,source_reload_seconds=source_time))
    if not build:
        raise FileNotFoundError(f"Prepare the frozen block cache for {split} first")
    start = perf_counter()
    syntax = source_syntax(sources)
    parse_seconds = perf_counter()-start
    cuts = {name:[block_ends(len(s.ids),u,s.boundaries,*LAYOUTS[name])
                  for s,(u,_,_) in zip(sources,syntax,strict=True)] for name in layouts}
    if encoder is None:
        encoder = load_context_encoder().requires_grad_(False)
    vectors,encoding = encode_blocks(encoder,sources,tokenizer,windows,cuts)
    del encoder
    if any(not torch.isfinite(v).all() for v in vectors.values()):
        raise ValueError("Nonfinite contextual block vectors")
    timing = dict(source_seconds=source_time,syntax_seconds=parse_seconds,encoding=encoding,
                  cached_bytes={name:v.nbytes for name,v in vectors.items()})
    if not fresh:
        cached.parent.mkdir(parents=True,exist_ok=True)
        tmp = cached.with_name(f".{cached.name}.{os.getpid()}.tmp")
        try:
            torch.save(dict(config=config,tokens=tokens,syntax=syntax,layouts=cuts,vectors=vectors,timing=timing),tmp)
            tmp.replace(cached)
        finally:
            tmp.unlink(missing_ok=True)
        print(split,"cached layouts",list(cuts),round(encoding["seconds"],2),"s",flush=True)
    return Corpus(rows,sources,windows,syntax,cuts,vectors,timing)


def plans(data,layout,contextual):
    return [Candidates(e,u,r,contextual=contextual)
            for e,(u,r,_) in zip(data.layouts[layout],data.syntax,strict=True)]


def block_data(data,layout,*,candidates=None,repair=None,offset=0):
    return BlockData(data.rows,data.sources,data.layouts[layout],data.vectors[layout],
                     plans=candidates,repair=repair,offset=offset)


def repair_targets(data,layout,edits):
    masses = [repair_block_mass(source,ends,annotation,(*EDIT_TYPES,*GAP_TYPES))
              for source,ends,annotation in zip(data.sources,data.layouts[layout],edits,strict=True)]
    return [m/m.sum().clamp_min(1) for m in masses]


def configuration(method,seed,layout,epochs,**kwargs):
    return dict(classifier_config("uniform_sampling"),method=method,seed=seed,layout=layout,epochs=epochs,
                model_id=BACKBONES["unixcoder"][0],model_revision=BACKBONES["unixcoder"][1],
                cross_block_width=WIDTH,latents=LATENTS,attention_heads=ATTENTION_HEADS,
                query_heads=QUERY_HEADS,index_dimension=INDEX_WIDTH,full_encoder_precision="BF16 SDPA",
                full_encoder_window_batch=CONTEXT_BATCH,historical_input_revision=HISTORICAL_REVISION,**kwargs)


def fitted(config,selected,history,valid):
    return dict(config=config,selected_epoch=selected["epoch"],validation=selected["validation"],
                train_seconds=sum(row["training_seconds"] for row in history),
                validation_seconds=sum(row["validation_seconds"] for row in history),
                peak_train_allocated_bytes=max(row["peak_allocated_bytes"] for row in history),
                train_bce_final3=sum(row.get("train_bce",row["train_loss"]) for row in history[-3:])/min(len(history),3),
                one_class_valid_epochs=sum(row["validation"]["tp"]+row["validation"]["fp"] in (0,len(valid))
                                           for row in history))


def state(seed,name):
    return torch.load(WEIGHTS/f"seed{seed}"/name/"best.pt",weights_only=True,map_location="cpu")


def coarse_model(seed,name):
    fit = read_json(RESULTS/f"seed{seed}"/name/"fit.json")
    cfg = fit["config"]
    model = CoarseClassifier(768,cfg["supervision"],cfg["repair_weight"],bypass=cfg["bypass"])
    model.load_state_dict(state(seed,name),strict=True)
    return model.cuda().eval()


def init_indexer(seed,layout,mode,weight,bypass):
    set_seed(seed)
    model = CoarseClassifier(768,mode,weight,bypass=bypass)
    missing = model.load_state_dict(state(seed,f"coarse/{layout}/global"),strict=False)
    if set(missing.missing_keys)!={"indexer.query.weight","indexer.key.weight"} or missing.unexpected_keys:
        raise ValueError("Global checkpoint could not initialize the indexer")
    return model.cuda()


def coarse_fit(seed,name,layout,train,valid,mode,weight,epochs,*,bypass=False,contextual=False):
    root,weights = RESULTS/f"seed{seed}"/name,WEIGHTS/f"seed{seed}"/name
    config = configuration(name,seed,layout,epochs,supervision=mode,repair_weight=weight,bypass=bypass,
                          contextual_keys=contextual)
    if (root/"fit.json").is_file():
        saved = read_json(root/"fit.json")
        if saved["config"]!=config:
            raise ValueError(f"Coarse settings changed for {name}")
        return saved
    if mode=="global":
        set_seed(seed)
        model = CoarseClassifier(768,mode).cuda()
    else:
        model = init_indexer(seed,layout,mode,weight,bypass)
    best,history = fit_classifier(model,train,valid,weights,root,config,collator=collate_blocks,epochs=epochs)
    result = fitted(config,best,history,valid)
    write_json(root/"fit.json",result)
    return result


@torch.inference_mode()
def coarse_predictions(model,data):
    values,probabilities = [],[]
    model.eval()
    for cpu in DataLoader(data,batch_size=BATCH_SIZE,collate_fn=collate_blocks):
        output = model(to_gpu(cpu),details=True)
        probabilities.extend(output["probability"].tolist())
        scores = output["importance"].cpu().numpy()
        values.extend(s[:int(n)].copy() for s,n in zip(scores,cpu["valid"].sum(1).tolist(),strict=True))
    return values,probabilities


def layout_diagnostics(data,spans,edits,*,anchors=None):
    lengths_ = [len(s.raw_ids) for s in data.sources]
    raw = [raw_runs(s.keep,selected) for s,selected in zip(data.sources,spans,strict=True)]
    retention = grouped_edit_retention(zip(lengths_,edits,raw,strict=True))
    quality = {}
    for group,(lo,hi) in GROUPS.items():
        chosen = [i for i,n in enumerate(lengths_) if lo<=n<=hi]
        width = [b-a for i in chosen for a,b in spans[i]]
        appearance,complete = 0,0
        touched_anchors,complete_anchors = 0,0
        for i in chosen:
            units,_,_ = data.syntax[i]
            if len(data.sources[i].ids)>BUDGET:
                appearance += sum(intersects(a,spans[i]) for a in units)
                complete += sum(contains(a,spans[i]) for a in units)
                if anchors is not None:
                    units_at_centers = [a for a in units if intersects(a,anchors[i])]
                    touched_anchors += len(units_at_centers)
                    complete_anchors += sum(contains(a,spans[i]) for a in units_at_centers)
        quality[group] = dict(functions=len(chosen),frag_count=len(width),mean_fragments=len(width)/max(1,len(chosen)),
                              mean_fragment_bpe=float(np.mean(width)) if width else None,
                              fragment_bpe_percentiles=np.percentile(width,[10,50,90,99]).tolist() if width else [],
                              relative_cut_statements=appearance-complete,covered_statements=complete,
                              touched_statements=appearance,statement_completion=complete/max(1,appearance))
        if anchors is not None:
            quality[group].update(anchor_statements=touched_anchors,complete_anchor_statements=complete_anchors,
                                  anchor_statement_completion=complete_anchors/touched_anchors if touched_anchors else None)
    return retention,quality


def select_layout(tokenizer):
    root = RESULTS/"seed42"
    if (root/"layout_trials.json").is_file():
        print("Layout selection already complete",flush=True)
        return
    train,valid = [corpus(s,tokenizer,LAYOUTS,build=True) for s in ("train","valid")]
    edits = paired_annotations(valid.sources)
    trials = []
    for layout in LAYOUTS:
        data = [block_data(s,layout) for s in (train,valid)]
        name = f"coarse/{layout}"
        global_fit = coarse_fit(SEED,f"{name}/global",layout,*data,"global",0.,GLOBAL_EPOCHS)
        indexed = coarse_fit(SEED,f"{name}/layout_probe",layout,*data,"task",0.,LAYOUT_INDEX_EPOCHS)
        model = coarse_model(SEED,f"{name}/layout_probe")
        rankings,scores = coarse_predictions(model,data[1])
        candidates = plans(valid,layout,False)
        spans = [plan.retrieve(p,source).spans for plan,p,source in zip(candidates,rankings,valid.sources,strict=True)]
        edit,quality = layout_diagnostics(valid,spans,edits)
        trials.append(dict(layout=layout,global_valid_mcc=global_fit["validation"]["mcc"],
                           valid_mcc=metrics(data[1].targets.tolist(),scores)["mcc"],retention=edit["long"]["actual_edits"]["fraction"],
                           completeness=quality["long"]["statement_completion"],fragments=quality["long"]["mean_fragments"],
                           blocks=len(train.vectors[layout])))
        del model
        print("Valid layout",trials[-1],flush=True)
    objectives = (("valid_mcc",2,True),("retention",1,True),("completeness",1,True),("fragments",1,False))
    for trial in trials:
        trial["rank_sum"] = sum(weight*(1+sum((other[field]>trial[field]) if high else (other[field]<trial[field])
                                            for other in trials)) for field,weight,high in objectives)
    best = min(trials,key=lambda v:(v["rank_sum"],-v["valid_mcc"],abs(LAYOUTS[v["layout"]][0]-64),v["layout"]))
    write_json(root/"layout_trials.json",dict(trials=trials,selected=best["layout"],
              selection="Valid coarse MCC rank x2 plus edit retention, completeness and fragment ranks",
              global_epochs=GLOBAL_EPOCHS,index_probe_epochs=LAYOUT_INDEX_EPOCHS))
    print("Selected layout",best,flush=True)


def selector_parameters(model):
    return tuple(model.indexer.parameters())


def loss_gradients(losses,parameters):
    values = {}
    for name,loss in losses.items():
        grads = torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
        values[name] = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                                 for g,p in zip(grads,parameters,strict=True)]).detach()
    return values


def gradient_summary(vectors):
    names = list(vectors[0])
    norms = {k:torch.stack([row[k].norm() for row in vectors]) for k in names}
    result = {k:dict(median=float(x.median()),nonzero=int(x.gt(0).sum())) for k,x in norms.items()}
    for i,name in enumerate(names):
        for other in names[i+1:]:
            use = (norms[name]>0)&(norms[other]>0)
            if not use.any():
                continue
            dots = torch.stack([v[name]@v[other] for v in vectors])[use]
            result[f"{other}_vs_{name}"] = dict(median_norm_ratio=float((norms[other][use]/norms[name][use]).median()),
                   median_cosine=float((dots/(norms[name][use]*norms[other][use])).median()))
    return dict(batches=len(vectors),components=result)


def probe_batches(data, count=64):
    rng = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(data),generator=rng)[:count*BATCH_SIZE].tolist()
    return DataLoader(Subset(data,indices),batch_size=BATCH_SIZE,collate_fn=collate_blocks)


def index_probe(model,train,weight):
    model.eval()
    vectors = []
    for cpu in probe_batches(train):
        batch = to_gpu(cpu,labels=True)
        out = model(batch,details=True)
        bce = out["loss_components"]["train_bce"]
        kl = repair_kl(out["importance"],batch["repair"])
        vectors.append(loss_gradients(dict(bce=bce,kl=kl,weighted_kl=weight*kl),selector_parameters(model)))
    return gradient_summary(vectors)


def index_quality(data,layout,ranks,edits):
    result = {}
    for kind,kinds in (("actual",EDIT_TYPES),*((kind,(kind,)) for kind in GAP_TYPES)):
        positive = [repair_block_mass(s,e,a,kinds).numpy()>0
                    for s,e,a in zip(data.sources,data.layouts[layout],edits,strict=True)]
        groups = {}
        for group,(lo,hi) in GROUPS.items():
            ids = [i for i,s in enumerate(data.sources) if lo<=len(s.raw_ids)<=hi]
            targets = [positive[i] for i in ids]
            n,total = sum(int(x.sum()) for x in targets),sum(map(len,targets))
            groups[group] = (region_metrics([ranks[i] for i in ids],targets,ks=(1,5,10,15))
                             if 0<n<total else dict(functions=len(targets),regions=total,positive_regions=n,ap=None))
        result[kind] = groups
    return result


def index_ablation(seed,layout,data,candidates,names):
    scores,layouts,probabilities = {},{},{}
    features = block_data(data,layout,candidates=candidates)
    for label,name in names.items():
        model = coarse_model(seed,name)
        scores[label],probabilities[label] = coarse_predictions(model,features)
        layouts[label] = [p.retrieve(s,source).spans
                          for p,s,source in zip(candidates,scores[label],data.sources,strict=True)]
        del model
    edits = paired_annotations(data.sources)
    result = {}
    for label,ranks in scores.items():
        retain,quality = layout_diagnostics(data,layouts[label],edits)
        result[label] = dict(indexer=index_quality(data,layout,ranks,edits),edit_retention=retain,selection=quality,
                             coarse_metrics=metrics(features.targets.tolist(),probabilities[label]))
    differences = {}
    for label in ("task","repair"):
        differences[label] = {}
        for group,(lo,hi) in GROUPS.items():
            selected = [(a,b) for a,b,s in zip(layouts[label],layouts["joint"],data.sources,strict=True)
                        if lo<=len(s.raw_ids)<=hi]
            jaccard = []
            for a,b in selected:
                intersection = overlap_bpe(a,b)
                jaccard.append(intersection/(lengths(a)+lengths(b)-intersection))
            differences[label][group] = dict(functions=len(selected),same_layout=sum(a==b for a,b in selected),
                                              bpe_jaccard_mean=float(np.mean(jaccard)) if selected else None)
    return dict(arms=result,comparison_with_joint=differences)


def prepare_indexers(tokenizer,seed):
    root = RESULTS/f"seed{seed}"
    layout = read_json(RESULTS/"seed42"/"layout_trials.json")["selected"]
    if (root/"index_policy.json").is_file() and (WEIGHTS/"blocks/test.pt").is_file():
        if seed!=SEED or (root/"index_ablation_test.json").is_file():
            print(seed,"index preparation complete",flush=True)
            return
    train,valid = [corpus(name,tokenizer,[layout]) for name in ("train","valid")]
    targets = repair_targets(train,layout,paired_annotations(train.sources))
    group = f"coarse/{layout}"
    coarse_fit(seed,f"{group}/global",layout,block_data(train,layout),block_data(valid,layout),"global",0.,GLOBAL_EPOCHS)
    before = perf_counter()
    train_plans,valid_plans = [plans(data,layout,True) for data in (train,valid)]
    planning = perf_counter()-before
    print(seed,"compiled Train/Valid syntax bundles",round(planning,2),"s",flush=True)
    tr,va = (block_data(train,layout,candidates=train_plans,repair=targets),
             block_data(valid,layout,candidates=valid_plans))
    weights = {}
    if seed==SEED:
        initial = init_indexer(seed,layout,"joint",0.,False)
        calibration = index_probe(initial,tr,0.)
        del initial
        ratio = calibration["components"]["kl_vs_bce"]["median_norm_ratio"]
        trials = []
        for target_ratio in TARGET_GRADIENT_RATIOS:
            weight = target_ratio/ratio
            name = f"{group}/joint_{target_ratio:g}"
            fit = coarse_fit(seed,name,layout,tr,va,"joint",weight,INDEX_EPOCHS,contextual=True)
            model = coarse_model(seed,name)
            probe = index_probe(model,tr,weight)
            del model
            trials.append(dict(name=name,weight=weight,target_ratio=target_ratio,
                         valid_mcc=fit["validation"]["mcc"],probe=probe,
                         weighted_ratio=probe["components"]["weighted_kl_vs_bce"]["median_norm_ratio"]))
        feasible = [v for v in trials if v["weighted_ratio"]<=1.]
        selected = min(feasible,key=lambda s:(-s["valid_mcc"],s["weighted_ratio"])) if feasible else min(
            trials,key=lambda s:(s["weighted_ratio"],-s["valid_mcc"]))
        weights["context"] = selected["name"]
        weight = selected["weight"]
        other_tr,other_va = block_data(train,layout,repair=targets),block_data(valid,layout)
        comparison = {}
        historical_weight = read_json(ARCHIVE/"seed42"/"summary.json")["policy"]["selected"]["weight"]
        for name,bypass,control_weight in (("block_context",True,historical_weight),("balanced",False,weight)):
            key = f"{group}/{name}"
            coarse_fit(seed,key,layout,other_tr,other_va,"joint",control_weight,INDEX_EPOCHS,bypass=bypass)
            weights[name] = key
            model = coarse_model(seed,key)
            comparison[name] = dict(repair_weight=control_weight,probe=index_probe(model,other_tr,control_weight))
            del model
        controls = {}
        for mode,value in (("task",0.),("repair",weight)):
            key = f"{group}/{mode}"
            coarse_fit(seed,key,layout,tr,va,mode,value,INDEX_EPOCHS,contextual=True)
            controls[mode] = key
        policy = dict(layout=layout,weight=weight,indexers=weights,calibration=calibration,trials=trials,
                      chosen=selected,bypass_comparison=comparison,controls=controls,
                      valid_selection="max Valid MCC among trained ratios at most 1; closest balance if none",
                      syntax_bundle_seconds=planning)
    else:
        weight = read_json(RESULTS/"seed42"/"index_policy.json")["weight"]
        name = f"{group}/joint"
        coarse_fit(seed,name,layout,tr,va,"joint",weight,INDEX_EPOCHS,contextual=True)
        policy = dict(layout=layout,weight=weight,indexers=dict(context=name),fixed_from_seed42=True,
                      syntax_bundle_seconds=planning)
    write_json(root/"index_policy.json",policy)
    print(seed,"fixed index policy",policy["layout"],policy["weight"],flush=True)
    test = corpus("test",tokenizer,[layout],build=seed==SEED)
    if seed==SEED:
        ablation = index_ablation(seed,layout,test,plans(test,layout,True),dict(policy["controls"],joint=weights["context"]))
        write_json(root/"index_ablation_test.json",ablation)


def loss_components(model, batch):
    out = model(batch)
    parts = out["loss_components"]
    return dict(aux_bce=parts["aux_train_bce"], final_bce=parts["train_bce"],
                kl=parts["aux_train_repair_kl"],
                weighted_kl=model.coarse.repair_weight*parts["aux_train_repair_kl"])


def changed_blocks(ends, removed, added):
    a,b = (block_mass(ends,spans) for spans in (removed,added))
    if min(a.sum(),b.sum())<=0:
        raise ValueError("A feedback comparison must change the visible code")
    change = b/b.sum()-a/a.sum()
    ids = change.nonzero().flatten()
    return ids.tolist(),change[ids].tolist()


@torch.inference_mode()
def feedback_pairs(model, data, seed, epoch):
    """Evaluate source layouts A and B against the same Train label and frozen model."""
    model.eval()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    rng = np.random.default_rng(np.random.SeedSequence([seed,epoch]))
    examples, deltas = {},[]
    tried = compare = touches = code_changed = snippets = calls = unrankable = 0
    for cpu in DataLoader(data,batch_size=BATCH_SIZE,collate_fn=collate_blocks):
        ids = cpu["sample_ids"].tolist()
        batch = to_gpu(cpu)
        coarse = model.coarse(batch,details=True)
        a = model.layouts(batch,coarse["importance"],uniform=False)
        scores = coarse["importance"].cpu().numpy()
        alternatives = [[],[]]
        for j,(i,layout) in enumerate(zip(ids,a,strict=True)):
            if not layout.centers:
                continue
            tried += 1
            source,candidate = model.examples[i]
            for k,(alt,removed,added) in enumerate(candidate.alternatives(layout,scores[j,:len(candidate.centers)],source,rng)):
                alternatives[k].append((j,i,alt,removed,added))
        active = sorted({j for variants in alternatives for j,*_ in variants})
        if not active:
            continue
        compare += len(active)
        snippets += len(active)
        calls += 1
        g,z = coarse["g"],coarse["z"]
        y = cpu["labels"].to(g.device,dtype=torch.float32)
        original = model.fine_logits([ids[j] for j in active],[a[j] for j in active],g[active],z[active])
        loss_a = torch.empty_like(y)
        loss_a[active] = F.binary_cross_entropy_with_logits(original,y[active],reduction="none")
        for variants in alternatives:
            if not variants:
                continue
            indices, samples, layouts, _, _ = zip(*variants,strict=True)
            indices = list(indices)
            logits_b = model.fine_logits(samples,layouts,g[indices],z[indices])
            delta = (F.binary_cross_entropy_with_logits(logits_b,y[indices],reduction="none")-loss_a[indices]).tolist()
            snippets += len(variants)
            calls += 1
            for (j,i,_,removed,added),d in zip(variants,delta,strict=True):
                deltas.append(d)
                code_changed += lengths(removed)
                if d == 0:
                    continue
                _,candidate = model.examples[i]
                ends = [b for _,b in candidate.centers]
                idx, w = changed_blocks(ends,removed,added)
                if not idx:
                    unrankable += 1
                    continue
                touches += len(idx)
                sign = math.copysign(1.,d)
                examples.setdefault(i,[]).append((abs(d),idx,[sign*x for x in w]))
    torch.cuda.synchronize()
    count = sum(map(len,examples.values()))
    costs = dict(seconds=perf_counter()-started,functions=len(data),need_selection=tried,
                 comparable_functions=compare,functions_with_nonzero_feedback=len(examples),comparisons=len(deltas),
                 tie_count=sum(d==0 for d in deltas),same_changed_block_distribution=unrankable,
                 replacements_preferred=sum(d<0 for d in deltas),
                 score_comparisons=count,scored_blocks_mean=touches/max(1,count),
                 changed_bpe_per_side_mean=code_changed/max(1,len(deltas)),
                 abs_delta_median=float(np.median(np.abs(deltas))) if deltas else None,
                 rank_layouts_encoded=snippets,fine_batch_calls=calls,fine_padded_positions=snippets*MAX_LENGTH,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated())
    return examples,costs


def ranking_loss(importance, ids, examples):
    size = importance.shape[1]
    regions,groups,contrast,weights = [],[],[],[]
    for j,i in enumerate(ids):
        for weight,index,sign in examples.get(i,()):
            regions.extend(j*size+k for k in index)
            groups.extend([len(weights)]*len(index))
            contrast.extend(sign)
            weights.append(weight)
    device = importance.device
    weight = torch.tensor(weights,device=device)
    logp = importance.flatten()[torch.tensor(regions,device=device,dtype=torch.long)].clamp_min(1e-12).log()
    change = torch.zeros_like(weight).index_add(0,torch.tensor(groups,device=device,dtype=torch.long),
                                               logp*torch.tensor(contrast,device=device))
    return (weight*F.softplus(change)).mean()


@torch.inference_mode()
def validation_layouts(model, valid):
    scores,spans = [],[]
    model.eval()
    for cpu in DataLoader(valid,batch_size=BATCH_SIZE,collate_fn=collate_blocks):
        out = model(to_gpu(cpu),details=True)
        scores.extend(out["probability"].tolist())
        spans.extend(layout.spans for layout in out["layouts"])
    return scores,spans


def update_indexer(model,train,valid,seed,epoch,*,smoke=False):
    model.eval()
    report = {}
    if not smoke:
        started = perf_counter()
        before,old = validation_layouts(model,valid)
        report["pre_feedback_valid_seconds"] = perf_counter()-started
    examples,generation = feedback_pairs(model,train,seed,epoch)
    report["generation"] = generation
    if not examples:
        raise ValueError("No nonzero Train feedback from the budget-matched layout comparisons")
    parameters = selector_parameters(model.coarse)
    optimizer = LOSVERAdamW(parameters,lr=LEARNING_RATE,eps=ADAM_EPSILON)
    subset = Subset(train,[i-train.offset for i in sorted(examples)])
    generator = torch.Generator().manual_seed(seed+epoch)
    loader = DataLoader(subset,batch_size=BATCH_SIZE,shuffle=True,generator=generator,collate_fn=collate_blocks)
    losses,gradients = [],[]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    for i,cpu in enumerate(loader):
        batch = to_gpu(cpu,labels=True)
        with torch.no_grad():
            g,context,_ = model.coarse.encoder(batch)
        p = model.coarse.indexer(g,context,batch)
        rank = ranking_loss(p,cpu["sample_ids"].tolist(),examples)
        if i < 4:
            components = loss_components(model,batch)
            gradients.append(loss_gradients(dict(rank=rank,**components),parameters))
        rank.backward()
        torch.nn.utils.clip_grad_norm_(parameters,MAX_GRAD_NORM,error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(rank.detach())
    torch.cuda.synchronize()
    report.update(ranking_seconds=perf_counter()-started,ranking_steps=len(loader),
                  ranking_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  ranking_loss=torch.stack(losses).mean().item(),gradients=gradient_summary(gradients))
    if not smoke:
        started = perf_counter()
        after,new = validation_layouts(model,valid)
        y = valid.targets.tolist()
        counts = [(a,b,len(model.examples[valid.offset+i][0].raw_ids))
                  for i,(a,b) in enumerate(zip(old,new,strict=True))]
        groups = {}
        for group,(lo,hi) in GROUPS.items():
            pairs = [(a,b) for a,b,n in counts if lo<=n<=hi]
            # Same source layout can still change probability via the continuous branch.
            groups[group] = dict(functions=len(pairs),changed=sum(a!=b for a,b in pairs))
        report.update(pre_feedback_valid=metrics(y,before),post_feedback_valid=metrics(y,after),
                      changed_valid_layouts=groups,post_feedback_valid_seconds=perf_counter()-started)
    return report


@torch.inference_mode()
def predict_layouts(model,data):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    model.eval()
    scores,ranks,layouts,slots = [],[],[],np.zeros(4)
    loader = DataLoader(data,batch_size=BATCH_SIZE,collate_fn=collate_blocks)
    for cpu in loader:
        result = model(to_gpu(cpu),details=True)
        scores.extend(result["probability"].tolist())
        layouts.extend(result["layouts"])
        values = result["importance"].cpu().numpy()
        ranks.extend(p[:n].copy() for p,n in zip(values,cpu["valid"].sum(1).tolist(),strict=True))
        slots += result["seconds"]
    fields = ("cross_block_seconds","index_seconds","retrieval_seconds","fine_encoding_seconds")
    cost = dict(zip(fields,slots.tolist(),strict=True),seconds=perf_counter()-start,windows=len(data),
                batch_calls=len(loader),unpadded_positions=sum(occupied(l.spans)+2 for l in layouts),
                padded_positions=len(data)*MAX_LENGTH,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    return scores,ranks,layouts,cost


def pair_results(rows,scores,raw_lengths,global_same,fine_same):
    result = {}
    for group,(lo,hi) in GROUPS.items():
        pairs = [i for i in range(0,len(rows),2) if lo<=max(raw_lengths[i:i+2])<=hi]
        ids = [i for a in pairs for i in (a,a+1)]
        if ids:
            p = paired_metrics([rows[i] for i in ids],[scores[i] for i in ids])
            result[group] = dict(p,identical_global_pair_features=sum(global_same[i//2] for i in pairs),
                                 identical_fine_pair_inputs=sum(fine_same[i//2] for i in pairs),
                                 identical_complete_pair_inputs=sum(global_same[i//2] and fine_same[i//2] for i in pairs))
    return result


def context_quality(data,layouts,control=None):
    result = {}
    for group,(lo,hi) in GROUPS.items():
        counters = {"clean_parse":{},"partial_parse":{}}
        functions = {"clean_parse":0,"partial_parse":0}
        impossible = {"clean_parse":0,"partial_parse":0}
        for i,(source,syntax) in enumerate(zip(data.sources,data.syntax,strict=True)):
            if not lo<=len(source.raw_ids)<=hi or len(source.ids)<=BUDGET:
                continue
            _,links,error = syntax
            cohort = "partial_parse" if error else "clean_parse"
            functions[cohort] += 1
            spans = layouts[i].spans
            other = control[i].spans if control is not None else spans
            for kind,anchor,context in links:
                if occupied(merge([anchor,context]))>BUDGET:
                    impossible[cohort] += 1
                    continue
                count = counters[cohort].setdefault(kind,[0,0,0,0])
                count[0] += 1
                if contains(anchor,spans) and contains(anchor,other):
                    count[1] += 1
                    count[2] += int(contains(context,spans))
                    count[3] += int(contains(context,other))
        result[group] = {}
        for cohort,by_kind in counters.items():
            by_kind["all"] = [sum(values) for values in zip(*by_kind.values())] if by_kind else [0]*4
            result[group][cohort] = dict(needs_selection=functions[cohort],over_budget_links=impossible[cohort],
                links={k:dict(feasible=n,complete_anchors=a,context_kept=b,context_fraction=b/a if a else None,
                             control_context_kept=c,control_context_fraction=c/a if a else None)
                       for k,(n,a,b,c) in by_kind.items()})
    return result


def evaluate_layouts(data,layout,candidates,tokenizer,prediction,root,name,*,planning,control=None):
    scores,ranks,layouts,fine_cost = prediction
    enc = data.timing["encoding"]
    raw_lengths = [len(s.raw_ids) for s in data.sources]
    positions = data.layouts[layout]
    source_runs = [raw_runs(s.keep,l.spans) for s,l in zip(data.sources,layouts,strict=True)]
    fine_ids = [pack_fixed(s.raw_ids,s.keep,l.spans,tokenizer)[0][0].numpy() for s,l in
                zip(data.sources,layouts,strict=True)]
    same_fine = [np.array_equal(a,b) for a,b in zip(fine_ids[::2],fine_ids[1::2],strict=True)]
    same_global = [np.array_equal(data.sources[i].ids,data.sources[i+1].ids)
                   and data.sources[i].keep==data.sources[i+1].keep
                   and data.windows[i]==data.windows[i+1] and positions[i]==positions[i+1]
                   and candidates[i].edges==candidates[i+1].edges for i in range(0,len(layouts),2)]
    rows = []
    for i,(row,source,value,entry) in enumerate(zip(data.rows,data.sources,scores,layouts,strict=True)):
        rows.append(dict(idx=row["idx"],sample_key=row["sample_key"],pair_key=row["pair_key"],target=row["target"],
                    original_bpe=raw_lengths[i],compacted_bpe=len(source.ids),probability=value,prediction=int(value>.5),
                    global_blocks=len(positions[i]),global_windows=len(data.windows[i]),
                    global_unpadded_positions=len(source.ids)+2*len(data.windows[i]),
                    global_padded_positions=enc["per_function_padded_positions"][i],syntax_error=data.syntax[i][2],
                    input_length=occupied(entry.spans)+2,code_bpe=lengths(entry.spans),
                    internal_separators=max(0,len(entry.spans)-1),encoding_runs=list(entry.spans),
                    source_runs=source_runs[i],center_blocks=list(entry.centers),
                    identical_global_pair_features=same_global[i//2],identical_fine_pair_input=same_fine[i//2],
                    identical_complete_pair_input=same_global[i//2] and same_fine[i//2]))
    root.mkdir(parents=True,exist_ok=True)
    write_prediction_rows(root/f"{name}.csv.gz",rows)
    # Pair versions are only read for diagnostics after saving this checkpoint's fixed predictions.
    edits = paired_annotations(data.sources)
    anchors = [([plan.centers[i] for i in entry.centers] if entry.centers else base_windows(source.boundaries)[0])
               if len(source.ids)>BUDGET else entry.spans
               for entry,plan,source in zip(layouts,candidates,data.sources,strict=True)]
    retain,quality = layout_diagnostics(data,[entry.spans for entry in layouts],edits,anchors=anchors)
    group_data = SimpleNamespace(metadata=rows,targets=torch.tensor([r["target"] for r in rows]))
    costs = dict(source_seconds=data.timing["source_seconds"],syntax_seconds=data.timing["syntax_seconds"],
                 candidate_seconds=planning,global_encoding={k:v for k,v in enc.items() if k!="per_function_padded_positions"},
                 cached_inference=fine_cost)
    costs["stage_sum_seconds"] = sum(costs[k] for k in ("source_seconds","syntax_seconds","candidate_seconds"))+enc["seconds"]+fine_cost["seconds"]
    for k in ("windows","batch_calls","unpadded_positions","padded_positions"):
        costs[k] = enc[k]+fine_cost[k]
    return dict(groups=grouped_metrics(group_data,scores),
                pairs=pair_results(data.rows,scores,raw_lengths,same_global,same_fine),cost=costs,
                edit_retention=retain,layout_quality=quality,context=context_quality(data,layouts),
                same_anchor_context=context_quality(data,layouts,control) if control is not None else None,
                indexer=index_quality(data,layout,ranks,edits))


def cold_test(model,tokenizer,layout,expected):
    """Measure source-to-prediction cost with both encoders already resident."""
    encoder = load_context_encoder().requires_grad_(False)
    model.eval()
    torch.cuda.synchronize()
    start = perf_counter()
    cold = corpus("test",tokenizer,[layout],build=True,fresh=True,encoder=encoder)
    planning = perf_counter()
    candidates = plans(cold,layout,True)
    planning_seconds = perf_counter()-planning
    prior,model.examples = model.examples,list(zip(cold.sources,candidates,strict=True))
    try:
        probabilities,_,layouts,cost = predict_layouts(model,block_data(cold,layout,candidates=candidates))
    finally:
        model.examples = prior
    total = perf_counter()-start
    changes = sum(actual.spans!=old.spans for actual,old in zip(layouts,expected[2],strict=True))
    difference = max(abs(a-b) for a,b in zip(probabilities,expected[0],strict=True))
    if changes or difference>1e-6:
        raise ValueError(f"Cold Test path changed {changes} fine inputs or probabilities by {difference}")
    enc = {k:v for k,v in cold.timing["encoding"].items() if k!="per_function_padded_positions"}
    return dict(seconds=total,candidate_seconds=planning_seconds,global_encoding=enc,
                source_seconds=cold.timing["source_seconds"],syntax_seconds=cold.timing["syntax_seconds"],
                downstream=cost,peak_allocated_bytes=max(enc["peak_allocated_bytes"],cost["peak_allocated_bytes"]),
                max_abs_probability_difference=difference,changed_layouts=changes,
                **{k:enc[k]+cost[k] for k in ("windows","batch_calls","unpadded_positions","padded_positions")})


def fork_warm(warm,weights,config):
    if (weights/"last.pt").is_file():
        return
    weights.mkdir(parents=True,exist_ok=True)
    # Same model, classifier optimizer, scheduler, and RNG at the eighth epoch.
    cp = torch.load(warm/"last.pt",map_location="cpu",weights_only=False)
    if len(cp["loop"])!=WARM_EPOCHS:
        raise ValueError("The shared warmup has not completed")
    cp["config"] = config
    if not (weights/"best.pt").exists():
        os.link(warm/"best.pt",weights/"best.pt")
    path = weights/"last.pt.tmp"
    torch.save(cp,path)
    path.replace(weights/"last.pt")


def fit_downstream(tokenizer,seed,methods):
    root,weights = RESULTS/f"seed{seed}",WEIGHTS/f"seed{seed}"
    policy = read_json(root/"index_policy.json")
    layout = policy["layout"]
    sources = {split:corpus(split,tokenizer,[layout]) for split in ("train","valid","test")}
    targets = repair_targets(sources["train"],layout,paired_annotations(sources["train"].sources))
    for contextual in (True,False):
        group = [m for m in methods if (m in ("context","full","uniform"))==contextual]
        if not group:
            continue
        offsets,datasets,prepared,times,examples = 0,{},{},{},[]
        for split,source in sources.items():
            start = perf_counter()
            candidate = plans(source,layout,contextual)
            times[split] = perf_counter()-start
            prepared[split] = candidate
            examples.extend(zip(source.sources,candidate,strict=True))
            datasets[split] = block_data(source,layout,candidates=candidate,
                                         repair=targets if split=="train" else None,offset=offsets)
            offsets += len(source.rows)
        train,valid = datasets["train"],datasets["valid"]
        print(seed,"built",("syntax bundles" if contextual else "independent blocks"),times,flush=True)
        for method in group:
            name = method if method in ("block_context","balanced") else "context"
            index = policy["indexers"][name]
            feedback = method in ("full","uniform")
            warm = "context_warm" if method in ("context","full") else method
            index_config = read_json(root/index/"fit.json")["config"]
            config = configuration(method,seed,layout,EPOCHS,indexer=index,repair_weight=index_config["repair_weight"],
                  fine_input="historical centered Uniform Sampling" if method=="uniform" else (
                      "context bundles by marginal value" if contextual else "ranked single blocks"),
                  contextual_keys=contextual,aux_bypass=method=="block_context",warm_epochs=WARM_EPOCHS,
                  feedback_epochs=list(FEEDBACK_EPOCHS) if feedback else [],warm_checkpoint=warm,
                  feedback_trials_per_pool=REPLACEMENT_TRIALS if feedback else None,
                  feedback_labels="Train only",joint_objective="final BCE + auxiliary BCE + fixed repair KL",
                  split_functions={name:len(data) for name,data in datasets.items()})
            location = root/method
            path = location/"metrics.json"
            if path.is_file():
                if read_json(path)["config"] != config:
                    raise ValueError(f"Already completed {seed}/{method} with different settings")
                continue
            coarse = coarse_model(seed,index)
            set_seed(seed)
            model = FusionClassifier(fp32_classifier(tokenizer).detector,coarse,examples,tokenizer,
                                     uniform=method=="uniform").cuda()
            if warm!=method:
                warm_config = dict(config,method=warm,feedback_epochs=[],feedback_trials_per_pool=None)
                if not (weights/warm/"last.pt").is_file() or not (root/warm/"complete.json").is_file():
                    best,history = fit_classifier(model,train,valid,weights/warm,root/warm,warm_config,
                                         collator=collate_blocks,epochs=EPOCHS,stop_after=WARM_EPOCHS)
                    write_json(root/warm/"complete.json",fitted(warm_config,best,history,valid))
                fork_warm(weights/warm,weights/method,config)

            def start_epoch(current,epoch):
                current.joint = epoch>WARM_EPOCHS
                if not current.joint or epoch not in FEEDBACK_EPOCHS:
                    return
                # Keep the subsequent fine-training data order matched across arms.
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    if feedback:
                        report = update_indexer(current,train,valid,seed,epoch)
                        print(seed,method,epoch,"feedback",report["generation"],flush=True)
                    else:
                        current.eval()
                        vectors = [loss_gradients(loss_components(current,to_gpu(cpu,labels=True)),
                                                 selector_parameters(current.coarse)) for cpu in probe_batches(train,4)]
                        report = dict(gradients=gradient_summary(vectors))
                write_json(location/"feedback"/f"epoch{epoch}.json",report)

            selected,history = fit_classifier(model,train,valid,weights/method,location,config,collator=collate_blocks,
                                               before_epoch=start_epoch)
            result = fitted(config,selected,history,valid)
            phases = {str(e):read_json(location/"feedback"/f"epoch{e}.json") for e in FEEDBACK_EPOCHS}
            result["feedback"] = phases
            result["feedback_train_seconds"] = sum(p["generation"]["seconds"]+p["ranking_seconds"]
                                                   for p in phases.values() if "generation" in p)
            result["feedback_validation_seconds"] = sum(p["pre_feedback_valid_seconds"]+p["post_feedback_valid_seconds"]
                                                       for p in phases.values() if "generation" in p)
            result["warm_train_seconds"] = sum(row["training_seconds"] for row in history[:WARM_EPOCHS])
            result["joint_train_seconds"] = sum(row["training_seconds"] for row in history[WARM_EPOCHS:])
            result["preprocessing"] = {k:dict(source_reload_seconds=data.timing["source_reload_seconds"],
                       cache_load_seconds=data.timing["cache_load_seconds"],candidates_seconds=times[k],
                       syntax_errors=sum(error for _,_,error in data.syntax),omitted_candidate_links={
                           kind:sum(plan.omitted[kind] for plan in prepared[k])
                           for kind in ("completion","condition","declaration","assignment")})
                       for k,data in sources.items()}
            splits,swap = {},{}
            for split in ("valid","test"):
                data,features,candidate = sources[split],datasets[split],prepared[split]
                scored = predict_layouts(model,features)
                swapped = None
                if method=="full":
                    model.uniform = True
                    try:
                        swapped = predict_layouts(model,features)
                    finally:
                        model.uniform = False
                    swap[split] = evaluate_layouts(data,layout,candidate,tokenizer,swapped,
                                  location/"same_checkpoint_uniform",split,planning=times[split],control=scored[2])
                splits[split] = evaluate_layouts(data,layout,candidate,tokenizer,scored,location,split,
                                      planning=times[split],control=swapped[2] if swapped else None)
                if split=="valid" and splits[split]["groups"]["overall"]!=selected["validation"]:
                    raise ValueError(f"{seed}/{method}: reloading failed to reproduce full Valid metrics")
                if split=="test" and method=="full" and seed==SEED:
                    result["cold_test"] = cold_test(model,tokenizer,layout,scored)
            result["split"] = splits
            if swap:
                result["same_checkpoint_uniform"] = swap
            write_json(path,result)
            print(seed,method,"complete at epoch",selected["epoch"],"Test MCC",splits["test"]["groups"]["overall"]["mcc"],
                  flush=True)
            del model,coarse
            gc.collect()


def summarize(tokenizer,seed):
    root = RESULTS/f"seed{seed}"
    names = METHODS if seed==SEED else ("context","full","uniform")
    results = {name:read_json(root/name/"metrics.json") for name in names}
    historical_dir = ARCHIVE/f"seed{seed}"
    historic = read_json(historical_dir/"summary.json")
    report = dict(seed=seed,new={name:dict(selected_epoch=data["selected_epoch"],test=data["split"]["test"]["groups"],
                                          pairs=data["split"]["test"]["pairs"],
                                          train_seconds=data["train_seconds"],
                                          feedback_train_seconds=data["feedback_train_seconds"])
                                 for name,data in results.items()},
                  same_checkpoint_uniform=results["full"]["same_checkpoint_uniform"]["test"]["groups"],
                  historical_joint=historic["methods"]["joint"]["split"]["test"]["groups"],
                  historical_uniform=historic["methods"]["uniform"]["split"]["test"]["groups"],
                  historical_uniform_sampling_seed42=historic["archived_uniform_sampling_seed42"]["split"]["test"]["groups"])
    layout = read_json(root/"index_policy.json")["layout"]
    if seed==SEED:
        paths = [(a,b) for a,b in (("context","balanced"),("full","context"),("full","uniform"))]
        data = corpus("test",tokenizer,[layout])

        def read_layouts(method):
            with gzip.open(root/method/"test.csv.gz","rt") as f:
                return [Layout(tuple(tuple(x) for x in json.loads(row["encoding_runs"])))
                        for record,row in zip(data.rows,csv.DictReader(f),strict=True)
                        if record["sample_key"]==row["sample_key"]]

        report["same_anchor"] = {}
        for method,control in paths:
            left,right = read_layouts(method),read_layouts(control)
            if min(len(left),len(right))!=len(data.rows):
                raise ValueError("Context comparison does not match the official Test members")
            report["same_anchor"][f"{method}_versus_{control}"] = context_quality(data,left,right)
    write_json(root/"summary.json",report)


def smoke(tokenizer):
    set_seed(SEED)
    rows = prime_source_rows(DATA,"train")
    extreme = max(range(len(rows)),key=lambda i:len(rows[i]["func"]))//2*2
    short = min(range(0,len(rows),2),key=lambda i:max(len(rows[i]["func"]),len(rows[i+1]["func"])))
    medium = [i for i in range(0,len(rows),2) if 4000<=len(rows[i]["func"])<=16000]
    indices = sorted({extreme,short,*np.random.default_rng(SEED).choice(medium,16,replace=False).tolist()})
    rows = [r for i in indices for r in rows[i:i+2]]
    sources = load_sources(rows,tokenizer)
    windows = [encoder_window_ends(s) for s in sources]
    syntax = source_syntax(sources)
    selected = RESULTS/"seed42"/"layout_trials.json"
    layout = read_json(selected)["selected"] if selected.is_file() else "statement64"
    cuts = {layout:[block_ends(len(s.ids),u,s.boundaries,*LAYOUTS[layout])
                    for s,(u,_,_) in zip(sources,syntax,strict=True)]}
    encoder = load_context_encoder().requires_grad_(False)
    vectors,encoding = encode_blocks(encoder,sources,tokenizer,windows,cuts)
    del encoder
    data = Corpus(rows,sources,windows,syntax,cuts,vectors,{})
    candidates = plans(data,layout,True)
    train = block_data(data,layout,candidates=candidates,repair=repair_targets(data,layout,paired_annotations(sources)))
    coarse = CoarseClassifier(768,"joint",.01).cuda()
    model = FusionClassifier(fp32_classifier(tokenizer).detector,coarse,list(zip(sources,candidates,strict=True)),
                             tokenizer).cuda()
    longest = max(range(len(sources)),key=lambda i:len(sources[i].raw_ids))
    if len(sources[longest].raw_ids)<200000:
        raise ValueError("Smoke test did not include the longest real source")
    cpu = collate_blocks([train[longest],train[(longest+1)%len(train)],train[0]])
    batch = to_gpu(cpu,labels=True)
    output = coarse(batch,details=True)
    parameters = selector_parameters(coarse)
    aux = output["loss_components"]["train_bce"]
    qgrad,broadcast = torch.autograd.grad(aux,(parameters[0],coarse.encoder.broadcast.in_proj_weight))
    padded = output["importance"][~batch["valid"]]
    if any(x.norm()==0 for x in (qgrad,broadcast)) or not len(padded) or padded.any():
        raise ValueError("Cross block gradients or padding mask are incorrect")
    torch.cuda.reset_peak_memory_stats()
    model.train()
    opt = LOSVERAdamW(model.parameters(),lr=LEARNING_RATE,eps=ADAM_EPSILON)
    warm = model(batch)
    warm["loss"].backward()
    if any(p.grad is not None for p in parameters) or model.fusion.weight.grad.norm()==0:
        raise ValueError("Fine warmup does not isolate the coarse branch")
    opt.step()
    opt.zero_grad(set_to_none=True)
    model.joint = True
    output = model(batch,details=True)
    fine_bce = output["loss_components"]["train_bce"]
    direct = loss_gradients(dict(final_bce=fine_bce),parameters)["final_bce"].norm().item()
    del output,fine_bce
    current = [p.detach().clone() for p in parameters]
    fine = model.fusion.weight.detach().clone()
    model.eval()
    report = update_indexer(model,train,None,SEED,9,smoke=True)
    if any(torch.equal(a,b) for a,b in zip(current,parameters,strict=True)) or not torch.equal(fine,model.fusion.weight):
        raise ValueError("Feedback must change the indexer while holding the classifier fixed")
    prior = parameters[0].detach().clone()
    model.train()
    current = model(batch,details=True)
    for sample,layout_,scores in zip(cpu["sample_ids"].tolist(),current["layouts"],
                                       current["importance"].detach().cpu().numpy(),strict=True):
        plan = candidates[sample]
        if plan.retrieve(scores[:len(plan.centers)],sources[sample])!=layout_:
            raise ValueError("Joint training reused a layout after an index update")
    current["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),MAX_GRAD_NORM,error_if_nonfinite=True)
    opt.step()
    if not direct or torch.equal(prior,parameters[0]):
        raise ValueError("Joint fine loss did not update the indexer through continuous features")
    n = len(sources[longest].ids)
    lengths_ = [occupied(l.spans)+2 for l in current["layouts"]]
    checks = dict(layout=layout,functions=len(rows),longest_original_bpe=len(sources[longest].raw_ids),
                  longest_compacted_bpe=n,longest_windows=len(windows[longest]),longest_blocks=len(cuts[layout][longest]),
                  longest_cross_window_blocks=len(set(windows[longest])-set(cuts[layout][longest])),
                  longest_block_endpoint=cuts[layout][longest][-1],fine_lengths=lengths_,
                  aux_query_gradient=qgrad.norm().item(),cross_block_gradient=broadcast.norm().item(),
                  continuous_final_bce_gradient=direct,feedback=report,
                  warm_loss=warm["loss"].item(),joint_loss=current["loss"].item(),
                  encoding={k:v for k,v in encoding.items() if k!="per_function_padded_positions"},
                  peak_allocated_bytes=torch.cuda.max_memory_allocated())
    write_json(RESULTS/"smoke.json",checks)
    print(json.dumps(checks),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=("smoke","select","indexes","fine","report","all"),default="all")
    parser.add_argument("--seed",type=int,choices=SEEDS,default=SEED)
    parser.add_argument("--methods",nargs="+",choices=METHODS)
    args = parser.parse_args()
    if args.phase!="report" and (torch.cuda.device_count()!=1 or "4090" not in torch.cuda.get_device_name()):
        raise ValueError("Select one RTX 4090 through CUDA_VISIBLE_DEVICES")
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    tokenizer = load_tokenizer("unixcoder",UNIX_BASE)
    methods = args.methods or (METHODS if args.seed==SEED else ("context","full","uniform"))
    if args.phase=="smoke":
        smoke(tokenizer)
    if args.phase=="select" or (args.phase=="all" and args.seed==SEED):
        select_layout(tokenizer)
    if args.phase in ("indexes","all"):
        prepare_indexers(tokenizer,args.seed)
    if args.phase in ("fine","all"):
        fit_downstream(tokenizer,args.seed,methods)
    expected = METHODS if args.seed==SEED else ("context","full","uniform")
    if args.phase=="report" or (args.phase=="all" and set(methods)==set(expected)):
        summarize(tokenizer,args.seed)


if __name__=="__main__":
    main()
