"""Complete matched baselines and diagnose the existing LongVul models."""
import argparse
import csv
from dataclasses import replace
import gzip
import io
import json
import subprocess

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import set_seed

import coarse_to_fine as center
import index_adaptation as adaptation
from coarse_encoding import BlockData, CoarseClassifier, FusionClassifier, collate_blocks, collate_fine
from coarse_retrieval import part_overlap, retrieve, uniform_layout
from compaction import BUDGET, REGIONS, PackedData, pack_fixed
from context_selection import gaps, occupied, visible_source, write_prediction_rows
from paired import BATCH_SIZE, ROOT, SEED, UNIX_BASE, UNIX_RESULTS, load_tokenizer, to_gpu
from predicted_diff import fp32_classifier
from rgcl.artifacts import read_json, write_json
from rgcl.metrics import metrics, paired_metrics
from window_inputs import BASE, DATA, GROUPS, SEEDS, prime_source_rows, seed_path

OUT = ROOT/"results"/"baseline_completion"
DIAG = ROOT/"results"/"next_stage_diagnosis"
ARCHIVED_UNIFORM = "e094c1ef8bd5e56b08f154a416f89a9eb9c3ecf6"
EXTERNAL = ("stagedvulbert","sea","vulnscout","codebert_uniform4")
INTERNAL = ("longvul","context_aware","historical_coarse_to_fine","matched_uniform")
PAIRS = ((42,"B"),(123456,"A"),(1234,"matched_uniform"))


def baseline_path(method,seed):
    if method=="codebert_uniform4":
        root = ROOT/"results"/"paired"/"uniform4"
    elif method=="context_aware":
        root = UNIX_RESULTS/method
    elif method=="historical_coarse_to_fine":
        return center.ARCHIVE/f"seed{seed}"/"joint"
    elif method in ("longvul","matched_uniform"):
        return center.RESULTS/f"seed{seed}"/center_method(method,seed)
    else:
        root = ROOT/"results"/method
    return seed_path(root,seed)


def center_method(method,seed):
    return ("global_fusion" if seed==SEED else "final") if method=="longvul" else method


def archived(path):
    return subprocess.check_output(["git","show",f"{ARCHIVED_UNIFORM}:{path}"],cwd=ROOT)


def original_uniform_predictions(split,rows):
    """The published CSV predated sample_key; verify its ordered (idx,target) join."""
    raw = archived(f"results/paired/uniform4/{split}.csv.gz")
    with gzip.open(io.BytesIO(raw),"rt") as stream:
        saved = list(csv.DictReader(stream))
    for source,entry in zip(rows,saved,strict=True):
        if (int(entry["idx"]),int(entry["target"]))!=(source["idx"],source["target"]):
            raise ValueError("Archived Uniform-4 does not match the official split order")
        entry["sample_key"] = source["sample_key"]
    return saved


def scores_for(method,seed,split,rows,codebert=None):
    if method=="codebert_uniform4" and seed==SEED:
        records = original_uniform_predictions(split,rows)
    else:
        records = center.read_predictions(baseline_path(method,seed)/f"{split}.csv.gz",rows)
    if codebert is not None and [int(r["original_bpe"]) for r in records]!=codebert:
        raise ValueError(f"{method} Seed {seed} {split} CodeBERT lengths disagree")
    return [float(r["probability"]) for r in records]


def cost_record(method,seed):
    path = baseline_path(method,seed)
    if method=="codebert_uniform4" and seed==SEED:
        m = json.loads(archived("results/paired/uniform4/metrics.json"))
        probe = json.loads(archived("results/paired/context_interventions/metrics.json"))["uniform4_memory_probe"]
    else:
        m = read_json(path/"metrics.json")
        probe = None
    test = m["split"]["test"]
    if method in ("longvul","matched_uniform","historical_coarse_to_fine"):
        measured = m.get("uncached_test_measurement",test["cost"])
        stage = test["cost"]
        return dict(train_seconds=m["train_seconds"],peak_train_bytes=m["peak_train_allocated_bytes"],
                    end_to_end_test_seconds=center.pipeline_seconds(measured),
                    test_windows=stage["windows"],test_batch_calls=stage["batch_calls"],
                    test_unpadded_positions=stage["unpadded_positions"],test_padded_positions=stage["padded_positions"],
                    peak_test_bytes=measured.get("peak_inference_allocated_bytes",stage.get("peak_allocated_bytes")))
    history = (json.loads(archived("results/paired/uniform4/epochs.json")) if probe else read_json(path/"epochs.json"))
    prep = test.get("preparation_seconds",m.get("preparation_seconds",{}).get("test"))
    extraction = m.get("feature_extraction",{})
    result = dict(train_seconds=m["training_seconds"],epochs_completed=len(history),selected_epoch=m["selected_epoch"],
                  peak_train_bytes=probe["train_allocated_bytes"] if probe else m["peak_train_allocated_bytes"],
                  test_model_seconds=test["inference_seconds"],test_preparation_seconds=prep,
                  end_to_end_test_seconds=prep+test["inference_seconds"],
                  peak_test_bytes=probe["inference_allocated_bytes"] if probe else test.get(
                      "peak_inference_allocated_bytes",test.get("peak_allocated_bytes")),
                  feature_extraction_seconds=sum(x["seconds"] for x in extraction.values()),
                  train_preparation_seconds=m.get("train_preparation_seconds",m.get("preparation_seconds",{}).get("train")),
                  costs_from_two_step_probe=probe is not None)
    if method in ("vulnscout","sea"):
        result["coverage"] = test["coverage"]
    elif method=="stagedvulbert":
        result["coverage"] = m["coverage"]["test"]
    return result


def verify_external_protocol(method,seed):
    if seed==SEED:
        return
    path = baseline_path(method,seed)
    saved = read_json(path/"config.json")
    canonical = (json.loads(archived("results/paired/uniform4/metrics.json"))["config"]
                 if method=="codebert_uniform4" else read_json(baseline_path(method,SEED)/"config.json"))
    ignore = {"seed","host","gpu","python","torch","transformers","cuda","dataset","selection","historical_head"}
    if saved["seed"]!=seed or any(saved.get(k)!=v for k,v in canonical.items() if k not in ignore):
        raise ValueError(f"{method}/{seed} protocol mismatch")
    history = read_json(path/"epochs.json")
    best = max(history,key=lambda e:e["validation"]["mcc"])
    saved_metrics = read_json(path/"metrics.json")
    if best["epoch"]!=saved_metrics["selected_epoch"]:
        raise ValueError(f"{method}/{seed}: history and checkpoint selection disagree")
    if method=="vulnscout":
        if len(history)!=saved["epochs"] and len(history)-best["epoch"]!=saved["patience"]:
            raise ValueError(f"{method}/{seed}: premature exit")
    elif len(history)!=saved["epochs"]:
        raise ValueError(f"{method}/{seed}: incomplete epochs")


def baseline_report():
    rows = {s:prime_source_rows(DATA,s) for s in ("valid","test")}
    codebert = {s:center.codebert_lengths(s,r) for s,r in rows.items()}
    methods,aggregate = {},{}
    for name in (*EXTERNAL,*INTERNAL):
        runs = {}
        for seed in SEEDS:
            if name in EXTERNAL:
                verify_external_protocol(name,seed)
            split_scores = {}
            for split in rows:
                scores = scores_for(name,seed,split,rows[split],codebert[split] if name in EXTERNAL else None)
                split_scores[split] = center.classification(rows[split],scores,codebert[split])
                if name in EXTERNAL:
                    legacy = (json.loads(archived("results/paired/uniform4/metrics.json")) if
                              name=="codebert_uniform4" and seed==SEED else read_json(baseline_path(name,seed)/"metrics.json"))
                    if split_scores[split]["groups"]!=legacy["split"][split]["groups"]:
                        raise ValueError(f"{name}/{seed}/{split} saved metrics fail recomputation")
            runs[str(seed)] = dict(split=split_scores,cost=cost_record(name,seed))
        methods[name] = runs
        view = [r["split"]["test"] for r in runs.values()]
        aggregate[name] = {g:{k:center.seed_stats([v[s][g][k] for v in view])
                               for s,keys in (("groups",("mcc","f1","precision","recall","auc","auprc")),
                                              ("pairs",("p_c","pair_ranking_accuracy")))
                               for k in keys} for g in GROUPS}
        keys = ("train_seconds","end_to_end_test_seconds","peak_train_bytes","peak_test_bytes")
        aggregate[name]["cost"] = {k:center.seed_stats([r["cost"].get(k) for r in runs.values()]) for k in keys}
    volume = {g:sum(a<=n<=b for n in codebert["test"]) for g,(a,b) in GROUPS.items()}
    write_json(OUT/"summary.json",dict(seeds=list(SEEDS),test_denominators=volume,
               pair_bucket="larger of the two CodeBERT source lengths",per_seed=methods,three_seed=aggregate,
               uniform4_seed42=ARCHIVED_UNIFORM))
    print("Published cohort verified, all baseline metrics recomputed",volume,flush=True)


def valid_threshold(y,p):
    # Decision partitions and the tie rule from the previous archived diagnostic.
    y,p = np.asarray(y),np.asarray(p)
    values = np.unique(np.r_[0.,p,1.])
    candidates = np.unique(np.r_[0.,.5,1.,(values[1:]+values[:-1])/2])
    order = np.argsort(p,kind="stable")
    below = np.searchsorted(p[order],candidates,side="right")
    fn = np.r_[0,y[order].cumsum()][below]
    tp,tn = y.sum()-fn,below-fn
    fp = len(y)-y.sum()-tn
    denom = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    score = np.divide(tp*tn-fp*fn,denom,out=np.zeros_like(denom),where=denom>0)
    return float(candidates[np.lexsort((candidates,np.abs(candidates-.5),-score))[0]])


def probability_summary(y,p,logits=None):
    y,p = np.asarray(y),np.asarray(p)
    if logits is None:
        bounded = np.clip(p,1e-15,1.-1e-15)
        logits = np.log(bounded/(1.-bounded))
    logits = np.asarray(logits)
    return {cohort:dict(probability=np.quantile(p[mask],[0.,.1,.5,.9,1.]).tolist(),
                        logit=np.quantile(logits[mask],[0.,.1,.5,.9,1.]).tolist(),
                        mean=float(p[mask].mean()),std=float(p[mask].std()),n=int(mask.sum()))
            for cohort,mask in (("all",np.ones(len(y),dtype=bool)),("positive",y==1),("negative",y==0))
            if mask.any()}


def threshold_row(y,p,t):
    predicted = np.asarray(p)>t
    report = metrics([int(v) for v in y],predicted.astype(float).tolist())
    return {k:report[k] for k in ("mcc","f1","precision","recall","tp","tn","fp","fn")}|{
        "threshold":t,"predicted_positive_fraction":float(predicted.mean())}


def threshold_report():
    rows = {s:prime_source_rows(DATA,s) for s in ("valid","test")}
    lengths_by_split = {s:center.codebert_lengths(s,rs) for s,rs in rows.items()}
    subjects = [(m,seed,baseline_path(m,seed)) for m in INTERNAL for seed in SEEDS]
    subjects += [(f"index_{method}",seed,adaptation.RESULTS/f"seed{seed}"/method)
                 for seed in SEEDS for method in ("ABCD" if seed==SEED else "A")]
    result = {}
    for label,seed,path in subjects:
        vectors = {s:[float(r["probability"]) for r in center.read_predictions(path/f"{s}.csv.gz",rs)]
                   for s,rs in rows.items()}
        targets = {s:np.array([r["target"] for r in rs]) for s,rs in rows.items()}
        choice = valid_threshold(targets["valid"],vectors["valid"])
        values = {}
        for split,(y,p) in ((k,(targets[k],np.array(v))) for k,v in vectors.items()):
            size = np.array(lengths_by_split[split])
            official = center.classification(rows[split],p.tolist(),size.tolist())
            split_groups = {}
            for g,(a,b) in GROUPS.items():
                subset = (size>=a)&(size<=b)
                max_pair = size.reshape(-1,2).max(1)
                pairs = (max_pair>=a)&(max_pair<=b)
                scaled = p.reshape(-1,2)[pairs]
                chosen = threshold_row(y[subset],p[subset],choice)
                chosen["p_c"] = float(((scaled[:,0]>choice)&(scaled[:,1]<=choice)).mean())
                split_groups[g] = dict(fixed=official["groups"][g],fixed_pairs=official["pairs"][g],
                    valid_chosen=chosen,probability=probability_summary(y[subset],p[subset]))
            values[split] = split_groups
        result[f"{label}_{seed}"] = dict(valid_threshold=choice,split=values)
    write_json(DIAG/"thresholds.json",dict(protocol="one full Valid threshold per model, tie nearest 0.5",
                                          ranking_metrics_use_original_scores=True,models=result))
    print("Strict 0.5 and Valid-only threshold diagnostics saved",flush=True)


def load_coarse(tag,seed):
    if tag in ("longvul","matched_uniform"):
        return center.load_index(seed,False)
    path = adaptation.RESULTS/f"seed{seed}"/tag
    trial = read_json(path/"index_policy.json")["selected"]["candidate"]
    weights = adaptation.WEIGHTS/f"seed{seed}"/tag/"index"/trial/"best.pt"
    model = CoarseClassifier(768,"joint",bypass=tag=="A")
    model.load_state_dict(torch.load(weights,weights_only=True,map_location="cpu"),strict=True)
    return model.cuda().eval().requires_grad_(False)


def selected_data(split,index,tokenizer,*,uniform=False,budget=96):
    features,p,_ = center.score_coarse(index,split,"global")
    data,_ = center.fine_data(split,features,p,tokenizer,budget,uniform=uniform)
    return data,p


def model_paths(tag,seed):
    if tag in ("longvul","matched_uniform"):
        name = center_method(tag,seed)
        return center.RESULTS/f"seed{seed}"/name,center.WEIGHTS/f"seed{seed}"/name
    return adaptation.RESULTS/f"seed{seed}"/tag,adaptation.WEIGHTS/f"seed{seed}"/tag/"fine"


def fresh_fine(tokenizer):
    return FusionClassifier(fp32_classifier(tokenizer).detector,"global").cuda()


def part_shares(spans,ends):
    sizes = np.diff([0,*ends])
    covered = np.zeros(len(ends),dtype=np.int32)
    for lo,hi in spans:
        for i,a,b in part_overlap(lo,hi,ends):
            covered[i] += b-a
    if (covered>sizes).any():
        raise ValueError("Overlapping layout falsely double counts source coverage")
    return covered/sizes


def auxiliary_weights(layout,uniform,ends,p):
    center_share = part_shares(layout.centers,ends)
    visible_share = part_shares(layout.spans,ends)
    uniform_share = part_shares(uniform.spans,ends)
    return dict(all=p,centers=p*center_share,visible=p*visible_share,
                visible_uniform=visible_share,uniform_layout=p*uniform_share,discarded=p*(1-visible_share)),(
                    float(np.count_nonzero(center_share)/len(ends)),float(p@center_share),float(p@visible_share))


def probability_metrics(rows,probs,mask,sizes):
    y = np.array([r["target"] for r in rows])
    n = np.asarray(sizes)
    p = np.asarray(probs,dtype=float)
    valid = np.asarray(mask,dtype=bool)
    result = {}
    for name,(lo,hi) in GROUPS.items():
        ids = np.flatnonzero(valid&(n>=lo)&(n<=hi))
        if not len(ids):
            continue
        pair_lengths = n.reshape(-1,2).max(1)
        pair_starts = np.flatnonzero(valid.reshape(-1,2).all(1)&(lo<=pair_lengths)&(pair_lengths<=hi))*2
        pair_indices = np.column_stack((pair_starts,pair_starts+1)).flatten()
        paired = paired_metrics([rows[i] for i in pair_indices],p[pair_indices].tolist()) if len(pair_indices) else None
        clipped = np.clip(p[ids],1e-15,1.-1e-15)
        bce = -np.mean(y[ids]*np.log(clipped)+(1-y[ids])*np.log1p(-clipped))
        result[name] = dict(metrics=metrics(y[ids].tolist(),p[ids].tolist()),pairs=paired,bce=float(bce))
    return result


@torch.inference_mode()
def auxiliary_interventions(split,index,data,ranks,codebert,tag):
    weighted,geometry = [],[]
    for source,ends,p,ex in zip(split.sources,split.ends,ranks,data.examples,strict=True):
        weights,details = auxiliary_weights(ex["layout"],uniform_layout(source),ends,p)
        weighted.append(weights)
        geometry.append(details)
    names = tuple(weighted[0])
    views = ("native","masked_g") if index.bypass else ("native",)
    rows = [dict(sample_key=r["sample_key"],target=r["target"],original_bpe=len(s.raw_ids),codebert_bpe=n,
                 center_block_fraction=geo[0],center_mass=geo[1],visible_mass=geo[2],
                 discarded_mass=1-geo[2])
            for r,s,n,geo in zip(split.rows,split.sources,codebert,geometry,strict=True)]
    offset = 0
    dataset = BlockData(split.rows,split.ends,split.vectors)
    for cpu in DataLoader(dataset,batch_size=BATCH_SIZE,collate_fn=collate_blocks):
        batch = to_gpu(cpu)
        out = index(batch,details=True)
        g = out["global_state"]
        n,padded = cpu["valid"].shape
        blocks = batch["blocks"].float()
        for variant in names:
            bank = np.zeros((n,padded),dtype=np.float32)
            available = []
            for i,weight in enumerate(weighted[offset:offset+n]):
                w = weight[variant]
                total = w.sum()
                available.append(total>1e-12)
                if available[-1]:
                    bank[i,:len(w)] = w/total
            z = out["weighted"] if variant=="all" else torch.bmm(torch.from_numpy(bank).cuda()[:,None,:],blocks).squeeze(1)
            for mode in views:
                _,logits = index.auxiliary(g if mode=="native" else torch.zeros_like(g),z)
                for i,p in enumerate(logits.sigmoid().tolist()):
                    rows[offset+i][f"{mode}_{variant}"] = p if available[i] else None
        offset += n
    fields = [f"{mode}_{name}" for mode in views for name in names]
    common = np.array([all(r[f] is not None for f in fields) for r in rows])
    output = {}
    for mode in views:
        base = np.array([r[f"{mode}_all"] for r in rows])
        for variant in names:
            field = f"{mode}_{variant}"
            p = np.array([r[field] if r[field] is not None else np.nan for r in rows])
            actual = np.isfinite(p)
            output[field] = dict(common=probability_metrics(split.rows,p,common,codebert),
                                 available=probability_metrics(split.rows,p,actual,codebert),
                                 probability_abs_delta_median=float(np.median(np.abs(p[actual]-base[actual]))))
    group_mass = {}
    for name,(lo,hi) in GROUPS.items():
        cohort = [r for r in rows if lo<=r["codebert_bpe"]<=hi]
        group_mass[name] = {k:float(np.mean([r[k] for r in cohort])) for k in
                            ("center_block_fraction","center_mass","visible_mass","discarded_mass")}|{"functions":len(cohort)}
    write_prediction_rows(DIAG/f"auxiliary_{tag}.csv.gz",rows)
    return dict(views=output,mean_mass=group_mass,common_functions=int(common.sum()),
                common_pairs=int(common.reshape(-1,2).all(1).sum()))


def outside_layout(spans,size,budget=BUDGET):
    """Form an exact-position layout from only the longest available gaps."""
    available = list(gaps(spans,0,size))
    if occupied(available)<budget or (budget%2==0 and all(b-a==1 for a,b in available)):
        return None
    selected,remaining = [],budget
    for a,b in sorted(available,key=lambda gap:(gap[0]-gap[1],gap[0])):
        sep = int(bool(selected))
        if remaining==1 and sep:
            # A second fragment costs two positions. Shift one selected body BPE.
            lo,hi = selected[0]
            if hi-lo<2:
                raise ValueError("A large complement must start with a nontrivial fragment")
            selected[0] = (lo,hi-1)
            selected.append((a,a+1))
            remaining = 0
        else:
            take = min(b-a,remaining-sep)
            if take:
                selected.append((a,a+take))
                remaining -= take+sep
        if not remaining:
            break
    out = sorted(selected)
    if occupied(out)!=budget:
        raise ValueError("Feasible outside layout did not fill its position budget")
    return out


def counterfactual_examples(split,tokenizer,data,ranks):
    layouts = {"context":data.examples,"independent":[],"uniform":[],"outside":[]}
    for src,ends,ranking,ex in zip(split.sources,split.ends,ranks,data.examples,strict=True):
        variants = dict(independent=retrieve(ranking,src,ends,0).spans,
                        uniform=uniform_layout(src).spans,
                        outside=outside_layout(ex["layout"].spans,len(src.ids)) if ex["length"]==512 else None)
        for name,spans in variants.items():
            if spans is None:
                layouts[name].append(None)
                continue
            packed,n = pack_fixed(src.raw_ids,src.keep,spans,tokenizer)
            if n!=ex["length"]:
                raise ValueError("Layout intervention changed the number of input positions")
            if name=="outside" and np.any(np.intersect1d(ex["fixed"][1].numpy(),packed[1].numpy())>=0):
                raise ValueError("Outside layout retained source from the native layout")
            layouts[name].append(dict(ex,fixed=packed,length=n))
    return layouts


@torch.inference_mode()
def fine_view(model,examples,*,inspect=False):
    indices = [i for i,ex in enumerate(examples) if ex is not None]
    dataset = [examples[i] for i in indices]
    values = [None]*len(examples)
    model.eval()
    cursor = 0
    for cpu in DataLoader(dataset,batch_size=BATCH_SIZE,collate_fn=collate_fine):
        batch = to_gpu(cpu)
        cls,residual = model.representations(batch)
        head = model.fine.classifier
        logit = head((cls+residual)[:,None])[:,0]
        if inspect:
            stats = torch.stack((logit,logit.sigmoid(),cls.norm(dim=1),residual.norm(dim=1),
                torch.nn.functional.cosine_similarity(cls,residual,dim=1),head(cls[:,None])[:,0].sigmoid(),
                head(residual[:,None])[:,0].sigmoid()),dim=1).tolist()
            names = ("logit","probability","cls_norm","residual_norm","cosine","cls_only","residual_only")
        else:
            stats,names = logit.sigmoid()[:,None].tolist(),("probability",)
        for row in stats:
            values[indices[cursor]] = dict(zip(names,row,strict=True))
            cursor += 1
    return values


def fine_interventions(split,model,layouts,codebert,path):
    valid = [r is not None for r in layouts["outside"]]
    results,rows = {},[dict(sample_key=r["sample_key"],target=r["target"],codebert_bpe=n,
                           outside_supported=v)
                       for r,n,v in zip(split.rows,codebert,valid,strict=True)]
    for name,examples in layouts.items():
        inference = fine_view(model,examples)
        p = np.array([row["probability"] if row is not None else np.nan for row in inference])
        coverage = [int(ex["fixed"][1].ge(0).sum()) if ex is not None else None for ex in examples]
        results[name] = dict(available=probability_metrics(split.rows,p,np.isfinite(p),codebert),
                             matched_outside=probability_metrics(split.rows,p,valid,codebert),
                             mean_matched_body_bpe=float(np.mean([n for n,v in zip(coverage,valid,strict=True) if v])))
        for row,prob,n in zip(rows,p,coverage,strict=True):
            row[name+"_probability"] = float(prob) if np.isfinite(prob) else None
            row[name+"_body_bpe"] = n
    write_prediction_rows(path,rows)
    return results


def fine_snapshots(split,model,data,results,weights,codebert,tag,seed):
    saved = center.read_predictions(results/"valid.csv.gz",split.rows)
    snapshots,distributions = {},{}
    y = [r["target"] for r in split.rows]
    for name in ("best","last"):
        path = weights/f"{name}.pt"
        state = torch.load(path,map_location="cpu",weights_only=name=="best",mmap=True)
        model.load_state_dict(state if name=="best" else state["model"],strict=True)
        del state
        diagnostics = fine_view(model,data.examples,inspect=True)
        p = [r["probability"] for r in diagnostics]
        if name=="best":
            parity = max(abs(value-float(r["probability"])) for value,r in zip(p,saved,strict=True))
            positions = [json.loads(r["source_runs"]) for r in saved]
            if parity>1e-6 or any([list(v) for v in visible_source(ex)]!=expected
                                  for ex,expected in zip(data.examples,positions,strict=True)):
                raise ValueError(f"{tag}/{seed}: native input or best probability mismatch {parity}")
        snapshots[name] = diagnostics
        distributions[name] = dict(probability_logit=probability_summary(y,p,[r["logit"] for r in diagnostics]),
             by_group=probability_metrics(split.rows,p,[True]*len(y),codebert),
             representations={key:float(np.median([r[key] for r in diagnostics]))
                              for key in ("cls_norm","residual_norm","cosine")},
             residual_removal_abs_delta_median=float(np.median([abs(r["probability"]-r["cls_only"]) for r in diagnostics])),
             fine_removal_abs_delta_median=float(np.median([abs(r["probability"]-r["residual_only"]) for r in diagnostics])),
             branch_only={branch:probability_metrics(split.rows,[r[branch] for r in diagnostics],[True]*len(y),codebert)
                          for branch in ("cls_only","residual_only")})
    rows = [dict(sample_key=r["sample_key"],target=r["target"],codebert_bpe=n,
              **{f"{time}_{k}":v for time,diag in snapshots.items() for k,v in diag[i].items()})
            for i,(r,n) in enumerate(zip(split.rows,codebert,strict=True))]
    write_prediction_rows(DIAG/f"fine_snapshots_{tag}_{seed}.csv.gz",rows)
    return dict(snapshots=distributions,max_best_probability_replay_error=parity)


def select_rows(split,ids):
    return replace(split,rows=[split.rows[i] for i in ids],sources=[split.sources[i] for i in ids],
                   ends=[split.ends[i] for i in ids])


def verify_uniform_inputs(tokenizer,splits):
    source = archived("compression.py").decode().replace("from dataset import BUDGET","")
    archived_module = {"BUDGET":BUDGET}
    exec(compile(source,f"{ARCHIVED_UNIFORM}:compression.py","exec"),archived_module)
    from compaction import tokenized_batches, prepare_function
    result = {}
    for name in splits:
        rows = prime_source_rows(DATA,name)
        archived_rows = original_uniform_predictions(name,rows) if name!="train" else None
        count = maximum = 0
        for members,texts,encoding in tokenized_batches(rows,tokenizer):
            for row,text,ids,offsets in zip(members,texts,encoding.input_ids,encoding.offset_mapping,strict=True):
                expected = archived_module["select"]("uniform4",text,ids,offsets)
                view = prepare_function(text,ids,offsets,tokenizer,alternatives=False)
                n = view["length"]
                original = [None if i<0 else i for i in view["fixed"][1,1:n-1].tolist()]
                if original!=expected or view["fixed"][0,:n].tolist()!=archived_module["model_ids"](ids,expected,tokenizer):
                    raise ValueError(f"Uniform-4 input differs from e094c1e at {row['sample_key']}")
                if archived_rows is not None and tuple(int(archived_rows[count][k]) for k in
                       ("original_bpe","input_length"))!=(len(ids),n):
                    raise ValueError("Archived Uniform-4 input length differs")
                count += 1
                maximum = max(maximum,len(ids))
        result[name] = dict(functions=count,maximum_original_bpe=maximum,exact_positions_and_ids=True)
        print("Uniform-4 archived inputs verified",name,count,flush=True)
    write_json(OUT/"uniform4_parity.json",result)


def evaluate_gpu():
    from diagnostics.optimization import reconstruct_order, run, subset_indices

    DIAG.mkdir(parents=True,exist_ok=True)
    tokenizer = load_tokenizer("unixcoder",UNIX_BASE)
    valid = center.load_split("valid",tokenizer)
    train = center.load_split("train",tokenizer)
    codebert = center.codebert_lengths("valid",valid.rows)
    method_views = {}
    report_file = DIAG/"auxiliary.json"
    if not report_file.is_file():
        for tag in ("longvul","A","C","D"):
            model = load_coarse(tag,SEED)
            data,ranks = selected_data(valid,model,tokenizer)
            method_views[tag] = auxiliary_interventions(valid,model,data,ranks,codebert,tag)
            del model,data,ranks
            print("Auxiliary intervention complete",tag,flush=True)
        write_json(report_file,dict(lengths="CodeBERT BPE",heads=method_views))
    subset = subset_indices(train.sources)
    comparisons = {}
    for seed,stalled in PAIRS:
        model = load_coarse("longvul",seed)
        normal,ranks = selected_data(valid,model,tokenizer)
        designs = counterfactual_examples(valid,tokenizer,normal,ranks)
        set_seed(seed)
        fine = fresh_fine(tokenizer)
        first,_ = reconstruct_order(len(train.rows),len(valid.rows))
        requested = sorted({*subset,*(i for batch in first for i in batch)})
        idx = {value:i for i,value in enumerate(requested)}
        probe_train = select_rows(train,requested)
        summaries = {}
        for tag in ("longvul",stalled):
            index = model if tag=="longvul" else load_coarse(tag,seed)
            uniform = tag=="matched_uniform"
            native,_ = selected_data(valid,index,tokenizer,uniform=uniform) if tag!="longvul" else (normal,ranks)
            paths = model_paths(tag,seed)
            observation = fine_snapshots(valid,fine,native,*paths,codebert,tag,seed)
            fine.load_state_dict(torch.load(paths[1]/"best.pt",map_location="cpu",weights_only=True,mmap=True))
            location = DIAG/f"fine_layout_{tag}_{seed}.csv.gz"
            observation["layouts"] = fine_interventions(valid,fine,designs,codebert,location)
            summaries[tag] = observation
            features,p,_ = center.score_coarse(index,train,"global")
            train_inputs,_ = center.fine_data(probe_train,features[requested],[p[i] for i in requested],
                                              tokenizer,96,uniform=uniform)
            run(seed,tag,lambda:fresh_fine(tokenizer),train_inputs,len(valid.rows),*paths,DIAG,
                [idx[i] for i in subset],[[idx[i] for i in batch] for batch in first],len(train.rows))
            print("Fine and optimization diagnostics complete",tag,seed,flush=True)
            del train_inputs,features,p,native,index
        comparisons[str(seed)] = summaries
        del model,fine
        write_json(DIAG/"fine.json",comparisons)
    # Forward the matched CodeBERT wrapper and its archived detector on the same real samples.
    from compaction import CompactData, collate
    from paired import CODEBERT_UNIFORM, make_model
    tokenizer = load_tokenizer("codebert",BASE)
    data = CompactData("valid",tokenizer)
    ids = sorted(range(len(data)),key=lambda i:data.examples[i]["raw"],reverse=True)[:BATCH_SIZE]
    cpu = collate([data[i] for i in ids])
    set_seed(SEED)
    model = make_model(tokenizer,CODEBERT_UNIFORM).eval()
    batch = to_gpu(cpu,labels=True)
    out = model(batch)
    raw = batch["fixed"][:,0].long()
    mask = torch.arange(raw.shape[1],device=raw.device)[None,:]<batch["lengths"][:,None]
    direct = model.detector(raw,mask.long(),None,None,labels=batch["labels"])
    torch.testing.assert_close(out["logits"],direct["logits"],rtol=0.,atol=0.)
    out["loss"].backward()
    gradient = float(model.detector.classifier.out_proj.weight.grad.norm())
    if gradient<=0:
        raise ValueError("Uniform-4 positive GPU gradient check failed")
    write_json(OUT/"uniform4_smoke.json",dict(functions=len(ids),gradient=gradient,
               maximum_original_bpe=max(int(data.examples[i]["raw"]) for i in ids),equal_detector_logits=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",required=True,choices=("cpu","gpu","report"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage=="cpu":
        verify_uniform_inputs(load_tokenizer("codebert",BASE),("train","valid","test"))
        threshold_report()
    elif args.stage=="report":
        baseline_report()
    else:
        if torch.cuda.device_count()!=1 or "4090" not in torch.cuda.get_device_name():
            raise ValueError("Select one RTX 4090")
        torch.backends.cudnn.deterministic = True
        evaluate_gpu()


if __name__=="__main__":
    main()
