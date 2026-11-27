"""Evaluate visible statement relations and fixed-checkpoint bias removal."""
from collections import Counter
import json
from pathlib import Path
import sys
from statistics import mean, median

import numpy as np
import torch
from torch.utils.data import DataLoader

if not __package__:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import coarse_to_fine as pipeline
from context_selection import contains, visible_source, write_prediction_rows
from diagnostics.context_retrieval import distance_band
from diagnostics.training_diagnosis import probability_metrics
from paired import BATCH_SIZE, SEED, UNIX_BASE, load_tokenizer, to_gpu
from relation_attention import RELATION_TYPES
from relation_encoding_experiment import BASELINE, CONTEXT_BUDGET, METHODS, RESULTS, WEIGHTS, load_index
from rgcl.artifacts import read_json, write_json
from rgcl.config import SEEDS
from window_inputs import GROUPS

DISTANCES = ("0","1-32","33-128","129-512","513+")


def relation_observations(split,data):
    rows = []
    for source,ex,(_,links,_) in zip(split.sources,data.examples,split.syntax,strict=True):
        total,kept = Counter(),Counter()
        relations = {(kind,*sorted((statement,context))) for kind,statement,context in links
                     if kind in RELATION_TYPES and statement!=context and statement[1]>statement[0]
                     and context[1]>context[0]}
        for kind,statement,context in relations:
            band = distance_band(source,statement,context)
            keys = [("type",kind),("distance",band),("all","all")]
            total.update(keys)
            if contains(statement,ex["layout"].spans) and contains(context,ex["layout"].spans):
                kept.update(keys)
        visible = len(ex["relations"])
        if visible!=kept["all","all"]:
            raise ValueError("Visible source relations do not match the mapped attention relations")
        farthest = next((band for band in reversed(DISTANCES) if kept["distance",band]),"none")
        rows.append((total,kept,visible,farthest))
    return rows


def coverage(observations,sizes):
    result = {}
    for group,(lo,hi) in GROUPS.items():
        total,kept = Counter(),Counter()
        functions,with_relations = 0,0
        for n,(available,selected,count,_) in zip(sizes,observations,strict=True):
            if lo<=n<=hi:
                functions += 1
                with_relations += count>0
                total.update(available)
                kept.update(selected)
        report = {"functions":functions,"with_visible_relations":with_relations,
                  "mean_visible_relations":kept["all","all"]/functions if functions else None}
        for dimension,names in (("type",RELATION_TYPES),("distance",DISTANCES),("all",("all",))):
            report[dimension] = {}
            for name in names:
                denominator = total[dimension,name]
                numerator = kept[dimension,name]
                report[dimension][name] = dict(available=denominator,visible=numerator,
                                                fraction=numerator/denominator if denominator else None)
        result[group] = report
    return result


@torch.inference_mode()
def observe(model,data):
    model.eval()
    rows = []
    for batch in DataLoader(data,batch_size=BATCH_SIZE,collate_fn=model.collator):
        out = model(to_gpu(batch),details=True)
        rows.extend(zip(out["probability"].tolist(),out["cls"].norm(dim=-1).tolist(),
                        out["fusion_residual"].norm(dim=-1).tolist(),strict=True))
    return list(zip(*rows,strict=True))


def cohorts(observations):
    counts = np.array([x[2] for x in observations])
    farthest = np.array([x[3] for x in observations])
    return {"visible_relations":counts>0, "no_visible_relations":counts==0,
            "1-4_relations":(counts>=1)&(counts<=4), "5-16_relations":(counts>=5)&(counts<=16),
            "17+_relations":counts>=17,
            **{f"farthest_{band}_bpe":farthest==band for band in DISTANCES}}


def group_diagnostics(rows,probabilities,cls,fusion,sizes,observations,*,zero=None):
    all_samples = np.ones(len(rows),dtype=bool)
    subsets = cohorts(observations)
    result = {name:probability_metrics(rows,probabilities,mask,sizes) for name,mask in subsets.items() if mask.any()}
    groups = {}
    if zero is not None:
        p,z = np.array(probabilities),np.array(zero)
        changed = np.abs(p-z)
        flipped = (p>.5)!=(z>.5)
        effect = dict(metrics=probability_metrics(rows,zero,all_samples,sizes),groups={})
    for group,(lo,hi) in GROUPS.items():
        ids = [i for i,n in enumerate(sizes) if lo<=n<=hi]
        if ids:
            groups[group] = dict(mean_cls_norm=mean(cls[i] for i in ids),median_cls_norm=median(cls[i] for i in ids),
                                 mean_global_residual_norm=mean(fusion[i] for i in ids))
            if zero is not None:
                effect["groups"][group] = dict(mean_probability_change=float(changed[ids].mean()),
                                               max_probability_change=float(changed[ids].max()),
                                               threshold_flips=int(flipped[ids].sum()))
    report = dict(subgroups=result,representation=groups)
    if zero is not None:
        report["zero_bias"] = effect
    return report


def replay_index(index,split,*,compare_devices):
    features,ranks,_ = pipeline.score_coarse(index,split,"global")
    devices = [torch.cuda.current_device()]
    if compare_devices and torch.cuda.device_count()==2:
        with torch.cuda.device(1):
            other_features,other_ranks,_ = pipeline.score_coarse(index.cuda(1),split,"global")
        index.cuda(0)
        torch.testing.assert_close(features,other_features,atol=0.,rtol=0.)
        for a,b in zip(ranks,other_ranks,strict=True):
            np.testing.assert_array_equal(a,b)
        devices.append(1)
    return features,ranks,devices


def audit_split(tokenizer,seed,index,methods,name):
    area = pipeline.seed_root(RESULTS,seed)
    weights = pipeline.seed_root(WEIGHTS,seed)
    split = pipeline.load_split(name,tokenizer)
    features,ranks,devices = replay_index(index,split,compare_devices=seed==SEED)
    data,_ = pipeline.fine_data(split,features,ranks,tokenizer,CONTEXT_BUDGET,with_relations=True)
    observations = relation_observations(split,data)
    sizes = dict(unixcoder=[len(s.raw_ids) for s in split.sources],codebert=pipeline.codebert_lengths(name,split.rows))
    write_json(area/f"{name}_relation_coverage.json",dict(index_replay_devices=devices,
        **{definition:coverage(observations,n) for definition,n in sizes.items()}))
    historical = pipeline.RESULTS/f"seed{seed}"/("global_fusion" if seed==SEED else "final")
    reference = pipeline.read_predictions(historical/f"{name}.csv.gz",split.rows)
    for method in methods:
        location = area/method
        config = read_json(location/"config.json")
        saved = pipeline.read_predictions(location/f"{name}.csv.gz",split.rows)
        for row,ref,ex in zip(saved,reference,data.examples,strict=True):
            if (int(row["input_length"])!=ex["length"]
                    or [tuple(span) for span in json.loads(row["source_runs"])]!=visible_source(ex)
                    or row["source_runs"]!=ref["source_runs"] or row["center_runs"]!=ref["center_runs"]):
                raise ValueError(f"Replayed layout differs for {method} {row['sample_key']}")
        model = pipeline.fine_classifier(tokenizer,config)
        model.load_state_dict(torch.load(weights/method/"best.pt",weights_only=True,map_location="cpu"),strict=True)
        probabilities,cls,fusion = observe(model,data)
        recorded = [float(x["probability"]) for x in saved]
        np.testing.assert_array_equal(probabilities,recorded)
        if method==BASELINE:
            np.testing.assert_array_equal(probabilities,[float(x["probability"]) for x in reference])
        zero = None
        relation_bias = None
        if model.relation_bias is not None:
            relation_bias = model.relation_weights()
            with torch.no_grad():
                model.relation_bias.zero_()
            zero,*_ = observe(model,data)
            no_relations = [i for i,x in enumerate(observations) if not x[2]]
            np.testing.assert_array_equal(np.array(probabilities)[no_relations],np.array(zero)[no_relations])
        rows = [dict(sample_key=r["sample_key"],target=r["target"],probability=p,cls_norm=z,
                     global_residual_norm=g,visible_relations=obs[2],farthest_relation_bpe=obs[3],
                     probability_at_zero_bias=None if zero is None else zero[i])
                for i,(r,p,z,g,obs) in enumerate(zip(split.rows,probabilities,cls,fusion,observations,strict=True))]
        write_prediction_rows(location/f"{name}_relation_diagnostics.csv.gz",rows)
        stats = {definition:group_diagnostics(split.rows,probabilities,cls,fusion,n,observations,zero=zero)
                 for definition,n in sizes.items()}
        write_json(location/f"{name}_relation_diagnostics.json",dict(stats,relation_bias=relation_bias))
        del model


def summarize(selected):
    averages = {}
    for method in (BASELINE,selected):
        runs = [read_json(RESULTS/f"seed{seed}"/method/"metrics.json") for seed in SEEDS]
        group_results = {}
        for definition in ("unixcoder","codebert"):
            sections = [r["split"]["test"][definition] for r in runs]
            group_results[definition] = {}
            for group in GROUPS:
                combined = {}
                for key,fields in (("groups",("mcc","f1","precision","recall","auc","auprc")),
                                   ("pairs",("p_c","pair_ranking_accuracy"))):
                    combined.update({f:pipeline.seed_stats([s[key][group][f] for s in sections]) for f in fields})
                for ed in ("actual_edits","insert_gap","delete_gap"):
                    combined[f"retention_{ed}"] = pipeline.seed_stats(
                        [s["edit_retention"][group][ed]["fraction"] for s in sections])
                group_results[definition][group] = combined
        group_results["cost"] = {key:pipeline.seed_stats([measure(r) for r in runs])
            for key,measure in (("train_seconds",lambda r:r["train_seconds"]),
                                ("training_peak_allocated_bytes",lambda r:r["peak_train_allocated_bytes"]),
                                ("test_seconds",lambda r:r["split"]["test"]["cost"]["measured_end_to_end_seconds"]),
                                ("test_peak_allocated_bytes",lambda r:r["split"]["test"]["cost"]["peak_inference_allocated_bytes"]))}
        averages[method] = group_results
    write_json(RESULTS/"summary.json",dict(seeds=list(SEEDS),selected=selected,three_seed=averages))


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    tokenizer = load_tokenizer("unixcoder",UNIX_BASE)
    selected = read_json(RESULTS/"method_policy.json")["selected"]
    for seed in SEEDS:
        methods = METHODS if seed==SEED else (BASELINE,selected)
        index = load_index(seed,768)
        if seed==SEED:
            train = pipeline.load_split("train",tokenizer)
            _,_,devices = replay_index(index,train,compare_devices=True)
            write_json(RESULTS/"seed42"/"shared_training_index.json",
                       dict(functions=len(train.rows),index_replay_devices=devices))
            del train
        for name in ("valid","test"):
            audit_split(tokenizer,seed,index,methods,name)
        print("Verified inputs, checkpoints and relation diagnostics for seed",seed,flush=True)
    summarize(selected)


if __name__=="__main__":
    main()
