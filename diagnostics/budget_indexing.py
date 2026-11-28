"""Compare budget-aware retrieval, unchanged controls, and the matched Oracle."""
import argparse
from bisect import bisect_left
import json
from statistics import median

import coarse_to_fine as pipeline
from budget_indexing import (ADAPTED_CONTROLS, ADAPTED_JOINT, BASE_METHODS, FUSION_DIAGNOSTICS, JOINT,
                             JOINT_DIAGNOSTIC, RANKING, RESULTS, RETRIEVAL, winning_method)
from coarse_retrieval import Layout
from context_selection import merge
from diagnostics.matched_oracle import CLASSIFICATION_METRICS, PAIRED_METRICS, method_path
from paired import SEED, UNIX_BASE, load_tokenizer
from predicted_diff import paired_annotations
from relation_encoding_experiment import RESULTS as ORIGINAL
from repair_oracle import raw_runs
from rgcl.artifacts import read_json, write_json
from rgcl.config import SEEDS
from window_inputs import GROUPS

ENCODERS = ("unixcoder","codebert")
COMPARATORS = ("predicted_fusion","oracle_fusion")
POSITIONS = ("sample_key","target","input_length","source_runs","center_runs","context_runs","fill_runs")


def saved_layout(source,record):
    visible = [tuple(span) for span in json.loads(record["source_runs"])]
    spans = merge((bisect_left(source.keep,a),bisect_left(source.keep,b)) for a,b in visible)
    if raw_runs(source.keep,spans)!=visible:
        raise ValueError("Saved source positions do not map to the historical compacted input")
    return Layout(spans,[],[],[])


def archive_retrieval(split,edits,records,sizes):
    layouts = (saved_layout(source,row) for source,row in zip(split.sources,records,strict=True))
    quality = list(pipeline.retrieval_quality_rows(split,layouts,edits))
    return {name:{group:pipeline.retrieval_quality([r for n,r in zip(lengths,quality,strict=True) if lo<=n<=hi])
                  for group,(lo,hi) in GROUPS.items()} for name,lengths in sizes.items()}


def subset(run,archived=None):
    test = run["split"]["test"]
    out = {}
    for length in ENCODERS:
        groups = test[length]
        quality = groups["retrieval"] if archived is None else archived[length]
        out[length] = {group:(
            {f:groups["groups"][group][f] for f in CLASSIFICATION_METRICS} |
            {f:groups["pairs"][group][f] for f in PAIRED_METRICS} | quality[group] |
            {f:groups["layout"][group][f] for f in ("mean_fragments","mean_fragment_bpe",
                "mean_body_bpe","mean_special_positions","fine_pair_collision_rate") if f in groups["layout"][group]})
            for group in GROUPS}
    cost = test["cost"]
    out["index_ap"] = {group:r["ap"] for group,r in test["indexer"].get("unixcoder",{}).get("actual",{}).items()}
    out["cost"] = dict(fine_train_seconds=run["train_seconds"],fine_train_peak_bytes=run["peak_train_allocated_bytes"],
                       test_seconds=cost["measured_end_to_end_seconds"],
                       test_peak_bytes=cost["peak_inference_allocated_bytes"],
                       test_token_positions=cost["unpadded_positions"],test_windows=cost["windows"],
                       test_retrieval_seconds=cost["retrieval_and_packing_seconds"])
    out["valid"] = run["validation"]
    out["convergence"] = {k:run[k] for k in ("one_class_valid_epochs","last_three_mean_train_loss")}
    return out


def aggregate(values):
    collect = lambda rows: {k:pipeline.seed_stats([r[k] for r in rows]) for k in rows[0]}
    output = {name:{g:collect([v[name][g] for v in values]) for g in GROUPS} for name in ENCODERS}
    output.update({k:collect([v[k] for v in values]) for k in ("cost","index_ap","convergence")})
    return output


def gap(a,b):
    keys = ("mcc","retention","omission","block_recall")
    return {length:{g:{k:b[length][g][k]-a[length][g][k] for k in keys} for g in GROUPS} for length in ENCODERS}


def gradient_medians(path):
    measurements = read_json(path)
    result = {}
    for state in ("initial","selected"):
        partitions = measurements[state]
        result[state] = {}
        for group,rows in dict(partitions,overall=[r for rs in partitions.values() for r in rs]).items():
            result[state][group] = {term:{f:median(r["terms"][term][f] for r in rows) for f in
                                        ("loss","gradient_norm","weighted_gradient_norm")}
                                    for term in rows[0]["terms"]}
    return result


def summarize(*,adapted_seeds=False):
    winner = winning_method()
    tokenizer = load_tokenizer("unixcoder",UNIX_BASE)
    split = pipeline.load_split("test",tokenizer)
    edits = paired_annotations(split.sources)
    sizes = dict(unixcoder=[len(s.raw_ids) for s in split.sources],codebert=pipeline.codebert_lengths("test",split.rows))
    baselines = {m:{} for m in COMPARATORS}
    methods = (*BASE_METHODS,JOINT)
    if read_json(RESULTS/"joint_policy.json")["train_adapted_joint"]:
        methods = (*methods,ADAPTED_JOINT,JOINT_DIAGNOSTIC)
    valid_choice = {}
    runs, deltas = {}, {}
    for seed in SEEDS:
        root = RESULTS/f"seed{seed}"
        for method in COMPARATORS:
            path = method_path(seed,method)
            saved = pipeline.read_predictions(path/"test.csv.gz",split.rows)
            baselines[method][str(seed)] = subset(read_json(path/"metrics.json"),archive_retrieval(split,edits,saved,sizes))
        requested = (winner,*ADAPTED_CONTROLS) if adapted_seeds else (winner,)
        for method in methods if seed==SEED else dict.fromkeys(requested):
            path = root/method/"fine"
            run = subset(read_json(path/"metrics.json"))
            index_root = root/FUSION_DIAGNOSTICS.get(method,method)/"index"
            index = read_json(index_root/"fit.json")
            run["index_valid"] = index["selected_retrieval"]
            if (index_root/"supervision_gradients.json").is_file():
                run["supervision_gradients"] = gradient_medians(index_root/"supervision_gradients.json")
            run["cost"].update(index_train_seconds=index["train_seconds"],index_peak_bytes=index["peak_train_allocated_bytes"],
                               index_valid_seconds=index["validation_seconds"],
                               shared_auxiliary_train_seconds=read_json(root/"auxiliary_global"/"fit.json")["train_seconds"])
            runs.setdefault(method,{})[str(seed)] = run
            deltas.setdefault(method,{})[str(seed)] = {
                "relative_to_predicted":gap(baselines["predicted_fusion"][str(seed)],run),
                "remaining_oracle_gap":gap(run,baselines["oracle_fusion"][str(seed)])}
            if method in (RETRIEVAL,RANKING):
                history = read_json(index_root/"epochs.json")
                best_mcc = max(history,key=lambda e:e["validation"]["mcc"])
                best_retrieval = max(history,key=lambda e:e["retrieval_validation"]["retention"])
                valid_choice.setdefault(method,{})[str(seed)] = dict(auxiliary_mcc_selection=best_mcc,
                                                                        retrieval_selection=best_retrieval)
                if method==RETRIEVAL:
                    previous = read_json(ORIGINAL/f"seed{seed}"/"coarse"/"block_index"/"epochs.json")
                    valid_choice[method][str(seed)]["same_training_as_original"] = all(
                        all(a[k]==b[k] for k in ("train_loss","train_bce","train_repair_kl","validation"))
                        for a,b in zip(history,previous,strict=True))
    matched = {}
    rows = {"valid":pipeline.prime_source_rows(pipeline.DATA,"valid"),"test":split.rows}
    for diagnosis,index in FUSION_DIAGNOSTICS.items():
        if diagnosis not in methods:
            continue
        matched[diagnosis] = {}
        seeds = SEEDS if adapted_seeds and diagnosis in ADAPTED_CONTROLS else (SEED,)
        for seed in seeds:
            root = RESULTS/f"seed{seed}"
            paths = [root/m/"fine" for m in (index,diagnosis)]
            index_fit = read_json(root/index/"index"/"fit.json")
            configs = [read_json(path/"config.json") for path in paths]
            if any(c["index_selected_epoch"]!=index_fit["selected_epoch"] for c in configs):
                raise ValueError(f"Seed {seed}: fine classifiers do not use the shared index epoch for {index}")
            matched[diagnosis][str(seed)] = []
            for fold,members in rows.items():
                first,second = (pipeline.read_predictions(path/f"{fold}.csv.gz",members) for path in paths)
                if any(any(a[k]!=b[k] for k in POSITIONS) for a,b in zip(first,second,strict=True)):
                    raise ValueError(f"Seed {seed} {fold}: Fusion diagnostics used different fine retrieval")
                matched[diagnosis][str(seed)].append(fold)
    past = read_json(pipeline.ROOT/"results"/"matched_oracle_512"/"comparisons.json")
    write_json(RESULTS/"summary.json",dict(
        per_seed=(baselines | runs),paired_gaps=deltas,valid_checkpoint_choices=valid_choice,
        three_seed={m:aggregate(list(values.values())) for m,values in (baselines | runs).items() if len(values)==3},
        external_baselines=past["external_baselines"],
        matched_fusion_layouts={m:folds[str(SEED)] for m,folds in matched.items()},
        **({"matched_fusion_layouts_by_seed":matched} if adapted_seeds else {}),
        selected=winner))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapted-seeds",action="store_true",help="Include the matched BCE + KL controls for all seeds")
    summarize(adapted_seeds=parser.parse_args().adapted_seeds)
