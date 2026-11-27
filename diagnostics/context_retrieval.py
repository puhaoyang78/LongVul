"""Audit context allocation for the two matched retrieval strategies."""
import json
from collections import Counter, defaultdict

import numpy as np
import torch

import coarse_to_fine as pipeline
from coarse_retrieval import part_overlap, retrieve, selected_statement_scores
from compaction import BUDGET
from context_retrieval_experiment import CONTEXT_BUDGET, INDEX, METHODS, RESULTS, load_index
from context_selection import contains, merge, occupied, source_parser
from index_adaptation import grouped_layout_change
from paired import UNIX_BASE, load_tokenizer
from repair_oracle import raw_runs
from rgcl.artifacts import read_json, write_json
from rgcl.config import SEEDS
from window_inputs import GROUPS


def distance_band(source,statement,context):
    first,end = source.keep[statement[0]],source.keep[statement[1]-1]+1
    context_first,context_end = source.keep[context[0]],source.keep[context[1]-1]+1
    distance = max(0,context_first-end,first-context_end)
    return "0" if not distance else "1-32" if distance<=32 else "33-128" if distance<=128 else (
        "129-512" if distance<=512 else "513+")


def score_bands(statement_scores):
    levels = sorted(set(statement_scores.values()))
    if len(levels)<2:
        return dict.fromkeys(statement_scores,"tied")
    bands = {score:("lower","middle","upper")[int(3*(rank+.5)/len(levels))]
             for rank,score in enumerate(levels)}
    return {statement:bands[score] for statement,score in statement_scores.items()}


def related_context_counts(source,ends,scores,layouts,links):
    centers = layouts[0].centers
    reserved = [merge([*x.centers,*x.context]) for x in layouts]
    levels = score_bands(selected_statement_scores(scores,ends,centers,links))
    counts = defaultdict(Counter)
    blocks = {i for a,b in centers for i,_,_ in part_overlap(a,b,ends)}
    highest = set(sorted(blocks,key=lambda i:(-scores[i],i))[:3])
    used_blocks = set()
    retained_blocks = [set() for _ in METHODS]
    for kind,statement,context in links:
        memberships = [("all","all"),("importance",levels[statement]),("relation",kind),
                       ("distance",distance_band(source,statement,context))]
        feasible = occupied(merge([statement,context]))<=BUDGET
        complete = [contains(statement,x.spans) for x in layouts]
        after_context = [contains(context,spans) for spans in reserved]
        final = [contains(context,x.spans) for x in layouts]
        for key in memberships:
            row = counts[key]
            row["candidates"] += 1
            row["individually_fits_fine_budget"] += feasible
            row["shared_complete_statement"] += feasible and all(complete)
            for method,kept,kept_final,whole_statement in zip(METHODS,after_context,final,complete,strict=True):
                row[f"{method}_after_context"] += kept
                row[f"{method}_in_fine_input"] += kept_final
                row[f"{method}_joint_in_fine_input"] += kept_final and whole_statement
                row[f"{method}_with_shared_complete_statement"] += feasible and all(complete) and kept_final
        related_blocks = {i for i,_,_ in part_overlap(*statement,ends) if i in blocks}
        used_blocks.update(related_blocks)
        for accepted,seen in zip(after_context,retained_blocks,strict=True):
            if accepted:
                seen.update(related_blocks)
    concentration = counts["concentration","selected_blocks"]
    concentration["with_candidate_context"] = len(used_blocks)
    concentration["top_three_with_candidate_context"] = len(used_blocks&highest)
    for method,seen in zip(METHODS,retained_blocks,strict=True):
        concentration[f"{method}_with_retained_context"] = len(seen)
        concentration[f"{method}_top_three_with_retained_context"] = len(seen&highest)
    return counts


def group_counts(observations,sizes):
    reports = {}
    for group,(lo,hi) in GROUPS.items():
        counts = defaultdict(Counter)
        for length,source_counts in zip(sizes,observations,strict=True):
            if lo<=length<=hi:
                for key,counts_by_key in source_counts.items():
                    counts[key].update(counts_by_key)
        report = defaultdict(dict)
        for (dimension,key),row in sorted(counts.items()):
            values = dict(row)
            if dimension=="concentration":
                eligible = row["with_candidate_context"]
                top = row["top_three_with_candidate_context"]
                for method in METHODS:
                    relevant = row[f"{method}_with_retained_context"]
                    high = row[f"{method}_top_three_with_retained_context"]
                    values[f"{method}_top_three_share"] = high/relevant if relevant else None
                    values[f"{method}_block_coverage"] = relevant/eligible if eligible else None
                    values[f"{method}_non_top_three_coverage"] = (relevant-high)/(eligible-top) if eligible>top else None
            else:
                candidates,shared = row["candidates"],row["shared_complete_statement"]
                for method in METHODS:
                    for stage in ("after_context","in_fine_input","joint_in_fine_input","with_shared_complete_statement"):
                        n = shared if stage=="with_shared_complete_statement" else candidates
                        values[f"{method}_{stage}_rate"] = row[f"{method}_{stage}"]/n if n else None
            report[dimension][key] = values
        reports[group] = dict(report)
    return reports


def verify_split(tokenizer,seed,name,area,index):
    split = pipeline.load_split(name,tokenizer)
    features,scores,_ = pipeline.score_coarse(index,split,"global")
    checked_devices = [torch.cuda.current_device()]
    if seed==SEEDS[-1] and torch.cuda.device_count()==2:
        with torch.cuda.device(1):
            other_features,other_scores,_ = pipeline.score_coarse(index.cuda(1),split,"global")
        index.cuda(0)
        for left,right in zip(scores,other_scores,strict=True):
            np.testing.assert_array_equal(left,right)
        torch.testing.assert_close(features,other_features,atol=0.,rtol=0.)
        checked_devices.append(1)
    saved = [pipeline.read_predictions(area/m/f"{name}.csv.gz",split.rows) for m in METHODS]
    previous_name = "global_fusion" if seed==SEEDS[0] else "final"
    previous = pipeline.read_predictions(pipeline.seed_root(pipeline.RESULTS,seed)/previous_name/f"{name}.csv.gz",
                                         split.rows)
    observed = []
    visible = [[],[]]
    equal_layouts = 0
    parser = source_parser()
    for source,ends,p,first,second in zip(split.sources,split.ends,scores,*saved,strict=True):
        links = []
        layouts = [retrieve(p,source,ends,CONTEXT_BUDGET,parser,candidate_relations=links),
                   retrieve(p,source,ends,CONTEXT_BUDGET,parser,importance_guided=True)]
        if layouts[0].centers!=layouts[1].centers:
            raise ValueError("Replayed code blocks differ")
        for x,(layout,record) in enumerate(zip(layouts,(first,second),strict=True)):
            source_runs = raw_runs(source.keep,layout.spans)
            visible[x].append(source_runs)
            fields = (("source_runs",source_runs),("center_runs",layout.centers),
                      ("context_runs",layout.context),("fill_runs",layout.fill))
            if any([tuple(span) for span in json.loads(record[key])]!=value for key,value in fields):
                raise ValueError(f"Replay does not match the saved layout for {record['sample_key']}")
        equal_layouts += layouts[0].spans==layouts[1].spans
        observed.append(related_context_counts(source,ends,p,layouts,links))
    codebert = pipeline.codebert_lengths(name,split.rows)
    unixcoder = [len(s.raw_ids) for s in split.sources]
    if any([int(item["codebert_bpe"]) for item in rows]!=codebert for rows in saved):
        raise ValueError("CodeBERT length assignment changed")
    result = dict(functions=len(split.rows),index_replay_devices=checked_devices,
                  same_selected_block_layouts=len(split.rows),same_fine_layouts=equal_layouts,
                  published_baseline_same_layouts=sum(x==[tuple(span) for span in json.loads(y["source_runs"])]
                                                       for x,y in zip(visible[0],previous,strict=True)),
                  published_baseline_same_probabilities=sum(a["probability"]==b["probability"]
                                                           for a,b in zip(saved[0],previous,strict=True)))
    for definition,sizes in (("unixcoder",unixcoder),("codebert",codebert)):
        result[definition] = dict(geometry=grouped_layout_change(*visible,sizes),
                                  relations=group_counts(observed,sizes))
    write_json(area/f"{name}_context_audit.json",result)


def summarize():
    names = [pipeline.seed_root(RESULTS,seed) for seed in SEEDS]
    runs = {method:[read_json(path/method/"metrics.json") for path in names] for method in METHODS}
    statistics = {}
    for method,results in runs.items():
        groups = {}
        for definition in ("unixcoder","codebert"):
            groups[definition] = {}
            for group in GROUPS:
                sections = [run["split"]["test"][definition] for run in results]
                values = {}
                for section,fields in (
                    ("groups",("mcc","f1","precision","recall","auc","auprc")),
                    ("pairs",("p_c","pair_ranking_accuracy")),
                ):
                    for field in fields:
                        values[field] = pipeline.seed_stats([r[section][group][field] for r in sections])
                for field in ("actual_edits","insert_gap","delete_gap"):
                    values[f"retention_{field}"] = pipeline.seed_stats(
                        [r["edit_retention"][group][field]["fraction"] for r in sections])
                reports = [r["layout"][group] for r in sections]
                for key in reports[0]:
                    if key.startswith("mean_") or key.endswith("_collision_rate"):
                        values[key] = pipeline.seed_stats([row[key] for row in reports])
                groups[definition][group] = values
        groups["cost"] = {}
        for label,measure in (
            ("fine_train_seconds",lambda r:r["train_seconds"]),
            ("test_seconds",lambda r:r["split"]["test"]["cost"]["measured_end_to_end_seconds"]),
            ("retrieval_seconds",lambda r:r["split"]["test"]["cost"]["retrieval_and_packing_seconds"]),
            ("test_gpu_bytes",lambda r:r["split"]["test"]["cost"]["peak_inference_allocated_bytes"]),
        ):
            groups["cost"][label] = pipeline.seed_stats([measure(r) for r in results])
        statistics[method] = groups
    difference = {definition:{group:pipeline.seed_stats([b-a for a,b in zip(
        *(statistics[method][definition][group]["mcc"]["seeds"] for method in METHODS),strict=True)])
        for group in GROUPS} for definition in ("unixcoder","codebert")}
    indices = [read_json(path/"coarse"/INDEX/"fit.json") for path in names]
    write_json(RESULTS/"summary.json",dict(seeds=list(SEEDS),three_seed=statistics,paired_seed_mcc_delta=difference,
               shared_index_training_seconds=pipeline.seed_stats([run["train_seconds"] for run in indices])))


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    tokenizer = load_tokenizer("unixcoder",UNIX_BASE)
    for seed in SEEDS:
        area = pipeline.seed_root(RESULTS,seed)
        index = load_index(seed,768)
        for name in ("valid","test"):
            verify_split(tokenizer,seed,name,area,index)
        print("Verified both layouts for seed",seed,flush=True)
    summarize()


if __name__=="__main__":
    main()
