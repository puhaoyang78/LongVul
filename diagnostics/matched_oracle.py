"""Audit matched Oracle and Predicted inputs and aggregate saved classification results."""
import json
from collections import Counter

import numpy as np
import torch

import coarse_to_fine as pipeline
from coarse_retrieval import CONTEXT_TYPES
from compaction import BUDGET
from context_selection import contains, intersects, merge, occupied, visible_source
from index_adaptation import grouped_layout_change
from matched_oracle_experiment import RESULTS, WEIGHTS, predicted_config, score_counts
from paired import ROOT, SEED, UNIX_BASE, load_tokenizer, prime_source_rows
from predicted_diff import paired_annotations
from relation_encoding_experiment import RESULTS as PREDICTED_RESULTS, WEIGHTS as PREDICTED_WEIGHTS, load_index
from repair_oracle import EDIT_TYPES, GAP_TYPES, overlap_bpe
from rgcl.artifacts import read_json, write_json
from rgcl.config import SEEDS
from window_inputs import DATA, GROUPS

METHODS = ("predicted_fusion","predicted_fine_only","oracle_fusion","oracle_fine_only")
CLASSIFICATION_METRICS = ("mcc","f1","precision","recall","auc","auprc")
PAIRED_METRICS = ("p_c","pair_ranking_accuracy")


def method_path(seed,method,*,checkpoint=False):
    if method.startswith("oracle_"):
        return pipeline.seed_root(WEIGHTS if checkpoint else RESULTS,seed)/method
    return pipeline.seed_root(PREDICTED_WEIGHTS if checkpoint else PREDICTED_RESULTS,seed)/method.replace(
        "predicted_","standard_",1)


def retrieval_observations(split,predicted,oracle):
    edits = paired_annotations(split.sources)
    syntax = split.syntax if split.syntax is not None else pipeline.source_syntax(split.sources)
    records = []
    for first,second,edit,(_,links,_) in zip(predicted.examples,oracle.examples,edits,syntax,strict=True):
        layouts = [first["layout"],second["layout"]]
        actual = merge(span for kind in EDIT_TYPES for span in edit[kind])
        a,b = visible_source(first),visible_source(second)
        row = Counter(functions=1,with_actual_modifications=bool(actual))
        if actual:
            row["predicted_completely_missed"] = overlap_bpe(a,actual)==0
            row["oracle_completely_missed"] = overlap_bpe(b,actual)==0
        if first["compacted_bpe"] > BUDGET:
            for kind,statement,context in links:
                if (kind in CONTEXT_TYPES and occupied(merge([statement,context]))<=BUDGET
                        and any(intersects(statement,x.centers) for x in layouts)
                        and all(contains(statement,x.spans) for x in layouts)):
                    row["shared_complete_statement_relations"] += 1
                    row["predicted_context_retained"] += contains(context,layouts[0].spans)
                    row["oracle_context_retained"] += contains(context,layouts[1].spans)
        records.append(row)
    return records


def grouped_retrieval(observations,sizes):
    groups = {}
    for group,(lo,hi) in GROUPS.items():
        counts = Counter()
        for length,row in zip(sizes,observations,strict=True):
            if lo<=length<=hi:
                counts.update(row)
        entry = dict(counts)
        n_edits = counts["with_actual_modifications"]
        n_relations = counts["shared_complete_statement_relations"]
        for method in ("predicted","oracle"):
            missed = counts[f"{method}_completely_missed"]
            retained = counts[f"{method}_context_retained"]
            entry[f"{method}_complete_omission"] = dict(missed=missed,eligible=n_edits,
                                                        fraction=missed/n_edits if n_edits else None)
            entry[f"{method}_shared_statement_context"] = dict(retained=retained,eligible=n_relations,
                                                                fraction=retained/n_relations if n_relations else None)
        groups[group] = entry
    return groups


def audit_split(tokenizer,seed,name,index):
    split = pipeline.load_split(name,tokenizer)
    features,predicted_ranks,_ = pipeline.score_coarse(index,split,"global")
    devices = [torch.cuda.current_device()]
    if seed==1234 and torch.cuda.device_count()==2:
        with torch.cuda.device(1):
            other_features,other_ranks,_ = pipeline.score_coarse(index.cuda(1),split,"global")
        index.cuda(0)
        torch.testing.assert_close(features,other_features,atol=0.,rtol=0.)
        for a,b in zip(predicted_ranks,other_ranks,strict=True):
            np.testing.assert_array_equal(a,b)
        devices.append(1)
    oracle_ranks = pipeline.oracle_block_scores(split)
    budget = predicted_config(seed)["context_budget"]
    predicted,_ = pipeline.fine_data(split,features,predicted_ranks,tokenizer,budget)
    oracle,_ = pipeline.fine_data(split,features,oracle_ranks,tokenizer,budget)
    for a,b in zip(predicted.examples,oracle.examples,strict=True):
        torch.testing.assert_close(a["coarse"],b["coarse"],atol=0.,rtol=0.)
        if a["length"]!=b["length"]:
            raise ValueError("Oracle and Predicted have different effective budgets")
    codebert = pipeline.codebert_lengths(name,split.rows)
    for method in METHODS:
        path = method_path(seed,method)
        data = oracle if method.startswith("oracle_") else predicted
        saved = pipeline.read_predictions(path/f"{name}.csv.gz",split.rows)
        config = read_json(path/"config.json")
        if bool(config.get("privileged_retrieval",False))!=method.startswith("oracle_"):
            raise ValueError("Artifact privileged information label differs from its ranking source")
        for record,example,n in zip(saved,data.examples,codebert,strict=True):
            layout = example["layout"]
            fields = (("source_runs",visible_source(example)),("center_runs",layout.centers),
                      ("context_runs",layout.context),("fill_runs",layout.fill))
            if (int(record["input_length"])!=example["length"] or int(record["codebert_bpe"])!=n
                    or any([tuple(span) for span in json.loads(record[k])]!=v for k,v in fields)):
                raise ValueError(f"Replayed positions differ for {record['sample_key']} in {method}")
        model = pipeline.fine_classifier(tokenizer,config)
        model.load_state_dict(torch.load(method_path(seed,method,checkpoint=True)/"best.pt",
                                         weights_only=True,map_location="cpu"),strict=True)
        scores,_ = pipeline.predict_fine(model,data)
        np.testing.assert_array_equal(scores,[float(x["probability"]) for x in saved])
        del model
    observations = retrieval_observations(split,predicted,oracle)
    layouts = [[visible_source(ex) for ex in data.examples] for data in (predicted,oracle)]
    result = dict(functions=len(split.rows),exact_matched_global_states=True,index_replay_devices=devices,
                  privileged_retrieval=True,oracle_scores=score_counts(oracle_ranks))
    for name_key,sizes in (("unixcoder",[len(s.raw_ids) for s in split.sources]),("codebert",codebert)):
        result[name_key] = dict(groups=grouped_retrieval(observations,sizes),
                                layout_jaccard=grouped_layout_change(*layouts,sizes))
    write_json(RESULTS/f"seed{seed}"/f"{name}_retrieval_audit.json",result)


def aggregate():
    runs = {m:[read_json(method_path(s,m)/"metrics.json") for s in SEEDS] for m in METHODS}
    audits = [read_json(RESULTS/f"seed{s}"/"test_retrieval_audit.json") for s in SEEDS]
    averages = {}
    for method,records in runs.items():
        output = {}
        ranking = method.split("_",1)[0]
        for definition in ("unixcoder","codebert"):
            tests = [r["split"]["test"][definition] for r in records]
            output[definition] = {}
            for group in GROUPS:
                values = {}
                for name,fields in (("groups",CLASSIFICATION_METRICS),("pairs",PAIRED_METRICS)):
                    values.update({f:pipeline.seed_stats([t[name][group][f] for t in tests]) for f in fields})
                for kind in ("actual_edits",*GAP_TYPES):
                    values[f"retention_{kind}"] = pipeline.seed_stats(
                        [r["edit_retention"][group][kind]["fraction"] for r in tests])
                layouts = [r["layout"][group] for r in tests]
                values.update({f:pipeline.seed_stats([layout[f] for layout in layouts])
                               for f in layouts[0] if f.startswith("mean_") or f.endswith("_collision_rate")})
                for f in ("complete_omission","shared_statement_context"):
                    values[f] = pipeline.seed_stats(
                        [a[definition]["groups"][group][f"{ranking}_{f}"]["fraction"] for a in audits])
                output[definition][group] = values
        output["cost"] = {f:pipeline.seed_stats([measure(r) for r in records]) for f,measure in (
            ("train_seconds",lambda r:r["train_seconds"]),
            ("peak_train_bytes",lambda r:r["peak_train_allocated_bytes"]),
            ("test_seconds",lambda r:r["split"]["test"]["cost"]["measured_end_to_end_seconds"]),
            ("test_retrieval_seconds",lambda r:r["split"]["test"]["cost"]["retrieval_and_packing_seconds"]),
            ("test_peak_bytes",lambda r:r["split"]["test"]["cost"]["peak_inference_allocated_bytes"]),
        )}
        output["convergence"] = {f:pipeline.seed_stats([r[f] for r in records])
                                 for f in ("last_three_mean_train_loss","one_class_valid_epochs")}
        averages[method] = output
    deltas = {}
    for name,a,b in (("fine_only_minus_fusion_predicted","predicted_fusion","predicted_fine_only"),
                     ("fine_only_minus_fusion_oracle","oracle_fusion","oracle_fine_only"),
                     ("oracle_minus_predicted_fusion","predicted_fusion","oracle_fusion"),
                     ("oracle_minus_predicted_fine_only","predicted_fine_only","oracle_fine_only")):
        deltas[name] = {definition:{g:pipeline.seed_stats([y-x for x,y in zip(
            averages[a][definition][g]["mcc"]["seeds"],averages[b][definition][g]["mcc"]["seeds"],strict=True)])
            for g in GROUPS} for definition in ("unixcoder","codebert")}
    write_json(RESULTS/"summary.json",dict(seeds=list(SEEDS),three_seed=averages,paired_seed_mcc_delta=deltas))


def archived_comparisons():
    rows = prime_source_rows(DATA,"test")
    codebert = pipeline.codebert_lengths("test",rows)
    directory = ROOT/"results"/"unixcoder_repair_oracle"
    retention = read_json(directory/"edit_retention.json")["groups"]
    archive = {}
    for method in ("oracle_diff","oracle_diff_context"):
        predictions = pipeline.read_predictions(directory/method/"test.csv.gz",rows)
        scores = [float(p["probability"]) for p in predictions]
        sizes = [int(p["original_bpe"]) for p in predictions]
        archive[method] = dict(privileged_test=True,seed=SEED,unixcoder=pipeline.classification(rows,scores,sizes),
                               codebert=pipeline.classification(rows,scores,codebert),
                               historical_edit_retention=retention[method])
    published = read_json(ROOT/"results"/"baseline_completion"/"summary.json")
    external = {m:published["three_seed"][m] for m in ("longvul","stagedvulbert","sea","vulnscout")}
    per_seed = {m:published["per_seed"][m] for m in external}
    write_json(RESULTS/"comparisons.json",dict(historical_oracles=archive,external_baselines=dict(
        test_denominators=published["test_denominators"],three_seed=external,per_seed=per_seed)))


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    tokenizer = load_tokenizer("unixcoder",UNIX_BASE)
    for seed in SEEDS:
        index = load_index(seed,768)
        for name in ("valid","test"):
            audit_split(tokenizer,seed,name,index)
        print("Verified matched score inputs and predictions for seed",seed,flush=True)
    aggregate()
    archived_comparisons()


if __name__=="__main__":
    main()
