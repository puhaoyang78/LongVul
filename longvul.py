"""LongVul at 1024 positions. Only the explicitly privileged Oracle arms use paired inputs."""
import argparse
from collections import Counter
import csv
import fcntl
import gc
import gzip
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
from time import perf_counter

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import set_seed

import budget_selection as selection
from compaction import REGIONS, collate, pack_fixed
from context_selection import contains, intersects, merge, occupied, source_parser
from dependence import (BATCH_BYTES, BATCH_FUNCTIONS, EDGE_TYPES, FRONTEND_ARGS, ISOLATE_BYTES,
                        ISOLATED_TIMEOUT_SECONDS, JAVA, MAX_DEFINITIONS, TIMEOUT_SECONDS, cached_graphs,
                        extract, joern_version, language_suffix)
from paired import SEED, UNIX_BASE, fit_classifier, make_model, predict, to_gpu
from region_locator import Localizer, Regions, region_metrics, score_regions, train_localizer, weak_targets
from repair_oracle import EDIT_TYPES, GAP_TYPES, line_edits, map_edits, overlap_bpe
from source_regions import Region, analyze, map_dependencies, original_runs, simplify, tokenize_sources
from window_inputs import DATA, GROUPS, ROOT, grouped_metrics
from rgcl.artifacts import write_json
from rgcl.config import (ADAM_EPSILON, BACKBONES, BATCH_SIZE, DATASETS, EPOCHS, LEARNING_RATE, MAX_GRAD_NORM,
                         SPLITS, WARMUP_FRACTION)
from rgcl.data import prime_source_rows
from rgcl.metrics import metrics
from rgcl.model import load_tokenizer
from rgcl.training import LOSVERAdamW

MAX_LENGTH = 1024
BUDGET = MAX_LENGTH-2
LENGTHS = {**GROUPS, "le1022": (0, 1022), "gt1022": (1023, float("inf"))}
RESULTS = ROOT / "results" / "longvul1024"
WEIGHTS = ROOT / "checkpoints" / "longvul1024"
GRAPH_CACHE = WEIGHTS / "native_cpg"
CONTROLS = ("uniform", "syntax", "oracle_diff", "oracle_syntax")
NATIVE = {"oracle_ddg": (1, 0), "oracle_cdg": (0, 1), "oracle_pdg": (1, 1),
          "oracle_pdg_1_2": (1, 2), "oracle_pdg_2_1": (2, 1)}
TUNED = ("oracle_pdg", "oracle_pdg_1_2", "oracle_pdg_2_1")
DEPLOYABLE = ("uniform", "syntax", "predicted", "longvul")


def prepare_graphs():
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    with (WEIGHTS / "joern.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (RESULTS / "joern.json").exists():
            build_graph_cache()


def build_graph_cache():
    costs = {}
    for split in SPLITS:
        statuses, edges, languages = Counter(), Counter(), Counter()
        times, rss, failures = Counter(), 0, []
        for batch, (graphs, cost) in enumerate(cached_graphs(prime_source_rows(DATA, split), GRAPH_CACHE / split, prepare=True)):
            statuses.update(g["status"] for g in graphs)
            languages.update(g["language"] for g in graphs)
            for graph in graphs:
                edges.update(graph["counts"])
            times["seconds"] += cost["seconds"]
            times["decode_seconds"] += cost.get("decode_seconds", 0.)
            edges["ddg_definition_limit_messages"] += cost.get("ddg_definition_limit_messages", 0)
            for stage in ("parse", "export"):
                if stage in cost:
                    times[f"{stage}_seconds"] += cost[stage]["seconds"]
                    rss = max(rss, cost[stage]["peak_rss_kib"])
            if "failure" in cost:
                failures.append(dict(batch=batch, functions=len(graphs), reason=cost["failure"].splitlines()[0]))
        costs[split] = dict(statuses=dict(statuses), languages=dict(languages), native_edges=dict(edges), timings=dict(times),
                            subprocess_peak_rss_kib=rss, batch_failures=failures)
        print("Joern complete", split, dict(statuses), flush=True)
    result = dict(version=joern_version(), source="original individual functions", splits=costs,
                  java=subprocess.check_output([str(JAVA/"bin/java"), "-version"], stderr=subprocess.STDOUT, text=True).splitlines()[0],
                  frontend="NEWC", frontend_args=FRONTEND_ARGS, max_definitions=MAX_DEFINITIONS,
                  batch_functions=BATCH_FUNCTIONS, batch_bytes=BATCH_BYTES, isolated_bytes=ISOLATE_BYTES,
                  timeout_seconds=TIMEOUT_SECONDS, isolated_timeout_seconds=ISOLATED_TIMEOUT_SECONDS)
    write_json(RESULTS / "joern.json", result)


def load_dependencies(rows, sources, split):
    start = perf_counter()
    graphs, aggregate, statuses = [], Counter(), Counter()
    iterator = ((g, cost) for batch, cost in cached_graphs(rows, GRAPH_CACHE / split) for g in batch)
    extreme = max(range(len(sources)), key=lambda i: sources[i].raw_bpe)
    for i, (source, (graph, cost)) in enumerate(zip(sources, iterator, strict=True)):
        mapped, stats = map_dependencies(source, graph)
        graphs.append(dict(edges=mapped, status=graph["status"], stats=stats))
        aggregate.update(stats)
        statuses[graph["status"]] += 1
        if i == extreme:
            largest = dict(idx=rows[i]["idx"], original_bpe=source.raw_bpe, simplified_bpe=len(source.ids),
                           status=graph["status"], mapping=stats, batch_timing={k:v for k,v in cost.items() if k != "failure"})
        for kind in EDGE_TYPES:
            aggregate[f"{kind}_usable_functions"] += graph["status"] == "ok" and bool(mapped[kind])
    return graphs, dict(aggregate, mapping_seconds=perf_counter()-start, statuses=dict(statuses), largest=largest)


def paired_edits(sources):
    """Called for privileged selection or held-out diagnostics, never by source-only inference."""
    simplified, original = [], []
    for left, right in zip(sources[::2], sources[1::2], strict=True):
        lines = [s.code.splitlines(keepends=True) for s in (left, right)]
        diff = line_edits(*lines)
        for side, source in enumerate((left, right)):
            original.append(map_edits(lines[side], source.raw_offsets, diff, side))
            simplified.append(map_edits(lines[side], source.source_offsets, diff, side)
                              if len(source.ids) else {kind: [] for kind in (*EDIT_TYPES, *GAP_TYPES)})
    return simplified, original


class SelectedData(Dataset):
    def __init__(self, rows, sources, tokenizer, method, *, weights=(0, 0), scores=None, graphs=None, edits=None,
                 budget=BUDGET, continuous_fill=False):
        started = perf_counter()
        self.metadata, self.examples, self.decisions = [], [], []
        self.targets = torch.tensor([row["target"] for row in rows])
        self.parser = source_parser() if method in {"syntax", "oracle_syntax"} else None
        uniform = torch.full((REGIONS, 4), -1, dtype=torch.int32)
        for i, (row, source) in enumerate(zip(rows, sources, strict=True)):
            graph, graph_status = selection.available_graph(graphs[i] if graphs is not None else None, weights)
            active_weights = weights if graph is not None else (0, 0)
            spans, anchors, steps, retained = selection.choose(
                source, method, budget, weights=active_weights, graph=graph,
                scores=scores[i] if scores is not None else None, edits=edits[i] if edits is not None else None,
                parser=self.parser, continuous_fill=continuous_fill)
            packed, length = pack_fixed(source.ids, range(len(source.ids)), spans, tokenizer, max_length=budget+2)
            if length != occupied(spans)+2 or any(not contains(span, spans) for span in (*retained, *anchors)):
                raise ValueError("Packed input cost or preselected coverage differs from selection")
            if len(source.ids) > budget and length != budget+2:
                raise ValueError("Long input failed to fill its position budget")
            self.examples.append(dict(fixed=packed, length=length, raw=source.raw_bpe, uniform=uniform, bank=None,
                                      label=row["target"]))
            self.metadata.append(dict(idx=row["idx"], sample_key=row["sample_key"], target=row["target"],
                                      original_bpe=source.raw_bpe, simplified_bpe=len(source.ids)))
            if any(weights):
                self.metadata[-1]["dependency_status"] = graph_status
                self.metadata[-1]["native_graph_status"] = graphs[i]["status"]
            self.decisions.append((spans, anchors, steps, retained))
        self.seconds = perf_counter()-started

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        return self.examples[index]


def collate_long(examples):
    batch = collate(examples)
    width = min(batch["fixed"].shape[-1], 8*((max(item["length"] for item in examples)+7)//8))
    batch["fixed"] = batch["fixed"][..., :width].contiguous()
    return batch


def report_data(data, sources, probabilities, path, *, include_pairs=False):
    rows = []
    probabilities = [None]*len(data) if probabilities is None else probabilities
    for meta, example, decisions, source, score in zip(data.metadata, data.examples, data.decisions, sources,
                                                        probabilities, strict=True):
        spans, anchors, steps, _ = decisions
        classification = {} if score is None else dict(probability=score, prediction=int(score > .5))
        rows.append(dict(**meta, **classification, input_length=example["length"],
                         source_bpe=sum(b-a for a,b in spans), source_runs=original_runs(source, spans),
                         simplified_runs=spans, oracle_anchors=anchors, optimizer_steps=steps))
    if include_pairs:
        active = [ex["fixed"][0, :ex["length"]].tolist() for ex in data.examples]
        for i in range(0, len(active), 2):
            same = active[i] == active[i+1]
            rows[i]["identical_pair_input"] = rows[i+1]["identical_pair_input"] = same
    if path is None:
        return rows
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})
    tmp.replace(path)
    return rows


def read_predictions(path):
    with gzip.open(path, "rt", newline="") as stream:
        return [{k: json.loads(v) if k in {"source_runs", "simplified_runs", "oracle_anchors"} else v
                 for k,v in row.items()} for row in csv.DictReader(stream)]


def input_report(rows, sources, graphs, simple_edits, raw_edits):
    diagnostics = []
    for row, source, graph, edits, raw in zip(rows, sources, graphs, simple_edits, raw_edits, strict=True):
        spans = row["simplified_runs"]
        kept = selection.region_ids(source.regions, spans)
        touched = selection.region_ids(source.regions, spans, touch=True)
        statements = {i for i,r in enumerate(source.regions) if r.kind == "statement"}
        edits_touch = selection.region_ids(source.regions, [r for ranges in edits.values() for r in ranges], touch=True)
        edges = {}
        for kind, native in graph["edges"].items():
            related = [(a,b) for a,b in native if b in edits_touch]
            edges[kind] = [len(native), sum(a in kept and b in kept for a,b in native),
                           len(related), sum(a in kept and b in kept for a,b in related)]
        totals = dict(raw)
        totals["actual_edits"] = merge([r for kind in EDIT_TYPES for r in raw[kind]])
        edits_kept = {kind: [overlap_bpe(intervals, row["source_runs"]), sum(b-a for a,b in intervals)]
                      for kind, intervals in totals.items()}
        quartiles = [source.raw_bpe*i//REGIONS for i in range(REGIONS+1)]
        diagnostics.append(dict(length=source.raw_bpe, simplified_bpe=len(source.ids),
                                pieces=[b-a for a,b in spans], specials=int(row["input_length"])-int(row["source_bpe"]),
                                input_length=int(row["input_length"]),
                                raw_retained=sum(b-a for a,b in row["source_runs"]),
                                covered_quartiles=sum(a < b and intersects((a,b), row["source_runs"])
                                                     for a,b in zip(quartiles, quartiles[1:])),
                                statements=[len(kept & statements), len(touched & statements), len(statements)],
                                edges=edges, edits=edits_kept))
    pairs = [(max(sources[i].raw_bpe, sources[i+1].raw_bpe), str(rows[i]["identical_pair_input"]) == "True")
             for i in range(0, len(rows), 2)] if "identical_pair_input" in rows[0] else []
    result = {}
    for name, (lo, hi) in LENGTHS.items():
        group = [r for r in diagnostics if lo <= r["length"] <= hi]
        if not group:
            continue
        fragments = [length for row in group for length in row["pieces"]]
        n, total = len(group), sum(fragments)
        counts = {kind: [sum(r["edits"][kind][i] for r in group) for i in range(2)]
                  for kind in diagnostics[0]["edits"]}
        pair_inputs = [same for length, same in pairs if lo <= length <= hi]
        full, touched, available = (sum(r["statements"][i] for r in group) for i in range(3))
        specials = sum(r["specials"] for r in group)
        raw, retained = sum(r["length"] for r in group), sum(r["raw_retained"] for r in group)
        result[name] = dict(functions=n, body_bpe=total, mean_body_bpe=total/n, fragments=len(fragments),
                            mean_fragments=len(fragments)/n, mean_fragment_bpe=float(np.mean(fragments)) if fragments else None,
                            median_fragment_bpe=float(np.median(fragments)) if fragments else None,
                            fragment_counts={label: sum(start <= v <= end for v in fragments) for label,start,end in
                                             (("1-8",1,8),("9-32",9,32),("33-128",33,128),("129+",129,float("inf")))},
                            special_tokens=specials, internal_separators=specials-2*n,
                            max_input_length=max(r["input_length"] for r in group),
                            original_bpe=dict(retained=retained, available=raw, fraction=retained/raw if raw else None),
                            mean_covered_source_quartiles=sum(r["covered_quartiles"] for r in group)/n,
                            pairs=len(pair_inputs),
                            identical_pair_inputs=sum(pair_inputs),
                            statements=dict(fully_retained=full, touched=touched, available=available,
                                            complete_among_touched=full/touched if touched else None),
                            edit_bpe={kind: dict(retained=a, available=b, recall=a/b if b else None)
                                      for kind,(a,b) in counts.items()},
                            native_relations={kind: dict(zip(("mapped", "both_retained", "to_edit_region",
                                                              "to_edit_region_both_retained"),
                                                             [sum(r["edges"][kind][i] for r in group) for i in range(4)]))
                                              for kind in EDGE_TYPES})
        if "dependency_status" in rows[0]:
            result[name]["dependency_status"] = dict(Counter(row["dependency_status"] for row in rows
                                                       if lo <= int(row["original_bpe"]) <= hi))
    return result


def matched_graph_metrics(rows, sources, graphs):
    both = [i for i,g in enumerate(graphs) if g["status"] == "ok" and all(g["edges"][kind] for kind in EDGE_TYPES)]
    groups = {"both_mapped": both, "both_mapped_needing_selection": [i for i in both if len(sources[i].ids) > BUDGET]}
    return {name: metrics([int(rows[i]["target"]) for i in indices], [float(rows[i]["probability"]) for i in indices])
            for name,indices in groups.items() if indices}


def config_for(method, weights, micro):
    config = dict(method=method, seed=SEED, epochs=EPOCHS, batch_size=BATCH_SIZE, micro_batch=micro,
                length=MAX_LENGTH, lr=LEARNING_RATE, eps=ADAM_EPSILON, optimizer="LOSVERAdamW", weight_decay=0.,
                warmup=WARMUP_FRACTION, grad_clip=MAX_GRAD_NORM, precision="BF16 autocast, FP32 parameters, SDPA",
                model_id=BACKBONES["unixcoder"][0], model_revision=BACKBONES["unixcoder"][1],
                split_functions=dict(zip(SPLITS, DATASETS["primevul_paired"], strict=True)),
                loss="independent function BCE", checkpoint="full validation MCC; earliest tie", threshold=.5,
                comparison=">", privileged_test=method.startswith("oracle_"),
                ddg_cdg_weights=list(weights), edge_hops=1, bridge_bpe=selection.BRIDGE_BPE,
                fragment_limit=selection.MAX_FRAGMENTS, new_fragment_cost=selection.NEW_FRAGMENT_COST,
                gpu=torch.cuda.get_device_name(),
                cuda=torch.version.cuda, python=platform.python_version(), torch=torch.__version__,
                transformers=transformers.__version__, joern=joern_version())
    if any(weights):
        config["missing_graph_policy"] = "same anchors and uniform fill; require all requested native relation types"
    return config


def loader(data, micro):
    return DataLoader(data, batch_size=micro, collate_fn=collate_long)


def timed_predict(model, data, micro):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = perf_counter()
    values, _ = predict(model, loader(data, micro), bf16=True)
    return values, dict(inference_seconds=perf_counter()-before,
                        peak_inference_allocated_bytes=torch.cuda.max_memory_allocated())


def fit_method(method, weights, micro, tokenizer, rows, sources, *, edits=None, scores=None, graphs=None):
    out, checkpoints = RESULTS / method, WEIGHTS / method
    config = config_for(method, weights, micro)
    done = out / "validation.json"
    if done.is_file():
        completed = json.loads(done.read_text())
        if completed["config"] != config or not (checkpoints / "best.pt").is_file():
            raise ValueError(f"Saved classifier run differs from configuration for {method}")
        return completed
    training = []
    for split in ("train", "valid"):
        training.append(SelectedData(rows[split], sources[split], tokenizer, method, weights=weights,
                                     edits=edits[split] if edits is not None else None,
                                     graphs=graphs[split] if graphs is not None else None,
                                     scores=scores[split] if scores is not None else None))
    train, valid = training
    print(method, "selection ready", round(train.seconds, 2), round(valid.seconds, 2), flush=True)
    set_seed(SEED)
    model = make_model(tokenizer, method, long_input=True)
    selected, history = fit_classifier(model, train, valid, checkpoints, out, config, micro_batch=micro,
                                       bf16=True, collator=collate_long)
    probability, cost = timed_predict(model, valid, micro)
    groups = grouped_metrics(valid, probability, LENGTHS)
    if groups["overall"] != selected["validation"]:
        raise ValueError(f"{method}: validation checkpoint metrics did not reproduce")
    report_data(valid, sources["valid"], probability, out / "valid.csv.gz")
    result = dict(config=config, epoch=selected["epoch"], groups=groups, **cost,
                  training_seconds=sum(r["training_seconds"] for r in history),
                  peak_training_allocated_bytes=max(r["peak_allocated_bytes"] for r in history),
                  selection_seconds=dict(train=train.seconds, valid=valid.seconds))
    if any(weights):
        result["dependency_status"] = {split: dict(Counter(row["dependency_status"] for row in data.metadata))
                                       for split, data in (("train", train), ("valid", valid))}
    write_json(done, result)
    print(method, "validation complete", selected["epoch"], groups["overall"]["mcc"], flush=True)
    return result


def test_method(method, weights, micro, tokenizer, rows, sources, *, edits=None, scores=None, graphs=None):
    out = RESULTS / method
    result_path = out / "test.json"
    if result_path.is_file():
        return json.loads(result_path.read_text()), read_predictions(out / "test.csv.gz")
    data = SelectedData(rows, sources, tokenizer, method, weights=weights, edits=edits, scores=scores, graphs=graphs)
    model = make_model(tokenizer, method, long_input=True, pretrained=False)
    model.load_state_dict(torch.load(WEIGHTS / method / "best.pt", weights_only=True, map_location="cpu"), strict=True)
    probability, cost = timed_predict(model, data, micro)
    result = dict(groups=grouped_metrics(data, probability, LENGTHS), **cost, selection_seconds=data.seconds,
                  privileged_test=method.startswith("oracle_"))
    predictions = report_data(data, sources, probability, out / "test.csv.gz", include_pairs=True)
    write_json(result_path, result)
    print(method, "test complete", result["groups"]["overall"]["mcc"], flush=True)
    return result, predictions


def summarize_source(sources, timings):
    summary = dict(**timings, functions=len(sources), raw_bpe=sum(s.raw_bpe for s in sources),
                   simplified_bpe=sum(len(s.ids) for s in sources),
                   original_gt1022=sum(s.raw_bpe > BUDGET for s in sources),
                   selection_gt1022=sum(len(s.ids) > BUDGET for s in sources),
                   tree_sitter_errors=sum(s.parse_error for s in sources),
                   lexical_status=dict(Counter(s.simplified_by for s in sources)))
    summary["saved_bpe"] = summary["raw_bpe"]-summary["simplified_bpe"]
    return summary


def quality_cohort(rows, tokenizer, parser):
    pairs = list(range(0, len(rows), 2))
    indices = sorted(set(pairs[:20]) | {pairs[i] for i in np.linspace(0, len(pairs)-1, num=96, dtype=int)})
    tags, picked = {}, set(indices[:5])
    for i in indices:
        codes = [r["func"] for r in rows[i:i+2]]
        if max(len(c.encode()) for c in codes) > 64_000:
            continue
        available = {language_suffix(c, parser) for c in codes}
        hunk = line_edits(*(c.splitlines(keepends=True) for c in codes))
        kinds = {entry[0] for entry in hunk}
        lengths = [len(seq) for seq in tokenizer(codes, add_special_tokens=False, verbose=False).input_ids]
        for kind, found in (("C", ".c" in available), ("C++", ".cpp" in available),
                            ("macro", any(re.search(r"(?m)^\s*#", c) for c in codes)),
                            ("typedef", any(re.search(r"\btypedef\b", c) for c in codes)),
                            ("pure_insert", kinds == {"insert"}), ("pure_delete", kinds == {"delete"}),
                            ("multi_block", len(hunk) > 1), ("short", max(lengths) <= BUDGET),
                            ("long", max(lengths) > BUDGET)):
            if found and kind not in tags:
                tags[kind] = i
                picked.add(i)
        if len(tags) == 9:
            break
    for frac in (.25, .5, .75, .98):
        picked.add(pairs[int(frac*(len(pairs)-1))])
    return [row for i in sorted(picked) for row in rows[i:i+2]], tags


def quality_check(tokenizer, parser):
    output = {}
    for split in SPLITS:
        rows = prime_source_rows(DATA, split)
        samples, tags = quality_cohort(rows, tokenizer, parser)
        codes = [row["func"] for row in samples]
        features, _ = tokenize_sources(codes, tokenizer, parser, BUDGET)
        native, timings = extract(codes)
        observations, totals = [], Counter()
        for row, source, graph in zip(samples, features, native, strict=True):
            edges, counts = map_dependencies(source, graph)
            totals.update(counts)
            for kind in EDGE_TYPES:
                totals[f"{kind}_native_functions"] += bool(graph["counts"].get(kind))
                totals[f"{kind}_mapped_functions"] += bool(edges[kind])
            observations.append(dict(idx=row["idx"], sample_key=row["sample_key"], original_bpe=source.raw_bpe,
                                     language=graph["language"], status=graph["status"], file_in_cpg=graph["file_in_cpg"],
                                     method_count=graph["methods"], counts=counts,
                                     has_cfg=bool(graph["counts"].get("CFG")),
                                     cpg_structure=graph["structure"]))
        if not all(totals[f"{kind}_mapped_functions"] for kind in EDGE_TYPES):
            raise ValueError(f"Joern quality cohort for {split} has no mapped edges of a required type")
        output[split] = dict(functions=len(samples), cases={kind: rows[i]["sample_key"] for kind,i in tags.items()},
                             statuses=dict(Counter(x["status"] for x in observations)), totals=dict(totals),
                             timing=timings, members=observations)
        print("Joern quality", split, len(samples), output[split]["statuses"],
              {kind: totals[f"{kind}_mapped_functions"] for kind in EDGE_TYPES}, flush=True)
    return output


def test_retention(method, test_rows, sources, tokenizer, graphs, simpl, raw, weights, scores):
    data = SelectedData(test_rows, sources, tokenizer, method, weights=weights,
                        edits=simpl if method.startswith("oracle_") else None, scores=scores,
                        graphs=graphs if any(weights) else None, budget=510)
    rows = [dict(simplified_runs=spans, source_runs=original_runs(source, spans), source_bpe=sum(b-a for a,b in spans),
                 input_length=ex["length"]) for source, ex, (spans, _, _, _) in
            zip(sources, data.examples, data.decisions, strict=True)]
    return dict(selection_seconds=data.seconds, groups=input_report(rows, sources, graphs, simpl, raw))


def run(tokenizer, micro):
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    graph_worker = None
    if not (RESULTS / "joern.json").is_file():
        with (WEIGHTS / "joern.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("Joining the active native graph extraction", flush=True)
            else:
                graph_worker = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "graphs"],
                                                start_new_session=True)
    parser = source_parser()
    rows, sources, source_summary, edits = {}, {}, {}, {}
    for split in ("train", "valid"):
        rows[split] = prime_source_rows(DATA, split)
        sources[split], elapsed = tokenize_sources([r["func"] for r in rows[split]], tokenizer, parser, BUDGET)
        source_summary[split] = summarize_source(sources[split], elapsed)
        before = perf_counter()
        edits[split], _ = paired_edits(sources[split])
        source_summary[split]["diff_seconds"] = perf_counter()-before
    write_json(RESULTS / "source.json", source_summary)
    validation = {}
    for method in CONTROLS:
        validation[method] = fit_method(method, (0,0), micro, tokenizer, rows, sources,
                                        edits=edits if method.startswith("oracle_") else None)
    targets = {split: weak_targets(sources[split], edits[split]) for split in ("train", "valid")}
    localizer, estimates, localizer_result = train_localizer(sources, targets, WEIGHTS / "localizer", RESULTS / "localizer")
    localizer.cpu()
    write_json(RESULTS / "localizer" / "metrics.json", localizer_result)
    if graph_worker is not None and graph_worker.wait():
        raise RuntimeError("Joern extraction process failed. See the native process log")
    if not (RESULTS / "joern.json").is_file():
        with (WEIGHTS / "joern.lock").open("r") as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            if not (RESULTS / "joern.json").is_file():
                raise RuntimeError("Native graph extraction finished without its result summary")
    graphs, graph_summary = {}, {}
    for split in ("train", "valid"):
        graphs[split], graph_summary[split] = load_dependencies(rows[split], sources[split], split)
        if any(not graph_summary[split][f"{kind}_usable_functions"] for kind in EDGE_TYPES):
            raise RuntimeError(f"{split}: no mapped native edges of a required type")
    write_json(RESULTS / "dependence_mapping.json", graph_summary)
    for method, weights in NATIVE.items():
        validation[method] = fit_method(method, weights, micro, tokenizer, rows, sources, edits=edits, graphs=graphs)
    winner = max(TUNED, key=lambda name: validation[name]["groups"]["overall"]["mcc"])
    chosen = NATIVE[winner]
    write_json(RESULTS / "weights.json", dict(candidates={name: dict(weights=NATIVE[name],
               mcc=validation[name]["groups"]["overall"]["mcc"], epoch=validation[name]["epoch"]) for name in TUNED},
               winner=winner, weights=chosen, selection_data="full Valid; prefer equal weights on ties",
               test_evaluation=["oracle_pdg", winner] if winner != "oracle_pdg" else [winner]))
    validation["predicted"] = fit_method("predicted", (0,0), micro, tokenizer, rows, sources, scores=estimates)
    validation["longvul"] = fit_method("longvul", chosen, micro, tokenizer, rows, sources, scores=estimates, graphs=graphs)
    del sources, targets, rows, edits, graphs, estimates
    gc.collect()

    test_rows = prime_source_rows(DATA, "test")
    test_sources, elapsed = tokenize_sources([r["func"] for r in test_rows], tokenizer, parser, BUDGET)
    source_summary["test"] = summarize_source(test_sources, elapsed)
    test_graphs, graph_summary["test"] = load_dependencies(test_rows, test_sources, "test")
    write_json(RESULTS / "dependence_mapping.json", graph_summary)
    localizer.cuda()
    unlabeled = Regions(test_sources)
    scores, seconds = score_regions(localizer, unlabeled)
    scores = unlabeled.regroup(scores)
    localizer.cpu()
    del unlabeled
    outcomes, saved = {}, {}
    for method in DEPLOYABLE:
        outcomes[method], saved[method] = test_method(method, chosen if method == "longvul" else (0,0), micro,
                                                       tokenizer, test_rows, test_sources,
                                                       scores=scores if method in {"longvul", "predicted"} else None,
                                                       graphs=test_graphs if method == "longvul" else None)
    # After all source-only predictions have been written, open counterpart information for diagnostics and oracles.
    before = perf_counter()
    simpl, raw = paired_edits(test_sources)
    source_summary["test"]["diff_seconds"] = perf_counter()-before
    write_json(RESULTS / "source.json", source_summary)
    labels = weak_targets(test_sources, simpl)
    localizer_result["splits"]["test"] = dict(region_metrics(scores, labels), inference_seconds=seconds)
    write_json(RESULTS / "localizer" / "metrics.json", localizer_result)
    privileged = dict(oracle_diff=(0,0), oracle_syntax=(0,0), oracle_ddg=(1,0), oracle_cdg=(0,1), oracle_pdg=(1,1))
    privileged[winner] = chosen
    for method, weights in privileged.items():
        outcomes[method], saved[method] = test_method(method, weights, micro, tokenizer, test_rows, test_sources,
                                                       edits=simpl, graphs=test_graphs if any(weights) else None)
    for method, predictions in saved.items():
        outcomes[method]["inputs"] = input_report(predictions, test_sources, test_graphs, simpl, raw)
        outcomes[method]["matched_graphs"] = matched_graph_metrics(predictions, test_sources, test_graphs)
        write_json(RESULTS / method / "test.json", outcomes[method])
    budgets = {}
    for method in ("uniform", "syntax", "oracle_diff", "predicted", "longvul"):
        budgets[method] = test_retention(method, test_rows, test_sources, tokenizer, test_graphs, simpl, raw,
                                        chosen if method == "longvul" else (0,0),
                                        scores if method in {"predicted", "longvul"} else None)
    write_json(RESULTS / "selection_at_512.json", dict(selection_only=True, length=512, methods=budgets))
    write_json(RESULTS / "summary.json", dict(weighted_oracle=winner, methods={method: dict(
        validation=validation[method]["groups"]["overall"], epoch=validation[method]["epoch"],
        test_groups=result["groups"], privileged_test=result["privileged_test"]) for method,result in outcomes.items()}))


def audit_fill(tokenizer):
    """Compare layouts on the completed run's Test inputs with the same region and graph choices."""
    if not (RESULTS / "summary.json").is_file():
        raise ValueError("The formal run must finish before comparing contiguous fill")
    config = json.loads((RESULTS / "weights.json").read_text())
    weights, oracle = tuple(config["weights"]), config["winner"]
    rows = prime_source_rows(DATA, "test")
    sources, _ = tokenize_sources([r["func"] for r in rows], tokenizer, source_parser(), BUDGET)
    graphs, _ = load_dependencies(rows, sources, "test")
    localizer = Localizer().cuda()
    localizer.load_state_dict(torch.load(WEIGHTS / "localizer" / "best.pt", weights_only=True, map_location="cpu"))
    regions = Regions(sources)
    scores, _ = score_regions(localizer, regions)
    scores = regions.regroup(scores)
    del localizer, regions
    print("Comparing continuous fill with the saved LongVul layouts", flush=True)
    data = {"longvul": SelectedData(rows, sources, tokenizer, "longvul", weights=weights, scores=scores,
                                    graphs=graphs, continuous_fill=True)}
    simpl, raw = paired_edits(sources)
    print("Comparing continuous fill with the saved weighted Oracle layouts", flush=True)
    data[oracle] = SelectedData(rows, sources, tokenizer, oracle, weights=weights, edits=simpl,
                                 graphs=graphs, continuous_fill=True)
    results = {}
    membership = ("idx", "sample_key", "target", "original_bpe", "simplified_bpe", "native_graph_status",
                  "dependency_status", "optimizer_steps")
    for method, selected in data.items():
        before = read_predictions(RESULTS / method / "test.csv.gz")
        after = report_data(selected, sources, None, None, include_pairs=True)
        changed_lengths = []
        for original, revised, (_, anchors, _, retained), source in zip(
                before, after, selected.decisions, sources, strict=True):
            old_spans = merge(original["simplified_runs"])
            if (any(str(original[key]) != str(revised[key]) for key in membership)
                    or merge(original["oracle_anchors"]) != anchors
                    or any(not contains(span, old_spans) for span in retained)):
                raise ValueError(f"{method}: fixed selection differs for {original['sample_key']}")
            if old_spans != revised["simplified_runs"]:
                if len(source.ids) <= BUDGET:
                    raise ValueError(f"{method}: a function inside the budget changed layout")
                changed_lengths.append(source.raw_bpe)
        results[method] = dict(privileged_test=method.startswith("oracle_"), preserved_prefill_functions=len(after),
                               selection_seconds=selected.seconds,
                               layouts_changed={key: sum(lo <= n <= hi for n in changed_lengths)
                                                for key, (lo, hi) in LENGTHS.items()},
                               formal=input_report(before, sources, graphs, simpl, raw),
                               continuous=input_report(after, sources, graphs, simpl, raw))
        print(method, "continuous fill compared", len(after), len(changed_lengths), flush=True)
    write_json(RESULTS / "continuous_fill.json", dict(selection_only=True, split="test", length=MAX_LENGTH,
                                                      ddg_cdg_weights=list(weights), methods=results))


def predict_source(code, tokenizer, *, filename=None, smoke=False):
    sources, source_timing = tokenize_sources([code], tokenizer, source_parser(), BUDGET)
    source = sources[0]
    graphs, timings = extract([code], filenames=[filename] if filename else None)
    edges, stats = map_dependencies(source, graphs[0])
    localizer = Localizer().cuda()
    model = make_model(tokenizer, "longvul", long_input=True, pretrained=smoke)
    weights = (1,1)
    if not smoke:
        weights = tuple(json.loads((RESULTS / "weights.json").read_text())["weights"])
        localizer.load_state_dict(torch.load(WEIGHTS / "localizer" / "best.pt", weights_only=True, map_location="cpu"))
        model.load_state_dict(torch.load(WEIGHTS / "longvul" / "best.pt", weights_only=True, map_location="cpu"))
    score, locator_seconds = score_regions(localizer, Regions([source]))
    start = perf_counter()
    graph, graph_status = selection.available_graph(dict(status=graphs[0]["status"], edges=edges), weights)
    spans, _, steps, _ = selection.choose(source, "longvul", BUDGET, weights=weights if graph is not None else (0,0),
                                       graph=graph, scores=score)
    fixed, length = pack_fixed(source.ids, range(len(source.ids)), spans, tokenizer, max_length=MAX_LENGTH)
    selection_seconds = perf_counter()-start
    batch = dict(fixed=fixed.unsqueeze(0), lengths=torch.tensor([length]), raw=torch.tensor([source.raw_bpe]),
                 uniform=torch.full((1,REGIONS,4),-1))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    model.eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        result = model(to_gpu(batch))
    probability = float(result["logits"].float().sigmoid().item())
    return dict(probability=probability, prediction=int(probability > .5), input_length=length,
                original_bpe=source.raw_bpe, simplified_bpe=len(source.ids), source_runs=original_runs(source, spans),
                graph_status=graphs[0]["status"], dependency_status=graph_status, graph_counts=stats,
                graph_timing=timings, source_timing=source_timing,
                locator_seconds=locator_seconds, selection_seconds=selection_seconds, greedy_steps=steps,
                inference_seconds=perf_counter()-start, peak_inference_allocated_bytes=torch.cuda.max_memory_allocated())


def smoke(tokenizer):
    parser = source_parser()
    source_check = ('#define SHIFT(v) ((v) + 1) /* keep macro text */\n'
                    'int f() {\n  auto s = R"x(a  // b\nc)x";\n'
                    '  const char *u = "/* still literal */";\n  return 2/**/+3; // removable\n}\n')
    comments, literals, _, _ = analyze(source_check, parser)
    simple, origin, synthetic, _ = simplify(source_check, comments, literals)
    assert "/* keep macro text */" in simple and 'R"x(a  // b\nc)x"' in simple and '"/* still literal */"' in simple
    assert "removable" not in simple and "/**/" not in simple and "2 +3" in simple
    assert simple.count("\n") == source_check.count("\n")
    assert all(simple[i] == source_check[j] for i,j in enumerate(origin) if not synthetic[i])
    spliced = "#define P(x) \\\n  (x)\n"
    assert simplify(spliced, [], [])[0] == spliced

    fixture = [Region(a,b,a,b,"statement") for a,b in ((0,5),(5,10),(50,54),(100,104))]
    graph = {"REACHING_DEF": [(0,3),(1,3),(2,3)], "CDG": [(0,3)]}
    plan, steps = selection.optimize(fixture, [(100,104)], {3}, graph, (1,1), 15, bridge=False)
    assert plan == [(0,10),(100,104)] and steps == 2, (plan, steps)
    scatter = [Region(i*6,i*6+3,i*6,i*6+3,"statement") for i in range(30)]
    edges = {"REACHING_DEF": [(i,29) for i in range(29)]}
    joined, _ = selection.optimize(scatter, [scatter[-1].span], {29}, edges, (1,0), 70)
    separated, _ = selection.optimize(scatter, [scatter[-1].span], {29}, edges, (1,0), 70, bridge=False)
    assert occupied(joined) <= 70 and len(joined) < len(separated), (joined, separated)

    cases = json.loads((ROOT / "results/unixcoder_repair_oracle/smoke.json").read_text())["cases"]
    members = prime_source_rows(DATA, "train")
    pairs = {tuple(r["idx"] for r in members[i:i+2]): members[i:i+2] for i in range(0,len(members),2)}
    checks = {}
    for name in ("multi_block", "pure_insert", "pure_delete"):
        pair = pairs[tuple(cases[name]["indices"])]
        sources, _ = tokenize_sources([r["func"] for r in pair], tokenizer, parser, BUDGET)
        simple, raw = paired_edits(sources)
        assert len(raw) == 2
        if name == "pure_insert":
            assert raw[0]["insert_gap"] and raw[1]["insert"] and not raw[0]["insert"]
        elif name == "pure_delete":
            assert raw[0]["delete"] and raw[1]["delete_gap"] and not raw[1]["delete"]
        else:
            assert len(line_edits(*(r["func"].splitlines(keepends=True) for r in pair))) > 1
        data = SelectedData(pair, sources, tokenizer, "oracle_diff", edits=simple)
        checks[name] = dict(lengths=[ex["length"] for ex in data.examples],
                            original_bpe=[s.raw_bpe for s in sources],
                            raw_edit_bpe=[{k: sum(b-a for a,b in spans) for k,spans in edits.items()} for edits in raw])

    example = "int scale(int x) {\n  int result = x + 1;\n  if (x > 2) {\n    result = result * 3;\n  }\n  return result;\n}\n"
    longest = sorted(members, key=lambda r: len(r["func"]), reverse=True)[:BATCH_SIZE]
    sources, source_time = tokenize_sources([r["func"] for r in longest], tokenizer, parser, BUDGET)
    graph_member = next(row for row in members if 15000 < len(row["func"]) < 20000)
    native, joern_time = extract([example, graph_member["func"]])
    known, _ = tokenize_sources([example], tokenizer, parser, BUDGET)
    edges, statistics = map_dependencies(known[0], native[0])
    assert any(known[0].code[known[0].regions[a].source_start:known[0].regions[a].source_end].strip().startswith("if")
               and "* 3" in known[0].code[known[0].regions[b].source_start:known[0].regions[b].source_end]
               for a,b in edges["CDG"])
    assert any("result =" in known[0].code[known[0].regions[a].source_start:known[0].regions[a].source_end]
               and "return result" in known[0].code[known[0].regions[b].source_start:known[0].regions[b].source_end]
               for a,b in edges["REACHING_DEF"])
    real, _ = tokenize_sources([graph_member["func"]], tokenizer, parser, BUDGET)
    real_edges, real_statistics = map_dependencies(real[0], native[1])
    set_seed(SEED)
    locator = Localizer().cuda()
    region_scores, locator_seconds = score_regions(locator, Regions(sources[:1]))
    real_scores, _ = score_regions(locator, Regions(real))
    del locator
    real_selection, _, _, _ = selection.choose(real[0], "longvul", BUDGET, weights=(1,1), scores=real_scores, graph=real_edges)
    assert occupied(real_selection)+2 <= MAX_LENGTH
    start = perf_counter()
    chosen, _, choices, _ = selection.choose(sources[0], "predicted", BUDGET, scores=region_scores)
    selection_seconds = perf_counter()-start
    assert occupied(chosen)+2 == MAX_LENGTH and len(chosen) <= selection.MAX_FRAGMENTS

    data = SelectedData(longest, sources, tokenizer, "uniform")

    def gpu_check(micro):
        set_seed(SEED)
        model = make_model(tokenizer, "uniform", long_input=True)
        assert model.detector.config.max_position_embeddings - tokenizer.pad_token_id - 1 >= MAX_LENGTH
        opt = LOSVERAdamW(model.parameters(), lr=LEARNING_RATE, eps=ADAM_EPSILON)
        batch = to_gpu(collate_long(data.examples[:micro]), labels=True)
        calls, times = [], []
        handle = model.detector.encoder.register_forward_pre_hook(lambda *args: calls.append(1))
        torch.cuda.reset_peak_memory_stats()
        for i in range(4):
            torch.cuda.synchronize()
            before = perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(batch)["loss"]
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite 1024-position BCE loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM, error_if_nonfinite=True)
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if i:
                times.append(perf_counter()-before)
        handle.remove()
        assert len(calls) == 4 and batch["fixed"].shape[-1] == MAX_LENGTH
        return dict(peak_bytes=torch.cuda.max_memory_allocated(), mean_step_seconds=float(np.mean(times)),
                    forwards_per_step=1, loss=float(loss.detach()),
                    attention=type(model.detector.encoder.encoder.layer[0].attention.self).__name__)

    failures = []
    for micro in (8, 4, 2, 1):
        try:
            gpu = gpu_check(micro)
            break
        except torch.cuda.OutOfMemoryError:
            failures.append(micro)
        gc.collect()
        torch.cuda.empty_cache()
    else:
        raise RuntimeError("One 1024-position training input does not fit the RTX 4090")
    quality = quality_check(tokenizer, parser)
    write_json(RESULTS / "smoke.json", dict(micro_batch=micro, oom_micro_batches=failures, gpu=gpu, alignment=checks,
               joern_version=joern_version(), joern_quality=quality,
               source=dict(**source_time, original_bpe=[s.raw_bpe for s in sources], simplified_bpe=[len(s.ids) for s in sources]),
               native_edges=statistics, native_timing=joern_time,
               real_graph=dict(idx=graph_member["idx"], status=native[1]["status"], statistics=real_statistics,
                               input_length=occupied(real_selection)+2, inference_without_counterpart=True), dynamic_example=plan,
               fragmentation=[len(separated),len(joined)], extreme=dict(idx=longest[0]["idx"],
                       method="predicted",
                       inference_without_counterpart=True, locator_seconds=locator_seconds,
                       selection_seconds=selection_seconds, steps=choices, length=occupied(chosen)+2,
                       fragments=len(chosen))))
    print("1024 smoke complete", micro, gpu, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("graphs", "smoke", "run", "predict", "audit-fill"))
    ap.add_argument("--source", type=Path)
    args = ap.parse_args()
    if args.action == "graphs":
        prepare_graphs()
        return
    if torch.cuda.device_count() != 1 or "4090" not in torch.cuda.get_device_name():
        raise ValueError("Select one RTX 4090 via CUDA_VISIBLE_DEVICES")
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    tokenizer = load_tokenizer("unixcoder", UNIX_BASE)
    if args.action == "predict":
        if args.source is None:
            ap.error("predict requires --source")
        print(json.dumps(predict_source(args.source.read_text(), tokenizer, filename=args.source.name), ensure_ascii=False))
    elif args.action == "smoke":
        smoke(tokenizer)
    elif args.action == "audit-fill":
        audit_fill(tokenizer)
    else:
        verified = json.loads((RESULTS / "smoke.json").read_text())
        if verified.get("joern_version") != joern_version() or verified.get("joern_quality", {}).keys() != set(SPLITS):
            raise ValueError("Complete the Joern quality smoke on Train, Valid and Test before training")
        run(tokenizer, verified["micro_batch"])


if __name__ == "__main__":
    main()
