"""Replace the 512-position Oracle-Diff anchors with source-only repair predictions."""
from collections import Counter
from dataclasses import dataclass
import json
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import set_seed

from compaction import BUDGET, PackedData, collate, line_boundaries, tokenized_batches
from context_selection import describe, input_layout, merge, occupied
from paired import (BACKBONES, BATCH_SIZE, DATA, MAX_LENGTH, ORACLE_RESULTS, PREDICTED_DIFF, ROOT, SEED, UNIX_RESULTS,
                    classifier_config, fit_classifier, grouped_metrics, make_model, predict, prime_source_rows, to_gpu)
from region_locator import Localizer, Regions, region_metrics, score_regions, train_localizer, weak_targets
from repair_oracle import EDIT_TYPES, compact_edits, line_edits, map_edits, overlap_bpe, pack_selected, raw_runs, repair_retention, reserve_blocks
from rgcl.artifacts import write_json
from source_regions import Region
from window_inputs import GROUPS

RESULTS = ROOT / "results" / "unixcoder_predicted_diff"
WEIGHTS = ROOT / "checkpoints" / "unixcoder_predicted_diff"
HISTORICAL_REVISION = "9d9536371b20ae3ea445254d8b4a4a6f36e252e1"
TOP_K = (4, 8)
ANCHOR_BUDGETS = (BUDGET//4, 3*BUDGET//8, BUDGET//2)
FIELDS = ("predicted_anchor_runs", "predicted_anchor_positions")


@dataclass(slots=True)
class CompactedSource:
    code: str
    raw_ids: np.ndarray
    offsets: np.ndarray
    keep: list[int]
    boundaries: list[int]
    ids: np.ndarray
    regions: list[Region]


def compact_source(code, raw_ids, offsets):
    keep, bounds = input_layout(code, raw_ids, offsets)
    # Short functions remain intact for classification but still have line-level locator targets.
    cuts = bounds if len(keep) > BUDGET else line_boundaries(code, offsets, keep, BUDGET)
    regions = [Region(a, b, offsets[keep[a]][0], offsets[keep[b-1]][1], "line")
               for a, b in zip(cuts, cuts[1:]) if a < b]
    ids = np.asarray(raw_ids, dtype=np.int32)
    return CompactedSource(code, ids, np.asarray(offsets, dtype=np.int32), keep, bounds, ids[keep], regions)


def load_sources(rows, tokenizer):
    return [compact_source(text, ids, offsets)
            for _, texts, encoding in tokenized_batches(rows, tokenizer)
            for text, ids, offsets in zip(texts, encoding.input_ids, encoding.offset_mapping, strict=True)]


def paired_annotations(sources):
    """Derive same-side raw-BPE labels only at supervised or audit call sites."""
    result = []
    for left, right in zip(sources[::2], sources[1::2], strict=True):
        lines = [source.code.splitlines(keepends=True) for source in (left, right)]
        edits = line_edits(*lines)
        result.extend(map_edits(lines[side], source.offsets, edits, side) for side, source in enumerate((left, right)))
    return result


def region_targets(sources, edits):
    mapped = [{kind: merge(spans) for kind, spans in compact_edits(source.keep, edit).items()}
              for source, edit in zip(sources, edits, strict=True)]
    return weak_targets(sources, mapped)


def prepare_example(source, scores, tokenizer, top_k, anchor_budget):
    if len(scores) != len(source.regions) or not 0 < anchor_budget <= BUDGET//2 or top_k <= 0:
        raise ValueError("Invalid source scores or predicted-anchor policy")
    anchors = []
    if len(source.ids) > BUDGET:
        ranked = np.argsort(-scores, kind="stable")[:top_k]
        anchors = reserve_blocks([], [source.regions[i].span for i in ranked], budget=anchor_budget)
    example = pack_selected(source.raw_ids, source.keep, source.boundaries, anchors, tokenizer)
    example.update(predicted_anchor_runs=raw_runs(source.keep, anchors), predicted_anchor_positions=occupied(anchors))
    return example


def calibrate(sources, scores, edits, tokenizer):
    """Freeze a single policy by exact raw edit coverage on Valid functions requiring selection."""
    cohort = [(source, values, merge([span for kind in EDIT_TYPES for span in edit[kind]]))
              for source, values, edit in zip(sources, scores, edits, strict=True) if len(source.ids) > BUDGET]
    available = sum(b-a for _, _, spans in cohort for a, b in spans)
    if not available:
        raise ValueError("Valid needs real edit BPE among functions exceeding the body budget")
    trials = []
    for top_k in TOP_K:
        for budget in ANCHOR_BUDGETS:
            retained = 0
            for source, values, edits in cohort:
                packed = prepare_example(source, values, tokenizer, top_k, budget)
                raw = packed["fixed"][1, :packed["length"]].tolist()
                retained += overlap_bpe(edits, merge([(i, i+1) for i in raw if i >= 0]))
            trials.append(dict(top_k=top_k, anchor_budget=budget, retained_bpe=retained, fraction=retained/available))
    best = max(trials, key=lambda row: (row["retained_bpe"], -row["anchor_budget"], -row["top_k"]))
    return dict(top_k=best["top_k"], anchor_budget=best["anchor_budget"], uniform_min_positions=BUDGET-best["anchor_budget"],
                split="valid", criterion="actual raw edit BPE coverage after packing; smaller budget then lower top-k on ties",
                functions=len(cohort), available_bpe=available, trials=trials)


class PredictedData(PackedData):
    def __init__(self, rows, sources, scores, tokenizer, policy):
        started = perf_counter()
        super().__init__(rows, (prepare_example(s, p, tokenizer, policy["top_k"], policy["anchor_budget"])
                                for s, p in zip(sources, scores, strict=True)))
        self.seconds = perf_counter()-started


def fragment_report(data):
    fragments = [list(Counter(i for i in x["fixed"][2, :x["length"]].tolist() if i > 0).values())
                 for x in data.examples]
    report = {}
    for name, (lo, hi) in GROUPS.items():
        indices = [i for i, row in enumerate(data.metadata) if lo <= row["original_bpe"] <= hi]
        lens = [n for i in indices for n in fragments[i]]
        report[name] = dict(functions=len(indices), fragments=len(lens), mean_fragments=len(lens)/len(indices),
                            mean_fragment_bpe=float(np.mean(lens)), median_fragment_bpe=float(np.median(lens)),
                            p25_fragment_bpe=float(np.percentile(lens, 25)), p75_fragment_bpe=float(np.percentile(lens, 75)),
                            internal_sep=sum(len(fragments[i])-1 for i in indices if fragments[i]),
                            mean_anchor_positions=sum(data.examples[i]["predicted_anchor_positions"] for i in indices)/len(indices),
                            longest_input=max(data.examples[i]["length"] for i in indices))
    return report


def evaluate(model, data, split, *, root=RESULTS, fields=FIELDS):
    loader = DataLoader(data, batch_size=BATCH_SIZE, collate_fn=collate)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    scores, _ = predict(model, loader)
    seconds, peak = perf_counter()-started, torch.cuda.max_memory_allocated()
    groups = grouped_metrics(data, scores)
    selection = describe(data, scores, root / f"{split}.csv.gz", paired=split == "test", include=fields)
    return dict(groups=groups, selection=selection, inference_seconds=seconds, peak_inference_allocated_bytes=peak,
                model_tokens=selection["overall"]["input_tokens"])


def compare_saved(result):
    history = dict.fromkeys(("uniform4", "uniform_sampling", "context_aware"), UNIX_RESULTS)
    history.update(dict.fromkeys(("oracle_diff", "oracle_diff_context"), ORACLE_RESULTS))
    saved = {name: json.loads((root / name / "metrics.json").read_text()) for name, root in history.items()}
    saved[PREDICTED_DIFF] = result
    retention = repair_retention(ORACLE_RESULTS, UNIX_RESULTS, extra_predictions={
        PREDICTED_DIFF: RESULTS / "test.csv.gz", "uniform4": UNIX_RESULTS / "uniform4" / "test.csv.gz"})
    comparison = dict(historical_revision=HISTORICAL_REVISION, methods={})
    for name, run in saved.items():
        test = run["split"]["test"]
        comparison["methods"][name] = dict(privileged_test=name.startswith("oracle_"), groups=test["groups"],
                                            selection=test["selection"], edit_retention=retention["groups"][name],
                                            selected_epoch=run["selected_epoch"], training_seconds=run["training_seconds"],
                                            classifier_test_seconds=test["inference_seconds"])
    write_json(RESULTS / "comparison.json", comparison)


def fp32_classifier(tokenizer):
    model = make_model(tokenizer, PREDICTED_DIFF)
    if (model.detector.encoder.config._attn_implementation != "eager"
            or any(p.dtype != torch.float32 for p in model.parameters())):
        raise ValueError("Predicted-Diff requires the historical FP32 eager classifier")
    return model


def run_predicted_diff(tokenizer):
    if (RESULTS / "metrics.json").is_file():
        print("predicted_diff already complete", flush=True)
        return
    started = perf_counter()
    rows, sources, edits, source_seconds, diff_seconds = {}, {}, {}, {}, {}
    for split in ("train", "valid"):
        rows[split] = prime_source_rows(DATA, split)
        before = perf_counter()
        sources[split] = load_sources(rows[split], tokenizer)
        source_seconds[split] = perf_counter()-before
        before = perf_counter()
        edits[split] = paired_annotations(sources[split])
        diff_seconds[split] = perf_counter()-before
    targets = {split: region_targets(sources[split], edits[split]) for split in ("train", "valid")}
    localizer, values, localization = train_localizer(sources, targets, WEIGHTS / "localizer", RESULTS / "localizer")
    localizer.cpu()
    before = perf_counter()
    policy = calibrate(sources["valid"], values["valid"], edits["valid"], tokenizer)
    policy["seconds"] = perf_counter()-before
    write_json(RESULTS / "policy.json", policy)
    print("Valid policy", policy["top_k"], policy["anchor_budget"], flush=True)
    train, valid = (PredictedData(rows[s], sources[s], values[s], tokenizer, policy) for s in ("train", "valid"))
    del rows, sources, edits, targets, values

    set_seed(SEED)
    model = fp32_classifier(tokenizer)
    config = classifier_config(PREDICTED_DIFF)
    model_id, model_revision = BACKBONES["unixcoder"]
    config.update(model_id=model_id, model_revision=model_revision, historical_input_revision=HISTORICAL_REVISION,
                  train_functions=len(train), valid_functions=len(valid), test_functions=1128,
                  locator_units="historical indentation compaction and line boundaries", privileged_test=False,
                  policy={k: policy[k] for k in ("top_k", "anchor_budget", "uniform_min_positions")})
    selected, history = fit_classifier(model, train, valid, WEIGHTS / "classifier", RESULTS, config)
    packed_seconds = dict(train=train.seconds, valid=valid.seconds)
    del train
    validation = evaluate(model, valid, "valid")
    if validation["groups"]["overall"] != selected["validation"]:
        raise ValueError("Reloaded classifier did not reproduce the selected Valid checkpoint")

    # Test uses the frozen locator and policy. Construct no paired annotations until predictions are saved.
    test_rows = prime_source_rows(DATA, "test")
    before = perf_counter()
    test_sources = load_sources(test_rows, tokenizer)
    source_seconds["test"] = perf_counter()-before
    model.cpu()
    torch.cuda.reset_peak_memory_stats()
    localizer.cuda()
    regions = Regions(test_sources)
    test_scores, seconds = score_regions(localizer, regions)
    test_scores = regions.regroup(test_scores)
    loc_peak = torch.cuda.max_memory_allocated()
    del regions, localizer
    test = PredictedData(test_rows, test_sources, test_scores, tokenizer, policy)
    packed_seconds["test"] = test.seconds
    evaluation = evaluate(model.cuda(), test, "test")
    evaluation["fragments"] = fragment_report(test)
    before = perf_counter()
    edits = paired_annotations(test_sources)
    diff_seconds["test"] = perf_counter()-before
    truth = region_targets(test_sources, edits)
    localization["splits"]["test"] = dict(region_metrics(test_scores, truth), inference_seconds=seconds,
                                          peak_inference_allocated_bytes=loc_peak)
    long_pairs = [(p, y) for source, p, y in zip(test_sources, test_scores, truth, strict=True)
                  if len(source.raw_ids) > BUDGET]
    localization["test_gt510"] = region_metrics(*zip(*long_pairs, strict=True))
    write_json(RESULTS / "localizer" / "metrics.json", localization)
    result = dict(config=config, selected_epoch=selected["epoch"], training_seconds=sum(r["training_seconds"] for r in history),
                  peak_train_allocated_bytes=max(r["peak_allocated_bytes"] for r in history),
                  source_seconds=source_seconds, packing_seconds=packed_seconds, diff_seconds=diff_seconds,
                  split=dict(valid=validation, test=evaluation), invoked_seconds=perf_counter()-started)
    compare_saved(result)
    write_json(RESULTS / "metrics.json", result)
    print("predicted_diff complete", selected["epoch"], evaluation["groups"]["overall"]["mcc"], flush=True)


def smoke_predicted_diff(tokenizer):
    set_seed(SEED)
    rows = sorted(prime_source_rows(DATA, "train"), key=lambda r: len(r["func"]), reverse=True)[:BATCH_SIZE]
    sources = load_sources(rows, tokenizer)
    regions, localizer = Regions(sources), Localizer().cuda()
    values, _ = score_regions(localizer, regions)
    scores = regions.regroup(values)
    del localizer, regions
    data = PredictedData(rows, sources, scores, tokenizer, dict(top_k=TOP_K[-1], anchor_budget=ANCHOR_BUDGETS[-1]))
    model = fp32_classifier(tokenizer)
    calls = []
    hook = model.detector.encoder.register_forward_pre_hook(lambda *args: calls.append(1))
    torch.cuda.reset_peak_memory_stats()
    report = model(to_gpu(collate(data.examples), labels=True), details=True)
    hook.remove()
    report["loss"].backward()
    if len(calls) != 1 or report["input_ids"].shape != (BATCH_SIZE, MAX_LENGTH) or not report["loss"].isfinite():
        raise ValueError("Expected one finite FP32 eager forward/backward at 512 positions")
    write_json(RESULTS / "smoke.json", dict(functions=data.metadata, encoder_calls=len(calls),
                                          shape=list(report["input_ids"].shape), loss=report["loss"].item(),
                                          peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    print("Predicted-Diff smoke passed", max(len(x.raw_ids) for x in sources), flush=True)
