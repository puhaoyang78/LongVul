"""Privileged 512-position anchor-replacement experiment, matched to Predicted-Diff."""
import csv
import gzip
import json
from random import Random
from time import perf_counter

import numpy as np
import torch
from transformers import set_seed

from compaction import BUDGET, PackedData
from context_selection import gaps, merge, occupied
from paired import BACKBONES, DATA, ORACLE_RESULTS, ROOT, SEED, UNIX_RESULTS, classifier_config, fit_classifier, prime_source_rows
from predicted_diff import (HISTORICAL_REVISION, RESULTS as PREDICTED_RESULTS, evaluate, fp32_classifier,
                            load_sources, paired_annotations)
from repair_oracle import (EDIT_TYPES, GAP_TYPES, compact_edits, grouped_edit_retention, pack_selected,
                           packed_edit_retention, raw_runs, reserve_blocks)
from rgcl.artifacts import write_json

RATES = (0, 25, 50, 75, 100)
THRESHOLDS = (.10, .12, .15)
RESULTS = ROOT / "results" / "oracle_noise_512"
WEIGHTS = ROOT / "checkpoints" / "oracle_noise_512"
FIELDS = ("oracle_anchor_runs", "assigned_anchor_runs", "anchor_positions", "replacement_attempts", "replaced_regions")


def oracle_regions(source, edits, policy):
    """Rank the same line units as Predicted-Diff by actual edit overlap, then gap overlap."""
    n = len(source.ids)
    if n <= BUDGET:
        return [], []
    mapped = compact_edits(source.keep, edits)
    real = merge([span for kind in EDIT_TYPES for span in mapped[kind]])
    proxy = merge([span for kind in GAP_TYPES for span in mapped[kind]])
    bounds = np.array([r.span for r in source.regions], dtype=np.int32)

    def counts(spans):
        delta = np.zeros(n+1, dtype=np.int32)
        for a, b in spans:
            delta[a] += 1
            delta[b] -= 1
        prefix = np.zeros(n+1, dtype=np.int32)
        prefix[1:] = np.cumsum(np.cumsum(delta[:-1]) > 0)
        return prefix[bounds[:, 1]]-prefix[bounds[:, 0]]

    actual, nearby = counts(real), counts(proxy)
    ranked = np.lexsort((bounds[:, 0], -nearby, -actual))
    candidates = [source.regions[i].span for i in ranked[:policy["top_k"]] if actual[i] or nearby[i]]
    return reserve_blocks([], candidates, budget=policy["anchor_budget"]), merge([*real, *proxy])


def matched_slots(size, anchors, edited, seed):
    """Reserve disjoint, equal-width alternatives in unchanged source for each anchor."""
    rng = Random(seed)

    def guarded(span):
        a, b = span
        return max(0, a-1), min(size, b+1)

    blocked = merge([*edited, *(guarded(s) for s in anchors)])
    slots = [None]*len(anchors)
    # Place wider spans first so short spans do not consume their only feasible gap.
    priorities = [rng.random() for _ in anchors]
    for i in sorted(range(len(anchors)), key=lambda i: (anchors[i][0]-anchors[i][1], i)):
        a, b = anchors[i]
        width = b-a
        possible = [(left, right-width+1) for left, right in gaps(blocked, 0, size) if right-left >= width]
        total = sum(hi-lo for lo, hi in possible)
        alternate = None
        if total:
            index = rng.randrange(total)
            for lo, hi in possible:
                if index < hi-lo:
                    alternate = lo+index, lo+index+width
                    blocked = merge([*blocked, guarded(alternate)])
                    break
                index -= hi-lo
        slots[i] = ((a, b), alternate, priorities[i])
    return slots


def prepare_split(split, tokenizer, policy):
    started = perf_counter()
    rows = prime_source_rows(DATA, split)
    sources = load_sources(rows, tokenizer)
    edits = paired_annotations(sources)
    plans = []
    for row, source, edit in zip(rows, sources, edits, strict=True):
        anchors, edited = oracle_regions(source, edit, policy)
        plans.append(matched_slots(len(source.ids), anchors, edited, f"{SEED}/{row['sample_key']}"))
    ranking = sorted((priority, i, j) for i, slots in enumerate(plans) for j, (_, _, priority) in enumerate(slots))
    return rows, sources, edits, plans, ranking, perf_counter()-started


def noise_data(prepared, tokenizer, rate):
    started = perf_counter()
    rows, sources, _, plans, ranked, _ = prepared
    # Use one order for all ratios. Rounding is over the split to avoid quantizing one-block functions.
    attempts = (len(ranked)*rate+50)//100
    chosen = {(i, j) for _, i, j in ranked[:attempts]}

    def examples():
        for i, (source, slots) in enumerate(zip(sources, plans, strict=True)):
            before, after, tries, done = [], [], 0, 0
            for j, (anchor, replacement, _) in enumerate(slots):
                attempted = (i,j) in chosen
                use_replacement = attempted and replacement is not None
                before.append(anchor)
                after.append(replacement if use_replacement else anchor)
                tries += attempted
                done += use_replacement
            spans = merge(after)
            if len(spans) != len(before) or occupied(spans) != occupied(before):
                raise ValueError("Noise changed the number or packed cost of the reserved blocks")
            example = pack_selected(source.raw_ids, source.keep, source.boundaries, spans, tokenizer)
            yield dict(example, oracle_anchor_runs=raw_runs(source.keep, before),
                       assigned_anchor_runs=raw_runs(source.keep, spans), anchor_positions=occupied(spans),
                       replacement_attempts=tries, replaced_regions=done)

    data = PackedData(rows, examples())
    replaced = sum(x["replaced_regions"] for x in data.examples)
    stats = dict(functions=len(data), anchor_regions=len(ranked), matchable_regions=sum(
        alt is not None for slots in plans for _, alt, _ in slots), attempted_regions=attempts,
        replaced_regions=replaced, unmatched_attempts=attempts-replaced,
        fraction=replaced/len(ranked) if ranked else None,
        total_anchor_positions=sum(x["anchor_positions"] for x in data.examples), packing_seconds=perf_counter()-started)
    return data, stats


def saved_retention(path, prepared):
    rows, sources, edits, *_ = prepared
    with gzip.open(path, "rt", newline="") as stream:
        predictions = list(csv.DictReader(stream))

    def observations():
        for row, source, edit, pred in zip(rows, sources, edits, predictions, strict=True):
            if row["sample_key"] != pred["sample_key"] or len(source.raw_ids) != int(pred["original_bpe"]):
                raise ValueError(f"Saved baseline differs in membership or tokenization: {path}")
            yield len(source.raw_ids), edit, json.loads(pred["source_runs"])

    return grouped_edit_retention(observations())


def budget_reference(prepared, tokenizer, policy):
    clean, stats = noise_data(prepared, tokenizer, 0)
    controls = dict(oracle_full=ORACLE_RESULTS / "oracle_diff/valid.csv.gz",
                    predicted_diff=PREDICTED_RESULTS / "valid.csv.gz",
                    uniform_sampling=UNIX_RESULTS / "uniform_sampling/valid.csv.gz")
    retention = {k: saved_retention(path, prepared) for k, path in controls.items()}
    retention["oracle_matched"] = packed_edit_retention(clean, prepared[2])
    write_json(RESULTS / "budget_reference.json", dict(split="valid", policy=policy, noise=stats, edit_retention=retention))
    print("Valid >510 BPE retained edits", {k: round(v["long"]["actual_edits"]["fraction"], 4)
                                           for k,v in retention.items()}, flush=True)


def crossing_report(points):
    report = {}
    for threshold in THRESHOLDS:
        observed = [x for x,y in points if y > threshold]
        crossed = []
        for (x0,y0), (x1,y1) in zip(points, points[1:]):
            if (y0 > threshold) != (y1 > threshold):
                crossed.append(x0 + (x1-x0)*(threshold-y0)/(y1-y0))
        report[str(threshold)] = dict(minimum_observed_retention=min(observed, default=None), interpolated_crossings=crossed)
    return report


def plot_curve(results):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4), layout="constrained", sharey=True)
    thresholds = {}
    for ax, group in zip(axes, ("overall", "long"), strict=True):
        thresholds[group] = {}
        for split, color in (("valid", "#52647a"), ("test", "#17765a")):
            points = [(row[split]["edit_retention"][group]["actual_edits"]["fraction"], row[split]["groups"][group]["mcc"])
                      for row in results]
            thresholds[group][split] = crossing_report(points)
            x, y = zip(*points, strict=True)
            ax.plot(np.array(x)*100, y, "o-", color=color, label=split.capitalize())
            if split == "test":
                for rate, (a,b) in zip(RATES, points, strict=True):
                    ax.annotate(f"{rate}%", (a*100,b), xytext=(3,6), textcoords="offset points", color=color, fontsize=8)
        for threshold in THRESHOLDS:
            ax.axhline(threshold, linewidth=.7, linestyle=":", color="#8c8c8c", zorder=0)
        ax.set(title="All functions" if group == "overall" else "Original length >510 BPE",
               xlabel="True modification BPE retained (%)", ylabel="MCC")
        ax.margins(x=.12, y=.18)
        ax.legend(frameon=False)
    fig.suptitle("Oracle anchor replacement under a 127-position cap")
    fig.savefig(RESULTS / "retention_mcc.png", dpi=180)
    plt.close(fig)
    return thresholds


def run_oracle_noise(tokenizer, *, budget_only=False):
    if not budget_only and (RESULTS / "summary.json").is_file():
        print("oracle_noise already complete", flush=True)
        return
    historical = json.loads((PREDICTED_RESULTS / "policy.json").read_text())
    policy = {key: historical[key] for key in ("top_k", "anchor_budget")}
    if policy != dict(top_k=4, anchor_budget=127):
        raise ValueError("Expected the archived Predicted-Diff policy of four units and 127 positions")
    valid = prepare_split("valid", tokenizer, policy)
    budget_reference(valid, tokenizer, policy)
    if budget_only:
        return
    train = prepare_split("train", tokenizer, policy)
    for rate in RATES:
        name = f"noise_{rate:03d}"
        root, weights = RESULTS / name, WEIGHTS / name
        if (root / "validation.json").is_file() and (weights / "best.pt").is_file():
            continue
        training, train_stats = noise_data(train, tokenizer, rate)
        validation, valid_stats = noise_data(valid, tokenizer, rate)
        config = classifier_config("oracle_diff")
        model_id, revision = BACKBONES["unixcoder"]
        config.update(method=name, replacement_percent=rate, model_id=model_id, model_revision=revision,
                      historical_input_revision=HISTORICAL_REVISION, privileged_test=True, policy=policy,
                      data_counts=dict(train=len(training), valid=len(validation), test=1128),
                      noise="nested seed-42 split order, unchanged same-width spans, one-BPE gaps, unmatched kept")
        set_seed(SEED)
        model = fp32_classifier(tokenizer)
        best, history = fit_classifier(model, training, validation, weights, root, config)
        result = evaluate(model, validation, "valid", root=root, fields=FIELDS)
        if result["groups"]["overall"] != best["validation"]:
            raise ValueError(f"{name}: reloaded validation metrics differ")
        result.update(epoch=best["epoch"], edit_retention=packed_edit_retention(validation, valid[2]),
                      noise=valid_stats, train_noise=train_stats,
                      training_seconds=sum(x["training_seconds"] for x in history),
                      peak_train_allocated_bytes=max(x["peak_allocated_bytes"] for x in history))
        write_json(root / "validation.json", result)
        print(name, "validation complete", best["epoch"], result["groups"]["overall"]["mcc"], flush=True)
        del model, training, validation

    train_seconds = train[-1]
    del train
    test = prepare_split("test", tokenizer, policy)
    results = []
    for rate in RATES:
        name = f"noise_{rate:03d}"
        root = RESULTS / name
        path = root / "test.json"
        if path.is_file():
            result = json.loads(path.read_text())
        else:
            data, stats = noise_data(test, tokenizer, rate)
            model = fp32_classifier(tokenizer)
            model.load_state_dict(torch.load(WEIGHTS / name / "best.pt", weights_only=True, map_location="cpu"), strict=True)
            result = evaluate(model, data, "test", root=root, fields=FIELDS)
            result.update(edit_retention=packed_edit_retention(data, test[2]), noise=stats)
            write_json(path, result)
            print(name, "test complete", result["groups"]["overall"]["mcc"], flush=True)
            del model, data
        results.append(dict(rate=rate, valid=json.loads((root / "validation.json").read_text()), test=result))
    references = {name: saved_retention(path, test) for name, path in {
        "oracle_full": ORACLE_RESULTS / "oracle_diff/test.csv.gz", "predicted_diff": PREDICTED_RESULTS / "test.csv.gz",
        "uniform_sampling": UNIX_RESULTS / "uniform_sampling/test.csv.gz"}.items()}
    thresholds = plot_curve(results)
    write_json(RESULTS / "summary.json", dict(seed=SEED, policy=policy, privileged_test=True, runs=results,
               preparing_seconds=dict(train=train_seconds, valid=valid[-1], test=test[-1]),
               test_reference_retention=references, mcc_crossings=thresholds))
