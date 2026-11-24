"""Learn the additional repair-token value of Tree-sitter-linked statement groups."""
import argparse
from bisect import bisect_left, bisect_right
import csv
from dataclasses import dataclass
import gzip
from itertools import islice
import json
from random import Random
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from compaction import BUDGET, REGIONS, PackedData
from context_selection import (base_windows, describe, fixed_example, gaps, link_retention, merge, occupied,
                               related_statements, source_parser, uniform_fill)
from paired import DATA, ROOT, SEED, UNIX_BASE, UNIX_RESULTS, fit_512_inputs, load_tokenizer, prime_source_rows
from predicted_diff import HISTORICAL_REVISION, RESULTS as DIFF_RESULTS, load_sources, paired_annotations
from region_locator import PARTS, Regions, region_metrics, score_regions, train_localizer
from repair_oracle import (CONTEXT_TYPES, EDIT_TYPES, GAP_TYPES, compact_edits, grouped_edit_retention,
                           overlap_bpe, packed_edit_retention, raw_runs)
from rgcl.artifacts import write_json
from source_regions import Region
from window_inputs import GROUPS

RESULTS = ROOT / "results" / "syntax_regions_512"
WEIGHTS = ROOT / "checkpoints" / "syntax_regions_512"
METHODS = ("random_region", "region_oracle", "predicted_region")
LIMITS = (1, 2, 4)
FIELDS = ("encoding_runs", "regions_selected", "additional_positions")


@dataclass(slots=True)
class Candidate:
    spans: tuple
    novel: tuple
    cost: int


@dataclass(slots=True)
class Plan:
    base: list
    limits: list
    candidates: list
    links: tuple
    linked_statements: int
    parse_error: bool | None


def plan_regions(source, parser):
    n = len(source.ids)
    if n <= BUDGET:
        return Plan([(0,n)] if n else [], [], [], (), 0, None)
    base, limits = base_windows(source.boundaries)
    base = merge(base)
    links, error = related_statements(source.code, source.offsets, source.keep, None, parser)
    by_anchor = {}
    for kind, anchor, context in sorted(links):
        if kind in CONTEXT_TYPES:
            by_anchor.setdefault(anchor, []).append((kind, anchor, context))
    candidates, feasible, seen = [], [], set()
    for anchor, edges in sorted(by_anchor.items()):
        spans = tuple(merge([anchor, *(link[2] for link in edges)]))
        joined = merge([*base, *spans])
        cost = occupied(joined)-occupied(base)
        if occupied(joined) > BUDGET:
            continue
        feasible.extend(edges)
        if joined == base or spans in seen:
            continue
        seen.add(spans)
        novel = tuple(seg for a,b in spans for seg in gaps(base, a, b))
        candidates.append(Candidate(spans, novel, cost))
    return Plan(base, limits, candidates, tuple(feasible), len(by_anchor), error)


def target_scores(source, plan, edits):
    """Density of distinct added edit/proxy BPE in the base-relative packed cost."""
    compact = compact_edits(source.keep, edits)
    real = merge([span for kind in EDIT_TYPES for span in compact[kind]])
    proxy = merge([span for kind in GAP_TYPES for span in compact[kind]])
    relevant = merge([*real, *proxy])
    return np.asarray([min(1., overlap_bpe(candidate.novel, relevant)/max(1, candidate.cost))
                       for candidate in plan.candidates], dtype=np.float32)


def locator_sources(sources, plans):
    """Pack each candidate's disjoint source runs in order for the existing embedding-bag locator."""
    result = []
    for source, plan in zip(sources, plans, strict=True):
        tokens, regions, cursor = [], [], 0
        for candidate in plan.candidates:
            size = 0
            for a,b in candidate.spans:
                tokens.append(source.ids[a:b])
                size += b-a
            regions.append(Region(cursor, cursor+size, 0, 0, "syntax_group"))
            cursor += size
        result.append(SimpleNamespace(ids=np.concatenate(tokens) if tokens else np.empty(0, dtype=np.int32), regions=regions))
    return result


def context_group_windows(sources, plans):
    """Map the four exact candidate pools onto their original encoder windows."""
    from locator_quality import whole_windows

    first = 0
    for source, plan in zip(sources, plans, strict=True):
        windows = list(whole_windows(source))
        stops = [hi for _,hi,_ in windows]
        requests = [[] for _ in windows]
        for index, candidate in enumerate(plan.candidates, first):
            size = sum(b-a for a,b in candidate.spans)
            cuts = [size*k//PARTS for k in range(PARTS+1)]
            origin = 0
            for a,b in candidate.spans:
                for part, (lo,hi) in enumerate(zip(cuts,cuts[1:])):
                    left, right = max(a, a+lo-origin), min(b, a+hi-origin)
                    window = bisect_right(stops, left)
                    while left < right:
                        end = min(right, stops[window])
                        requests[window].append((index*PARTS+part, left, end))
                        left = end
                        window += 1
                origin += b-a
        first += len(plan.candidates)
        for (lo,hi,_), pool in zip(windows, requests, strict=True):
            yield source, lo, hi, pool


def contextual_group_vectors(model, sources, plans, tokenizer, split):
    from locator_quality import context_encoder_batches

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    groups = sum(len(plan.candidates) for plan in plans)
    sums = torch.zeros((groups*PARTS, model.config.hidden_size), dtype=torch.float32)
    counts = torch.zeros(groups*PARTS, dtype=torch.int32)
    body = calls = windows = padded = 0
    stream = context_group_windows(sources, plans)
    for batch, cumulative, slots in context_encoder_batches(model, tokenizer, stream):
        parts = ((part, row, 1+a-lo, 1+b-lo) for row, (_,lo,_,pool) in enumerate(batch) for part,a,b in pool)
        while chunk := list(islice(parts, 8192)):
            part, row, a, b = torch.tensor(chunk, dtype=torch.int64).T
            pooled = cumulative[row.cuda(), b.cuda()] - cumulative[row.cuda(), a.cuda()]
            sums.index_add_(0, part, pooled.cpu())
            counts.index_add_(0, part, (b-a).int())
        body += sum(hi-lo for _,lo,hi,_ in batch)
        calls += 1
        windows += len(batch)
        padded += slots
    expected = sum(b-a for plan in plans for candidate in plan.candidates for a,b in candidate.spans)
    if body != sum(len(s.ids) for s in sources) or int(counts.sum()) != expected:
        raise ValueError("Full-function context windows or candidate pools lost source BPE")
    sums.div_(counts.clamp_min(1)[:, None])
    vectors = sums.reshape(groups, PARTS, model.config.hidden_size).half()
    torch.cuda.synchronize()
    stats = dict(seconds=perf_counter()-started, regions=groups, windows=windows, encoder_calls=calls,
                 body_bpe=body, padded_positions=padded, pooled_feature_bytes=vectors.nbytes,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated())
    print("contextual syntax groups", split, groups, "windows", windows, "seconds", round(stats["seconds"], 2), flush=True)
    return vectors, stats


def choose(plan, scores, limit, *, positive_only=False):
    if len(plan.candidates) != len(scores):
        raise ValueError("Scores differ from the ordered syntax candidate list")
    kept, count = plan.base, 0
    for index in np.argsort(-scores, kind="stable"):
        if count == limit or (positive_only and scores[index] <= 0):
            break
        union = merge([*kept, *plan.candidates[index].spans])
        if union != kept and occupied(union) <= BUDGET:
            kept = union
            count += 1
    return kept, count


def make_data(rows, sources, plans, values, tokenizer, limit, *, positive_only=False, relations=True):
    def examples():
        blank = torch.full((REGIONS, 4), -1, dtype=torch.int32)
        for source, plan, scores in zip(sources, plans, values, strict=True):
            kept, count = choose(plan, scores, limit, positive_only=positive_only)
            spans = uniform_fill(kept, source.boundaries, plan.limits, contiguous=False) if plan.limits else kept
            example = fixed_example(source.raw_ids, source.keep, spans, tokenizer, blank)
            example.update(encoding_runs=spans, regions_selected=count, additional_positions=occupied(kept)-occupied(plan.base))
            if relations:
                example.update(links=link_retention(plan.links, spans), parse_error=plan.parse_error)
            yield example

    return PackedData(rows, examples())


def split_inputs(split, tokenizer, parser):
    started = perf_counter()
    rows = prime_source_rows(DATA, split)
    sources = load_sources(rows, tokenizer)
    read = perf_counter()-started
    started = perf_counter()
    plans = [plan_regions(source, parser) for source in sources]
    costs = dict(source_seconds=read, syntax_and_groups_seconds=perf_counter()-started)
    print("syntax candidates", split, sum(len(p.candidates) for p in plans), flush=True)
    return rows, sources, plans, costs


def candidate_coverage(sources, plans, annotations):
    base, upper = [], []
    for source, plan, edits in zip(sources, plans, annotations, strict=True):
        union = merge([*plan.base, *(span for c in plan.candidates for span in c.spans)])
        base.append((len(source.raw_ids), edits, raw_runs(source.keep, plan.base)))
        upper.append((len(source.raw_ids), edits, raw_runs(source.keep, union)))
    return dict(base_windows=grouped_edit_retention(base), eligible_candidate_union=grouped_edit_retention(upper),
                eligible_groups=sum(len(p.candidates) for p in plans), linked_statements=sum(p.linked_statements for p in plans),
                parsed_with_errors=sum(p.parse_error is True for p in plans),
                long_functions_with_groups=sum(bool(p.candidates) for p in plans))


def archived_controls(rows, sources, plans, annotations):
    roots = dict(uniform_sampling=UNIX_RESULTS/"uniform_sampling", context_aware=UNIX_RESULTS/"context_aware",
                 predicted_diff=DIFF_RESULTS)
    out = {}
    for method, root in roots.items():
        with gzip.open(root/"test.csv.gz", "rt") as stream:
            saved = list(csv.DictReader(stream))
        observations, links = [], []
        for row, source, plan, edits, prediction in zip(rows, sources, plans, annotations, saved, strict=True):
            length = len(source.raw_ids)
            if prediction["sample_key"] != row["sample_key"] or int(prediction["original_bpe"]) != length:
                raise ValueError(f"Archived membership mismatch for {method}")
            raw = json.loads(prediction["source_runs"])
            compact = [(bisect_left(source.keep, a), bisect_left(source.keep, b)) for a,b in raw]
            observations.append((length, edits, raw))
            links.append((length, link_retention(plan.links, compact)))
        relations = {}
        for group, (lo,hi) in GROUPS.items():
            part = [count for n,count in links if lo <= n <= hi]
            relations[group] = {kind: dict(available=sum(c[kind][0] for c in part),
                                       both_kept=sum(c[kind][2] for c in part)) for kind in CONTEXT_TYPES}
        info = json.loads((root/"metrics.json").read_text())
        out[method] = dict(groups=info["split"]["test"]["groups"], selection=info["split"]["test"]["selection"],
                           edit_retention=grouped_edit_retention(observations), shared_candidate_relations=relations,
                           preparation_seconds=info.get("preparation_seconds"),
                           inference_seconds=info["split"]["test"]["inference_seconds"])
    return out


def run(tokenizer, seed, methods, *, contextual=False):
    parser = source_parser()
    output = RESULTS / "contextual" if contextual else RESULTS
    weights = WEIGHTS / "contextual" if contextual else WEIGHTS
    feature_config = dict(target="union of added edit and empty-side neighbor BPE divided by actual added positions", base_quota=88)
    if contextual:
        from locator_quality import CONTEXT_BATCH, load_context_encoder
        from paired import BACKBONES

        started = perf_counter()
        encoder = load_context_encoder()
        torch.cuda.synchronize()
        encoder_load_seconds = perf_counter()-started
        name, revision = BACKBONES["unixcoder"]
        feature_config.update(backbone=name, revision=revision, frozen_encoder=True, input_window=BUDGET+2,
                              batch_encoder=CONTEXT_BATCH, precision="bf16 SDPA, fp16 pooled features, fp32 head")
    vectors = {} if contextual else None
    splits, edits, truth, loc_inputs = {}, {}, {}, {}
    for split in ("train", "valid"):
        splits[split] = split_inputs(split, tokenizer, parser)
        _, sources, plans, costs = splits[split]
        edits[split] = paired_annotations(sources)
        truth[split] = [target_scores(s,p,e) for s,p,e in zip(sources, plans, edits[split], strict=True)]
        if contextual:
            vectors[split], costs["full_function_encoding"] = contextual_group_vectors(encoder, sources, plans, tokenizer, split)
            loc_inputs[split] = [SimpleNamespace(regions=p.candidates) for p in plans]
        else:
            started = perf_counter()
            loc_inputs[split] = locator_sources(sources, plans)
            costs["locator_token_bags_seconds"] = perf_counter()-started
    model, prediction, loc_result = train_localizer(loc_inputs, truth, weights/"locator", output/"locator",
        vectors=vectors, soft_targets=True, feature_config=feature_config)
    del loc_inputs, vectors

    rows,sources,plans,_ = splits["valid"]
    candidates = []
    for limit in LIMITS:
        data = make_data(rows, sources, plans, prediction["valid"], tokenizer, limit, relations=False)
        retained = packed_edit_retention(data, edits["valid"])["long"]["actual_edits"]
        candidates.append(dict(limit=limit, **retained))
    limit = max(candidates, key=lambda c: (c["retained_bpe"],-c["limit"]))["limit"]
    policy = dict(selected_limit=limit, validation=candidates, base_quota=88, sampler=HISTORICAL_REVISION, ranking_seed=SEED)
    print("Valid selected syntax-group limit", limit, flush=True)

    splits["test"] = split_inputs("test", tokenizer, parser)
    rows, sources, plans, test_costs = splits["test"]
    if contextual:
        test_vectors, test_costs["full_function_encoding"] = contextual_group_vectors(encoder, sources, plans, tokenizer, "test")
        regions = Regions([SimpleNamespace(regions=p.candidates) for p in plans], vectors=test_vectors)
        del encoder, test_vectors
    else:
        started = perf_counter()
        regions = Regions(locator_sources(sources, plans))
        test_costs["locator_token_bags_seconds"] = perf_counter()-started
    flat, test_costs["locator_scoring_seconds"] = score_regions(model, regions)
    prediction["test"] = regions.regroup(flat)
    del model, regions

    datasets = {m: {} for m in METHODS}
    selection_costs = {m: {} for m in METHODS}
    for split, (rows, sources, plans, _) in splits.items():
        started = perf_counter()
        random = []
        for row, plan in zip(rows, plans, strict=True):
            rng = Random(f"{SEED}/{row['sample_key']}")
            random.append(np.asarray([rng.random() for _ in plan.candidates]))
        datasets["random_region"][split] = make_data(rows, sources, plans, random, tokenizer, limit)
        selection_costs["random_region"][split] = perf_counter()-started
        started = perf_counter()
        datasets["predicted_region"][split] = make_data(rows, sources, plans, prediction[split], tokenizer, limit)
        selection_costs["predicted_region"][split] = perf_counter()-started

    # Both deployable Test layouts are fixed before deriving the Test repair annotation.
    _, sources, plans, _ = splits["test"]
    edits["test"] = paired_annotations(sources)
    truth["test"] = [target_scores(s,p,e) for s,p,e in zip(sources, plans, edits["test"], strict=True)]
    loc_result["splits"]["test"] = region_metrics(prediction["test"], [y > 0 for y in truth["test"]])
    for split, (rows, sources, plans, _) in splits.items():
        started = perf_counter()
        datasets["region_oracle"][split] = make_data(rows, sources, plans, truth[split], tokenizer, limit, positive_only=True)
        selection_costs["region_oracle"][split] = perf_counter()-started

    root = output / f"seed{seed}"
    report = dict(policy=policy, locator=loc_result, preprocessing={k:v[-1] for k,v in splits.items()},
                  selection_seconds=selection_costs,
                  coverage={split: candidate_coverage(splits[split][1], splits[split][2], edits[split])
                            for split in ("valid", "test")}, inputs={})
    for method,sets in datasets.items():
        report["inputs"][method] = {split: dict(edit_retention=packed_edit_retention(sets[split], edits[split]),
            selection=describe(sets[split], None, None, paired=True)) for split in ("valid", "test")}
    rows,sources,plans,_ = splits["test"]
    report["archived_test_controls"] = archived_controls(rows, sources, plans, edits["test"])
    if contextual:
        report.update(encoder_load_seconds=encoder_load_seconds)
    write_json(root/"inputs.json", report)
    print("syntax region input verification complete", flush=True)
    if contextual:
        return
    del splits, edits, truth, prediction

    result = {}
    for method in methods:
        result[method] = fit_512_inputs(tokenizer, datasets[method], root/method, WEIGHTS/f"seed{seed}"/method,
            method=method, seed=seed, configuration=dict(study="syntax_regions", policy=policy,
            privileged_test=method == "region_oracle"), fields=FIELDS)
    write_json(root/"summary.json", dict(seed=seed, methods=result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--contextual-locator-only", action="store_true",
                        help="Compare frozen full-function window features without retraining a classifier.")
    args = parser.parse_args()
    if torch.cuda.device_count() != 1 or "4090" not in torch.cuda.get_device_name():
        raise ValueError("Select one RTX 4090 using CUDA_VISIBLE_DEVICES")
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    run(load_tokenizer("unixcoder", UNIX_BASE), args.seed, [m for m in METHODS if m in args.methods],
        contextual=args.contextual_locator_only)


if __name__ == "__main__":
    main()
