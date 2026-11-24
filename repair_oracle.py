"""Privileged PrimeVul experiment. Both revisions are required on every split, including Test."""
from bisect import bisect_left, bisect_right
import csv
from difflib import SequenceMatcher
import gzip
from itertools import accumulate
import json

import torch

from compaction import BUDGET, REGIONS, region_edges
from context_selection import (fixed_example, gaps, input_layout, merge, occupied, prefer_context,
                               related_statements, uniform_fill)
from window_inputs import GROUPS

METHODS = ("oracle_diff", "oracle_diff_context")
EDIT_TYPES = ("replace", "delete", "insert")
GAP_TYPES = ("insert_gap", "delete_gap")
CONTEXT_TYPES = ("condition", "declaration", "assignment")
PREDICTION_FIELDS = ("repair_bpe_spans", "oracle_anchor_runs", "parse_error")


def line_edits(left, right):
    """Align exact physical lines and keep the two ranges of each edit block."""
    prefix = 0
    while prefix < min(len(left), len(right)) and left[prefix] == right[prefix]:
        prefix += 1
    a, b = len(left), len(right)
    while a > prefix and b > prefix and left[a-1] == right[b-1]:
        a -= 1
        b -= 1
    diff = SequenceMatcher(None, left[prefix:a], right[prefix:b], autojunk=False)
    return [(kind, i+prefix, j+prefix, k+prefix, l+prefix)
            for kind, i, j, k, l in diff.get_opcodes() if kind != "equal"]


def map_edits(lines, offsets, edits, side):
    """Return raw BPE intervals, separating genuine edits from empty-side context."""
    positions = list(accumulate(map(len, lines), initial=0))
    starts, ends = zip(*offsets)
    mapped = {kind: [] for kind in (*EDIT_TYPES, *GAP_TYPES)}
    for kind, *ranges in edits:
        lo, hi = ranges[2*side:2*side+2]
        if lo == hi:
            lo, hi = max(0, lo-1), min(len(lines), lo+1)
            kind = f"{kind}_gap"
        # A token straddling a changed line counts once even if it spans several lines.
        start = bisect_right(ends, positions[lo])
        end = bisect_left(starts, positions[hi])
        if start < end:
            mapped[kind].append((start, end))
    return {kind: merge(spans) for kind, spans in mapped.items()}


def reserve_blocks(base, blocks, *, budget=BUDGET):
    """Fit source blocks, sharing an overflowing budget across their locations."""
    combined = merge([*base, *blocks])
    if occupied(combined) <= budget:
        return combined

    blocks = [chunk for start, end in merge(blocks) for chunk in gaps(base, start, end)]
    room = budget - occupied(base)
    count = min(len(blocks), (room + int(not base))//2)
    if count == 0:
        return base
    if count < len(blocks):
        indices = [len(blocks)//2] if count == 1 else [i*(len(blocks)-1)//(count-1) for i in range(count)]
        blocks = [blocks[i] for i in indices]
    # Reserve EOS slots first. Small edits fit in full; larger edits share the rest.
    room -= count - int(not base)
    kept = list(base)
    for i, (start, end) in enumerate(sorted(blocks, key=lambda p: (p[1]-p[0], p[0]))):
        width = min(end-start, room//(count-i))
        left = start + (end-start-width)//2
        kept.append((left, left+width))
        room -= width
    return merge(kept)


def prepare_pair(left, right, tokenizer, *, method, parser):
    """Construct both privileged inputs before either function enters the classifier."""
    lines = [item[0].splitlines(keepends=True) for item in (left, right)]
    edits = line_edits(*lines)
    return tuple(prepare_side(source, tokenizer, map_edits(lines[side], source[2], edits, side),
                              with_context=method == "oracle_diff_context", parser=parser)
                 for side, source in enumerate((left, right)))


def compact_edits(keep, edits):
    return {kind: [(bisect_left(keep, lo), bisect_left(keep, hi)) for lo, hi in spans]
            for kind, spans in edits.items()}


def raw_runs(keep, spans):
    return merge([(int(i), int(i)+1) for lo, hi in spans for i in keep[lo:hi]])


def pack_selected(ids, keep, boundaries, selected, tokenizer):
    """The 9d95363 layout, from reserved blocks through source-order packing."""
    if len(keep) <= BUDGET:
        spans = [(0, len(keep))]
    else:
        edges = [boundaries[i] for i in region_edges(boundaries)]
        spans = uniform_fill(selected, boundaries, list(zip(edges, edges[1:])), contiguous=False)
    return fixed_example(ids, keep, spans, tokenizer, torch.full((REGIONS, 4), -1, dtype=torch.int32))


def prepare_side(source, tokenizer, edits, *, with_context, parser):
    code, ids, offsets = source
    keep, boundaries = input_layout(code, ids, offsets)
    mapped = compact_edits(keep, edits)
    anchors = reserve_blocks([], [span for kind in EDIT_TYPES for span in mapped[kind]])
    anchors = reserve_blocks(anchors, [span for kind in GAP_TYPES for span in mapped[kind]])
    selected, parse_error = anchors, None
    if len(keep) > BUDGET and with_context and anchors and occupied(anchors) < BUDGET:
        links, parse_error = related_statements(code, offsets, keep, anchors, parser)
        selected = prefer_context(anchors, {link for link in links if link[0] in CONTEXT_TYPES})
    example = pack_selected(ids, keep, boundaries, selected, tokenizer)
    example.update(repair_bpe_spans=edits, oracle_anchor_runs=raw_runs(keep, anchors),
                   parse_error=parse_error)
    return example


def overlap_bpe(left, right):
    i = j = total = 0
    while i < len(left) and j < len(right):
        a, b = left[i]
        c, d = right[j]
        total += max(0, min(b, d)-max(a, c))
        if b <= d:
            i += 1
        else:
            j += 1
    return total


def grouped_edit_retention(observations):
    """Summarize original-BPE visibility for (length, raw annotations, visible runs)."""
    per_function = []
    for length, edits, visible in observations:
        edits = dict(edits, actual_edits=merge([span for kind in EDIT_TYPES for span in edits[kind]]))
        counts = {kind: (overlap_bpe(spans, visible), sum(hi-lo for lo, hi in spans))
                  for kind, spans in edits.items()}
        per_function.append((length, counts))
    groups = {}
    for group, (lo, hi) in GROUPS.items():
        members = [counts for length, counts in per_function if lo <= length <= hi]
        kinds = {}
        for kind in (*EDIT_TYPES, *GAP_TYPES, "actual_edits"):
            retained = sum(m[kind][0] for m in members)
            available = sum(m[kind][1] for m in members)
            kinds[kind] = dict(retained_bpe=retained, available_bpe=available,
                               fraction=retained/available if available else None)
        groups[group] = kinds
    return groups


def packed_edit_retention(data, annotations):
    def observations():
        for ex, edits in zip(data.examples, annotations, strict=True):
            raw = ex["fixed"][1, :ex["length"]].tolist()
            yield ex["raw"], edits, merge([(i, i+1) for i in raw if i >= 0])

    return grouped_edit_retention(observations())


def repair_retention(oracle_root, baseline_root, *, extra_predictions=None):
    """Score saved test selections against the same raw edit intervals."""
    paths = {method: baseline_root / method / "test.csv.gz" for method in ("uniform_sampling", "context_aware")}
    paths.update({method: oracle_root / method / "test.csv.gz" for method in METHODS})
    paths.update(extra_predictions or {})
    predictions = {}
    for method, path in paths.items():
        with gzip.open(path, "rt", newline="") as handle:
            predictions[method] = list(csv.DictReader(handle))
    reference = predictions["oracle_diff"]
    for first, second in zip(reference, predictions["oracle_diff_context"], strict=True):
        if any(first[key] != second[key] for key in ("repair_bpe_spans", "oracle_anchor_runs")):
            raise ValueError("the two repair oracles disagree on edits or reserved anchors")
    result = {}
    for method, rows in predictions.items():
        observations = []
        for repair, selected in zip(reference, rows, strict=True):
            if any(repair[key] != selected[key] for key in ("idx", "sample_key", "target", "original_bpe")):
                raise ValueError(f"{method}: test membership or tokenizer differs")
            observations.append((int(selected["original_bpe"]), json.loads(repair["repair_bpe_spans"]),
                                 json.loads(selected["source_runs"])))
        result[method] = grouped_edit_retention(observations)
    return dict(privileged_methods=METHODS, split="test", groups=result)
