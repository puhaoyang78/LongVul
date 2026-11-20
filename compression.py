"""Single-function input selection. Positions refer to the original CodeBERT BPE."""
from bisect import bisect_left, bisect_right
from collections import Counter
from itertools import accumulate
import re

from dataset import BUDGET

METHODS = ("head", "head_tail", "uniform4", "compact")
INDENT = re.compile(r"^[ \t]+", re.MULTILINE)
SPLICE = re.compile(r"\\[ \t]*\r?\n")


def line_layout(code, ids, offsets):
    # Preserve a separator on each indented line. Raw literals and line splices
    # can make physical indentation significant inside C/C++ tokens.
    keep = list(range(len(ids)))
    if 'R"' not in code and not SPLICE.search(code):
        starts = [s for s, _ in offsets]
        discard = set()
        for match in INDENT.finditer(code):
            for i in range(bisect_left(starts, match.start()), bisect_left(starts, match.end() - 1)):
                s, e = offsets[i]
                if s < e <= match.end() - 1:
                    discard.add(i)
        keep = [i for i in keep if i not in discard]
    n = len(keep)
    if n <= BUDGET:
        return keep, [0, n]

    # Form physical-line units. Divide oversized lines into pieces small enough
    # to compete for a window, preferring whitespace boundaries.
    boundaries = [0]
    unit_limit = (BUDGET - 3) // 8
    last_space = 0
    for j, i in enumerate(keep):
        s, e = offsets[i]
        piece = code[s:e]
        if piece[:1].isspace():
            last_space = j
        if j + 1 - boundaries[-1] > unit_limit:
            cut = last_space if last_space - boundaries[-1] >= unit_limit // 2 else j
            boundaries.append(cut)
        if '\n' in piece:
            boundaries.append(j + 1)
    if boundaries[-1] != n:
        boundaries.append(n)

    return keep, boundaries


def four_windows(ids, keep, boundaries, *, uniform):
    n = len(keep)
    if n <= BUDGET:
        return keep
    # Divide by position for coverage, score contiguous runs by within-function
    # bigram rarity, and keep original order. Reserve one EOS per possible gap.
    if not uniform:
        seq = [ids[i] for i in keep]
        grams = list(zip([-1, *seq[:-1]], seq))
        counts = Counter(grams)
        weights = {g: c ** -0.5 for g, c in counts.items()}
        prefix = [0., *accumulate(weights[g] for g in grams)]
    edges = [0]
    for k in range(1, 4):
        p = bisect_left(boundaries, n * k / 4)
        edges.append(min((p - 1, p), key=lambda t: abs(boundaries[t] - n * k / 4)))
    edges.append(len(boundaries) - 1)
    positions = []
    previous_end = 0
    for k, (lo, hi) in enumerate(zip(edges, edges[1:])):
        quota = (BUDGET - 3) * (k + 1) // 4 - (BUDGET - 3) * k // 4
        center = (boundaries[lo] + boundaries[hi]) / 2
        best_value = None
        for start in boundaries[lo:hi]:
            end = boundaries[bisect_right(boundaries, start + quota, lo, hi + 1) - 1]
            if uniform:
                # Choose the line-aligned window that covers most of the
                # central quota, then break ties by proximity to its center.
                overlap = min(end, center + quota / 2) - max(start, center - quota / 2)
                value = (overlap, -abs((start + end) / 2 - center))
            else:
                value = prefix[end] - prefix[start]
            if best_value is None or value > best_value:
                best_value, a, b = value, start, end
        if positions and previous_end != a:
            positions.append(None)
        positions.extend(keep[a:b])
        previous_end = b
    return positions


def select(method, code, ids, offsets=None):
    n = len(ids)
    if method == "head" or n <= BUDGET:
        return list(range(min(BUDGET, n)))
    if method == "head_tail":
        half = BUDGET // 2
        return [*range(half), *range(n - (BUDGET - half), n)]
    if method not in ("uniform4", "compact"):
        raise ValueError(method)
    keep, boundaries = line_layout(code, ids, offsets)
    return four_windows(ids, keep, boundaries, uniform=method == "uniform4")


def model_ids(ids, positions, tok):
    return [tok.cls_token_id, *(ids[i] if i is not None else tok.sep_token_id for i in positions),
            tok.sep_token_id]
