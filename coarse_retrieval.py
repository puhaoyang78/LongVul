"""Center-first retrieval and the shared source-aligned block partitions."""
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

import numpy as np

from compaction import BUDGET
from context_selection import (base_windows, contains, gaps, merge, occupied,
                               prefer_context, related_statements, uniform_fill)


CONTEXT_TYPES = {"completion", "condition", "declaration", "assignment"}


def block_ends(length, units, lines, target, aligned):
    if not aligned:
        return [*range(target, length, target), length]
    # A cut on a nested boundary would still split its containing statement.
    safe, covered_until = [], 0
    for a, b in sorted(set(units)):
        if a >= covered_until:
            safe.append(a)
            if covered_until:
                safe.append(covered_until)
        covered_until = max(covered_until, b)
    preferred = sorted({*safe, covered_until, length})
    ends, start = [], 0
    while length-start > target:
        minimum = start+max(1, target*3//4)
        maximum = min(length, start+target*3//2)
        center = start+target
        eligible = preferred[bisect_left(preferred, minimum):bisect_right(preferred, maximum)]
        if not eligible:
            eligible = lines[bisect_left(lines, minimum):bisect_right(lines, maximum)]
        cut = min(eligible, key=lambda x: (abs(x-center), x)) if eligible else center
        ends.append(cut)
        start = cut
    if start < length:
        ends.append(length)
    return ends


def part_overlap(lo, hi, ends):
    i = bisect_right(ends, lo)
    while lo < hi:
        stop = min(ends[i], hi)
        yield i, lo, stop
        lo, i = stop, i+1


def uncovered(spans, covered):
    return [part for a,b in spans for part in gaps(covered,a,b)]


def lengths(spans):
    return sum(b-a for a,b in spans)


@dataclass(slots=True)
class Layout:
    spans: list
    centers: list
    context: list
    fill: list


def ranked_fill(blocks, ranked, selected):
    """Use remaining positions in score order, preserving the historical partial-block rule."""
    spans = list(selected)
    for index in ranked:
        for lo,hi in gaps(spans,*blocks[index]):
            extended = merge([*spans,(lo,hi)])
            if occupied(extended) <= BUDGET:
                spans = extended
            else:
                room = BUDGET-occupied(spans)
                left = room-int(bool(spans) and not any(b==lo for _,b in spans))
                right = room-int(bool(spans) and not any(a==hi for a,_ in spans))
                if max(left,right) > 0:
                    span = (lo,lo+left) if left>=right else (hi-right,hi)
                    spans = merge([*spans,span])
            if occupied(spans) == BUDGET:
                return spans
    raise ValueError("Ranked retrieval did not fill the available budget")


def retrieve(scores, source, ends, context_budget, parser=None):
    if len(scores) != len(ends) or not 0 <= context_budget < BUDGET:
        raise ValueError("Invalid scores or context budget")
    length = len(source.ids)
    if length <= BUDGET:
        return Layout([(0,length)],[],[],[])
    blocks = list(zip([0,*ends[:-1]],ends))
    ranked = np.argsort(-scores,kind="stable")
    centers = []
    for index in ranked:
        candidate = merge([*centers,blocks[index]])
        if occupied(candidate) <= BUDGET-context_budget:
            centers = candidate
        if occupied(centers) == BUDGET-context_budget:
            break
        # Zero reserve follows the historical policy: trim the first over-budget block.
        if not context_budget and candidate != centers:
            break
    if not context_budget:
        spans = ranked_fill(blocks,ranked,[])
        return Layout(spans,centers,[],uncovered(spans,centers))

    links, _ = related_statements(source.code,source.offsets,source.keep,centers,parser)
    selected = prefer_context(centers,(link for link in links if link[0] in CONTEXT_TYPES),
                              budget=occupied(centers)+context_budget)
    spans = selected if occupied(selected)==BUDGET else ranked_fill(blocks,ranked,selected)
    return Layout(spans,centers,uncovered(selected,centers),uncovered(spans,selected))


def uniform_layout(source):
    if len(source.ids) <= BUDGET:
        return Layout([(0,len(source.ids))],[],[],[])
    base,limits = base_windows(source.boundaries)
    spans = uniform_fill(base,source.boundaries,limits,contiguous=False)
    return Layout(spans,merge(base),[],uncovered(spans,merge(base)))
