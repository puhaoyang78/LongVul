"""Statement-aligned compression and context-aware retrieval within the fine budget."""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from compaction import BUDGET
from context_selection import base_windows, contains, gaps, merge, occupied, uniform_fill

PRIORITY = {"completion": 0, "condition": 1, "declaration": 2, "assignment": 3}
SIZES = (32, 64, 128)
REPLACEMENT_TRIALS = 8
LAYOUTS = {f"{style}{size}": (size, style == "statement")
           for size in SIZES for style in ("fixed", "statement")}


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
    spans: tuple
    centers: tuple = ()


class Candidates:
    """Each center has a source-only bundle and sparse links to its context blocks."""
    def __init__(self, ends, units, links, *, contextual):
        self.centers = list(zip([0,*ends[:-1]], ends))
        self.contextual = contextual
        self.omitted = Counter()
        self.edges = []
        self.bundle = [(span,) for span in self.centers]
        if contextual:
            by_center = defaultdict(set)
            for a,b in units:
                for i,lo,hi in part_overlap(a,b,ends):
                    if (lo,hi) != (a,b):
                        by_center[i].add(("completion",a,b))
            for kind,anchor,context in links:
                if kind in PRIORITY:
                    for i,_,_ in part_overlap(*anchor,ends):
                        by_center[i].add((kind,*context))
            for i, center in enumerate(self.centers):
                lo,hi = center
                links_ = sorted(by_center[i], key=lambda r: (
                    PRIORITY[r[0]], max(0, lo-r[2], r[1]-hi),r[1],r[2]))
                spans = [center]
                for kind,a,b in links_:
                    if contains((a,b),spans):
                        continue
                    expanded = merge([*spans,(a,b)])
                    if occupied(expanded) <= BUDGET:
                        spans = expanded
                    else:
                        self.omitted[kind] += 1
                self.bundle[i] = tuple(spans)
                weights = Counter()
                for a,b in uncovered(spans,[center]):
                    for j,left,right in part_overlap(a,b,ends):
                        weights[j] += right-left
                mass = sum(weights.values())
                self.edges.extend((i,j,n/mass) for j,n in sorted(weights.items()))
        self.starts,self.stops = np.asarray([span for r in self.bundle for span in r], dtype=np.int32).T
        sizes = np.asarray([len(r) for r in self.bundle],dtype=np.int32)
        self.offsets = np.r_[0,sizes.cumsum()[:-1]]
        self.center_starts,self.center_stops = np.asarray(self.centers,dtype=np.int32).T

    def overlap(self, spans, starts, stops, *, separators=False):
        if not spans:
            return np.zeros_like(starts)
        lo,hi = np.asarray(spans,dtype=np.int32).T
        if separators:
            hi = hi+1
        sizes = hi-lo
        prefix = np.r_[0,sizes.cumsum()[:-1]]
        a = (np.searchsorted(lo, starts, side="right")-1).clip(min=0)
        b = (np.searchsorted(lo, stops, side="right")-1).clip(min=0)
        return (prefix[b]+(stops-lo[b]).clip(0,sizes[b])
                -prefix[a]-(starts-lo[a]).clip(0,sizes[a]))

    def marginal(self, selected, *, separators):
        stops = self.stops+int(separators)
        hit = self.overlap(selected,self.starts,stops,separators=separators)
        mass = np.add.reduceat(stops-self.starts-hit,self.offsets)
        return mass-int(separators and not selected)

    def novel_centers(self, selected):
        covered = self.overlap(selected,self.center_starts,self.center_stops)
        return covered < self.center_stops-self.center_starts

    def retrieve(self, scores, source):
        n = len(source.ids)
        if len(scores) != len(self.bundle) or not np.isfinite(scores).all():
            raise ValueError("Code block scores do not match the candidates")
        if n <= BUDGET:
            return Layout(((0,n),))
        selected, ids = [], []
        while occupied(selected) < BUDGET:
            cost = self.marginal(selected,separators=True)
            can_take = (cost <= BUDGET-occupied(selected)) & self.novel_centers(selected)
            if not can_take.any():
                break
            priority = scores/np.maximum(cost,1) if self.contextual else scores
            i = int(np.where(can_take,priority,-np.inf).argmax())
            selected = merge([*selected,*self.bundle[i]])
            ids.append(i)
        if not ids:
            raise ValueError("Every code layout must have an affordable region")
        # Grow available space beside selected spans using the existing physical units.
        spans = uniform_fill(selected,source.boundaries,[],contiguous=True)
        return Layout(tuple(spans),tuple(ids))

    def alternatives(self, layout, scores, source, rng):
        """Swap one bundle and keep both the BPE and SEP budgets matched."""
        if not layout.centers:
            return []
        body, budget = lengths(layout.spans),occupied(layout.spans)
        for departing in sorted(layout.centers,key=lambda i:(scores[i],i)):
            retained = [i for i in layout.centers if i!=departing]
            shared = merge([s for i in retained for s in self.bundle[i]])
            cost = self.marginal(shared,separators=True)
            code = self.marginal(shared,separators=False)
            eligible = ((cost<=budget-occupied(shared)) & (code<=body-lengths(shared)) & self.novel_centers(shared))
            eligible[list(layout.centers)] = False
            choices = np.flatnonzero(eligible)
            rank = sorted(choices.tolist(),key=lambda i:(-scores[i]/max(cost[i],1),i))
            boundary, others = rank[:REPLACEMENT_TRIALS],rank[REPLACEMENT_TRIALS:]
            exploration = rng.choice(others,min(REPLACEMENT_TRIALS,len(others)),replace=False).tolist() if others else []
            out = []
            # Refill beside each alternative's bundles using exactly the original packing policy.
            # Match the final body width as well as occupancy so SEP count cannot explain feedback.
            for pool in (boundary,exploration):
                for i in pool:
                    initial = merge([*shared,*self.bundle[i]])
                    if len(initial)<len(layout.spans):
                        continue
                    spans = tuple(uniform_fill(initial,source.boundaries,[],contiguous=True))
                    if spans==layout.spans or lengths(spans)!=body or occupied(spans)!=budget:
                        continue
                    out.append((Layout(spans,tuple([*retained,i])),
                                uncovered(layout.spans,spans),uncovered(spans,layout.spans)))
                    break
            if out:
                return out
        return []


def uniform_layout(source):
    n = len(source.ids)
    if n <= BUDGET:
        return Layout(((0,n),))
    base, limits = base_windows(source.boundaries)
    return Layout(tuple(uniform_fill(base,source.boundaries,limits,contiguous=False)))
