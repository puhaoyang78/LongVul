"""Select whole regions and one-hop native dependence edges under the BPE budget."""
from bisect import bisect_left, bisect_right
from collections import defaultdict
from math import ceil, sqrt

from compaction import BODY_BUDGET, REGIONS, choose_candidates, line_boundaries, region_edges
from context_selection import BASE_QUOTAS, merge, occupied, prefer_context, related_statements, uniform_fill
from dependence import EDGE_TYPES
from repair_oracle import CONTEXT_TYPES, EDIT_TYPES, GAP_TYPES, reserve_blocks

MAX_FRAGMENTS = 12
BRIDGE_BPE = 8
NEW_FRAGMENT_COST = 8


def available_graph(graph, weights):
    """Withhold dependency context when any edge type required by the arm is unavailable."""
    if not any(weights):
        return None, "not_used"
    if graph is None:
        raise ValueError("A dependency method requires a recorded extraction outcome")
    if graph["status"] != "ok":
        return None, "parse_or_graph_failed"
    if any(weight and not graph["edges"][kind] for kind, weight in zip(EDGE_TYPES, weights, strict=True)):
        return None, "no_effective_dependence"
    return graph["edges"], "available"


def physical_limits(boundaries):
    edges = [boundaries[i] for i in region_edges(boundaries)]
    return list(zip(edges, edges[1:]))


def uniform(source, budget, cuts, parser=None):
    quota = (budget - REGIONS + 1) * BASE_QUOTAS[0] // BODY_BUDGET
    windows, limits = choose_candidates(cuts, alternatives=False, quotas=(quota,)*REGIONS)
    spans = [choices[0] for choices in windows]
    if parser is not None:
        links, _ = related_statements(source.code, source.source_offsets, range(len(source.ids)), spans, parser)
        spans = prefer_context(spans, links, budget=budget)
    return uniform_fill(spans, cuts, limits, budget=budget)


def repair_anchors(edits, budget):
    actual = [span for kind in EDIT_TYPES for span in edits[kind]]
    proxy = [span for kind in GAP_TYPES for span in edits[kind]]
    base = reserve_blocks([], actual, budget=budget)
    return reserve_blocks(base, proxy, budget=budget)


def region_ids(regions, spans, *, touch=False):
    starts = [r.start for r in regions]
    ends = [r.end for r in regions]
    return visible_ids(starts, ends, spans, touch=touch)


def visible_ids(starts, ends, spans, *, touch=False):
    found = set()
    for a, b in spans:
        lo = bisect_right(ends, a) if touch else bisect_left(starts, a)
        hi = bisect_left(starts, b) if touch else bisect_right(ends, b)
        found.update(range(lo, hi))
    return found


def bridge_spans(spans):
    result = []
    for a, b in spans:
        if result and a-result[-1][1] <= BRIDGE_BPE:
            result[-1] = (result[-1][0], b)
        else:
            result.append((a, b))
    return result


def usable(source, region):
    return bool(source.code[region.source_start:region.source_end].strip(" \t\v\r\n{};"))


def global_regions(source, budget):
    """Cover length-dependent bands with whole regions, without slot quotas."""
    n = len(source.ids)
    count = min(MAX_FRAGMENTS//2, ceil(n/budget))
    candidates = [r for r in source.regions if usable(source, r) and r.end-r.start <= budget]
    bands = []
    for i in range(count):
        a, b = n*i/count, n*(i+1)/count
        center = (a+b)/2
        here = [r for r in candidates if a <= (r.start+r.end)/2 < b]
        if here:
            here.sort(key=lambda r: (abs((r.start+r.end)/2-center), r.end-r.start, r.start))
            bands.append(here)
    reserved = []
    for here in bands:
        smallest = min(here, key=lambda r: r.end-r.start).span
        packed = merge([*(span for _, span in reserved), smallest])
        if occupied(packed) <= budget:
            reserved.append((here, smallest))
    for i, (here, _) in enumerate(reserved):
        other = [span for j, (_, span) in enumerate(reserved) if j != i]
        for region in here:
            planned = merge([*other, region.span])
            if occupied(planned) <= budget:
                reserved[i] = (here, region.span)
                break
    return merge([span for _,span in reserved]), candidates


def predicted_candidates(source, candidates, scores, budget):
    available = {r.start for r in candidates}
    ordering = sorted((i for i, r in enumerate(source.regions) if r.start in available),
                      key=lambda i: (-float(scores[i]), source.regions[i].start))
    seeds, seen = set(), 0
    for i in ordering:
        region = source.regions[i]
        seeds.add(i)
        seen += region.end-region.start
        if seen >= budget*2:
            break
    return seeds


def optimize(regions, base, seeds, graph, weights, budget, *, scores=None, bridge=True):
    """Recompute closed-edge gain, actual packed cost and spatial novelty at each step."""
    starts, ends = [r.start for r in regions], [r.end for r in regions]
    edges, incident = [], defaultdict(set)
    actions = {(i,) for i in seeds} if scores is not None else set()
    for kind, weight in zip(EDGE_TYPES, weights, strict=True):
        if weight:
            for u, v in graph.get(kind, ()):
                if v not in seeds:
                    continue
                incident[u].add(len(edges))
                incident[v].add(len(edges))
                edges.append((u, v, weight))
                actions.add((u,))
                actions.add(tuple(sorted((u, v))))
    actions = sorted(actions)
    chosen = merge(base)
    limit = max(MAX_FRAGMENTS, len(chosen))
    selected = visible_ids(starts, ends, chosen)
    closed = {i for i, (u, v, _) in enumerate(edges) if u in selected and v in selected}
    additions = 0
    while actions:
        room, best = occupied(chosen), None
        current = set(chosen)
        observed = set()
        for action in actions:
            if all(i in selected for i in action):
                continue
            if sum(ends[i]-starts[i] for i in action if i not in selected) > budget:
                continue
            merged = merge([*chosen, *(regions[i].span for i in action)])
            if occupied(merged) > budget:
                continue
            with_bridges = bridge_spans(merged) if bridge else merged
            planned = with_bridges if occupied(with_bridges) <= budget else merged
            if len(planned) > limit:
                continue
            identity = tuple(planned)
            if identity in observed:
                continue
            observed.add(identity)
            newly = visible_ids(starts, ends, [span for span in planned if span not in current]) - selected
            if not newly:
                continue
            endpoints = newly | selected
            possible = set().union(*(incident[i] for i in newly)) - closed
            gain = {i for i in possible if edges[i][0] in endpoints and edges[i][1] in endpoints}
            value = sum(edges[i][2] for i in gain)
            if scores is not None:
                for i in sorted(newly & seeds):
                    center = (starts[i]+ends[i])/2
                    distance = min((max(lo-center, center-hi, 0) for lo, hi in chosen), default=ends[-1])
                    value += (float(scores[i])+distance/max(1, ends[-1]))*sqrt(ends[i]-starts[i])
            cost = occupied(planned)-room
            fragment_cost = NEW_FRAGMENT_COST*max(0, len(planned)-len(chosen))
            key = (value/max(1, cost+fragment_cost), len(gain), -len(planned), -cost, -action[0])
            if value > 0 and (best is None or key > best[0]):
                best = key, planned, newly, gain
        if best is None:
            break
        _, chosen, newly, gain = best
        selected.update(newly)
        closed.update(gain)
        additions += 1
    return chosen, additions


def choose(source, method, budget, *, weights=(0, 0), edits=None, scores=None, graph=None, parser=None):
    """The deployable methods accept scores and source-only graphs, never edit annotations."""
    privileged = method.startswith("oracle_")
    if privileged != (edits is not None):
        raise ValueError("Only a privileged Oracle may receive paired edits")
    if (method in {"predicted", "longvul"}) != (scores is not None):
        raise ValueError("Predicted methods require source-only localizer scores")
    n = len(source.ids)
    anchors = repair_anchors(edits, budget) if privileged else []
    if n <= budget:
        return ([(0, n)] if n else []), anchors, 0
    lines = source.boundaries if source.boundary_budget == budget else line_boundaries(
        source.code, source.source_offsets, range(n), budget)
    if method in {"uniform", "syntax"}:
        return uniform(source, budget, lines, parser if method == "syntax" else None), [], 0
    if method == "oracle_syntax":
        spans = anchors
        if anchors:
            links, _ = related_statements(source.code, source.source_offsets, range(n), anchors, parser)
            spans = prefer_context(anchors, {link for link in links if link[0] in CONTEXT_TYPES}, budget=budget)
        return uniform_fill(spans, lines, physical_limits(lines), budget=budget), anchors, 0
    steps = 0
    if scores is not None:
        base, candidates = global_regions(source, budget)
        seeds = predicted_candidates(source, candidates, scores, budget)
        spans, steps = optimize(source.regions, base, seeds, graph or {}, weights, budget, scores=scores)
    elif any(weights):
        seeds = region_ids(source.regions, anchors, touch=True)
        spans, steps = optimize(source.regions, anchors, seeds, graph, weights, budget)
    else:
        spans = anchors
    cuts = [0, *(region.end for region in source.regions)] if any(weights) or scores is not None else lines
    limit = max(MAX_FRAGMENTS, len(anchors)) if any(weights) or scores is not None else None
    return uniform_fill(spans, cuts, physical_limits(lines), budget=budget,
                        fragment_limit=limit), anchors, steps
