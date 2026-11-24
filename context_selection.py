"""Fixed-budget UniXcoder inputs and lexical statement-retention diagnostics."""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import csv
import gzip
from itertools import accumulate
import json

import torch

from compaction import BUDGET, MAX_LENGTH, REGIONS, aligned_windows, choose_candidates, line_layout, pack_fixed
from window_inputs import GROUPS

METHODS = ("uniform4", "uniform_sampling", "context_aware")
BASE_QUOTAS = (88,) * REGIONS
RELATED_WEIGHTS = {"condition": 3, "declaration": 2, "assignment": 2, "completion": 2, "neighbor": 1}
CONTROL = {"if_statement", "for_statement", "for_range_loop", "while_statement", "do_statement", "switch_statement"}
STATEMENT = {"declaration", "expression_statement", "return_statement", "break_statement", "continue_statement",
             "goto_statement", "throw_statement", "co_return_statement"}


def source_parser():
    from tree_sitter import Language, Parser
    from tree_sitter_cpp import language

    return Parser(Language(language()))


def intersects(span, spans):
    return any(span[0] < end and start < span[1] for start, end in spans)


def contains(span, spans):
    return any(start <= span[0] and span[1] <= end for start, end in spans)


def merge(spans):
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        elif start < end:
            merged.append((start, end))
    return merged


def occupied(spans):
    return sum(end - start for start, end in spans) + max(0, len(spans) - 1)


def gaps(spans, lo, hi):
    for start, end in spans:
        if start >= hi:
            break
        if lo < start:
            yield lo, start
        lo = max(lo, end)
    if lo < hi:
        yield lo, hi


def extend_context(spans, boundaries, limits, budget, fragment_limit, unit_at):
    """Complete retained units, then grow their contiguous neighbors."""
    size = boundaries[-1]
    if unit_at is None:
        def unit_at(position):
            cut = bisect_right(boundaries, position)
            return boundaries[cut-1], boundaries[cut]

    def add(unit):
        nonlocal spans
        expanded = merge([*spans, unit])
        if expanded == spans or occupied(expanded) > budget:
            return False
        if fragment_limit is not None and len(expanded) > max(fragment_limit, len(spans)):
            return False
        spans = expanded
        return True

    for start, end in tuple(spans):
        for position in (start, end-1):
            add(unit_at(position))

    # Seed at most one complete unit in an uncovered band, up to the number
    # of bands in total. Additional budget grows these and the original spans.
    for lo, hi in limits:
        if len(spans) >= len(limits):
            break
        if lo >= hi or intersects((lo, hi), spans):
            continue
        center = (lo+hi)/2
        i = bisect_left(boundaries, center)
        units = {unit_at(boundaries[j]) for j in (i-1, i) if 0 <= j < len(boundaries)-1}
        for a, b in sorted(units, key=lambda p: (abs((p[0]+p[1])/2-center), p[0], p[1])):
            if lo <= a < b <= hi and add((a, b)):
                break

    while occupied(spans) < budget:
        changed = False
        for start, end in tuple(spans):
            if start > 0:
                changed |= add(unit_at(start-1))
            if end < size:
                changed |= add(unit_at(end))
        if not changed:
            break
    return spans


def uniform_fill(spans, boundaries, limits, *, budget=BUDGET, fragment_limit=None, contiguous=True, unit_at=None):
    spans = merge(spans)
    budget = min(budget, boundaries[-1])
    if occupied(spans) > budget:
        raise ValueError("selected code exceeds the token budget")
    if contiguous:
        spans = extend_context(spans, boundaries, limits, budget, fragment_limit, unit_at)
    else:
        # Reproduce the centered windows used to train the archived classifiers.
        for i, (left, right) in enumerate(limits):
            quota = (budget - occupied(spans)) // (len(limits) - i) - 1
            if quota <= 0:
                continue
            lo, hi = max(gaps(spans, left, right), key=lambda p: p[1]-p[0], default=(left, left))
            first, last = bisect_left(boundaries, lo), bisect_right(boundaries, hi) - 1
            windows = aligned_windows(boundaries, first, last, (lo+hi)/2, quota)
            best = max((candidate for candidate in windows if candidate[0][0] < candidate[0][1]),
                       key=lambda candidate: candidate[1], default=None)
            if best is not None:
                augmented = merge([*spans, best[0]])
                if fragment_limit is None or len(augmented) <= fragment_limit:
                    spans = augmented

    # Spend any residue next to an existing span, preferring a physical cut
    # within budget. With no spans this creates the first contiguous window.
    while (room := budget - occupied(spans)) > 0:
        left, right = max(gaps(spans, 0, boundaries[-1]), key=lambda p: p[1]-p[0])
        if left:
            cap = min(left+room, right)
            edge = boundaries[bisect_right(boundaries, cap)-1]
            span = (left, edge if edge > left else cap)
        else:
            cap = max(left, right-room)
            edge = boundaries[bisect_left(boundaries, cap)]
            span = (edge if edge < right else cap, right)
        spans = merge([*spans, span])
    if contiguous:
        # A one-BPE seam replaces one in-body SEP, even at full occupancy.
        seams = [(end, start) for (_, end), (start, _) in zip(spans, spans[1:]) if start-end == 1]
        spans = merge([*spans, *seams])
    return spans


def declarator_name(node):
    while node is not None and node.type != "identifier":
        if node.type == "reference_declarator" and node.named_child_count == 1:
            node = node.named_children[0]
        else:
            node = node.child_by_field_name("declarator")
    return node


def control_header(node):
    body = node.child_by_field_name("body") or node.child_by_field_name("consequence")
    if body is None or body.is_missing:
        return None
    if node.type == "do_statement":
        marker = next((part for part in node.children if part.type == "while"), None)
        if marker is not None and not node.has_error:
            return (marker.start_byte, node.end_byte), (body.start_byte, body.end_byte)
        return None
    prefix = [part for part in node.children if part.end_byte <= body.start_byte]
    condition = next((part for part in reversed(prefix) if part.type in {"condition_clause", ")"}), None)
    if condition is not None and not condition.is_missing and not any(part.has_error for part in prefix):
        return (node.start_byte, condition.end_byte), (body.start_byte, node.end_byte)
    return None


def related_statements(code, offsets, keep, base, parser):
    """Find syntax links from retained regions, or from all statements when base is None."""
    source = code.encode("utf-8")
    root = parser.parse(source).root_node
    if len(source) == len(code):
        starts, ends = zip(*(offsets[i] for i in keep))
    else:
        byte_at = list(accumulate((len(c.encode("utf-8")) for c in code), initial=0))
        starts, ends = zip(*((byte_at[offsets[i][0]], byte_at[offsets[i][1]]) for i in keep))

    def bpe(start, end):
        return bisect_right(ends, start), bisect_left(starts, end)

    base_bytes = [(starts[start], ends[end-1]) for start, end in base] if base is not None else None
    scopes = [(None, None)]
    definitions, anchors, named_sites = defaultdict(list), [], set()
    stack = [(root, 0, None)]
    while stack:
        node, scope, anchor = stack.pop()
        kind = node.type
        if kind in CONTROL or kind in {"compound_statement", "else_clause", "lambda_expression"}:
            header = control_header(node) if kind in CONTROL else None
            outer = scope
            scope = len(scopes)
            scopes.append((outer, header))
            anchor = None
            if header:
                span = bpe(*header[0])
                if base is not None and intersects(span, base):
                    anchor = (span, scope, None, {})
                    anchors.append(anchor)
        if kind in STATEMENT:
            anchor = None
            if not node.has_error:
                span = bpe(node.start_byte, node.end_byte)
                if span[0] < span[1] and (base is None or intersects(span, base)):
                    anchor = (span, scope, node, {})
                    anchors.append(anchor)
                if kind == "declaration":
                    names = [declarator_name(d) for d in node.children_by_field_name("declarator")]
                    for name in (d for d in names if d is not None):
                        named_sites.add(name.start_byte)
                        definitions[scope, name.text, "declaration"].append(span)
                elif kind == "expression_statement" and node.named_child_count:
                    expr = node.named_children[0]
                    lhs = expr.child_by_field_name("left") if expr.type == "assignment_expression" else None
                    if lhs is not None and lhs.type == "identifier":
                        definitions[scope, lhs.text, "assignment"].append(span)
        if kind == "identifier" and anchor is not None and node.start_byte not in named_sites:
            if base_bytes is None or intersects((node.start_byte, node.end_byte), base_bytes):
                span = bpe(node.start_byte, node.end_byte)
                if (base is None or contains(span, base)) and contains(span, [anchor[0]]):
                    anchor[3].setdefault(node.text, span[0])
        stack.extend((child, scope, anchor) for child in reversed(node.named_children))

    links = set()
    for anchor, scope, stmt, names in anchors:
        if base is not None and not contains(anchor, base):
            links.add(("completion", anchor, anchor))
        cursor = scope
        while cursor is not None:
            cursor, header = scopes[cursor]
            if header and contains(anchor, [bpe(*header[1])]):
                links.add(("condition", anchor, bpe(*header[0])))
        if base is not None and stmt is not None:
            for attr in ("prev_named_sibling", "next_named_sibling"):
                neighbor = getattr(stmt, attr)
                while neighbor is not None and neighbor.type == "comment":
                    neighbor = getattr(neighbor, attr)
                if neighbor is not None and neighbor.type in STATEMENT and not neighbor.has_error:
                    links.add(("neighbor", anchor, bpe(neighbor.start_byte, neighbor.end_byte)))
        for name, use in names.items():
            cursor, assignment = scope, None
            while cursor is not None:
                decls = definitions.get((cursor, name, "declaration"), ())
                writes = definitions.get((cursor, name, "assignment"), ())
                di = bisect_right(decls, (anchor[0], -1)) - 1
                wi = bisect_right(writes, (anchor[0], -1)) - 1
                if assignment is None and wi >= 0 and writes[wi][1] <= use:
                    assignment = writes[wi]
                if di >= 0 and decls[di][1] <= use:
                    decl = decls[di]
                    links.add(("declaration", anchor, decl))
                    if assignment is not None and assignment[0] < decl[1]:
                        assignment = None
                    break
                cursor = scopes[cursor][0]
            if assignment is not None:
                links.add(("assignment", anchor, assignment))

    return {link for link in links if link[2][0] < link[2][1] and
            (base is None or not contains(link[2], base))}, root.has_error


def prefer_context(base, links, *, budget=BUDGET):
    weights = Counter()
    for kind, _, span in links:
        weights[span] += RELATED_WEIGHTS[kind]
    spans = merge(base)
    size = occupied(spans)
    priced = [(occupied(merge([*spans, span]))-size, span) for span in weights]
    priced.sort(key=lambda pair: (-weights[pair[1]]/max(1, pair[0]), pair[0], pair[1]))
    for _, span in priced:
        if contains(span, spans):
            continue
        extended = merge([*spans, span])
        if occupied(extended) <= budget:
            spans = extended
    return spans


def input_layout(code, ids, offsets):
    return line_layout(code, ids, offsets) if len(ids) > BUDGET else (list(range(len(ids))), [0, len(ids)])


def base_windows(boundaries):
    windows, limits = choose_candidates(boundaries, alternatives=False, quotas=BASE_QUOTAS)
    return [choices[0] for choices in windows], limits


def link_retention(links, spans):
    linked = {kind: [0, 0, 0] for kind in RELATED_WEIGHTS}
    retained = merge(spans)
    for kind, anchor, context in links:
        context_kept = contains(context, retained)
        linked[kind][0] += 1
        linked[kind][1] += int(context_kept)
        linked[kind][2] += int(context_kept and contains(anchor, retained))
    return linked


def prepare(code, ids, offsets, tokenizer, *, method, parser=None, analyze=False):
    keep, boundaries = input_layout(code, ids, offsets)
    links, parse_error = set(), None
    if len(keep) <= BUDGET:
        base = spans = [(0, len(keep))]
    else:
        base, limits = base_windows(boundaries)
        if analyze or method == "context_aware":
            links, parse_error = related_statements(code, offsets, keep, base, parser)
        if method == "uniform4":
            windows, _ = choose_candidates(boundaries, alternatives=False)
            spans = [choices[0] for choices in windows]
        else:
            selected = prefer_context(base, links) if method == "context_aware" else base
            spans = uniform_fill(selected, boundaries, limits)
    uniform = torch.full((REGIONS, 4), -1, dtype=torch.int32)
    for i, (start, end) in enumerate(base):
        if start < end:
            uniform[i] = torch.tensor((start, end, keep[start], keep[end-1]+1))
    example = fixed_example(ids, keep, spans, tokenizer, uniform, exact=method != "uniform4")
    if analyze:
        example.update(links=link_retention(links, spans), parse_error=parse_error)
    return example


def fixed_example(ids, keep, spans, tokenizer, uniform, *, exact=True):
    fixed, length = pack_fixed(ids, keep, spans, tokenizer)
    if exact and len(keep) > BUDGET and length != MAX_LENGTH:
        raise ValueError(f"input failed to fill the {MAX_LENGTH}-token budget")
    return dict(fixed=fixed, length=length, raw=len(ids), uniform=uniform, bank=None, compacted_bpe=len(keep))


def visible_source(example):
    if "original_runs" in example:
        return example["original_runs"]
    positions = example["fixed"][1, :example["length"]].tolist()
    return merge([(i, i+1) for i in positions if i >= 0])


def write_prediction_rows(path, rows):
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    temporary.replace(path)


def describe(data, scores, path, *, paired, include=()):
    rows, inputs = [], []
    if scores is None:
        scores = [None]*len(data)
    for row, ex, score in zip(data.metadata, data.examples, scores, strict=True):
        # Exact source runs include the holes left by removed indentation.
        runs = visible_source(ex)
        inputs.append(ex["fixed"][0, :ex["length"]].tolist())
        base = ex["uniform"].tolist()
        classify = {} if score is None else dict(probability=score, prediction=int(score > .5))
        rows.append(dict(**row, **classify, input_length=ex["length"],
                         source_bpe=sum(b-a for a,b in runs), source_runs=runs, base_windows=[part for part in base if part[0] >= 0],
                         compacted_bpe=ex["compacted_bpe"]))
        if paired and "links" in ex:
            rows[-1].update(parse_error=ex["parse_error"], links=ex["links"])
        rows[-1].update((key, ex[key]) for key in include)
    if paired and [row["target"] for row in rows] != [1, 0] * (len(rows)//2):
        raise ValueError("test pair order differs from the original paired release")
    pairs = [(max(a["original_bpe"], b["original_bpe"]), x == y) for a, b, x, y in
             zip(rows[::2], rows[1::2], inputs[::2], inputs[1::2], strict=True)] if paired else []
    for i, same in enumerate(same for _, same in pairs):
        rows[2*i]["identical_pair_input"] = rows[2*i+1]["identical_pair_input"] = same
    if path is not None:
        write_prediction_rows(path, rows)

    summary = {}
    for name, (lo, hi) in GROUPS.items():
        group = [row for row in rows if lo <= row["original_bpe"] <= hi]
        summary[name] = dict(functions=len(group), input_tokens=sum(row["input_length"] for row in group),
                             source_tokens=sum(row["source_bpe"] for row in group),
                             full_budget=sum(row["input_length"] == MAX_LENGTH for row in group))
        if paired:
            matches = [same for length, same in pairs if lo <= length <= hi]
            summary[name].update(pairs=len(matches), identical_inputs=sum(matches),
                                 identical_fraction=sum(matches)/len(matches) if matches else None)
            if "links" in rows[0]:
                counts = {kind: [sum(row["links"][kind][i] for row in group) for i in range(3)]
                          for kind in RELATED_WEIGHTS}
                summary[name].update(syntax_analyzed=sum(row["parse_error"] is not None for row in group),
                                     syntax_errors=sum(row["parse_error"] is True for row in group),
                                     relations={k: dict(available=a, context_kept=b, both_kept=c)
                                                for k, (a, b, c) in counts.items()})
    return summary
