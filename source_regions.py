"""C/C++ source simplification and reversible UniXcoder coordinates."""
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass
from itertools import accumulate
import re
from time import perf_counter

import numpy as np

from compaction import line_boundaries
from context_selection import CONTROL, STATEMENT, control_header, merge
from dependence import EDGE_TYPES, physical_lines, source_position

SPLICE_LINE = re.compile(r"(?:\\|\?\?/)[ \t\v\f]*(?:\r\n|\n|\r)")
PROTECTED_LINE = re.compile(r"(?:\A|(?<=[\r\n]))[^\r\n]*(?:#|%:|\?\?=)[^\r\n]*")
HSPACE = re.compile(r"[ \t]+")
QUOTES = re.compile("[\"']")
FIRST_TOKEN = re.compile(r"[\w$]+|\S")
LITERALS = {"string_literal", "raw_string_literal", "char_literal", "system_lib_string", "number_literal"}
PHYSICAL_LINE = re.compile(r"(?:\A|(?<=[\r\n]))[^\r\n]*(?:\r\n|\r|\n)")
LINE_SENSITIVE = re.compile(r"\b(?:__LINE__|__COLUMN__|__builtin_LINE|__builtin_COLUMN|source_location)\b")


@dataclass(slots=True)
class Region:
    start: int
    end: int
    source_start: int
    source_end: int
    kind: str

    @property
    def span(self):
        return self.start, self.end


@dataclass(slots=True)
class Source:
    code: str
    ids: np.ndarray
    raw_offsets: np.ndarray
    offsets: np.ndarray
    source_offsets: np.ndarray
    origin: np.ndarray
    synthetic: np.ndarray
    lines: list
    regions: list
    boundaries: list
    boundary_budget: int
    simplified_by: str
    parse_error: bool

    @property
    def raw_bpe(self):
        return len(self.raw_offsets)


def standalone_null(node):
    """Only bare compound lists under a function body can lose a null statement."""
    block = node.parent
    if block is None or block.type != "compound_statement":
        return False
    while block.parent is not None and block.parent.type == "compound_statement":
        block = block.parent
    owner = block.parent
    return owner is not None and owner.type == "function_definition" and owner.child_by_field_name("body") == block


def analyze(code, parser, *, null_statements=None):
    raw = code.encode("utf-8")
    tree = parser.parse(raw)
    if len(raw) == len(code):
        def char(position):
            return position
    else:
        positions = list(accumulate((len(c.encode("utf-8")) for c in code), initial=0))

        def char(position):
            return bisect_left(positions, position)

    comments, literals, statements = [], [], []
    pending = [tree.root_node]
    while pending:
        node = pending.pop()
        kind = node.type
        lo, hi = char(node.start_byte), char(node.end_byte)
        if kind == "comment":
            comments.append((lo, hi))
            continue
        if kind in LITERALS and not node.has_error:
            literals.append((lo, hi))
            continue
        if kind in CONTROL:
            header = control_header(node)
            if header is not None:
                statements.append((*map(char, header[0]), "header"))
        elif kind in STATEMENT and not node.has_error:
            statements.append((lo, hi, "statement"))
            if (null_statements is not None and kind == "expression_statement" and code[lo:hi] == ";"
                    and standalone_null(node)):
                null_statements.append(lo)
        elif kind == "function_definition":
            body = node.child_by_field_name("body")
            if body is not None:
                statements.append((lo, char(body.start_byte), "header"))
        pending.extend(reversed(node.named_children))
    return comments, literals, statements, tree.root_node.has_error


def simplify(code, comments, literals, *, drop_blank=False):
    """Protect literal/directive characters and preserve lexical token boundaries."""
    if SPLICE_LINE.search(code):
        return code, np.arange(len(code), dtype=np.int32), np.zeros(len(code), dtype=bool), "line_splice"
    protected = np.zeros(len(code), dtype=bool)
    for lo, hi in [*literals, *(m.span() for m in PROTECTED_LINE.finditer(code))]:
        protected[lo:hi] = True
    covered = protected.copy()
    for lo, hi in comments:
        covered[lo:hi] = True
    if any(not covered[m.start()] for m in QUOTES.finditer(code)):
        return code, np.arange(len(code), dtype=np.int32), np.zeros(len(code), dtype=bool), "ambiguous_literal"
    retain, synthetic = np.ones(len(code), dtype=bool), np.zeros(len(code), dtype=bool)
    for match in HSPACE.finditer(code):
        lo, hi = match.span()
        if covered[lo:hi].any():
            continue
        at_edge = lo == 0 or code[lo-1] in "\r\n" or hi == len(code) or code[hi] in "\r\n"
        retain[lo if at_edge else lo+1:hi] = False
    for lo, hi in comments:
        if protected[lo:hi].any():
            continue
        retain[lo:hi] = False
        retain[lo] = synthetic[lo] = True
        for newline in re.finditer(r"[\r\n]", code[lo:hi]):
            retain[lo+newline.start()] = True
    index = np.flatnonzero(retain).astype(np.int32, copy=False)
    substituted = synthetic[index]
    simple = "".join(" " if flag else code[i] for i, flag in zip(index.tolist(), substituted.tolist(), strict=True))
    if drop_blank and not (PROTECTED_LINE.search(code) or LINE_SENSITIVE.search(code)):
        kept = np.ones(len(simple), dtype=bool)
        for match in PHYSICAL_LINE.finditer(simple):
            lo, hi = match.span()
            if not match.group().strip(" \t\v\f\r\n") and not protected[index[lo:hi]].any():
                kept[lo:hi] = False
        if not kept.all():
            simple = "".join(c for c, keep in zip(simple, kept.tolist(), strict=True) if keep)
            index, substituted = index[kept], substituted[kept]
    if SPLICE_LINE.search(simple):
        return code, np.arange(len(code), dtype=np.int32), np.zeros(len(code), dtype=bool), "possible_new_splice"
    return simple, index, substituted, "lexical"


def regions_for(statements, offsets, source_offsets, boundaries, budget):
    starts, ends = source_offsets.T
    n = len(offsets)
    regions = []

    def append(lo, hi, kind, raw=None):
        if hi <= lo:
            return
        # Break only oversized spans and unparsed gaps at the existing physical-line cuts.
        pieces = [lo, hi]
        if hi-lo > budget or kind == "line":
            pieces = [lo, *boundaries[bisect_right(boundaries, lo):bisect_left(boundaries, hi)], hi]
            kind = "line"
            raw = None
        regions.extend(Region(a, b, *(raw or (int(starts[a]), int(ends[b-1]))), kind)
                       for a, b in zip(pieces, pieces[1:]) if a < b)

    cursor = 0
    for start, end, kind in sorted(set(statements), key=lambda s: (s[0], -s[1])):
        lo = int(ends.searchsorted(start, side="right"))
        hi = int(starts.searchsorted(end))
        if hi <= cursor or hi <= lo:
            continue
        append(cursor, lo, "line")
        append(max(cursor, lo), hi, kind if cursor <= lo else "line", (start, end))
        cursor = hi
    append(cursor, n, "line")
    return regions


def tokenize_sources(codes, tokenizer, parser, budget):
    """Tokenize original and simplified text in batches; annotations are source-only."""
    sources = []
    started = perf_counter()
    timings = Counter()
    for begin in range(0, len(codes), 32):
        batch = codes[begin:begin+32]
        before = perf_counter()
        parsed = [analyze(code, parser) for code in batch]
        timings["syntax_seconds"] += perf_counter()-before
        before = perf_counter()
        compacted = [simplify(code, comments, literals) for code, (comments, literals, _, _) in zip(batch, parsed, strict=True)]
        timings["simplification_seconds"] += perf_counter()-before
        before = perf_counter()
        encoded = tokenizer([*batch, *(x[0] for x in compacted)], add_special_tokens=False,
                            return_offsets_mapping=True, return_attention_mask=False, verbose=False)
        timings["tokenization_seconds"] += perf_counter()-before
        before = perf_counter()
        for i, (code, (simple, origin, synthetic, reason), (_, _, statements, error)) in enumerate(zip(batch, compacted, parsed, strict=True)):
            ids = np.asarray(encoded.input_ids[len(batch)+i], dtype=np.int32)
            raw_offsets = np.asarray(encoded.offset_mapping[i], dtype=np.int32).reshape(-1, 2)
            offsets = np.asarray(encoded.offset_mapping[len(batch)+i], dtype=np.int32).reshape(-1, 2)
            if not len(ids):
                # A comment-only source can have no tokens after lexical simplification.
                source_offsets = offsets.copy()
                boundaries = [0]
                regions = []
            else:
                if (offsets[:, 0] >= offsets[:, 1]).any():
                    raise ValueError("Missing source offsets in UniXcoder BPE")
                source_offsets = np.column_stack((origin[offsets[:, 0]], origin[offsets[:, 1]-1]+1))
                boundaries = line_boundaries(simple, offsets, range(len(ids)), budget)
                regions = regions_for(statements, offsets, source_offsets, boundaries, budget)
            lines = list(accumulate(map(len, code.splitlines(keepends=True)), initial=0))
            sources.append(Source(code, ids, raw_offsets, offsets, source_offsets, origin, synthetic, lines, regions,
                                  boundaries, budget, reason, error))
        timings["region_mapping_seconds"] += perf_counter()-before
    timings["seconds"] = perf_counter()-started
    return sources, dict(timings)


def original_runs(source, spans):
    visible = np.zeros(len(source.code), dtype=bool)
    for lo, hi in spans:
        start, end = int(source.offsets[lo, 0]), int(source.offsets[hi-1, 1])
        visible[source.origin[start:end][~source.synthetic[start:end]]] = True
    missing = np.concatenate(([0], np.cumsum(~visible, dtype=np.int32)))
    begins, ends = source.raw_offsets.T
    tokens = np.flatnonzero((ends > begins) & (missing[begins] == missing[ends]))
    return merge([(int(i), int(i)+1) for i in tokens])


def map_dependencies(source, graph):
    """Project native endpoints onto contiguous source regions using Joern coordinates."""
    stats = Counter(joern_nodes=graph["joern_nodes"], endpoint_nodes=graph["endpoint_nodes"],
                    parse_problems=graph["counts"].get("parse_problems", 0),
                    preprocessor_statements=graph["counts"].get("preprocessor_statements", 0))
    mapped, points = {}, {}
    eligible = [i for i,r in enumerate(source.regions) if r.kind != "line"]
    begins = [source.regions[i].source_start for i in eligible]
    ends = [source.regions[i].source_end for i in eligible]
    lines = physical_lines(source.code)
    endpoint_ids = {node for _, u, v in graph["edges"] for node in (u, v)}
    structural = {"METHOD", "METHOD_RETURN", "METHOD_PARAMETER_OUT", "BLOCK"}
    for node, (kind, line, col, code, name, _) in graph["nodes"].items():
        if node in endpoint_ids and not 0 < line < len(lines):
            stats["missing_line"] += 1
        if kind in structural:
            stats["non_statement_node"] += node in endpoint_ids
            continue
        first = FIRST_TOKEN.search(code)
        if not first:
            continue
        tokens = {first.group()}
        if name and kind in {"LOCAL", "METHOD_PARAMETER_IN", "FIELD_IDENTIFIER"}:
            tokens.add(name)
        point = source_position(source.code, lines, line, col)
        if point is None and 0 < line < len(lines) and col == 0:
            stats["missing_column"] += node in endpoint_ids
            lo, hi = lines[line-1:line+1]
            exact = [lo+m.start() for m in FIRST_TOKEN.finditer(source.code[lo:hi]) if m.group() in tokens]
            if len(exact) == 1:
                point = exact[0]
        if point is not None:
            points[node] = point
            observed = FIRST_TOKEN.match(source.code, point)
            left, right = bisect_right(ends, point), bisect_right(begins, point)
            if observed and observed.group() in tokens and right-left == 1:
                mapped[node] = eligible[left]
                stats["directly_mapped_nodes"] += node in endpoint_ids

    for node in sorted(endpoint_ids-mapped.keys()):
        kind, _, _, _, _, parent = graph["nodes"][node]
        if kind in structural:
            continue
        while parent in graph["nodes"]:
            parent_kind, _, _, _, _, next_parent = graph["nodes"][parent]
            if parent_kind in structural or parent_kind == "CONTROL_STRUCTURE":
                break
            if parent in mapped:
                region = source.regions[mapped[parent]]
                point = points.get(node)
                if point is None or region.source_start <= point < region.source_end:
                    mapped[node] = mapped[parent]
                    stats["ast_owned_nodes"] += 1
                break
            parent = next_parent
    stats["unmapped_coordinate"] = len(endpoint_ids-mapped.keys())
    edge_sets = {kind: set() for kind in EDGE_TYPES}
    for kind, left, right in graph["edges"]:
        stats[f"{kind}_native"] += 1
        if left not in mapped or right not in mapped:
            stats[f"{kind}_unmapped"] += 1
        elif mapped[left] == mapped[right]:
            stats[f"{kind}_within_region"] += 1
        else:
            stats[f"{kind}_mapped_native"] += 1
            edge_sets[kind].add((mapped[left], mapped[right]))
    stats["mapped_nodes"] = sum(n in mapped for n in endpoint_ids)
    for kind, edges in edge_sets.items():
        stats[f"{kind}_region_edges"] = len(edges)
        stats[f"{kind}_missing_graph"] = int(not stats[f"{kind}_native"])
    shape = graph["structure"]
    if graph["status"] == "ok":
        if not stats["CDG_native"]:
            stats["CDG_empty_with_control"] = int(bool(shape.get("CONTROL_STRUCTURE")))
            stats["CDG_empty_without_control"] = int(not shape.get("CONTROL_STRUCTURE"))
        if not stats["REACHING_DEF_native"]:
            defs = bool(shape.get("LOCAL") or shape.get("METHOD_PARAMETER_IN"))
            stats["REACHING_DEF_empty_with_local_or_parameter"] = int(defs)
            stats["REACHING_DEF_empty_without_local_or_parameter"] = int(not defs)
    return {kind: sorted(edges) for kind, edges in edge_sets.items()}, dict(stats)
