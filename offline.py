"""Align release pairs only after independently constructing their model inputs."""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import re
from time import perf_counter

from compression import METHODS, model_ids, select
from dataset import BUDGET, DATA, SPLITS, tokenizer, write
from rgcl.data import prime_source_rows

# Lexical edit proxy: identifiers/numbers and non-whitespace symbols. The
# coordinates allow a split identifier to count only if all its pieces survive.
LEX = re.compile(r"\w+|[^\w\s]")


def edit_spans(a, b):
    lines = [s.splitlines(keepends=True) for s in (a, b)]
    starts = []
    for side in lines:
        p = [0]
        for line in side:
            p.append(p[-1] + len(line))
        starts.append(p)
    spans = [[], []]
    for tag, i, j, k, l in SequenceMatcher(None, *lines, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        tokens = [list(LEX.finditer(src, pos[x], pos[y])) for src, pos, x, y in
                  ((a, starts[0], i, j), (b, starts[1], k, l))]
        words = [[m.group() for m in ts] for ts in tokens]
        for op, ai, aj, bi, bj in SequenceMatcher(None, *words, autojunk=False).get_opcodes():
            if op != "equal":
                spans[0].extend(m.span() for m in tokens[0][ai:aj])
                spans[1].extend(m.span() for m in tokens[1][bi:bj])
    return spans


def bpe_ranges(offsets, spans):
    ends = [e for _, e in offsets]
    starts = [s for s, _ in offsets]
    out = [(bisect_right(ends, a), bisect_left(starts, b)) for a, b in spans]
    if any(a >= b for a, b in out):
        raise ValueError("unmapped edit")
    return out


def audit():
    tok = tokenizer()
    for split in SPLITS:
        begin = perf_counter()
        rows = prime_source_rows(DATA.parent, split)
        counts = {m: defaultdict(Counter) for m in METHODS}
        for pair in range(0, len(rows), 2):
            left, right = rows[pair:pair + 2]
            if any(left[k] != right[k] for k in ("project", "commit_id")):
                raise ValueError(f"release pair disagreement: {split}/{pair}")
            codes = [left["func"], right["func"]]
            enc = tok(codes, add_special_tokens=False, return_attention_mask=False,
                      return_offsets_mapping=True, verbose=False)
            ids, offsets = enc["input_ids"], enc["offset_mapping"]
            views = {m: [select(m, *args) for args in zip(codes, ids, offsets)] for m in METHODS}
            edits = [bpe_ranges(o, spans) for o, spans in zip(offsets, edit_spans(*codes))]
            group = "long" if max(map(len, ids)) > BUDGET else "short"
            head_different = None
            for m, sides in views.items():
                inputs = [model_ids(x, s, tok) for x, s in zip(ids, sides)]
                different = inputs[0] != inputs[1]
                if m == "head":
                    head_different = different
                c = Counter(pairs=1, different=int(different), recovered=int(different and not head_different),
                            regressed=int(head_different and not different),
                            source_bpe=sum(map(len, ids)), input_including_special=sum(map(len, inputs)))
                evidence = 0
                for key, source, positions, changes in zip(("before", "after"), ids, sides, edits):
                    kept = {i for i in positions if i is not None}
                    ordered = [i for i in positions if i is not None]
                    if len(kept) != len(ordered) or sorted(kept) != ordered or len(positions) > BUDGET:
                        raise ValueError("invalid selected positions")
                    c["source_bpe_kept"] += len(kept)
                    c["suffix_bpe_kept"] += sum(i >= BUDGET for i in kept)
                    c["suffix_bpe"] += max(0, len(source) - BUDGET)
                    c[key + "_edits"] += len(changes)
                    hits = 0
                    for a, b in changes:
                        hit = all(i in kept for i in range(a, b))
                        hits += hit
                        if b > BUDGET:
                            c["prefix_missed_edits"] += 1
                            c["recovered_edits"] += hit
                    c[key + "_kept"] += hits
                    evidence += hits
                c["pairs_with_edit"] += bool(edits[0] or edits[1])
                c["pairs_with_visible_edit"] += bool(evidence)
                c["different_without_visible_edit"] += bool(different and not evidence)
                for name in ("all", group):
                    counts[m][name].update(c)
        write(f"offline_{split}.json", dict(seconds=perf_counter()-begin, methods=counts))
        print(split, {m: dict(c["long"]) for m, c in counts.items()}, flush=True)


if __name__ == "__main__":
    audit()
