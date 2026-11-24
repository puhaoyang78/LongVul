"""Compact source BPEs and construct line-aligned windows in four regions."""
from bisect import bisect_left, bisect_right
from functools import partial
from itertools import chain
import re

import numpy as np
import torch
from torch.utils.data import Dataset

from window_inputs import DATA
from rgcl.config import MAX_LENGTH
from rgcl.data import prime_source_rows

BUDGET = MAX_LENGTH - 2
REGIONS = 4
MAX_CANDIDATES = 8
BODY_BUDGET = BUDGET - (REGIONS - 1)
QUOTAS = tuple(BODY_BUDGET * (k + 1) // REGIONS - BODY_BUDGET * k // REGIONS for k in range(REGIONS))
WIDTH = max(QUOTAS)
INDENT = re.compile(r"^[ \t]+", re.MULTILINE)
SPLICE = re.compile(r"\\[ \t]*\r?\n")


def line_layout(code, ids, offsets):
    # Uniform-4 from e094c1e. Keep a separator per line, and keep indentation
    # verbatim in functions containing raw string literals or line splices.
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
    return keep, line_boundaries(code, offsets, keep, BUDGET)


def line_boundaries(code, offsets, keep, budget):
    boundaries = [0]
    unit_limit = max(8, (budget - (REGIONS - 1)) // 8)
    last_space = 0
    for j, i in enumerate(keep):
        s, e = offsets[i]
        piece = code[s:e]
        if piece[:1].isspace():
            last_space = j
        if j + 1 - boundaries[-1] > unit_limit:
            cut = last_space if last_space - boundaries[-1] >= unit_limit // 2 else j
            boundaries.append(cut)
        if "\n" in piece:
            boundaries.append(j + 1)
    if boundaries[-1] != len(keep):
        boundaries.append(len(keep))
    return boundaries


def aligned_windows(boundaries, lo, hi, center, quota):
    for start in boundaries[lo:hi]:
        end = boundaries[bisect_right(boundaries, start + quota, lo, hi + 1) - 1]
        overlap = min(end, center + quota / 2) - max(start, center - quota / 2)
        yield (start, end), (overlap, -abs((start + end) / 2 - center))


def region_edges(boundaries):
    n = boundaries[-1]
    edges = [0]
    for k in range(1, REGIONS):
        p = bisect_left(boundaries, n * k / REGIONS)
        edges.append(min((p - 1, p), key=lambda t: abs(boundaries[t] - n * k / REGIONS)))
    edges.append(len(boundaries) - 1)
    return edges


def choose_candidates(boundaries, *, alternatives, quotas=QUOTAS):
    edges = region_edges(boundaries)
    windows, limits = [], []
    for (lo, hi), quota in zip(zip(edges, edges[1:]), quotas, strict=True):
        left, right = boundaries[lo], boundaries[hi]
        center = (left + right) / 2
        legal = []
        best, best_value = None, None
        for span, value in aligned_windows(boundaries, lo, hi, center, quota):
            if alternatives:
                legal.append(span)
            if best_value is None or value > best_value:
                best, best_value = span, value
        if best is None or best[0] >= best[1]:
            raise ValueError(f"no nonempty window in region {left}:{right}")
        chosen = [best]  # slot 0 stays the Uniform-4 window
        if alternatives:
            starts = [start for start, _ in legal]
            last = max(left, right - quota)
            for k in range(MAX_CANDIDATES - 1):
                target = left + (last - left) * k / (MAX_CANDIDATES - 2)
                p = bisect_left(starts, target)
                near = min((max(0, p - 1), min(len(starts) - 1, p)), key=lambda i: abs(starts[i] - target))
                candidate = legal[near]
                if candidate not in chosen:
                    chosen.append(candidate)
        windows.append(chosen)
        limits.append((left, right))
    return windows, limits


def pack_fixed(ids, keep, spans, tokenizer, *, max_length=MAX_LENGTH):
    fixed_ids = [tokenizer.cls_token_id]
    fixed_positions, fixed_parts = [-1], [0]
    previous_end = None
    for part, (start, end) in enumerate(spans, 1):
        if start == end:
            continue
        positions = keep[start:end]
        if previous_end is not None and previous_end != start:
            fixed_ids.append(tokenizer.sep_token_id)
            fixed_positions.append(-1)
            fixed_parts.append(0)
        fixed_ids.extend(ids[pos] for pos in positions)
        fixed_positions.extend(positions)
        fixed_parts.extend([part] * len(positions))
        previous_end = end
    fixed_ids.append(tokenizer.sep_token_id)
    fixed_positions.append(-1)
    fixed_parts.append(0)
    length = len(fixed_ids)
    if length > max_length:
        raise ValueError(f"packed input exceeds the {max_length}-position budget")
    pad = max_length - length
    fixed = np.array([fixed_ids + [tokenizer.pad_token_id]*pad,
                      fixed_positions + [-1]*pad, fixed_parts + [0]*pad], dtype=np.int32)
    return torch.from_numpy(fixed), length


def prepare_function(code, ids, offsets, tokenizer, *, alternatives):
    raw = len(ids)
    if raw <= BUDGET:
        keep, boundaries = list(range(raw)), [0, raw]
    else:
        keep, boundaries = line_layout(code, ids, offsets)
    if len(keep) <= BUDGET:
        windows, limits = [[(0, len(keep))]], [(0, len(keep))]
    else:
        windows, limits = choose_candidates(boundaries, alternatives=alternatives)

    uniform = np.full((REGIONS, 4), -1, dtype=np.int32)
    specs = np.full((REGIONS, MAX_CANDIDATES, 4), -1, dtype=np.int32)
    bank = originals = None
    if alternatives and len(keep) > BUDGET:
        shape = (REGIONS, MAX_CANDIDATES, WIDTH)
        bank = np.full(shape, tokenizer.pad_token_id, dtype=np.int32)
        originals = np.full(shape, -1, dtype=np.int32)
    for k, candidates in enumerate(windows):
        for j, (start, end) in enumerate(candidates):
            positions = keep[start:end]
            if not positions:
                continue
            selected_ids = [ids[pos] for pos in positions]
            specs[k, j] = (start, end, positions[0], positions[-1] + 1)
            if bank is not None:
                bank[k, j, :len(positions)] = selected_ids
                originals[k, j, :len(positions)] = positions
            if j == 0:
                uniform[k] = specs[k, j]
    fixed, length = pack_fixed(ids, keep, [candidates[0] for candidates in windows], tokenizer)
    return dict(fixed=fixed, length=length, raw=raw, uniform=torch.from_numpy(uniform),
                bank=None if bank is None else (torch.from_numpy(bank), torch.from_numpy(originals),
                                                 torch.from_numpy(specs)),
                limits=limits, candidate_counts=(specs[..., 0] >= 0).sum(1).tolist())


def tokenized_batches(rows, tokenizer):
    for start in range(0, len(rows), 32):
        batch = rows[start:start+32]
        texts = [row["func"] for row in batch]
        encoding = tokenizer(texts, add_special_tokens=False, return_offsets_mapping=True,
                             return_attention_mask=False, verbose=False)
        yield batch, texts, encoding


class PackedData(Dataset):
    def __init__(self, rows, examples):
        self.targets = torch.tensor([r["target"] for r in rows])
        self.metadata, self.examples = [], []
        for row, example in zip(rows, examples, strict=True):
            example["label"] = row["target"]
            self.examples.append(example)
            self.metadata.append(dict(idx=row["idx"], sample_key=row["sample_key"],
                                      target=row["target"], original_bpe=example["raw"]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class CompactData(PackedData):
    def __init__(self, split, tokenizer, *, alternatives=False, prepare=None, prepare_pair=None):
        rows = prime_source_rows(DATA, split)
        if prepare is None:
            prepare = partial(prepare_function, alternatives=alternatives)

        def examples():
            for _, texts, encoding in tokenized_batches(rows, tokenizer):
                sources = zip(texts, encoding.input_ids, encoding.offset_mapping, strict=True)
                if prepare_pair is None:
                    yield from (prepare(*source, tokenizer) for source in sources)
                else:
                    yield from chain.from_iterable(prepare_pair(a, b, tokenizer)
                                                   for a, b in zip(sources, sources, strict=True))

        super().__init__(rows, examples())


def collate(examples):
    selectable = [(i, ex["bank"]) for i, ex in enumerate(examples) if ex["bank"] is not None]
    batch = dict(fixed=torch.stack([ex["fixed"] for ex in examples]),
                 raw=torch.tensor([ex["raw"] for ex in examples]),
                 lengths=torch.tensor([ex["length"] for ex in examples]),
                 uniform=torch.stack([ex["uniform"] for ex in examples]),
                 labels=torch.tensor([ex["label"] for ex in examples]))
    if selectable:
        indices, bank = zip(*selectable)
        batch["indices"] = torch.tensor(indices)
        batch["bank"] = tuple(torch.stack(xs) for xs in zip(*bank))
    return batch
