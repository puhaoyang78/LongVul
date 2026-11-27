"""PrimeVul inputs and per-function coverage for the two window baselines."""
import csv
import gzip
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / "data"
BASE = ROOT.parents[2] / "resource/codebert-base"
UPSTREAM = ROOT / "vendor"
sys.path.insert(0, str(ROOT.parent / "RGCL"))
from rgcl.data import prime_source_rows
from rgcl.config import SEEDS
from rgcl.metrics import metrics

sys.path.insert(0, str(UPSTREAM / "SEA"))

SCOUT_LENGTH = 512
SCOUT_OVERLAP = 256
SEA_WIDTH, SEA_STEP = 256, 128
# get_args() sets code_length=window_size+20; data_flow_length defaults to 16.
SEA_LENGTH = SEA_WIDTH + 20 + 16
SEA_TRAIN_WINDOWS = 6
GROUPS = {
    "overall": (0, float("inf")), "long": (511, float("inf")), "short": (0, 510),
    "511-1020": (511, 1020), "1021-2040": (1021, 2040), "2041+": (2041, float("inf")),
}
COUNTERS = ("observations", "covered_bpe_total", "fully_covered_observations",
            "plm_windows", "plm_batch_calls")


def seed_path(path, seed):
    return path if seed == SEEDS[0] else path / f"seed{seed}"


def sea_input(tokenizer, ids):
    from two_stage_utils import sliding_window

    n = len(ids)
    head = (0, min(n, SEA_LENGTH - 2))
    windows = [(start, start + len(piece)) for start, piece in sliding_window(range(n), SEA_WIDTH, SEA_STEP)]
    spans = [head, *windows] if len(windows) > 1 else [head]
    input_ids = torch.full((len(spans), SEA_LENGTH), tokenizer.pad_token_id, dtype=torch.int32)
    for i, (start, end) in enumerate(spans):
        body = ids[start:end]
        input_ids[i, :len(body) + 2] = torch.tensor([tokenizer.cls_token_id, *body, tokenizer.sep_token_id])
    return input_ids, spans


def scout_input(tokenizer, text):
    encoded = tokenizer(text, return_tensors="pt", max_length=SCOUT_LENGTH, truncation=True,
                        stride=SCOUT_OVERLAP, return_overflowing_tokens=True, padding="max_length")
    step = SCOUT_LENGTH - 2 - SCOUT_OVERLAP
    spans = [(i * step, i * step + int(size) - 2)
             for i, size in enumerate(encoded.attention_mask.sum(dim=1))]
    return encoded.input_ids.int(), spans


def covered_length(spans):
    covered = right = 0
    for start, end in sorted(spans):
        covered += max(0, end - max(start, right))
        right = max(right, end)
    return covered


class WindowData:
    def __init__(self, method, split, tokenizer):
        self.method = method
        self.split = split
        self.examples, self.metadata = [], []
        rows = prime_source_rows(DATA, split)
        self.targets = torch.tensor([r["target"] for r in rows])
        for start in range(0, len(rows), 48):
            batch = rows[start:start + 48]
            texts = [r["func"] for r in batch]
            if method == "sea":
                bodies = tokenizer(texts, add_special_tokens=False, return_attention_mask=False,
                                   verbose=False)["input_ids"]
                examples = (sea_input(tokenizer, ids) for ids in bodies)
            else:
                examples = (scout_input(tokenizer, text) for text in texts)
            for row, (ids, spans) in zip(batch, examples, strict=True):
                original_bpe = max(end for _, end in spans)
                if not original_bpe or covered_length(spans) != original_bpe:
                    raise ValueError(f"{method}/{split}/{row['idx']}: invalid source coverage")
                if method == "sea" and int(ids.ne(tokenizer.pad_token_id).sum()) != sum(b-a+2 for a,b in spans):
                    raise ValueError(f"{split}/{row['idx']}: reserved PAD inside SEA source")
                self.metadata.append(dict(idx=row["idx"], sample_key=row["sample_key"], target=row["target"],
                                          original_bpe=original_bpe, available_windows=len(spans)))
                self.examples.append((ids, spans))

    def __len__(self):
        return len(self.metadata)

    def empty_counters(self):
        return np.zeros((len(self), len(COUNTERS)), dtype=np.int64)

    def batch(self, indices, limit, counters, *, training=False):
        inputs, lengths, counts = [], [], []
        offset = 0
        for index in indices:
            ids, spans = self.examples[index]
            selected = list(range(len(spans)))
            if self.method == "sea":
                if training and len(spans) > SEA_TRAIN_WINDOWS:
                    selected = [random.choice(selected) for _ in range(SEA_TRAIN_WINDOWS)]
                if not training:
                    # Author's retrieval predict() includes the prefix view twice.
                    selected.insert(0, 0)
            spans_used = [spans[i] for i in selected]
            inputs.append(ids[selected])
            lengths.extend(end-start+2 for start,end in spans_used)
            count = len(selected)
            counts.append(count)
            covered = covered_length(spans_used)
            calls = (offset + count - 1) // limit - offset // limit + 1
            counters[index] += (1, covered, int(covered == self.metadata[index]["original_bpe"]), count, calls)
            offset += count
        return torch.cat(inputs), torch.tensor(lengths), counts


def grouped_metrics(data, scores, groups=GROUPS):
    targets = data.targets.tolist()
    lengths = np.array([row["original_bpe"] for row in data.metadata])
    result = {}
    for name, (lo, hi) in groups.items():
        indices = np.flatnonzero((lo <= lengths) & (lengths <= hi))
        result[name] = metrics([targets[i] for i in indices], [scores[i] for i in indices])
    return result


def save_rows(path, data, counters, scores=None):
    columns = [*data.metadata[0], *COUNTERS]
    if scores is not None:
        columns += ["probability", "prediction"]
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for index, row in enumerate(data.metadata):
            value = dict(row, **dict(zip(COUNTERS, counters[index].tolist(), strict=True)))
            if scores is not None:
                value.update(probability=scores[index], prediction=int(scores[index] > .5))
            writer.writerow(value)
    tmp.replace(path)


def coverage(data, counters, batch_calls):
    raw = np.array([r["original_bpe"] for r in data.metadata])
    visits, covered, complete, windows, calls = counters.T
    return dict(functions=len(data), observations=int(visits.sum()),
                fully_covered_observations=int(complete.sum()), fully_covered_functions=int((complete > 0).sum()),
                source_bpe_observed=int(raw @ visits), covered_bpe=int(covered.sum()),
                token_coverage=float(covered.sum() / (raw @ visits)), max_covered_bpe=int(covered.max()) if visits.max()==1 else None,
                plm_windows=int(windows.sum()), plm_batch_calls=batch_calls,
                max_plm_windows_per_function=int(windows.max()),
                max_plm_batch_calls_per_function=int(calls.max()))
