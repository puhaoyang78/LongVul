"""Release paths, tokenization and streaming truncation census."""
import itertools
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("RAYON_NUM_THREADS", "6")
ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent.parent / "data" / "PrimeVul_original"
MODEL = ROOT.parents[2] / "resource" / "codebert-base"
RGCL = ROOT.parent / "RGCL"
sys.path.insert(0, str(RGCL))
from rgcl.model import load_tokenizer
from rgcl.config import SPLITS, MAX_LENGTH
from rgcl.artifacts import write_json

BUDGET = MAX_LENGTH - 2
RESULTS = ROOT / "results"


def tokenizer():
    return load_tokenizer("codebert", MODEL)


def rows(split):
    with (DATA / f"primevul_{split}.jsonl").open() as stream:
        for line in stream:
            yield json.loads(line)


def batches(items, size):
    it = iter(items)
    while batch := list(itertools.islice(it, size)):
        yield batch


def write(name, data):
    write_json(RESULTS / name, data)


def census():
    import numpy as np
    tok = tokenizer()
    result = {}
    for split in SPLITS:
        lengths = [[], []]
        for batch in batches(rows(split), 64):
            encoded = tok([r["func"] for r in batch], add_special_tokens=False,
                          return_attention_mask=False, verbose=False)["input_ids"]
            for r, ids in zip(batch, encoded):
                lengths[r["target"]].append(len(ids))
        result[split] = {}
        for key, values in zip(("benign", "vulnerable", "all"), [*lengths, lengths[0] + lengths[1]]):
            n = np.array(values)
            result[split][key] = dict(functions=len(n), truncated=int((n > BUDGET).sum()),
                source_tokens=int(n.sum()), lost_tokens=int(np.maximum(n - BUDGET, 0).sum()),
                median=float(np.median(n)), p90=float(np.percentile(n, 90)), max=int(n.max()))
        write("truncation.json", result)
        print(split, result[split], flush=True)


if __name__ == "__main__":
    census()
