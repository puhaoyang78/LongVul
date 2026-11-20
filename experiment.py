"""One-epoch CodeBERT comparison on the published natural PrimeVul splits."""
import argparse
import csv
import gzip
import platform
import sys
from time import perf_counter

import numpy as np
import torch
import transformers
from transformers import get_linear_schedule_with_warmup, set_seed

from compression import METHODS, model_ids, select
from dataset import BUDGET, DATA, MAX_LENGTH, MODEL, RESULTS, ROOT, SPLITS, batches, rows, tokenizer, write
from rgcl.config import ADAM_EPSILON, BATCH_SIZE, LEARNING_RATE, MAX_GRAD_NORM, WARMUP_FRACTION
from rgcl.data import prime_source_rows
from rgcl.model import build_model
from rgcl.training import LOSVERAdamW

sys.path.insert(0, str(ROOT.parent / "vul-detect"))
from vulnmechanism.data import balanced_subset
from vulnmechanism.metrics import metrics, select_threshold

SEED = 42
TRAIN_SIZE = 2048
EVAL_BATCH = 32


def prepare(split, method, tok):
    source = prime_source_rows(DATA.parent, "test") if split == "test_paired" else rows(split)
    split_rows = [dict(idx=r["idx"], label=r["target"], func=r["func"]) for r in source]
    if split == "train":
        split_rows = balanced_subset(split_rows, TRAIN_SIZE, SEED)
    ids = np.full((len(split_rows), MAX_LENGTH), tok.pad_token_id, dtype=np.int32)
    n = np.empty(len(ids), dtype=np.int32)
    original = np.empty_like(n)
    labels = [r["label"] for r in split_rows]
    keys = [r["idx"] for r in split_rows]
    i = 0
    for batch in batches(split_rows, 64):
        enc = tok([r["func"] for r in batch], add_special_tokens=False, verbose=False,
                  return_attention_mask=False, return_offsets_mapping=method == "compact")
        offsets = enc.get("offset_mapping", [None] * len(batch))
        for row, source, mapping in zip(batch, enc["input_ids"], offsets):
            pos = select(method, row["func"], source, mapping)
            view = model_ids(source, pos, tok)
            original[i], n[i] = len(source), len(view)
            ids[i, :len(view)] = view
            i += 1
    return ids, n, original, labels, keys


def batch_tensors(data, idx, device):
    ids, lengths, _, labels, _ = data
    n = lengths[idx]
    width = int(n.max())
    x = torch.as_tensor(ids[idx, :width], device=device, dtype=torch.long)
    mask = torch.as_tensor(np.arange(width) < n[:, None], device=device)
    y = torch.tensor([labels[i] for i in idx], device=device)
    return x, mask, y


def train_epoch(model, data, device):
    model.train()
    optimizer = LOSVERAdamW(model.parameters(), lr=LEARNING_RATE, eps=ADAM_EPSILON)
    order = np.random.default_rng(SEED).permutation(len(data[0])).reshape(-1, BATCH_SIZE)
    scheduler = get_linear_schedule_with_warmup(optimizer, len(order) * WARMUP_FRACTION, len(order))
    losses = []
    for idx in order:
        x, mask, y = batch_tensors(data, idx, device)
        with torch.autocast("cuda", torch.bfloat16):
            loss = model(x, mask, None, None, labels=y)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.detach())
    return torch.stack(losses).mean().item()


@torch.inference_mode()
def predict(model, data, device):
    model.eval()
    scores = np.empty(len(data[0]), dtype=np.float32)
    # Same membership and batch order for all methods, grouped by raw length.
    for idx in batches(np.argsort(data[2], kind="stable"), EVAL_BATCH):
        x, mask, _ = batch_tensors(data, idx, device)
        with torch.autocast("cuda", torch.bfloat16):
            logits = model(x, mask, None, None)["logits"]
        scores[idx] = logits.float().sigmoid().cpu().numpy()
    return scores.tolist()


def save_predictions(method, split, data, scores):
    path = RESULTS / f"{method}_{split}.csv.gz"
    with gzip.open(path.with_suffix(".tmp"), "wt") as stream:
        writer = csv.writer(stream)
        writer.writerow(("idx", "target", "original_bpe", "input_length", "probability"))
        writer.writerows(zip(data[4], data[3], data[2].tolist(), data[1].tolist(), scores))
    path.with_suffix(".tmp").replace(path)


def run(method):
    begin = perf_counter()
    set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    device = torch.device("cuda:0")
    tok = tokenizer()
    model = build_model("codebert", "plm_classifier", model_source=MODEL).to(device)
    config = dict(method=method, seed=SEED, epochs=1, train_size=TRAIN_SIZE, train_batch=BATCH_SIZE,
                  eval_batch=EVAL_BATCH, lr=LEARNING_RATE, eps=ADAM_EPSILON, optimizer="LOSVERAdamW",
                  weight_decay=0., warmup_fraction=WARMUP_FRACTION, grad_clip=MAX_GRAD_NORM,
                  threshold_selection="valid MCC, F1, accuracy, then proximity to 0.5; grid .05:.01:.95",
                  max_length=MAX_LENGTH, model=str(MODEL), precision="bf16 autocast",
                  torch=torch.__version__, transformers=transformers.__version__,
                  numpy=np.__version__, python=platform.python_version(), gpu=torch.cuda.get_device_name(device),
                  cuda=torch.version.cuda)
    out = dict(config=config, seconds={}, split={})
    for split in (*SPLITS, "test_paired"):
        start = perf_counter()
        data = prepare(split, method, tok)
        out["seconds"][split + "_preparation"] = perf_counter() - start
        original = data[2]
        result = dict(count=len(original), positives=sum(data[3]), long=int((original > BUDGET).sum()),
                      model_tokens=int(data[1].sum()))
        out["split"][split] = result
        torch.cuda.synchronize()
        start = perf_counter()
        if split == "train":
            result["loss"] = train_epoch(model, data, device)
            result["optimizer_steps"] = len(original) // BATCH_SIZE
            result["source_indices"] = data[4]
        else:
            scores = predict(model, data, device)
        torch.cuda.synchronize()
        out["seconds"][split + "_model"] = perf_counter() - start
        if split != "train":
            if split == "valid":
                threshold, _ = select_threshold(data[3], scores)
            result["metrics"] = metrics(data[3], scores, threshold)
            result["at_half"] = metrics(data[3], scores)
            for label, mask in (("long", original > BUDGET), ("short", original <= BUDGET)):
                idx = np.flatnonzero(mask)
                result[label + "_metrics"] = metrics([data[3][i] for i in idx], [scores[i] for i in idx], threshold)
            if split == "test_paired":
                before, after = np.asarray(scores[::2]), np.asarray(scores[1::2])
                bound = np.float32(threshold)
                result["pairs"] = dict(count=len(before), both_correct=int(((before >= bound) & (after < bound)).sum()),
                                       ranked=int((before > after).sum()), ties=int((before == after).sum()))
            save_predictions(method, split, data, scores)
        out["seconds"]["wall_to_date"] = perf_counter() - begin
        write(f"{method}.json", out)
        print(method, split, out["seconds"], result.get("metrics", {}).get("mcc"), flush=True)
        del data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=METHODS)
    run(parser.parse_args().method)
