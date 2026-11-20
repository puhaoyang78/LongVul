"""Single-function BCE on the original RGCL PrimeVul paired release."""
import argparse
import math
import platform
from time import perf_counter

import numpy as np
import torch
import transformers
from safetensors import safe_open
from torch.utils.data import DataLoader, TensorDataset
from transformers import get_linear_schedule_with_warmup, set_seed

from context import BINS, CONTEXT_METHODS, RANK, ContextClassifier, context_data, training_unigram
from compression import METHODS
from dataset import BUDGET, DATA, MAX_LENGTH, MODEL, RESULTS, RGCL, ROOT, encode_rows, save_predictions, tokenizer
from rgcl.artifacts import read_json, read_jsonl, write_json
from rgcl.config import ADAM_EPSILON, BATCH_SIZE, EPOCHS, LEARNING_RATE, MAX_GRAD_NORM, WARMUP_FRACTION
from rgcl.data import prime_source_rows
from rgcl.evaluation import verify_predictions
from rgcl.metrics import metrics, paired_metrics
from rgcl.model import build_model, load_checkpoint
from rgcl.training import LOSVERAdamW, save_training_state, load_training_state, restore_rng

SEED = 42
OUT = RESULTS / "paired"
ARCHIVE = RGCL / "results/main/codebert/primevul_paired/seed42/plm_classifier"
HEAD_WEIGHT = RGCL / "checkpoints/codebert/primevul_paired/seed42/plm_classifier.safetensors"


def prepare(split, method, tok, noise=None):
    start = perf_counter()
    rows = prime_source_rows(DATA.parent, split)
    if method in CONTEXT_METHODS:
        data, loader = context_data(rows, method, tok, split == "train", noise)
    else:
        data = encode_rows([dict(idx=r["idx"], label=r["target"], func=r["func"]) for r in rows], method, tok)
        loader = None
    if loader is None:
        loader = data_loader(data, train=split == "train")
    return rows, data, loader, perf_counter()-start


def data_loader(data, train=False):
    ids, lengths, _, labels, _ = data
    if not ((lengths >= 2) & (lengths <= MAX_LENGTH)).all():
        raise ValueError("input outside model budget")
    tensors = TensorDataset(torch.from_numpy(ids).long(),
                            torch.from_numpy(np.arange(MAX_LENGTH) < lengths[:, None]).long(),
                            torch.tensor(labels))
    return DataLoader(tensors, batch_size=BATCH_SIZE, shuffle=train)


def forward(model, batch, with_labels=False):
    x, mask, y, *extra = batch
    args = [v.cuda() for v in extra] if extra else [None, None]
    return model(x.cuda(), mask.cuda(), *args, labels=y.cuda() if with_labels else None)


@torch.inference_mode()
def predict(model, loader):
    model.eval()
    scores = []
    for batch in loader:
        scores.extend(forward(model, batch)["probability"].tolist())
    return scores


def groups(data, scores):
    n = data[2]
    groups = {"overall": n >= 0, "long": n > BUDGET, "short": n <= BUDGET,
              "511-1020": (n > BUDGET) & (n <= 2*BUDGET),
              "1021-2040": (n > 2*BUDGET) & (n <= 4*BUDGET), "2041+": n > 4*BUDGET}
    return {name: metrics([data[3][i] for i in np.flatnonzero(mask)],
                          [scores[i] for i in np.flatnonzero(mask)]) for name, mask in groups.items()}


def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    losses = []
    for batch in loader:
        loss = forward(model, batch, True)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        losses.append(loss.detach())
    return torch.stack(losses).mean().item()


def run(method, reuse_head=False):
    if reuse_head and method != "head":
        raise ValueError("--reuse-head requires head")
    set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    if torch.cuda.device_count() != 1 or "4090" not in torch.cuda.get_device_name():
        raise ValueError("select one RTX 4090 via CUDA_VISIBLE_DEVICES")
    run_dir = ROOT / "checkpoints" / "paired" / method
    result_dir = OUT / method
    summary = result_dir / "metrics.json"
    if summary.exists():
        print(f"Completed result exists: {summary}", flush=True)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    tok = tokenizer()
    model = (load_checkpoint(HEAD_WEIGHT, model_source=MODEL) if reuse_head else
             build_model("codebert", "plm_classifier", model_source=MODEL))
    if method in ("context_real", "context_random"):
        model = ContextClassifier(model)
    model.cuda()
    config = dict(method=method, seed=SEED, epochs=EPOCHS, batch_size=BATCH_SIZE, length=MAX_LENGTH,
                  lr=LEARNING_RATE, eps=ADAM_EPSILON, optimizer="LOSVERAdamW", weight_decay=0.,
                  warmup=WARMUP_FRACTION, grad_clip=MAX_GRAD_NORM, precision="fp32 eager",
                  model=str(MODEL), loss="function BCE", class_weight=1., threshold=.5, comparison=">",
                  selection="highest full validation MCC; earliest tie", gpu=torch.cuda.get_device_name(),
                  cuda=torch.version.cuda, torch=torch.__version__, transformers=transformers.__version__,
                  python=platform.python_version(), dataset=str(DATA), historical_head=reuse_head)
    if method in CONTEXT_METHODS:
        config.update(context_slots=1, context_bins=BINS, projection_rank=RANK if method != "context_empty" else 0,
                      context_source={"context_empty": "zero residual", "context_real": "omitted source BPEs",
                                      "context_random": "train unigram, matched omitted bin counts"}[method])
    start = perf_counter()
    noise = training_unigram(tok) if method == "context_random" else None
    noise_seconds = perf_counter() - start if noise is not None else 0.
    vr, vd, vl, valid_prep = prepare("valid", method, tok, noise)
    if reuse_head:
        history = read_jsonl(ARCHIVE / "epochs.jsonl")
        best = max(history, key=lambda r:r["validation"]["mcc"])
        with safe_open(HEAD_WEIGHT, framework="pt") as f:
            epoch = int(f.metadata()["epoch"])
        if ([r["epoch"] for r in history] != list(range(1, EPOCHS+1)) or best["epoch"] != epoch or
                read_json(ARCHIVE / "selection.json")["epoch"] != epoch):
            raise ValueError("RGCL checkpoint does not match the 16-epoch validation MCC winner")
        train_prep = None
    else:
        _, _, loader, train_prep = prepare("train", method, tok, noise)
        steps = EPOCHS * len(loader)
        opt = LOSVERAdamW(model.parameters(), lr=LEARNING_RATE, eps=ADAM_EPSILON)
        scheduler = get_linear_schedule_with_warmup(opt, steps * WARMUP_FRACTION, steps)
        history = []
        if (run_dir / "last.pt").exists():
            history, rng = load_training_state(run_dir, model, opt, config, scheduler=scheduler)
            restore_rng(rng)
        for epoch in range(len(history)+1, EPOCHS+1):
            torch.cuda.synchronize()
            start = perf_counter()
            torch.cuda.reset_peak_memory_stats()
            loss = train_epoch(model, loader, opt, scheduler)
            peak = torch.cuda.max_memory_allocated()
            train_s = perf_counter() - start
            start = perf_counter()
            scores = predict(model, vl)
            valid_s = perf_counter() - start
            valid = metrics(vd[3], scores)
            previous = max((r["validation"]["mcc"] for r in history), default=-math.inf)
            if valid["mcc"] > previous:
                temporary = run_dir / "best.tmp"
                torch.save(model.state_dict(), temporary)
                temporary.replace(run_dir / "best.pt")
            history.append(dict(epoch=epoch, optimizer_steps=len(loader), train_loss=loss, peak_allocated_bytes=peak,
                                validation=valid, training_seconds=train_s, validation_seconds=valid_s))
            save_training_state(run_dir, model, opt, config, history, scheduler=scheduler)
            write_json(result_dir / "epochs.json", history)
            print(method, epoch, f"loss={loss:.5f} valid_mcc={valid['mcc']:.4f} train_s={train_s:.1f}", flush=True)
        best = max(history, key=lambda r:r["validation"]["mcc"])
        del opt, scheduler
        model.load_state_dict(torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True))
    write_json(result_dir / "epochs.json", history)
    result = dict(config=config, selected_epoch=best["epoch"], train_preparation_seconds=train_prep,
                  training_seconds=sum(r["training_seconds"] for r in history), split={})
    if not reuse_head and all("peak_allocated_bytes" in r for r in history):
        result["peak_train_allocated_bytes"] = max(r["peak_allocated_bytes"] for r in history)
    if noise is not None:
        result["train_unigram_seconds"] = noise_seconds
    for split in ("valid", "test"):
        rows, data, loader, prep = (vr, vd, vl, valid_prep) if split == "valid" else prepare(split, method, tok, noise)
        torch.cuda.synchronize()
        start = perf_counter()
        torch.cuda.reset_peak_memory_stats()
        scores = predict(model, loader)
        elapsed = perf_counter() - start
        value = dict(groups=groups(data, scores), pairs=paired_metrics(rows, scores),
                     preparation_seconds=prep, inference_seconds=elapsed,
                     peak_inference_allocated_bytes=torch.cuda.max_memory_allocated(), model_tokens=int(data[1].sum()))
        if reuse_head:
            current = [dict(sample_key=r["sample_key"], pair_key=r["pair_key"], target=r["target"],
                            probability=p, prediction=int(p > .5)) for r,p in zip(rows,scores)]
            value["replay"] = verify_predictions(current, read_jsonl(ARCHIVE / f"{split}.predictions.jsonl"))
        if split == "valid" and abs(value["groups"]["overall"]["mcc"] - best["validation"]["mcc"]) > 1e-10:
            raise ValueError("reloaded model differs from MCC-selected validation epoch")
        save_predictions(result_dir / f"{split}.csv.gz", data, scores)
        result["split"][split] = value
    write_json(summary, result)
    print(method, "complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=(*METHODS, *CONTEXT_METHODS))
    parser.add_argument("--reuse-head", action="store_true")
    args = parser.parse_args()
    run(args.method, args.reuse_head)
