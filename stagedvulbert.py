"""Fine-tune the authors' function classifier on RGCL's original PrimeVul pairs."""
import argparse
import csv
import gzip
import math
import os
import platform
import subprocess
import sys
from itertools import islice
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import torch
import transformers
from torch.utils.data import DataLoader
from transformers import (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          RobertaTokenizerFast, get_linear_schedule_with_warmup, set_seed)

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "vendor/StagedVulBERT"
RGCL = ROOT.parent / "RGCL"
sys.path[:0] = [str(UPSTREAM), str(RGCL)]
from Entry.StagedBert_vul import TextDataset, convert_examples_to_features
from Models.StagedModel_vul import Model
from rgcl.artifacts import write_json
from rgcl.data import prime_source_rows
from rgcl.metrics import metrics
from window_inputs import SEEDS, seed_path

REVISION = "d589175880ff99b0192f4ded54f5455951a696fe"
RESOURCE = ROOT.parents[2] / "resource"
BASE = RESOURCE / "codebert-base"
MSP = RESOURCE / "StagedVulBERT/pretrain_model.bin"
DATA = ROOT.parents[1] / "data"
OUT = ROOT / "results/stagedvulbert"
WEIGHTS = ROOT / "checkpoints/stagedvulbert"
EPOCHS, MICRO, BATCH, SEED = 16, 8, 32, 42
LR, EPS = 2e-5, 1e-8


def build(args, tok):
    cfg = RobertaConfig.from_pretrained(BASE, local_files_only=True)
    cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_labels = 6, 12, 1
    cfg._attn_implementation = "eager"  # upstream Transformers 4.34 attention
    model = Model(RobertaForSequenceClassification(cfg), RobertaForSequenceClassification(cfg), cfg, tok, args)
    source = torch.load(MSP, map_location="cpu", weights_only=True)
    target = model.state_dict()
    matched = sorted(source.keys() & target.keys())
    unmatched = sorted(source.keys() - target.keys())
    missing = sorted(target.keys() - source.keys())
    wrong = [k for k in matched if source[k].shape != target[k].shape or source[k].dtype != target[k].dtype]
    for prefix in ("TEtransformer.", "SETransformer."):
        backbone = [k for k in target if k.startswith(prefix)]
        if len(backbone) != 105 or not all(k in matched for k in backbone):
            raise ValueError(f"incomplete MSP initialization for {prefix}")
    if wrong or any(not k.startswith(("roberta.", "classifier.")) for k in missing):
        raise ValueError(f"unexpected initialization mismatch: {wrong}, {missing}")
    if any(k not in ("Q", "K") and not k.startswith("mslmTask.") for k in unmatched):
        raise ValueError(f"unrecognized MSP-only keys: {unmatched}")
    model.load_state_dict({k: source[k] for k in matched}, strict=False)
    for k in matched:
        if not torch.equal(target[k], source[k]) or not torch.isfinite(target[k]).all():
            raise ValueError(f"MSP copy/finite check failed for {k}")
    audit = dict(file=MSP.name, bytes=MSP.stat().st_size, source_tensors=len(source), target_tensors=len(target),
                 matched_elements=sum(source[k].numel() for k in matched), matched=matched, missing=missing,
                 source_only=unmatched, mismatched=wrong)
    return model.cuda(), audit


class PrimeFunctions(TextDataset):
    """Preserve the upstream converter and masks; return only the function target."""
    def __init__(self, split, tok, length_tok, args):
        self.args = args
        self.examples, self.metadata = [], []
        rows = prime_source_rows(DATA, split)
        for start in range(0, len(rows), 32):
            batch = rows[start:start + 32]
            raw_lengths = length_tok([r["func"] for r in batch], add_special_tokens=False,
                                     return_attention_mask=False, return_length=True, verbose=False)["length"]
            for row, raw in zip(batch, raw_lengths):
                ex = convert_examples_to_features((row["func"], row["target"], None, tok, args))
                total = len(ex.input_tokens) - 2
                ex.input_tokens = None  # data-only field, unused by forward
                segments = [s for s in ex.tokens_features if s is not None]
                sizes = [len(s.row_idx) - 1 for s in segments]
                covered = sum(sizes)
                if not (0 <= covered <= total and 1 <= len(sizes) <= 4 and max(sizes) <= 511):
                    raise ValueError(f"bad coverage at {split}/{row['idx']}")
                if ex.row_num + 1 > 512:
                    raise ValueError(f"statement position limit exceeded at {split}/{row['idx']}")
                self.metadata.append(dict(idx=row["idx"], sample_key=row["sample_key"], target=row["target"],
                                          original_bpe=raw, staged_bpe=total, covered_bpe=covered, segments=len(sizes),
                                          empty_segments=sizes.count(0), statements=ex.row_num))
                self.examples.append(ex)

    def __getitem__(self, index):
        ex = self.examples[index]
        return [self.get_feature(s) for s in ex.tokens_features], ex.label_c


def move(features):
    # Upstream uses the segment-present flags to index its CPU row_map, too.
    return [[t.cuda() for t in seg[:-1]] + [seg[-1]] for seg in features]


def optimizer_for(model):
    groups = [dict(params=[], weight_decay=0.), dict(params=[], weight_decay=0.)]
    for n, p in model.named_parameters():
        groups[int("bias" in n or "LayerNorm.weight" in n)]["params"].append(p)
    return torch.optim.AdamW(groups, lr=LR, eps=EPS)


def smoke(model, dataset):
    # Stress four occupied TE passes with actual training rows. No validation/test labels.
    longest = sorted(range(len(dataset)), key=lambda i: dataset.metadata[i]["covered_bpe"], reverse=True)[:MICRO]
    features, labels = next(iter(DataLoader(dataset, batch_size=MICRO, sampler=longest)))
    opt = optimizer_for(model)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    loss, prob, _ = model(move(features), labels=labels.cuda())
    loss.backward()
    watch = [model.TEtransformer.roberta.encoder.layer[0].attention.self.query.weight,
             model.SETransformer.roberta.encoder.layer[0].attention.self.query.weight,
             model.classifier.out_proj.weight]
    grad = [p.grad.norm().item() for p in watch]
    if not all(math.isfinite(v) and v > 0 for v in grad) or not torch.isfinite(prob).all():
        raise ValueError("nonfinite smoke output or missing active gradient")
    before = [p.detach().clone() for p in watch]
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    opt.step()
    if any(torch.equal(p, initial) for p, initial in zip(watch, before)):
        raise ValueError("optimizer did not update TE, SE or function head")
    return dict(idx=[dataset.metadata[i]["idx"] for i in longest], loss=loss.item(), gradient_norms=grad,
                peak_allocated_bytes=torch.cuda.max_memory_allocated())


@torch.inference_mode()
def predict(model, dataset):
    model.eval()
    scores = []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    for features, labels in DataLoader(dataset, batch_size=MICRO):
        _, prob, _ = model(move(features))
        scores.extend(prob[:, 0, 1].tolist())
    return scores, perf_counter()-start, torch.cuda.max_memory_allocated()


def report(split, data, scores, seconds, peak, *, out):
    rows = data.metadata
    with gzip.open(out / f"{split}.csv.gz", "wt") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*rows[0], "probability", "prediction"])
        writer.writeheader()
        writer.writerows(dict(r, probability=p, prediction=int(p > .5)) for r, p in zip(rows, scores, strict=True))
    groups = {}
    for name, low, high in [("overall", 0, math.inf), ("long", 511, math.inf), ("short", 0, 510),
                            ("511-1020", 511, 1020), ("1021-2040", 1021, 2040), ("2041+", 2041, math.inf)]:
        members = [(r["target"], p) for r, p in zip(rows, scores) if low <= r["original_bpe"] <= high]
        labels, values = zip(*members)
        groups[name] = metrics(list(labels), list(values))
    return dict(groups=groups, inference_seconds=seconds, peak_allocated_bytes=peak)


def coverage(data):
    rows = data.metadata
    return dict(functions=len(rows), max_original_bpe=max(r["original_bpe"] for r in rows),
                max_staged_bpe=max(r["staged_bpe"] for r in rows), max_covered_bpe=max(r["covered_bpe"] for r in rows),
                total_staged_bpe=sum(r["staged_bpe"] for r in rows), total_covered_bpe=sum(r["covered_bpe"] for r in rows),
                nominal_over_four=sum(math.ceil((r["staged_bpe"]+1)/512)>4 for r in rows),
                truncated=sum(r["covered_bpe"] < r["staged_bpe"] for r in rows),
                truncated_with_four=sum(r["segments"]==4 and r["covered_bpe"]<r["staged_bpe"] for r in rows),
                zero_body_functions=sum(not r["covered_bpe"] for r in rows),
                functions_with_empty_segment=sum(r["empty_segments"]>0 for r in rows))


def run(smoke_only, seed=SEED):
    out,weights = seed_path(OUT,seed),seed_path(WEIGHTS,seed)
    args = SimpleNamespace(block_size=512, seg_num=4, device=torch.device("cuda:0"))
    if torch.cuda.device_count() != 1 or "4090" not in torch.cuda.get_device_name():
        raise ValueError("select a single RTX 4090")
    rev = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
    if rev != REVISION:
        raise ValueError("wrong upstream revision")
    out.mkdir(parents=True, exist_ok=True)
    weights.mkdir(parents=True, exist_ok=True)
    if (out / "metrics.json").exists() and not smoke_only:
        raise FileExistsError("completed experiment already exists")
    torch.set_num_threads(4)
    set_seed(seed)
    tok = RobertaTokenizer.from_pretrained(BASE, local_files_only=True)
    length_tok = RobertaTokenizerFast.from_pretrained(BASE, local_files_only=True)
    model, audit = build(args, tok)
    write_json(out / "initialization.json", audit)
    print("Verified MSP tensors:", len(audit["matched"]), flush=True)
    start = perf_counter()
    train = PrimeFunctions("train", tok, length_tok, args)
    train_prep = perf_counter()-start
    smoke_result = smoke(model, train)
    write_json(out / "smoke.json", smoke_result)
    print("Optimizer smoke", smoke_result, flush=True)
    if smoke_only:
        return
    del model
    torch.cuda.empty_cache()
    set_seed(seed)
    model, _ = build(args, tok)
    start = perf_counter()
    valid = PrimeFunctions("valid", tok, length_tok, args)
    valid_prep = perf_counter()-start
    loader = DataLoader(train, shuffle=True, batch_size=MICRO)
    steps = math.ceil(len(train)/BATCH)
    opt = optimizer_for(model)
    scheduler = get_linear_schedule_with_warmup(opt, steps*EPOCHS//5, steps*EPOCHS)
    cfg = dict(upstream=REVISION, seed=seed, epochs=EPOCHS, micro_batch=MICRO, effective_batch=BATCH,
               accumulation=BATCH//MICRO, steps_per_epoch=steps, warmup_steps=steps*EPOCHS//5, lr=LR,
               optimizer="torch.optim.AdamW", eps=EPS, weight_decay=0, clip_norm=1., scheduler="linear",
               block_size=512, seg_num=4, layers_per_stage=6, precision="FP32 eager", threshold=.5,
               selection="highest full validation MCC, earliest tie", loss="official two-class cross entropy",
               torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
               transformers=transformers.__version__, python=platform.python_version(), host=platform.node())
    write_json(out / "config.json", cfg)
    history, best = [], -math.inf
    for epoch in range(1, EPOCHS+1):
        model.train()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = perf_counter()
        count, updates, total_loss = 0, 0, 0.
        it = iter(loader)
        while batches := list(islice(it, BATCH//MICRO)):
            n = sum(len(y) for _, y in batches)
            opt.zero_grad(set_to_none=True)
            for features, y in batches:
                loss, _, _ = model(move(features), labels=y.cuda())
                (loss*(len(y)/n)).backward()
                total_loss += loss.detach().item()*len(y)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            scheduler.step()
            count += n
            updates += 1
        torch.cuda.synchronize()
        seconds, peak = perf_counter()-start, torch.cuda.max_memory_allocated()
        if count != len(train) or updates != steps:
            raise ValueError("missing functions or optimizer steps")
        opt.zero_grad(set_to_none=True)
        scores, valid_time, _ = predict(model, valid)
        metric = metrics([r["target"] for r in valid.metadata], scores)
        if metric["mcc"] > best:
            best = metric["mcc"]
            tmp = weights / "best.tmp"
            torch.save(model.state_dict(), tmp)
            os.replace(tmp, weights / "best.pt")
        history.append(dict(epoch=epoch, loss=total_loss/count, functions=count, updates=updates,
                            seconds=seconds, peak_allocated_bytes=peak, valid_seconds=valid_time, validation=metric))
        write_json(out / "epochs.json", history)
        print(epoch, "valid MCC", metric["mcc"], "train seconds", seconds, flush=True)
    del opt, scheduler
    selected = max(history, key=lambda r:r["validation"]["mcc"])
    model.load_state_dict(torch.load(weights / "best.pt", map_location="cpu", weights_only=True), strict=True)
    start = perf_counter()
    test = PrimeFunctions("test", tok, length_tok, args)
    test_prep = perf_counter()-start
    result = dict(selected_epoch=selected["epoch"], training_seconds=sum(r["seconds"] for r in history),
                  preparation_seconds=dict(train=train_prep, valid=valid_prep, test=test_prep),
                  peak_train_allocated_bytes=max(r["peak_allocated_bytes"] for r in history),
                  coverage={s:coverage(d) for s,d in (("train",train),("valid",valid),("test",test))}, split={})
    for split, data in (("valid", valid), ("test", test)):
        result["split"][split] = report(split, data, *predict(model, data),out=out)
    if result["split"]["valid"]["groups"]["overall"] != selected["validation"]:
        raise ValueError("best checkpoint validation replay differs")
    write_json(out / "metrics.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--seed",type=int,choices=SEEDS,default=SEED)
    args = parser.parse_args()
    run(args.smoke_only,args.seed)
