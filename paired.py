"""Train the compaction ablations with LongVul's paired-release BCE protocol."""
import argparse
import csv
import gzip
import json
import math
import platform
from time import perf_counter

import numpy as np
import torch
import transformers
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, set_seed

from compaction import BUDGET, MAX_CANDIDATES, QUOTAS, REGIONS, CompactData, collate
from csp_model import CHANNELS, DISTANCE_BINS, POSITION_BINS, TEMPERATURE, CompactionClassifier, pack
from window_inputs import BASE, ROOT, grouped_metrics
from rgcl.artifacts import write_json
from rgcl.config import ADAM_EPSILON, BATCH_SIZE, EPOCHS, LEARNING_RATE, MAX_GRAD_NORM, MAX_LENGTH, WARMUP_FRACTION
from rgcl.metrics import metrics
from rgcl.model import build_model, load_tokenizer
from rgcl.training import LOSVERAdamW, load_training_state, restore_rng, save_training_state

SEED = 42
METHODS = {"ab": (True, False), "ac": (False, True), "abc": (True, True)}
RESULTS = ROOT / "results" / "csp"
WEIGHTS = ROOT / "checkpoints" / "csp"


def make_model(tokenizer, method):
    select_windows, use_positions = METHODS[method]
    base = build_model("codebert", "plm_classifier", model_source=BASE)
    return CompactionClassifier(base, tokenizer, select_windows=select_windows, use_positions=use_positions).cuda()


def to_gpu(batch, *, labels=False):
    return {k: tuple(x.cuda() for x in v) if k == "bank" else v.cuda()
            for k, v in batch.items() if labels or k != "labels"}


def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    losses, gradients = [], []
    selector = list(model.conv.parameters()) + list(model.scorer.parameters()) if model.scorer is not None else []
    for batch in loader:
        loss = model(to_gpu(batch, labels=True))["loss"]
        loss.backward()
        if "bank" in batch:
            gradients.append(torch.linalg.vector_norm(torch.stack([p.grad.detach().norm() for p in selector])))
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM, error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        losses.append(loss.detach())
    result = dict(train_loss=torch.stack(losses).mean().item())
    if gradients:
        norms = torch.stack(gradients)
        result.update(selector_batches=len(gradients), selector_nonzero_batches=int(norms.gt(0).sum().item()),
                      selector_mean_grad_norm=norms.mean().item())
    return result


@torch.inference_mode()
def predict(model, loader, *, trace=False):
    model.eval()
    scores, traces = [], []
    for batch in loader:
        report = model(to_gpu(batch), details=trace)
        scores.extend(report["probability"].tolist())
        if trace:
            vectors = [report[k].tolist() for k in ("input_lengths", "spans", "choices")]
            traces.extend(zip(*vectors, strict=True))
    return scores, traces


def describe_selection(data, scores, traces, path):
    decisions = []
    for row, features, score, (length, chosen, choices) in zip(data.metadata, data.examples, scores, traces, strict=True):
        uniform = features["uniform"].tolist()
        active = [(c, u) for c, u in zip(chosen, uniform, strict=True) if c[0] >= 0]
        intersect = sum(max(0, min(c[1], u[1])-max(c[0], u[0])) for c, u in active)
        selected_size = sum(c[1]-c[0] for c, _ in active)
        uniform_size = sum(u[1]-u[0] for _, u in active)
        differences = sum(c != u for c, u in active)
        decisions.append(dict(**row, input_length=length, probability=score, prediction=int(score > .5),
                              selected_windows=chosen, uniform_windows=uniform,
                              selected_candidate=choices, candidate_counts=features["candidate_counts"],
                              region_limits=features["limits"], changed_regions=differences,
                              uniform_overlap_bpe=intersect, window_jaccard=intersect/(selected_size+uniform_size-intersect)))
    temporary = path.with_suffix(".tmp")
    with gzip.open(temporary, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=decisions[0].keys())
        writer.writeheader()
        for decision in decisions:
            writer.writerow({k: json.dumps(v) if isinstance(v, list) else v for k, v in decision.items()})
    temporary.replace(path)

    summary = {}
    for name, minimum in (("long", BUDGET+1), ("2041+", 2041)):
        rows = [r for r in decisions if r["original_bpe"] >= minimum]
        split = [r for r in rows if len(r["region_limits"]) == REGIONS]
        movable = [r for r in split if max(r["candidate_counts"]) > 1]
        histogram = np.zeros((REGIONS, MAX_CANDIDATES), dtype=int)
        center = [0] * REGIONS
        counts = [0] * REGIONS
        for row in movable:
            for k, ((start, end, _, _), (left, right), n) in enumerate(
                    zip(row["selected_windows"], row["region_limits"], row["candidate_counts"], strict=True)):
                if n > 1:
                    relative = ((start+end)/2-left)/(right-left)
                    histogram[k, min(MAX_CANDIDATES-1, int(MAX_CANDIDATES*relative))] += 1
                    center[k] += int(row["selected_candidate"][k] == 0)
                    counts[k] += 1
        summary[name] = dict(functions=len(rows), four_regions=len(split), movable_functions=len(movable),
                             functions_changed=sum(r["changed_regions"] > 0 for r in rows),
                             movable_functions_changed=sum(r["changed_regions"] > 0 for r in movable),
                             mean_window_jaccard=float(np.mean([r["window_jaccard"] for r in rows])),
                             movable_regions_by_quarter=counts, uniform_selections_by_quarter=center,
                             chosen_centers_by_quarter=histogram.tolist())
    return summary


def smoke(tokenizer):
    data = CompactData("train", tokenizer, alternatives=True)
    longest = sorted(range(len(data)), key=lambda i: data.metadata[i]["original_bpe"], reverse=True)[:BATCH_SIZE]
    batch = to_gpu(collate([data[i] for i in longest]), labels=True)
    model = make_model(tokenizer, "abc")
    base = model.detector
    windows = batch["bank"]
    fixed = torch.zeros_like(windows[-1][..., 0, 0], dtype=torch.long)
    packed = pack(*windows, fixed, model.special)
    for derived, reference in zip(packed[:5], (*batch["fixed"].unbind(1), batch["lengths"], batch["uniform"]), strict=True):
        if not torch.equal(derived, reference):
            raise ValueError("candidate 0 differs from Uniform-4 packing")
    model.eval()
    short = next(ex for ex in data.examples if ex["raw"] <= BUDGET)
    short_batch = to_gpu(collate([short]))
    with torch.inference_mode():
        hard = model(batch, details=True)
        repeated = model(batch, details=True)
        assert torch.equal(hard["input_ids"], repeated["input_ids"])
        for report in (hard, model(short_batch, details=True)):
            original = base(report["input_ids"], report["attention"], None, None)
            torch.testing.assert_close(report["logits"], original["logits"], rtol=0, atol=0)
        assert torch.equal(short_batch["fixed"][:, 0].long(), model(short_batch, details=True)["input_ids"])

    model.train()
    calls, inputs = [], []
    def record(module, args, kwargs):
        calls.append(1)
        inputs.append(kwargs["inputs_embeds"].detach())

    handle = base.encoder.register_forward_pre_hook(record, with_kwargs=True)
    torch.cuda.reset_peak_memory_stats()
    out = model(batch, details=True)
    handle.remove()
    if len(calls) != 1 or out["input_lengths"].max().item() > MAX_LENGTH:
        raise ValueError("more than one PLM call or an over-budget sequence")
    torch.testing.assert_close(inputs[0], base.encoder.get_input_embeddings()(out["input_ids"]), rtol=0, atol=0)
    out["loss"].backward()
    checked = {"cnn": model.conv.weight, "scorer": model.scorer[-1].weight,
               "position": model.position[0].weight, "codebert": base.encoder.encoder.layer[0].attention.self.query.weight}
    norms = {k: v.grad.norm().item() if v.grad is not None else 0. for k, v in checked.items()}
    if not math.isfinite(out["loss"].item()) or not all(math.isfinite(v) and v > 0. for v in norms.values()):
        raise ValueError(f"invalid BCE gradients on real long functions: {norms}")
    saved = [p.detach().clone() for p in checked.values()]
    optimizer = LOSVERAdamW(model.parameters(), lr=LEARNING_RATE, eps=ADAM_EPSILON)
    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM, error_if_nonfinite=True)
    optimizer.step()
    if any(torch.equal(p, before) for p, before in zip(checked.values(), saved, strict=True)):
        raise ValueError("optimizer failed to update an active component")
    result = dict(functions=[data.metadata[i] for i in longest], input_lengths=out["input_lengths"].tolist(),
                  candidate_counts=[data[i]["candidate_counts"] for i in longest],
                  encoder_calls=len(calls), max_candidate_token_slots=windows[0][0].numel(),
                  candidate_embedding_shape=list(windows[0].shape)+[base.config.hidden_size],
                  loss=out["loss"].item(), bce_gradient_norms=norms,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated())
    write_json(RESULTS / "smoke.json", result)
    print("CSP smoke passed", result["functions"][0], norms, round(result["peak_allocated_bytes"]/2**30, 3), flush=True)


def measure_data(split, tokenizer, alternatives):
    start = perf_counter()
    data = CompactData(split, tokenizer, alternatives=alternatives)
    return data, perf_counter()-start


def run(method, tokenizer):
    root = RESULTS / method
    if (root / "metrics.json").exists():
        print(method, "already complete", flush=True)
        return
    weights = WEIGHTS / method
    weights.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    model = make_model(tokenizer, method)
    use_selection, use_positions = METHODS[method]
    train, train_prep = measure_data("train", tokenizer, use_selection)
    valid, valid_prep = measure_data("valid", tokenizer, use_selection)
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    valid_loader = DataLoader(valid, batch_size=BATCH_SIZE, collate_fn=collate)
    optimizer = LOSVERAdamW(model.parameters(), lr=LEARNING_RATE, eps=ADAM_EPSILON)
    steps = len(train_loader) * EPOCHS
    schedule = get_linear_schedule_with_warmup(optimizer, steps*WARMUP_FRACTION, steps)
    config = dict(method=method, seed=SEED, epochs=EPOCHS, batch_size=BATCH_SIZE, length=MAX_LENGTH,
                  lr=LEARNING_RATE, eps=ADAM_EPSILON, optimizer="LOSVERAdamW", weight_decay=0.,
                  warmup=WARMUP_FRACTION, grad_clip=MAX_GRAD_NORM, precision="fp32 eager",
                  model=str(BASE), loss="function BCE", class_weight=1., threshold=.5, comparison=">",
                  selection="highest full validation MCC, earliest tie", gpu=torch.cuda.get_device_name(),
                  cuda=torch.version.cuda, torch=torch.__version__, transformers=transformers.__version__,
                  python=platform.python_version(), quotas=QUOTAS, candidates_per_region=MAX_CANDIDATES,
                  selector_channels=CHANNELS if use_selection else None,
                  gumbel_temperature=TEMPERATURE if use_selection else None,
                  source_position_bins=POSITION_BINS if use_positions else None,
                  distance_bins=DISTANCE_BINS if use_positions else None)
    write_json(root / "config.json", config)
    history = []
    if (weights / "last.pt").exists():
        history, rng = load_training_state(weights, model, optimizer, config, scheduler=schedule)
        restore_rng(rng)
    best = max((row["validation"]["mcc"] for row in history), default=-math.inf)
    for epoch in range(len(history)+1, EPOCHS+1):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        train_stats = train_epoch(model, train_loader, optimizer, schedule)
        seconds, peak = perf_counter()-start, torch.cuda.max_memory_allocated()
        start = perf_counter()
        scores, _ = predict(model, valid_loader)
        valid_seconds = perf_counter()-start
        validation = metrics(valid.targets.tolist(), scores)
        if validation["mcc"] > best:
            best = validation["mcc"]
            temporary = weights / "best.tmp"
            torch.save(model.state_dict(), temporary)
            temporary.replace(weights / "best.pt")
        history.append(dict(epoch=epoch, optimizer_steps=len(train_loader), **train_stats,
                            peak_allocated_bytes=peak, validation=validation,
                            training_seconds=seconds, validation_seconds=valid_seconds))
        save_training_state(weights, model, optimizer, config, history, scheduler=schedule)
        write_json(root / "epochs.json", history)
        print(method, epoch, f"loss={train_stats['train_loss']:.5f} valid_mcc={validation['mcc']:.4f} train_s={seconds:.1f}",
              flush=True)
    del optimizer, schedule
    selected = max(history, key=lambda r: r["validation"]["mcc"])
    model.load_state_dict(torch.load(weights / "best.pt", weights_only=True, map_location="cpu"), strict=True)
    test, test_prep = measure_data("test", tokenizer, use_selection)
    result = dict(config=config, selected_epoch=selected["epoch"],
                  training_seconds=sum(row["training_seconds"] for row in history),
                  peak_train_allocated_bytes=max(row["peak_allocated_bytes"] for row in history),
                  preparation_seconds=dict(train=train_prep, valid=valid_prep, test=test_prep), split={})
    for split, data in (("valid", valid), ("test", test)):
        loader = DataLoader(data, batch_size=BATCH_SIZE, collate_fn=collate)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        scores, traces = predict(model, loader, trace=True)
        seconds, peak = perf_counter()-start, torch.cuda.max_memory_allocated()
        groups = grouped_metrics(data, scores)
        if split == "valid" and groups["overall"] != selected["validation"]:
            raise ValueError("reloaded checkpoint does not reproduce selected validation metrics")
        selection = describe_selection(data, scores, traces, root / f"{split}.csv.gz")
        result["split"][split] = dict(groups=groups, selection=selection,
                                      inference_seconds=seconds, peak_inference_allocated_bytes=peak,
                                      model_tokens=sum(length for length, _, _ in traces))
    write_json(root / "metrics.json", result)
    print(method, "complete", selected["epoch"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("methods", choices=METHODS, nargs="*")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    if not args.methods and not args.smoke_only:
        parser.error("choose one or more ablations or --smoke-only")
    if torch.cuda.device_count() != 1 or "4090" not in torch.cuda.get_device_name():
        raise ValueError("select one RTX 4090 via CUDA_VISIBLE_DEVICES")
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    tokenizer = load_tokenizer("codebert", BASE)
    if args.smoke_only:
        set_seed(SEED)
        smoke(tokenizer)
    else:
        for method in args.methods:
            run(method, tokenizer)
