"""Run the published VulnScout and SEA long-code baselines on PrimeVul_original."""
import argparse
import math
import platform
import subprocess
import sys
from time import perf_counter

import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup, set_seed
import transformers

from window_inputs import (BASE, ROOT, UPSTREAM, SCOUT_LENGTH, SCOUT_OVERLAP, SEA_LENGTH, SEA_STEP,
                           SEA_TRAIN_WINDOWS, SEA_WIDTH, SEEDS, WindowData, coverage, grouped_metrics, save_rows, seed_path)
from rgcl.artifacts import write_json
from rgcl.metrics import metrics
from rgcl.training import LOSVERAdamW, load_training_state, restore_rng, save_training_state

sys.path.insert(0, str(UPSTREAM / "VulnScout"))

REVISIONS = {
    "vulnscout": ("VulnScout", "2834486c1750fe800ea875311b3fa403265bae4f"),
    "sea": ("SEA", "8bb61b1baabeb5da3b46005380d6cabdeda53a5c"),
}
SEED = 42
SEA_BATCH, SEA_MICRO, EVAL_BATCH = 48, 8, 64
SEA_TRAIN_LIMIT, EVAL_LIMIT = SEA_TRAIN_WINDOWS * SEA_MICRO, 64
SCOUT_BATCH, SCOUT_EPOCHS, SCOUT_PATIENCE, SEA_EPOCHS = 64, 100, 10, 15


class LongCodeModel(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.method = method
        encoder, loading = AutoModel.from_pretrained(BASE, local_files_only=True, output_loading_info=True,
                                                      attn_implementation="sdpa")
        unexpected = set(loading["unexpected_keys"]) - {"embeddings.position_ids"}
        if loading["missing_keys"] or loading["mismatched_keys"] or unexpected or loading.get("error_msgs"):
            raise ValueError(f"CodeBERT did not load cleanly: {loading}")
        self.encoder = encoder
        self.load_audit = dict(loading, parameters=sum(p.numel() for p in encoder.parameters()))
        width = self.encoder.config.hidden_size
        if method == "vulnscout":
            from Experiment.mlp_classifier import MLPClassifier

            self.encoder.requires_grad_(False).eval()
            self.head = MLPClassifier(embedding_dim=width)
        else:
            from Model.Attention import Attention_1

            self.fusion = Attention_1(width, with_ave=True, mul=False, common_type="one_layer")
            self.head = nn.Linear(width, 1)

    def encode(self, ids, lengths, counts, limit):
        chunks = []
        for block, active in zip(ids.split(limit), lengths.split(limit), strict=True):
            block = block.cuda().long()
            if self.method == "vulnscout":
                active = active.cuda()[:, None]
                mask = torch.arange(block.size(1), device=block.device) < active
                states = self.encoder(block, attention_mask=mask).last_hidden_state
                chunks.append((states * mask.unsqueeze(-1)).sum(1) / active)
            else:
                chunks.append(self.encoder(block, attention_mask=block.ne(1)).pooler_output)
        windows = torch.cat(chunks).split(counts)
        if self.method == "vulnscout":
            return torch.stack([vectors.mean(0) for vectors in windows])
        return torch.cat([self.fusion(vectors.unsqueeze(0)) for vectors in windows])

    def forward(self, representations):
        return self.head(representations).flatten()


@torch.no_grad()
def evaluate(model, data, *, features_only=False):
    model.eval()
    counters = data.new_counters()
    outputs, calls = [], 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    for indices in DataLoader(range(len(data)), batch_size=EVAL_BATCH):
        ids, lengths, counts = data.batch(indices.tolist(), EVAL_LIMIT, counters)
        vectors = model.encode(ids, lengths, counts, EVAL_LIMIT)
        outputs.append((vectors if features_only else model(vectors).sigmoid()).cpu())
        calls += math.ceil(len(ids) / EVAL_LIMIT)
    return torch.cat(outputs), counters, calls, perf_counter()-start, torch.cuda.max_memory_allocated()


def save_model(path, state):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def smoke(model, train, criterion):
    # Use the actual longest function and stress the largest training micro-batch.
    longest = sorted(range(len(train)), key=lambda i: train.metadata[i]["original_bpe"], reverse=True)
    index = longest[0]
    counters = train.new_counters()
    model.eval()
    actual_calls = []
    hook = model.encoder.register_forward_pre_hook(lambda module, args: actual_calls.append(len(args[0])))
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        ids, lengths, counts = train.batch([index], EVAL_LIMIT, counters)
        embedding = model.encode(ids, lengths, counts, EVAL_LIMIT)
        if not torch.isfinite(embedding).all() or embedding.norm() == 0:
            raise ValueError("missing or nonfinite longest-function embedding")
    hook.remove()
    assert len(actual_calls) == counters[index, -1] and sum(actual_calls) == len(ids)
    assert max(actual_calls) <= EVAL_LIMIT and counters[index, 1] == train.metadata[index]["original_bpe"]
    inference_peak = torch.cuda.max_memory_allocated()

    model.train()
    if model.method == "vulnscout":
        model.encoder.eval()
        batch = longest[:2]
        with torch.no_grad():
            ids, lengths, counts = train.batch(batch, EVAL_LIMIT, train.new_counters())
            vectors = model.encode(ids, lengths, counts, EVAL_LIMIT)
    else:
        batch = longest[:SEA_MICRO]
    torch.cuda.reset_peak_memory_stats()
    if model.method == "sea":
        ids, lengths, counts = train.batch(batch, SEA_TRAIN_LIMIT, train.new_counters(), training=True)
        vectors = model.encode(ids, lengths, counts, SEA_TRAIN_LIMIT)
    target = train.targets[batch].cuda().float()
    loss = criterion(model(vectors), target)
    loss.backward()
    parameters = {"head": next(model.head.parameters())}
    if model.method == "sea":
        parameters.update(encoder=model.encoder.encoder.layer[0].attention.self.query.weight,
                          attention=model.fusion.embedding_common[0].weight)
    else:
        assert all(p.grad is None and not p.requires_grad for p in model.encoder.parameters())
    norms = {key: p.grad.norm().item() if p.grad is not None else 0. for key, p in parameters.items()}
    if not math.isfinite(loss.item()) or not all(math.isfinite(v) and v > 0 for v in norms.values()):
        raise ValueError(f"invalid smoke gradients: {norms}")
    opt = make_optimizer(model)
    before = {key: p.detach().clone() for key, p in parameters.items()}
    opt.step()
    if any(torch.equal(p, before[k]) for k, p in parameters.items()):
        raise ValueError("smoke optimizer did not update active parameters")
    return dict(longest_function=train.metadata[index], encoded_windows=sum(actual_calls),
                encoder_calls=len(actual_calls), maximum_windows_per_call=max(actual_calls),
                full_forward_peak_allocated_bytes=inference_peak, training_functions=len(batch),
                loss=loss.item(), gradient_norms=norms, training_peak_allocated_bytes=torch.cuda.max_memory_allocated())


def make_optimizer(model):
    if model.method == "vulnscout":
        return torch.optim.Adam(model.head.parameters(), lr=1e-4, weight_decay=1e-5)
    return LOSVERAdamW(model.parameters(), lr=2e-5, eps=1e-8, weight_decay=0.)


def train_epoch(model, data, loader, optimizer, scheduler, criterion, counters, cached):
    model.train()
    if cached is not None:
        model.encoder.eval()
    losses, calls, windows, seen = [], 0, 0, 0
    for indices in loader:
        optimizer.zero_grad(set_to_none=True)
        seen += len(indices)
        targets = data.targets[indices].cuda().float()
        if cached is not None:
            loss = criterion(model(cached[indices].cuda()), targets)
            loss.backward()
            losses.append(loss.detach() * len(indices))
        else:
            for micro, labels in zip(indices.split(SEA_MICRO), targets.split(SEA_MICRO), strict=True):
                ids, lengths, counts = data.batch(micro.tolist(), SEA_TRAIN_LIMIT, counters, training=True)
                loss = criterion(model(model.encode(ids, lengths, counts, SEA_TRAIN_LIMIT)), labels)
                (loss * (len(micro) / len(indices))).backward()
                losses.append(loss.detach() * len(micro))
                calls += math.ceil(len(ids) / SEA_TRAIN_LIMIT)
                windows += len(ids)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if cached is None:
            scheduler.step()
    if cached is not None:
        scheduler.step()
    if seen != len(data):
        raise ValueError("training did not consume the full draw count")
    return torch.stack(losses).sum().item() / seen, calls, windows


@torch.no_grad()
def cached_scores(model, features):
    model.eval()
    return torch.cat([model(block.cuda()).sigmoid().cpu() for block in features.split(SCOUT_BATCH)])


def run(method, smoke_only, seed=SEED):
    root = seed_path(ROOT / "results" / method,seed)
    weights = seed_path(ROOT / "checkpoints" / method,seed)
    if not smoke_only and (root / "metrics.json").exists():
        raise FileExistsError(f"completed result already exists in {root}")
    if torch.cuda.device_count() != 1 or "4090" not in torch.cuda.get_device_name():
        raise ValueError("select a single RTX 4090 with CUDA_VISIBLE_DEVICES")
    directory, revision = REVISIONS[method]
    installed = subprocess.check_output(["git", "-C", str(UPSTREAM / directory), "rev-parse", "HEAD"],
                                        text=True).strip()
    if installed != revision:
        raise ValueError(f"{directory}: expected {revision}, got {installed}")
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    set_seed(seed)
    scout = method == "vulnscout"
    tokenizer = AutoTokenizer.from_pretrained(BASE, local_files_only=True, use_fast=True)
    prep = {}
    start = perf_counter()
    train = WindowData(method, "train", tokenizer)
    prep["train"] = perf_counter()-start
    print(method, "Train prepared", len(train), round(prep["train"], 2), "seconds", flush=True)
    class_counts = train.targets.bincount(minlength=2)
    if scout:
        from Experiment.losses import AdaptiveWeightedFocalLoss

        # RGCL validates a balanced 1:1 label ratio for all official splits.
        ratio = (class_counts[0] / class_counts[1]).item()
        criterion = AdaptiveWeightedFocalLoss(train_ratio=ratio, valid_ratio=1, test_ratio=1)
    else:
        criterion = nn.BCEWithLogitsLoss()
    root.mkdir(parents=True, exist_ok=True)
    weights.mkdir(parents=True, exist_ok=True)
    model = LongCodeModel(method).cuda()
    write_json(root / "initialization.json", model.load_audit)
    if smoke_only or not (weights / "last.pt").exists():
        result = smoke(model, train, criterion)
        write_json(root / "smoke.json", result)
        print(method, "smoke", result, flush=True)
        if smoke_only:
            return
        del model
        torch.cuda.empty_cache()
        set_seed(seed)
        model = LongCodeModel(method).cuda()

    start = perf_counter()
    valid = WindowData(method, "valid", tokenizer)
    prep["valid"] = perf_counter()-start
    batch = SCOUT_BATCH if scout else SEA_BATCH
    epochs = SCOUT_EPOCHS if scout else SEA_EPOCHS
    config = dict(method=method, upstream=revision, seed=seed, epochs=epochs, train_batch=batch,
                  eval_batch=EVAL_BATCH, eval_window_batch=EVAL_LIMIT, train_micro=batch if scout else SEA_MICRO,
                  train_window_batch=0 if scout else SEA_TRAIN_LIMIT, max_train_windows_per_function=None if scout else SEA_TRAIN_WINDOWS,
                  window_length=SCOUT_LENGTH if scout else SEA_WIDTH, stride=SCOUT_OVERLAP if scout else SEA_STEP,
                  stride_type="tokenizer overlap" if scout else "window advance", input_length=SCOUT_LENGTH if scout else SEA_LENGTH,
                  frozen_encoder=scout, base=str(BASE), loss="adaptive focal, alpha=.5 gamma=2" if scout else "function BCE",
                  optimizer="Adam" if scout else "Transformers AdamW", lr=1e-4 if scout else 2e-5,
                  eps=1e-8, weight_decay=1e-5 if scout else 0., clip_norm=None if scout else 1.,
                  schedule="cosine, eta_min=1e-6" if scout else "linear, 10 warmup updates",
                  train_sampler="inverse-frequency replacement" if scout else "shuffle without replacement",
                  patience=SCOUT_PATIENCE if scout else None, threshold=.5, comparison=">",
                  selection="full validation MCC, earliest tie", precision="FP32 SDPA",
                  torch=torch.__version__, transformers=transformers.__version__, python=platform.python_version(),
                  host=platform.node(), gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda)
    write_json(root / "config.json", config)
    extracted, train_cache, valid_cache = {}, None, None
    train_counters, train_calls = train.new_counters(), 0
    if scout:
        for data in (train, valid):
            features, counters, calls, seconds, peak = evaluate(model, data, features_only=True)
            extracted[data.split] = dict(seconds=seconds, peak_allocated_bytes=peak, coverage=coverage(data, counters, calls))
            if data is train:
                train_cache, train_counters, train_calls = features, counters, calls
            else:
                valid_cache = features
            print(method, data.split, "feature extraction", round(seconds, 2), "seconds", flush=True)
        sampler = WeightedRandomSampler(1. / class_counts[train.targets].float(), len(train))
        loader = DataLoader(range(len(train)), batch_size=batch, sampler=sampler)
    else:
        loader = DataLoader(range(len(train)), batch_size=batch, shuffle=True)

    opt = make_optimizer(model)
    schedule = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-6) if scout else
                get_linear_schedule_with_warmup(opt, 10, len(loader)*epochs))
    history = []
    if (weights / "last.pt").exists():
        progress, rng = load_training_state(weights, model.head if scout else model, opt, config, scheduler=schedule)
        history, train_counters, train_calls = progress
        restore_rng(rng)
    best = max((r["validation"]["mcc"] for r in history), default=-math.inf)
    best_epoch = max(history, key=lambda r: r["validation"]["mcc"])["epoch"] if history else 0
    for epoch in range(len(history) + 1, epochs + 1):
        if scout and history and history[-1]["epoch"] - best_epoch >= SCOUT_PATIENCE:
            break
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        loss, calls, windows = train_epoch(model, train, loader, opt, schedule, criterion, train_counters, train_cache)
        torch.cuda.synchronize()
        seconds, peak = perf_counter()-start, torch.cuda.max_memory_allocated()
        train_calls += calls
        opt.zero_grad(set_to_none=True)
        val_start = perf_counter()
        scores = cached_scores(model, valid_cache) if scout else evaluate(model, valid)[0]
        validation = metrics(valid.targets.tolist(), scores.tolist())
        improved = validation["mcc"] > best
        if improved:
            best, best_epoch = validation["mcc"], epoch
            save_model(weights / "best.pt", model.head.state_dict() if scout else model.state_dict())
        history.append(dict(epoch=epoch, loss=loss, seconds=seconds, updates=len(loader), functions=len(train),
                            peak_allocated_bytes=peak, plm_batch_calls=calls, plm_windows=windows,
                            validation=validation, validation_seconds=perf_counter()-val_start))
        save_training_state(weights, model.head if scout else model, opt, config,
                            (history, train_counters, train_calls), scheduler=schedule)
        write_json(root / "epochs.json", history)
        print(method, epoch, "validation MCC", round(validation["mcc"], 6),
              "train seconds", round(seconds, 2), "peak GiB", round(peak / 2**30, 2), flush=True)

    del opt, schedule
    state = torch.load(weights / "best.pt", map_location="cpu", weights_only=True)
    (model.head if scout else model).load_state_dict(state, strict=True)
    model.eval()
    save_rows(root / "train_coverage.csv.gz", train, train_counters)
    result = dict(selected_epoch=best_epoch, epochs_completed=len(history),
                  training_seconds=sum(r["seconds"] for r in history),
                  peak_train_allocated_bytes=max(r["peak_allocated_bytes"] for r in history),
                  feature_extraction=extracted, preparation_seconds=prep,
                  train_coverage=coverage(train, train_counters, train_calls), split={})
    start = perf_counter()
    test = WindowData(method, "test", tokenizer)
    prep["test"] = perf_counter()-start
    for data in (valid, test):
        scores, counters, calls, seconds, peak = evaluate(model, data)
        scores = scores.tolist()
        groups = grouped_metrics(data, scores)
        if data is valid and groups["overall"] != history[best_epoch-1]["validation"]:
            raise ValueError("validation checkpoint replay differs")
        result["split"][data.split] = dict(groups=groups, inference_seconds=seconds, peak_allocated_bytes=peak,
                                           coverage=coverage(data, counters, calls))
        save_rows(root / f"{data.split}.csv.gz", data, counters, scores)
    write_json(root / "metrics.json", result)
    print(method, "complete", best_epoch, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=REVISIONS)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--seed",type=int,choices=SEEDS,default=SEED)
    args = parser.parse_args()
    run(args.method,args.smoke_only,args.seed)
