"""Fixed-checkpoint interventions on the single context slot and baseline memory measurement."""
import csv
import gzip

import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup, set_seed

from context import ContextClassifier, training_unigram
from dataset import BUDGET, DATA, MODEL, ROOT, encode_rows, save_predictions, tokenizer
from paired import OUT, SEED, data_loader, groups, predict, prepare, train_epoch
from rgcl.artifacts import read_json, write_json
from rgcl.config import ADAM_EPSILON, BATCH_SIZE, EPOCHS, LEARNING_RATE, WARMUP_FRACTION
from rgcl.data import prime_source_rows
from rgcl.metrics import paired_metrics
from rgcl.model import build_model
from rgcl.training import LOSVERAdamW


def baseline_peak(tok):
    # Historical Uniform-4 lacks allocator telemetry. Measure a full-size forward
    # and two optimizer updates, so the second starts with initialized moments.
    model = build_model("codebert", "plm_classifier", model_source=MODEL).cuda()
    rows = prime_source_rows(DATA.parent, "train")[:2 * BATCH_SIZE]
    data = encode_rows([dict(idx=r["idx"], label=r["target"], func=r["func"]) for r in rows], "uniform4", tok)
    loader = data_loader(data)
    optimizer = LOSVERAdamW(model.parameters(), lr=LEARNING_RATE, eps=ADAM_EPSILON)
    history = read_json(OUT / "uniform4/epochs.json")
    steps = EPOCHS * history[0]["optimizer_steps"]
    scheduler = get_linear_schedule_with_warmup(optimizer, steps * WARMUP_FRACTION, steps)
    torch.cuda.reset_peak_memory_stats()
    train_epoch(model, loader, optimizer, scheduler)
    peak = torch.cuda.max_memory_allocated()
    del optimizer, scheduler
    torch.cuda.reset_peak_memory_stats()
    predict(model, loader)
    return dict(gpu=torch.cuda.get_device_name(), batch_size=BATCH_SIZE, updates=len(loader),
                train_allocated_bytes=peak, inference_allocated_bytes=torch.cuda.max_memory_allocated())


def changes(data, actual, intervention):
    raw = data[2]
    real, other = np.asarray(actual), np.asarray(intervention)
    labels = np.asarray(data[3])
    result = {}
    for name, mask in (("long", raw > BUDGET), ("short", raw <= BUDGET)):
        a, b, y = real[mask], other[mask], labels[mask]
        correct_a = (a > 0.5) == y
        correct_b = (b > 0.5) == y
        result[name] = dict(samples=len(a), mean_abs_probability_change=float(np.mean(np.abs(a-b))),
                            max_abs_probability_change=float(np.max(np.abs(a-b))),
                            correct_only_real=int(np.sum(correct_a & ~correct_b)),
                            correct_only_intervention=int(np.sum(~correct_a & correct_b)))
    return result


def same_visible_pairs(data, scores):
    # RGCL orders each pair as vulnerable, then repaired. This is post-prediction.
    inputs = data[0]
    long = data[2] > BUDGET
    same = (inputs[::2] == inputs[1::2]).all(axis=1) & (long[::2] | long[1::2])
    a, b = np.asarray(scores).reshape(-1, 2)[same].T
    return dict(long_pairs=int((long[::2] | long[1::2]).sum()), same_visible=int(same.sum()),
                vulnerable_higher=int((a > b).sum()), repaired_higher=int((a < b).sum()),
                equal_probability=int((a == b).sum()), both_correct=int(((a > .5) & (b <= .5)).sum()))


def run():
    if torch.cuda.device_count() != 1 or "4090" not in torch.cuda.get_device_name():
        raise ValueError("select one RTX 4090 via CUDA_VISIBLE_DEVICES")
    set_seed(SEED)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    tok = tokenizer()
    output = OUT / "context_interventions"
    result = dict(uniform4_memory_probe=baseline_peak(tok), split={})
    detector = build_model("codebert", "plm_classifier", model_source=MODEL)
    real = ContextClassifier(detector).cuda()
    # Both objects reference the same encoder and classification head.
    real.load_state_dict(torch.load(ROOT / "checkpoints/paired/context_real/best.pt", map_location="cpu", weights_only=True))
    result["real_selected_epoch"] = read_json(OUT / "context_real/metrics.json")["selected_epoch"]
    noise = training_unigram(tok)
    for split in ("valid", "test"):
        rows, data, loader, real_prep = prepare(split, "context_real", tok)
        actual = predict(real, loader)
        with gzip.open(OUT / "context_real" / f"{split}.csv.gz", "rt") as stream:
            original = list(csv.DictReader(stream))
        if data[4] != [int(r["idx"]) for r in original]:
            raise ValueError(f"{split} sample mismatch")
        np.testing.assert_allclose(actual, [float(r["probability"]) for r in original], atol=1e-6, rtol=0)
        result["split"][split] = {"real_preparation_seconds": real_prep,
                                  "same_visible_pairs": {"real": same_visible_pairs(data, actual)}}
        for method, classifier, distribution in (("context_empty", detector, None), ("context_random", real, noise)):
            _, other, loader, _ = prepare(split, method, tok, distribution)
            if any(not np.array_equal(x, y) for x, y in zip(data, other)):
                raise ValueError(f"{method} changed the retained input or members")
            scores = predict(classifier, loader)
            delta = changes(data, actual, scores)
            if delta["short"]["max_abs_probability_change"] > 1e-6:
                raise ValueError("intervention changed a function without omissions")
            result["split"][split][method] = dict(groups=groups(data, scores), pairs=paired_metrics(rows, scores),
                                                  compared_with_real=delta)
            result["split"][split]["same_visible_pairs"][method] = same_visible_pairs(data, scores)
            save_predictions(output / f"{split}_{method}.csv.gz", data, scores)
    write_json(output / "metrics.json", result)


if __name__ == "__main__":
    run()
