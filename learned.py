"""Learn a view's change in downstream BCE from randomized train-only comparisons."""
from bisect import bisect_left, bisect_right
from time import perf_counter

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
import torch

from compression import four_windows, line_layout
from dataset import BUDGET, DATA, MAX_LENGTH, MODEL, ROOT, batches, save_predictions, tokenizer
from paired import OUT, SEED, data_loader, groups, predict
from rgcl.artifacts import read_json, write_json
from rgcl.data import prime_source_rows
from rgcl.metrics import paired_metrics
from rgcl.model import build_model

DIM = 65536


def alternatives(code, ids, offsets, sep):
    if len(ids) <= BUDGET:
        return ids, []
    keep, boundaries = line_layout(code, ids, offsets)
    ref_pos = four_windows(ids, keep, boundaries, uniform=True)
    reference = [sep if i is None else ids[i] for i in ref_pos]
    if len(keep) <= BUDGET:
        return reference, []
    # Tile the compacted source at line boundaries. Anchor the final window
    # at the end so that a short remainder does not receive an unfair budget.
    local = []
    seq = np.asarray(ids)[keep]
    a = 0
    while a < len(seq):
        b = boundaries[bisect_right(boundaries, a + BUDGET) - 1]
        if b == len(seq):
            a = boundaries[bisect_left(boundaries, b - BUDGET)]
        local.append(seq[a:b])
        a = b
    return reference, local


def features(view):
    ids = np.asarray(view, dtype=np.int64)
    # Distribution of ordered adjacent BPE pairs; no vocabulary fit on held-out code.
    previous = np.empty_like(ids)
    previous[0] = 50265  # outside CodeBERT vocabulary
    previous[1:] = ids[:-1]
    return ((previous * 1000003) ^ (ids * 9176 + ids // 7)) % DIM


def feature_matrix(views):
    cols = [features(v) for v in views]
    size = np.array([len(x) for x in cols])
    return csr_matrix((np.repeat(1. / size, size), np.concatenate(cols),
                       np.r_[0, size.cumsum()]), shape=(len(views), DIM))


def gain(reference_features, candidate, weights, bias):
    return weights[features(candidate)].mean() - reference_features + bias


def choose(reference, local, weights, bias):
    if not local:
        return reference, False
    base = weights[features(reference)].mean()
    gains = [gain(base, v, weights, bias) for v in local]
    i = int(np.argmax(gains))
    return (local[i], True) if gains[i] > 0 else (reference, False)


def loss(labels, probabilities):
    p = np.clip(probabilities, 1e-7, 1-1e-7)
    return -np.where(np.asarray(labels) == 1, np.log(p), np.log1p(-p))


def pack(rows, original, views, tok):
    ids = np.full((len(rows), MAX_LENGTH), tok.pad_token_id, dtype=np.int32)
    sizes = np.array([len(v)+2 for v in views])
    if sizes.max() > MAX_LENGTH:
        raise ValueError("selected view exceeds 512")
    for i, v in enumerate(views):
        ids[i, 0] = tok.cls_token_id
        ids[i, 1:len(v)+1] = v
        ids[i, len(v)+1] = tok.sep_token_id
    return ids, sizes, np.array(original), [r["target"] for r in rows], [r["idx"] for r in rows]


def encoded(split, tok):
    for batch in batches(prime_source_rows(DATA.parent, split), 32):
        codes = [r["func"] for r in batch]
        out = tok(codes, add_special_tokens=False, return_attention_mask=False,
                  return_offsets_mapping=True, verbose=False)
        for row, ids, offsets in zip(batch, out["input_ids"], out["offset_mapping"]):
            ref, local = alternatives(row["func"], ids, offsets, tok.sep_token_id)
            yield row, len(ids), ref, local


def sample_probes(split, tok, model):
    rng = np.random.default_rng(SEED)
    rows, original, reference, sampled = [], [], [], []
    for row, n, ref, local in encoded(split, tok):
        if local:
            rows.append(row)
            original.append(n)
            reference.append(ref)
            sampled.append(local[rng.integers(len(local))])
    scores = [predict(model, data_loader(pack(rows, original, view, tok))) for view in (reference, sampled)]
    X = feature_matrix(sampled) - feature_matrix(reference)
    y = [r["target"] for r in rows]
    target = loss(y, scores[0]) - loss(y, scores[1])
    return X, target


def evaluate(split, model, tok, weights, bias, directory):
    start = perf_counter()
    rows, original, views, changed = [], [], [], []
    for row, n, ref, local in encoded(split, tok):
        view, use_local = choose(ref, local, weights, bias)
        rows.append(row)
        original.append(n)
        views.append(view)
        changed.append(use_local)
    data = pack(rows, original, views, tok)
    loader = data_loader(data)
    prep = perf_counter() - start
    torch.cuda.synchronize()
    start = perf_counter()
    scores = predict(model, loader)
    elapsed = perf_counter() - start
    save_predictions(directory / f"{split}.csv.gz", data, scores)
    return dict(groups=groups(data, scores), pairs=paired_metrics(rows, scores), focused=sum(changed),
                preparation_seconds=prep, inference_seconds=elapsed, model_tokens=int(data[1].sum()),
                mean_bce=float(loss(data[3], scores).mean()))


def run():
    if torch.cuda.device_count() != 1 or "4090" not in torch.cuda.get_device_name():
        raise ValueError("select one RTX 4090 via CUDA_VISIBLE_DEVICES")
    torch.set_num_threads(4)
    # Use the validation winner of the four completed controls as the frozen classifier.
    methods = ("head", "head_tail", "uniform4", "compact")
    best = max(methods, key=lambda m: read_json(OUT/m/"metrics.json")["split"]["valid"]["groups"]["overall"]["mcc"])
    if best != "uniform4":
        raise ValueError("This experiment uses the stage-one validation winner Uniform-4")
    tok = tokenizer()
    model = build_model("codebert", "plm_classifier", model_source=MODEL, pretrained=False)
    root = ROOT / "checkpoints/paired"
    model.load_state_dict(torch.load(root/best/"best.pt", map_location="cpu", weights_only=True))
    model.cuda().eval()
    start = perf_counter()
    X, target = sample_probes("train", tok, model)
    train_seconds = perf_counter() - start
    V, valid_gain = sample_probes("valid", tok, model)
    permuted = np.random.default_rng(SEED).permutation(target)
    for name, y in (("learned", target), ("shuffled", permuted)):
        directory = OUT / name
        start = perf_counter()
        reg = Ridge(alpha=1., solver="lsqr", tol=1e-6).fit(X, y)
        fits = perf_counter() - start
        path = root / name
        path.mkdir(exist_ok=True)
        np.savez(path/"selector.npz", weights=reg.coef_, bias=reg.intercept_)
        pred = reg.predict(V)
        corr = spearmanr(pred, valid_gain).statistic
        result = dict(config=dict(method=name, base=best, frozen=True, seed=SEED, alpha=1.,
                                  hashed_bigram_dimensions=DIM, threshold=.5, comparison=">"),
                      probe_train_functions=len(y), probe_train_encoder_calls=2*len(y),
                      probe_construction_seconds=train_seconds, fit_seconds=fits,
                      valid_probe=dict(functions=len(pred), spearman=float(corr) if np.isfinite(corr) else None,
                                       mse=float(np.mean((pred-valid_gain)**2)),
                                       train_mean_predictor_mse=float(np.mean((target.mean()-valid_gain)**2))),
                      split={s:evaluate(s, model, tok, reg.coef_, reg.intercept_, directory)
                             for s in ("valid","test")})
        write_json(directory/"metrics.json",result)
        print(name,result["valid_probe"],result["split"]["valid"]["groups"]["long"]["mcc"],flush=True)


if __name__ == "__main__":
    run()
