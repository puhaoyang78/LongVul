"""Statement-level repair relatedness from frozen UniXcoder word embeddings."""
from itertools import accumulate
from time import perf_counter

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import set_seed

from budget_selection import region_ids
from paired import SEED, UNIX_BASE, LOSVERAdamW, load_training_state, restore_rng, save_training_state
from rgcl.artifacts import write_json

PARTS = 4
CHANNELS = 64
EPOCHS = 16
BATCH = 512
LR = .001


class RegionHead(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.projection = nn.Linear(width, CHANNELS)
        self.conv = nn.Conv1d(CHANNELS, CHANNELS, 3, padding=1)
        self.score = nn.Linear(CHANNELS*2, 1)

    def forward(self, bags):
        local = F.gelu(self.projection(bags)).reshape(-1, PARTS, CHANNELS).transpose(1, 2)
        encoded = F.gelu(self.conv(local))
        return self.score(torch.cat([encoded.mean(-1), encoded.amax(-1)], dim=1)).squeeze(-1)


class Localizer(RegionHead):
    def __init__(self):
        state = torch.load(UNIX_BASE / "pytorch_model.bin", weights_only=True, map_location="cpu", mmap=True)
        table = state["embeddings.word_embeddings.weight"]
        super().__init__(table.shape[1])
        self.register_buffer("tokens", table, persistent=False)

    def forward(self, tokens, offsets):
        # Four contiguous embedding bags see every token without an N x 768 tensor.
        bags = F.embedding_bag(tokens, self.tokens, offsets, mode="mean", include_last_offset=True)
        return super().forward(bags)


class Regions:
    def __init__(self, sources, truth=None, *, vectors=None):
        self.sources = sources
        self.limits = list(accumulate((len(s.regions) for s in sources), initial=0))
        self.truth = np.concatenate(truth) if truth is not None else None
        self.vectors = vectors
        self.index = [(i, r.start, r.end) for i, source in enumerate(sources) for r in source.regions] if vectors is None else None
        if vectors is not None and (len(vectors) != len(self) or vectors.ndim != 3 or vectors.shape[1] != PARTS):
            raise ValueError("Contextual vectors must align with the ordered source regions")

    def __len__(self):
        return self.limits[-1]

    def __getitem__(self, i):
        return i

    def collate(self, indices):
        labels = torch.from_numpy(self.truth[indices]).float() if self.truth is not None else None
        if self.vectors is not None:
            return self.vectors[indices], None, labels
        chunks, lengths = [], []
        for index in indices:
            i, a, b = self.index[index]
            chunks.append(self.sources[i].ids[a:b])
            lengths.extend((b-a)*k//PARTS - (b-a)*(k-1)//PARTS for k in range(1, PARTS+1))
        ids = torch.from_numpy(np.concatenate(chunks))
        offsets = torch.tensor(list(accumulate(lengths, initial=0)), dtype=torch.int32)
        return ids, offsets, labels

    def loader(self, *, shuffle=False):
        return DataLoader(self, batch_size=BATCH, shuffle=shuffle, collate_fn=self.collate)

    def regroup(self, values):
        return [values[a:b] for a,b in zip(self.limits, self.limits[1:])]


def weak_targets(sources, edits):
    labels = []
    for source, annotation in zip(sources, edits, strict=True):
        included = region_ids(source.regions, [s for spans in annotation.values() for s in spans], touch=True)
        labels.append(np.asarray([i in included for i in range(len(source.regions))], dtype=bool))
    return labels


def region_logits(model, values, offsets):
    values = values.cuda()
    return model(values, offsets.cuda()) if offsets is not None else model(values.float())


@torch.inference_mode()
def score_regions(model, data):
    model.eval()
    torch.cuda.synchronize()
    started = perf_counter()
    values = [region_logits(model, x, offsets).sigmoid().cpu().numpy() for x, offsets, _ in data.loader()]
    scores = np.concatenate(values) if values else np.empty(0, dtype=np.float32)
    if not np.isfinite(scores).all():
        raise ValueError("Nonfinite localizer prediction")
    return scores, perf_counter()-started


def region_metrics(predictions, labels, *, ks=(1, 5, 10)):
    y, p = np.concatenate(labels), np.concatenate(predictions)
    positives = int(y.sum())
    result = dict(functions=len(labels), regions=len(y), positive_regions=positives,
                  functions_with_positive=sum(bool(label.any()) for label in labels),
                  ap=float(average_precision_score(y, p)), auc=float(roc_auc_score(y, p)))
    for k in ks:
        hits = [int(label[np.argsort(-score, kind="stable")[:k]].sum())
                for score, label in zip(predictions, labels, strict=True) if label.any()]
        result[f"top{k}"] = dict(hit_functions=sum(x > 0 for x in hits), recall=sum(hits)/positives,
                                hit_rate=sum(x > 0 for x in hits)/len(hits))
    return result


def train_localizer(sources, targets, weights, results, *, vectors=None, feature_config=None):
    """Train/Valid weak labels are supplied explicitly; scores require only source."""
    weights.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    train, valid = (Regions(sources[split], targets[split], vectors=vectors[split] if vectors is not None else None)
                    for split in ("train", "valid"))
    positive = int(train.truth.sum())
    if not 0 < positive < len(train):
        raise ValueError("Repair localizer requires positive and negative training regions")
    ratio = (len(train)-positive)/positive
    config = dict(seed=SEED, batch=BATCH, epochs=EPOCHS, lr=LR, parts=PARTS, channels=CHANNELS,
                  source_model=str(UNIX_BASE), frozen_input_embeddings=True, pos_weight=ratio,
                  target="region intersects same-side historical edit or empty-side neighboring lines",
                  selection="full validation region average precision, earliest tie")
    if feature_config is not None:
        config.update(feature_config)
    write_json(results / "config.json", config)
    set_seed(SEED)
    model = (Localizer() if vectors is None else RegionHead(train.vectors.shape[-1])).cuda()
    optimizer = LOSVERAdamW(model.parameters(), lr=LR, eps=1e-8)
    history = []
    if (weights / "last.pt").exists():
        history, state = load_training_state(weights, model, optimizer, config)
        restore_rng(state)
    best = max((x["valid_ap"] for x in history), default=-1.)
    pos_weight = torch.tensor(ratio, device="cuda")
    for epoch in range(len(history)+1, EPOCHS+1):
        torch.cuda.reset_peak_memory_stats()
        model.train()
        before = perf_counter()
        losses = []
        for inputs, offsets, labels in train.loader(shuffle=True):
            logits = region_logits(model, inputs, offsets)
            loss = F.binary_cross_entropy_with_logits(logits, labels.cuda(), pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(loss.detach())
        loss = float(torch.stack(losses).mean())
        train_seconds, peak = perf_counter()-before, torch.cuda.max_memory_allocated()
        p, valid_seconds = score_regions(model, valid)
        ap = float(average_precision_score(valid.truth, p))
        if ap > best:
            best = ap
            tmp = weights / "best.tmp"
            torch.save(model.state_dict(), tmp)
            tmp.replace(weights / "best.pt")
        history.append(dict(epoch=epoch, loss=loss, train_seconds=train_seconds, valid_seconds=valid_seconds,
                            valid_ap=ap, peak_allocated_bytes=peak))
        save_training_state(weights, model, optimizer, config, history)
        write_json(results / "epochs.json", history)
        print("localizer", epoch, f"loss={loss:.4f} valid_ap={ap:.4f} train_s={train_seconds:.1f}", flush=True)
    del optimizer
    model.load_state_dict(torch.load(weights / "best.pt", weights_only=True, map_location="cpu"), strict=True)
    state = dict(config=config, selected_epoch=max(history, key=lambda r: r["valid_ap"])["epoch"],
                 training_seconds=sum(r["train_seconds"] for r in history), invoked_seconds=perf_counter()-started,
                 peak_allocated_bytes=max(r["peak_allocated_bytes"] for r in history), splits={})
    predictions = {}
    for split, data in (("train", train), ("valid", valid)):
        p, seconds = score_regions(model, data)
        predictions[split] = data.regroup(p)
        state["splits"][split] = dict(region_metrics(predictions[split], targets[split]), inference_seconds=seconds)
    return model, predictions, state
