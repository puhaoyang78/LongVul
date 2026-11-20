"""One reserved position for a retained-code-conditioned summary of omitted BPEs."""
import math
import zlib

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers.models.roberta.modeling_roberta import create_position_ids_from_input_ids

from compression import select
from dataset import BUDGET, DATA, MAX_LENGTH, batches
from rgcl.config import BATCH_SIZE
from rgcl.data import prime_source_rows

CONTEXT_METHODS = ("context_empty", "context_real", "context_random")
BINS = 16
RANK = 32


def source_view(code, ids, offsets):
    pos = select("uniform4", code, ids, offsets)
    if len(ids) > BUDGET and len(pos) == BUDGET:
        pos = pos[:-1]  # Uniform-4 ends with a source token
    visible = np.asarray([i for i in pos if i is not None], dtype=np.int64)
    missing = np.ones(len(ids), dtype=bool)
    missing[visible] = False
    return pos, np.flatnonzero(missing)


def training_unigram(tok):
    vocab_size = len(tok)
    counts = np.zeros(vocab_size, dtype=np.int64)
    for batch in batches(prime_source_rows(DATA.parent, "train"), 64):
        source = tok([r["func"] for r in batch], add_special_tokens=False, verbose=False,
                     return_attention_mask=False)["input_ids"]
        counts += np.bincount(np.concatenate(source), minlength=vocab_size)
    return counts / counts.sum()


def bags(ids, missing, vocab_size, noise, main):
    # Counts are constants; the lookup table used by embedding_bag stays trainable.
    bins = (BINS * missing) // len(ids)
    mass = np.bincount(bins, minlength=BINS)
    if noise is None:
        tokens = np.asarray(ids)[missing]
    else:
        seed = zlib.crc32(np.asarray(main, dtype="<i4").tobytes(), 42)
        tokens = np.random.default_rng(seed).choice(vocab_size, len(missing), p=noise)
    unique, counts = np.unique(bins * vocab_size + tokens, return_counts=True)
    owners = unique // vocab_size
    weight = counts / mass[owners]
    return (torch.from_numpy(unique % vocab_size), torch.from_numpy(weight.astype(np.float32)),
            torch.from_numpy(np.bincount(owners, minlength=BINS)), torch.from_numpy(mass))


def collate(examples):
    x, masks, y, source_masks, slots, contexts = zip(*examples)
    tokens, weights, sizes, mass = zip(*contexts)
    boundaries = F.pad(torch.cat(sizes).cumsum(0), (1, 0))
    return (torch.stack(x), torch.stack(masks), torch.tensor(y), torch.stack(source_masks),
            torch.tensor(slots), torch.cat(tokens), torch.cat(weights), boundaries, torch.stack(mass))


def context_data(rows, method, tok, train, noise=None):
    if (noise is not None) != (method == "context_random"):
        raise ValueError("random context requires a training-only token distribution")
    vocab_size = len(tok)
    inputs = np.full((len(rows), MAX_LENGTH), tok.pad_token_id, dtype=np.int32)
    n = np.empty(len(rows), dtype=np.int32)
    raw = np.empty_like(n)
    labels = [r["target"] for r in rows]
    keys = [r["idx"] for r in rows]
    examples = []
    i = 0
    for batch in batches(rows, 32):
        enc = tok([r["func"] for r in batch], add_special_tokens=False, return_offsets_mapping=True,
                  return_attention_mask=False, verbose=False)
        for row, ids, offsets in zip(batch, enc["input_ids"], enc["offset_mapping"]):
            pos, missing = source_view(row["func"], ids, offsets)
            body = [ids[j] if j is not None else tok.sep_token_id for j in pos]
            main = [tok.cls_token_id, *body]
            slot = len(main) if len(missing) else 0
            if slot:
                main.append(tok.sep_token_id)
            main.append(tok.sep_token_id)
            raw[i], n[i] = len(ids), len(main)
            inputs[i, :len(main)] = main
            if method != "context_empty":
                source_mask = torch.zeros(MAX_LENGTH, dtype=torch.bool)
                source_mask[1:1+len(pos)] = torch.tensor([j is not None for j in pos], dtype=torch.bool)
                examples.append((torch.from_numpy(inputs[i]).long(), torch.arange(MAX_LENGTH).lt(len(main)).long(),
                                 labels[i], source_mask, slot, bags(ids, missing, vocab_size, noise, main)))
            i += 1
    data = inputs, n, raw, labels, keys
    return data, (DataLoader(examples, batch_size=BATCH_SIZE, shuffle=train, collate_fn=collate)
                  if examples else None)


class ContextClassifier(nn.Module):
    def __init__(self, detector):
        super().__init__()
        self.encoder = detector.encoder
        self.classifier = detector.classifier
        with torch.random.fork_rng(devices=[]):
            self.project = nn.Linear(self.encoder.config.hidden_size, RANK, bias=False)

    def forward(self, ids, mask, source_mask, slots, tokens, weights, offsets, mass, *, labels=None):
        table = self.encoder.get_input_embeddings()
        main = table(ids)
        means = F.embedding_bag(tokens, table.weight, offsets, mode="sum", per_sample_weights=weights,
                                include_last_offset=True).reshape(ids.size(0), BINS, -1)
        retained = (main * source_mask.unsqueeze(-1)).sum(1) / source_mask.sum(1, keepdim=True).clamp_min(1)
        # Normalize only the scoring inputs; values remain in the pretrained word space.
        width = (main.size(-1),)
        query = self.project(F.layer_norm(retained, width))
        keys = self.project(F.layer_norm(means, width))
        logits = (keys * query[:, None]).sum(-1) / math.sqrt(RANK)
        logits = logits + mass.clamp_min(1).float().log()
        logits = logits.masked_fill(mass == 0, torch.finfo(logits.dtype).min)
        attention = logits.softmax(-1) * (mass > 0)
        summary = (means * attention.unsqueeze(-1)).sum(1)
        # A slot of zero occurs only when the function has no omitted positions.
        add = F.one_hot(slots, ids.size(1)).to(main.dtype) * slots.ne(0)[:, None]
        main = main + add.unsqueeze(-1) * summary[:, None]
        positions = create_position_ids_from_input_ids(ids, table.padding_idx)
        states = self.encoder(inputs_embeds=main, attention_mask=mask, position_ids=positions)[0]
        score = self.classifier(states)[:, 0]
        out = dict(probability=score.sigmoid(), logits=score)
        if labels is not None:
            out["loss"] = F.binary_cross_entropy_with_logits(score, labels.float())
        return out
