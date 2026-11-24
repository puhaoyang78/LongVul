"""Full-function block pooling, masked indexing, and fixed-budget fine encoding."""
from itertools import accumulate
import math
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from compaction import BUDGET, collate
from context_selection import merge, occupied
from locator_quality import context_encoder_batches, whole_windows
from repair_oracle import compact_edits

BLOCK = 32
QUERY_HEADS = 4
INDEX_WIDTH = 128


def block_offsets(sources):
    return list(accumulate((math.ceil(len(s.ids)/BLOCK) for s in sources), initial=0))


def encoder_window_ends(source):
    ends = [hi for _,hi,_ in whole_windows(source)]
    if not ends or ends[-1] != len(source.ids):
        raise ValueError("Context windows must cover the complete, nonempty Simple input")
    return ends


def window_blocks(lo, hi, first):
    return [(first+i, max(lo, i*BLOCK), min(hi, (i+1)*BLOCK)) for i in range(lo//BLOCK, (hi-1)//BLOCK+1)]


def encode_blocks(model, sources, tokenizer, window_ends):
    """Pool each 32-BPE block exactly once, including pieces split by windows."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    offsets = block_offsets(sources)
    sums = torch.zeros((offsets[-1], model.config.hidden_size), dtype=torch.float32)
    counts = torch.zeros(offsets[-1], dtype=torch.int32)
    per_function_padding = [0]*len(sources)
    windows = ((s,lo,hi,(i, offsets[i])) for i,(s,ends) in enumerate(zip(sources,window_ends,strict=True))
               for lo,hi in zip([0,*ends[:-1]],ends))
    calls = sequences = 0
    for batch, prefix, padded in context_encoder_batches(model, tokenizer, windows):
        requests = []
        width = padded//len(batch)
        for row, (_, lo, hi, (function, first)) in enumerate(batch):
            requests.extend((index,row,1+a-lo,1+b-lo) for index,a,b in window_blocks(lo,hi,first))
            per_function_padding[function] += width
        index, row, a, b = torch.tensor(requests, dtype=torch.long).T
        row_cuda, a_cuda, b_cuda = (x.cuda() for x in (row,a,b))
        sums.index_add_(0, index, (prefix[row_cuda,b_cuda]-prefix[row_cuda,a_cuda]).cpu())
        counts.index_add_(0, index, (b-a).int())
        calls += 1
        sequences += len(batch)
    expected = torch.full_like(counts, BLOCK)
    expected[torch.tensor(offsets[1:])-1] = torch.tensor([(len(s.ids)-1)%BLOCK+1 for s in sources], dtype=counts.dtype)
    if not torch.equal(counts, expected):
        raise ValueError("Context window pooling lost or duplicated source tokens")
    vectors = sums.div_(counts[:, None]).half()
    torch.cuda.synchronize()
    body = int(counts.sum())
    return vectors, dict(seconds=perf_counter()-started, body_bpe=body, blocks=len(vectors), windows=sequences,
                         batch_calls=calls, unpadded_positions=body+2*sequences, padded_positions=sum(per_function_padding),
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(), cached_vector_bytes=vectors.nbytes,
                         per_function_padded_positions=per_function_padding)


def repair_block_mass(source, edits, kinds):
    mapped = compact_edits(source.keep, edits)
    counts = torch.zeros(math.ceil(len(source.ids)/BLOCK), dtype=torch.float32)
    for lo,hi in merge(span for kind in kinds for span in mapped[kind]):
        for index,a,b in window_blocks(lo,hi,0):
            counts[index] += b-a
    return counts


class BlockData(Dataset):
    def __init__(self, rows, sources, vectors, repair_targets=None):
        self.offsets = block_offsets(sources)
        self.targets = torch.tensor([r["target"] for r in rows])
        self.vectors = vectors
        self.repair_targets = repair_targets
        if self.offsets[-1] != len(vectors) or len(rows) != len(sources):
            raise ValueError("Cached block vectors do not align with source membership")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        a,b = self.offsets[i:i+2]
        return self.vectors[a:b], self.targets[i], None if self.repair_targets is None else self.repair_targets[i]


def collate_blocks(examples):
    vectors, targets, repair = zip(*examples, strict=True)
    padded = pad_sequence(vectors, batch_first=True)
    valid = torch.arange(padded.shape[1])[None, :] < torch.tensor([len(v) for v in vectors])[:, None]
    batch = dict(blocks=padded, valid=valid, labels=torch.stack(targets))
    if repair[0] is not None:
        batch["repair"] = pad_sequence(repair, batch_first=True)
    return batch


def global_pool(blocks, valid):
    mean = blocks.masked_fill(~valid[..., None], 0).sum(1)/valid.sum(1, keepdim=True)
    maximum = blocks.masked_fill(~valid[..., None], -torch.inf).amax(1)
    return torch.cat((mean, maximum), dim=-1)


class ContextIndexer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.query = nn.Linear(2*width, QUERY_HEADS*INDEX_WIDTH, bias=False)
        self.key = nn.Linear(width, INDEX_WIDTH, bias=False)

    def forward(self, blocks, valid, global_state):
        query = self.query(F.layer_norm(global_state, (global_state.shape[-1],))).reshape(-1, QUERY_HEADS, INDEX_WIDTH)
        key = self.key(F.layer_norm(blocks, (blocks.shape[-1],)))
        score = torch.bmm(query, key.transpose(1,2))/math.sqrt(INDEX_WIDTH)
        return score.masked_fill(~valid[:, None, :], -torch.inf).softmax(-1).mean(1)


class CoarseClassifier(nn.Module):
    def __init__(self, width, supervision, repair_weight=0.):
        super().__init__()
        self.width = width
        self.supervision = supervision
        self.repair_weight = repair_weight
        self.representation = nn.Sequential(nn.LayerNorm(3*width), nn.Linear(3*width,width), nn.Tanh())
        self.head = nn.Sequential(nn.Dropout(.1), nn.Linear(width,1))
        self.indexer = None if supervision == "global" else ContextIndexer(width)

    def forward(self, batch, *, details=False):
        blocks, valid = batch["blocks"].float(), batch["valid"]
        g = global_pool(blocks, valid)
        if self.indexer is None:
            z = g[:, :self.width]
        else:
            p = self.indexer(blocks, valid, g)
            pooling = p.detach() if self.supervision == "repair" else p
            z = torch.bmm(pooling[:, None, :], blocks).squeeze(1)
        coarse = self.representation(torch.cat((g,z), dim=-1))
        logits = self.head(coarse).squeeze(-1)
        output = dict(logits=logits, probability=logits.sigmoid())
        if details:
            output["coarse"] = coarse
            if self.indexer is not None:
                output["importance"] = p
        if "labels" in batch:
            bce = F.binary_cross_entropy_with_logits(logits, batch["labels"].float())
            loss, parts = bce, dict(train_bce=bce)
            if self.repair_weight:
                q = batch["repair"]
                kl = F.kl_div(p.clamp_min(1e-12).log(), q, reduction="none").sum()/q.any(-1).sum().clamp_min(1)
                parts["train_repair_kl"] = kl
                loss = bce+self.repair_weight*kl
            output.update(loss=loss, loss_components=parts)
        return output


def retrieve(scores, length, *, budget=BUDGET):
    if len(scores) != math.ceil(length/BLOCK):
        raise ValueError("Indexer scores do not align with Simple code blocks")
    if length <= budget:
        return [(0,length)]
    selected = []
    for index in np.argsort(-scores, kind="stable"):
        lo, hi = int(index)*BLOCK, min(length, (int(index)+1)*BLOCK)
        merged = merge([*selected, (lo,hi)])
        if occupied(merged) <= budget:
            selected = merged
        else:
            room = budget-occupied(selected)
            width_left = room-int(bool(selected) and not any(b==lo for _,b in selected))
            width_right = room-int(bool(selected) and not any(a==hi for a,_ in selected))
            if max(width_left,width_right) > 0:
                span = (lo,lo+width_left) if width_left >= width_right else (hi-width_right,hi)
                selected = merge([*selected,span])
        if occupied(selected) == budget:
            break
    if occupied(selected) != budget:
        raise ValueError("Retrieval could not fill its exact token and SEP budget")
    return selected


def collate_fine(examples):
    batch = collate(examples)
    batch["coarse"] = torch.stack([ex["coarse"] for ex in examples])
    return batch


class FusionClassifier(nn.Module):
    def __init__(self, fine):
        super().__init__()
        self.fine = fine
        self.fusion = nn.Linear(fine.config.hidden_size, fine.config.hidden_size, bias=False)
        nn.init.zeros_(self.fusion.weight)

    def forward(self, batch, *, details=False):
        ids = batch["fixed"][:,0].long()
        mask = torch.arange(ids.shape[1], device=ids.device)[None,:] < batch["lengths"][:,None]
        cls = self.fine.encoder(input_ids=ids, attention_mask=mask.long()).last_hidden_state[:,0]
        mixed = cls+self.fusion(batch["coarse"])
        logits = self.fine.classifier(mixed[:,None])[:,0]
        output = dict(logits=logits, probability=logits.sigmoid())
        if "labels" in batch:
            output["loss"] = F.binary_cross_entropy_with_logits(logits, batch["labels"].float())
        return output
