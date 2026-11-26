"""Full-function block pooling, masked indexing, and fixed-budget fine encoding."""
from itertools import accumulate
import math
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from compaction import collate
from coarse_retrieval import part_overlap
from context_selection import merge
from locator_quality import context_encoder_batches, whole_windows
from repair_oracle import compact_edits

BLOCK = 32
QUERY_HEADS = 4
INDEX_WIDTH = 128


def encoder_window_ends(source):
    ends = [hi for _,hi,_ in whole_windows(source)]
    if not ends or ends[-1] != len(source.ids):
        raise ValueError("Context windows must cover the complete, nonempty Simple input")
    return ends


def encode_blocks(model, sources, tokenizer, window_ends, blocks):
    """Pool each block from every overlapping nonpadding encoder window."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    offsets = list(accumulate(map(len,blocks),initial=0))
    width = model.config.hidden_size
    sums = torch.zeros((offsets[-1],width),dtype=torch.float32)
    counts = torch.zeros(offsets[-1],dtype=torch.int32)
    padding = [0]*len(sources)
    windows = ((source,lo,hi,i) for i,(source,ends) in enumerate(zip(sources,window_ends,strict=True))
               for lo,hi in zip([0,*ends[:-1]],ends))
    calls = sequences = 0
    for batch,prefix,padded in context_encoder_batches(model,tokenizer,windows):
        for _,_,_,i in batch:
            padding[i] += padded//len(batch)
        requests = [(offsets[i]+block,row,1+a-lo,1+b-lo)
                    for row,(_,lo,hi,i) in enumerate(batch) for block,a,b in part_overlap(lo,hi,blocks[i])]
        index,row,a,b = torch.tensor(requests,dtype=torch.long).T
        row_cuda,a_cuda,b_cuda = (x.cuda() for x in (row,a,b))
        sums.index_add_(0,index,(prefix[row_cuda,b_cuda]-prefix[row_cuda,a_cuda]).cpu())
        counts.index_add_(0,index,(b-a).int())
        calls += 1
        sequences += len(batch)
    sizes = torch.tensor([b-a for ends in blocks for a,b in zip([0,*ends[:-1]],ends)],dtype=torch.int32)
    if not torch.equal(sizes,counts):
        raise ValueError("Encoder window pooling lost or duplicated source tokens")
    vectors = sums.div_(sizes[:,None]).half()
    torch.cuda.synchronize()
    body = sum(len(s.ids) for s in sources)
    return vectors,dict(seconds=perf_counter()-started,body_bpe=body,blocks=len(vectors),windows=sequences,batch_calls=calls,
                        unpadded_positions=body+2*sequences,padded_positions=sum(padding),
                        per_function_padded_positions=padding,cached_vector_bytes=vectors.nbytes,peak_allocated_bytes=torch.cuda.max_memory_allocated())


def block_mass(ends, spans):
    counts = torch.zeros(len(ends),dtype=torch.float32)
    for lo,hi in merge(spans):
        for i,a,b in part_overlap(lo,hi,ends):
            counts[i] += b-a
    return counts


def repair_block_mass(source, ends, edits, kinds):
    mapped = compact_edits(source.keep,edits)
    return block_mass(ends,(span for kind in kinds for span in mapped[kind]))


class BlockData(Dataset):
    def __init__(self, rows, ends, vectors, repair_targets=None):
        self.offsets = list(accumulate(map(len,ends),initial=0))
        self.targets = torch.tensor([r["target"] for r in rows])
        self.vectors = vectors
        self.repair_targets = repair_targets
        if self.offsets[-1] != len(vectors) or len(rows) != len(ends):
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

    def forward(self, blocks, valid, global_state, *, per_head=False):
        query = self.query(F.layer_norm(global_state, (global_state.shape[-1],))).reshape(-1, QUERY_HEADS, INDEX_WIDTH)
        key = self.key(F.layer_norm(blocks, (blocks.shape[-1],)))
        score = torch.bmm(query, key.transpose(1,2))/math.sqrt(INDEX_WIDTH)
        heads = score.masked_fill(~valid[:, None, :], -torch.inf).softmax(-1)
        return heads if per_head else heads.mean(1)


class CoarseClassifier(nn.Module):
    def __init__(self, width, supervision, repair_weight=0., *, bypass=True):
        super().__init__()
        self.width = width
        self.supervision = supervision
        self.repair_weight = repair_weight
        self.bypass = bypass
        inputs = 3*width if bypass else width
        self.representation = nn.Sequential(nn.LayerNorm(inputs), nn.Linear(inputs,width), nn.Tanh())
        self.head = nn.Sequential(nn.Dropout(.1), nn.Linear(width,1))
        self.indexer = None if supervision == "global" else ContextIndexer(width)

    def auxiliary(self, g, z):
        features = torch.cat((g,z),dim=-1) if self.bypass else z
        coarse = self.representation(features)
        return coarse,self.head(coarse).squeeze(-1)

    def forward(self, batch, *, details=False, inspect=False):
        blocks, valid = batch["blocks"].float(), batch["valid"]
        g = global_pool(blocks, valid)
        if self.indexer is None:
            z = g[:, :self.width]
        else:
            scores = self.indexer(blocks, valid, g, per_head=inspect)
            p = scores.mean(1) if inspect else scores
            pooling = p.detach() if self.supervision == "repair" else p
            z = torch.bmm(pooling[:, None, :], blocks).squeeze(1)
        coarse,logits = self.auxiliary(g,z)
        output = dict(logits=logits, probability=logits.sigmoid())
        if details:
            output["coarse"] = coarse
            if self.indexer is not None:
                output.update(importance=p, global_state=g, weighted=z)
        if inspect and self.indexer is not None:
            _,uniform_logits = self.auxiliary(g,g[:, :self.width])
            output.update(head_importance=scores, uniform_probability=uniform_logits.sigmoid())
        if "labels" in batch:
            bce = F.binary_cross_entropy_with_logits(logits, batch["labels"].float())
            loss, parts = bce, dict(train_bce=bce)
            if self.repair_weight:
                q = batch["repair"]
                kl = repair_kl(p,q)
                parts["train_repair_kl"] = kl
                loss = bce+self.repair_weight*kl
            output.update(loss=loss, loss_components=parts)
        return output


def repair_kl(p, q):
    return F.kl_div(p.clamp_min(1e-12).log(), q, reduction="none").sum()/q.any(-1).sum().clamp_min(1)


def collate_fine(examples):
    batch = collate(examples)
    batch["coarse"] = torch.stack([ex["coarse"] for ex in examples])
    return batch


class FusionClassifier(nn.Module):
    def __init__(self, fine, fusion="original"):
        super().__init__()
        self.fine = fine
        width = fine.config.hidden_size
        self.fusion = nn.Linear(2*width if fusion == "global" else width, width, bias=False)
        nn.init.zeros_(self.fusion.weight)

    def representations(self, batch):
        ids = batch["fixed"][:,0].long()
        mask = torch.arange(ids.shape[1], device=ids.device)[None,:] < batch["lengths"][:,None]
        cls = self.fine.encoder(input_ids=ids, attention_mask=mask.long()).last_hidden_state[:,0]
        return cls,self.fusion(batch["coarse"])

    def forward(self, batch, *, details=False):
        cls,residual = self.representations(batch)
        mixed = cls+residual
        logits = self.fine.classifier(mixed[:,None])[:,0]
        output = dict(logits=logits, probability=logits.sigmoid())
        if "labels" in batch:
            output["loss"] = F.binary_cross_entropy_with_logits(logits, batch["labels"].float())
        return output
