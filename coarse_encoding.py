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
from coarse_retrieval import block_ends, center_blocks, part_overlap
from context_selection import merge
from locator_quality import context_encoder_batches, whole_windows
from repair_oracle import EDIT_TYPES, GAP_TYPES, compact_edits
from relation_attention import RELATION_TYPES, additive_relation_mask, relation_bits

BLOCK = 32
QUERY_HEADS = 4
INDEX_WIDTH = 128


def encoder_window_ends(source):
    ends = [hi for _,hi,_ in whole_windows(source)]
    if not ends or ends[-1] != len(source.ids):
        raise ValueError("Context windows must cover the complete, nonempty Simple input")
    return ends


def window_block_requests(windows,blocks,offsets):
    return torch.tensor([(offsets[i]+block,row,1+a-lo,1+b-lo)
                         for row,(_,lo,hi,i) in enumerate(windows)
                         for block,a,b in part_overlap(lo,hi,blocks[i])],dtype=torch.long).T


def encode_blocks(model, sources, tokenizer, window_ends, blocks, *, token_features=None):
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
    token_copy_seconds = 0.
    token_offsets = list(accumulate((len(s.ids) for s in sources),initial=0))
    for batch,values,padded in context_encoder_batches(model,tokenizer,windows,prefix_sum=token_features is None):
        prefix = values
        if token_features is not None:
            prefix = F.pad(values.float().cumsum(1),(0,0,1,0))
            start_copy = perf_counter()
            hidden = values.cpu()
            for row,(_,lo,hi,i) in enumerate(batch):
                start = token_offsets[i]
                token_features[start+lo:start+hi].copy_(hidden[row,1:1+hi-lo])
            token_copy_seconds += perf_counter()-start_copy
        for _,_,_,i in batch:
            padding[i] += padded//len(batch)
        index,row,a,b = window_block_requests(batch,blocks,offsets)
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
    result = dict(seconds=perf_counter()-started,body_bpe=body,blocks=len(vectors),windows=sequences,batch_calls=calls,
                        unpadded_positions=body+2*sequences,padded_positions=sum(padding),
                        per_function_padded_positions=padding,cached_vector_bytes=vectors.nbytes,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    if token_features is not None:
        result.update(token_feature_bytes=token_features.nbytes,token_transfer_and_copy_seconds=token_copy_seconds)
    return vectors,result


def block_mass(ends, spans):
    counts = torch.zeros(len(ends),dtype=torch.float32)
    for lo,hi in merge(spans):
        for i,a,b in part_overlap(lo,hi,ends):
            counts[i] += b-a
    return counts


def repair_block_mass(source, ends, edits, kinds):
    mapped = compact_edits(source.keep,edits)
    return block_mass(ends,(span for kind in kinds for span in mapped[kind]))


def repair_block_distribution(source,ends,edits):
    mass = repair_block_mass(source,ends,edits,(*EDIT_TYPES,*GAP_TYPES))
    return mass/mass.sum().clamp_min(1)


class BlockData(Dataset):
    def __init__(self, rows, ends, vectors, repair_targets=None, *, token_features=None):
        self.offsets = list(accumulate(map(len,ends),initial=0))
        self.targets = torch.tensor([r["target"] for r in rows])
        self.vectors = vectors
        self.repair_targets = repair_targets
        self.token_features = token_features
        if token_features is not None:
            self.token_offsets = list(accumulate((e[-1] for e in ends),initial=0))
            if self.token_offsets[-1] != len(token_features) or any(
                e[:-1] != list(range(BLOCK,e[-1],BLOCK)) for e in ends
            ):
                raise ValueError("Token matching requires one frozen state per BPE and fixed 32-BPE blocks")
        if self.offsets[-1] != len(vectors) or len(rows) != len(ends):
            raise ValueError("Cached block vectors do not align with source membership")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        a,b = self.offsets[i:i+2]
        example = self.vectors[a:b], self.targets[i], None if self.repair_targets is None else self.repair_targets[i]
        if self.token_features is not None:
            start,end = self.token_offsets[i:i+2]
            return (*example,self.token_features[start:end])
        return example


def collate_blocks(examples):
    vectors,targets,repair,*token_features = zip(*examples,strict=True)
    padded = pad_sequence(vectors, batch_first=True)
    valid = torch.arange(padded.shape[1])[None, :] < torch.tensor([len(v) for v in vectors])[:, None]
    batch = dict(blocks=padded, valid=valid, labels=torch.stack(targets))
    if repair[0] is not None:
        batch["repair"] = pad_sequence(repair, batch_first=True)
    if token_features:
        frozen, = token_features
        counts = torch.tensor([len(v) for v in vectors])
        offsets = list(accumulate(counts.tolist(),initial=0))
        # Pack only existing blocks. Rectangular Token padding would replicate outliers across the batch.
        tokens = frozen[0].new_zeros((offsets[-1]*BLOCK,frozen[0].shape[-1]))
        sizes = torch.full((offsets[-1],),BLOCK,dtype=torch.long)
        for start,end,states in zip(offsets[:-1],offsets[1:],frozen,strict=True):
            tokens[start*BLOCK:start*BLOCK+len(states)].copy_(states)
            sizes[end-1] = len(states)-(end-start-1)*BLOCK
        batch.update(tokens=tokens,block_sizes=sizes,owners=torch.arange(len(counts)).repeat_interleave(counts))
    return batch


def global_pool(blocks, valid):
    mean = blocks.masked_fill(~valid[..., None], 0).sum(1)/valid.sum(1, keepdim=True)
    maximum = blocks.masked_fill(~valid[..., None], -torch.inf).amax(1)
    return torch.cat((mean, maximum), dim=-1)


class ContextIndexer(nn.Module):
    def __init__(self, width, matching="block"):
        super().__init__()
        if matching not in ("block","token_mean","token_logmeanexp"):
            raise ValueError(f"Unknown matching mode {matching}")
        self.matching = matching
        self.query = nn.Linear(2*width, QUERY_HEADS*INDEX_WIDTH, bias=False)
        self.key = nn.Linear(width, INDEX_WIDTH, bias=False)

    def forward(self, blocks, valid, global_state, *, per_head=False, token_batch=None):
        query = self.query(F.layer_norm(global_state, (global_state.shape[-1],))).reshape(-1, QUERY_HEADS, INDEX_WIDTH)
        if self.matching == "block":
            key = self.key(F.layer_norm(blocks, (blocks.shape[-1],)))
            score = torch.bmm(query, key.transpose(1,2))/math.sqrt(INDEX_WIDTH)
        else:
            tokens,sizes,owners = (token_batch[field] for field in ("tokens","block_sizes","owners"))
            key = self.key(F.layer_norm(tokens.float(),(tokens.shape[-1],))).reshape(-1,BLOCK,INDEX_WIDTH)
            token_scores = torch.bmm(query[owners],key.transpose(1,2))/math.sqrt(INDEX_WIDTH)
            mask = torch.arange(BLOCK,device=sizes.device) < sizes[:,None]
            if self.matching == "token_mean":
                pooled = token_scores.masked_fill(~mask[:,None,:],0).sum(-1)/sizes[:,None]
            else:
                pooled = token_scores.masked_fill(~mask[:,None,:],-torch.inf).logsumexp(-1)-sizes[:,None].log()
            score = query.new_zeros((*valid.shape,QUERY_HEADS))
            score[valid] = pooled
            score = score.transpose(1,2)
        heads = score.masked_fill(~valid[:, None, :], -torch.inf).softmax(-1)
        return heads if per_head else heads.mean(1)


class CoarseClassifier(nn.Module):
    def __init__(self, width, supervision, repair_weight=0., *, bypass=True, matching="block",
                 ranking_budget=None, ranking_weight=0.):
        super().__init__()
        self.width = width
        self.supervision = supervision
        self.repair_weight = repair_weight
        self.bypass = bypass
        self.ranking_budget = ranking_budget
        self.ranking_weight = ranking_weight
        inputs = 3*width if bypass else width
        self.representation = nn.Sequential(nn.LayerNorm(inputs), nn.Linear(inputs,width), nn.Tanh())
        self.head = nn.Sequential(nn.Dropout(.1), nn.Linear(width,1))
        self.indexer = None if supervision == "global" else ContextIndexer(width,matching)

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
            scores = self.indexer(blocks, valid, g, per_head=inspect, token_batch=batch)
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
                kl = repair_kl(p,batch["repair"])
                parts["train_repair_kl"] = kl
                loss = loss+self.repair_weight*kl
            if self.ranking_weight:
                ranking = budget_ranking_loss(p,batch["repair"],batch["body_lengths"],self.ranking_budget)
                parts["train_repair_ranking"] = ranking
                loss = loss+self.ranking_weight*ranking
            output.update(loss=loss, loss_components=parts)
        return output


def repair_kl(p, q):
    return F.kl_div(p.clamp_min(1e-12).log(), q, reduction="none").sum()/q.any(-1).sum().clamp_min(1)


def budget_ranking_loss(probabilities, repair, body_lengths, context_budget):
    """Compare missed repair blocks with unmarked blocks bought by the center budget."""
    choices = probabilities.detach().cpu().numpy()
    labels = repair.detach().cpu().numpy()
    losses = []
    for i,length in enumerate(body_lengths.tolist()):
        if length<=BUDGET:
            continue
        ends = block_ends(length,(),(),BLOCK,False)
        blocks = list(zip([0,*ends[:-1]],ends))
        ranked = np.argsort(-choices[i,:len(blocks)],kind="stable")
        _, selected = center_blocks(blocks,ranked,context_budget)
        selected = np.asarray(selected,dtype=np.int64)
        positives = np.flatnonzero(labels[i,:len(blocks)]>0)
        missed = positives[~np.isin(positives,selected)]
        unmarked = selected[labels[i,selected]==0]
        if missed.size and unmarked.size:
            # log(p) preserves deployed rank order and cancels score scale from long functions.
            log_score = probabilities[i].clamp_min(1e-12).log()
            mass = repair[i,missed]
            costs = F.softplus(log_score[unmarked][None,:]-log_score[missed][:,None]).mean(-1)
            losses.append((costs*mass).sum()/mass.sum())
    return torch.stack(losses).mean() if losses else probabilities.sum()*0.


def collate_fine(examples):
    batch = collate(examples)
    batch["coarse"] = torch.stack([ex["coarse"] for ex in examples])
    return batch


def collate_relation_fine(examples):
    batch = collate_fine(examples)
    batch["relations"] = relation_bits(examples)
    return batch


class FusionClassifier(nn.Module):
    def __init__(self, fine, fusion="original", *, relation_attention=False):
        super().__init__()
        self.fine = fine
        width = fine.config.hidden_size
        projection = nn.Linear(2*width if fusion in ("global","none") else width, width, bias=False)
        nn.init.zeros_(projection.weight)
        # Keep the seeded draw for the subsequent data shuffle in all matched arms.
        self.fusion = None if fusion=="none" else projection
        self.relation_bias = nn.Parameter(torch.zeros(len(RELATION_TYPES))) if relation_attention else None

    @property
    def collator(self):
        return collate_relation_fine if self.relation_bias is not None else collate_fine

    def relation_weights(self):
        return dict(zip(RELATION_TYPES,self.relation_bias.detach().tolist(),strict=True))

    def representations(self, batch):
        ids = batch["fixed"][:,0].long()
        mask = torch.arange(ids.shape[1], device=ids.device)[None,:] < batch["lengths"][:,None]
        attention_mask = mask.long() if self.relation_bias is None else additive_relation_mask(
            batch["relations"],mask,self.relation_bias)
        cls = self.fine.encoder(input_ids=ids,attention_mask=attention_mask).last_hidden_state[:,0]
        return cls,torch.zeros_like(cls) if self.fusion is None else self.fusion(batch["coarse"])

    def forward(self, batch, *, details=False):
        cls,residual = self.representations(batch)
        mixed = cls+residual
        logits = self.fine.classifier(mixed[:,None])[:,0]
        output = dict(logits=logits, probability=logits.sigmoid())
        if details:
            output.update(cls=cls,fusion_residual=residual)
        if "labels" in batch:
            output["loss"] = F.binary_cross_entropy_with_logits(logits, batch["labels"].float())
        return output
