"""Variable block pooling, linear-cost cross-block attention, and fine classification."""
from contextlib import nullcontext
from itertools import accumulate
import math
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from coarse_retrieval import part_overlap, uniform_layout
from compaction import MAX_LENGTH, pack_fixed
from context_selection import merge, occupied
from locator_quality import context_encoder_batches, whole_windows
from repair_oracle import compact_edits

WIDTH = 256
LATENTS = 8
ATTENTION_HEADS = 4
QUERY_HEADS = 4
INDEX_WIDTH = 128


def encoder_window_ends(source):
    ends = [hi for _,hi,_ in whole_windows(source)]
    if not ends or ends[-1] != len(source.ids):
        raise ValueError("Context windows must cover the complete, nonempty Simple input")
    return ends


def encode_blocks(model, sources, tokenizer, window_ends, layouts):
    """Pool several blockings from the same nonoverlapping PLM passes."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    offsets = {name:list(accumulate(map(len,ends),initial=0)) for name,ends in layouts.items()}
    width = model.config.hidden_size
    sums = {name:torch.zeros((cuts[-1],width),dtype=torch.float32) for name,cuts in offsets.items()}
    counts = {name:torch.zeros(cuts[-1],dtype=torch.int32) for name,cuts in offsets.items()}
    padding = [0]*len(sources)
    windows = ((source,lo,hi,i) for i,(source,ends) in enumerate(zip(sources,window_ends,strict=True))
               for lo,hi in zip([0,*ends[:-1]],ends))
    calls = sequences = 0
    for batch,prefix,padded in context_encoder_batches(model,tokenizer,windows):
        for _,_,_,i in batch:
            padding[i] += padded//len(batch)
        for name,ends in layouts.items():
            requests = [(offsets[name][i]+block,row,1+a-lo,1+b-lo)
                        for row,(_,lo,hi,i) in enumerate(batch) for block,a,b in part_overlap(lo,hi,ends[i])]
            index,row,a,b = torch.tensor(requests,dtype=torch.long).T
            row_cuda,a_cuda,b_cuda = (x.cuda() for x in (row,a,b))
            sums[name].index_add_(0,index,(prefix[row_cuda,b_cuda]-prefix[row_cuda,a_cuda]).cpu())
            counts[name].index_add_(0,index,(b-a).int())
        calls += 1
        sequences += len(batch)
    vectors = {}
    for name,ends in layouts.items():
        sizes = torch.tensor([b-a for e in ends for a,b in zip([0,*e[:-1]],e)],dtype=torch.int32)
        if not torch.equal(sizes,counts[name]):
            raise ValueError(f"Pooling lost or duplicated tokens in {name}")
        vectors[name] = sums[name].div_(sizes[:,None]).half()
    torch.cuda.synchronize()
    body = sum(len(s.ids) for s in sources)
    return vectors,dict(seconds=perf_counter()-started,body_bpe=body,windows=sequences,batch_calls=calls,
                        unpadded_positions=body+2*sequences,padded_positions=sum(padding),
                        per_function_padded_positions=padding,peak_allocated_bytes=torch.cuda.max_memory_allocated())


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
    def __init__(self, rows, sources, ends, vectors, *, plans=None, repair=None, offset=0):
        self.offsets = list(accumulate(map(len,ends),initial=0))
        self.targets = torch.tensor([row["target"] for row in rows])
        self.positions = [torch.tensor([(s.keep[a]+s.keep[b-1])/2 for a,b in zip([0,*e[:-1]],e)])
                          for s,e in zip(sources,ends,strict=True)]
        self.vectors, self.repair, self.offset = vectors,repair,offset
        self.context = {}
        if plans is not None:
            for i,plan in enumerate(plans):
                if plan.edges:
                    anchor,related,weights = zip(*plan.edges)
                    self.context[i] = (torch.bincount(torch.tensor(anchor),minlength=len(ends[i])),
                                       torch.tensor(related,dtype=torch.long),torch.tensor(weights))
        if len(vectors)!=self.offsets[-1] or len(rows)!=len(ends):
            raise ValueError("Block cache does not match dataset membership")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self,i):
        a,b = self.offsets[i:i+2]
        return (self.vectors[a:b],self.positions[i],self.targets[i],self.offset+i,self.context.get(i),
                self.repair[i] if self.repair is not None else None)


def collate_blocks(examples):
    vectors,positions,targets,ids,context,repair = zip(*examples,strict=True)
    blocks = pad_sequence(vectors,batch_first=True)
    width = blocks.shape[1]
    batch = dict(blocks=blocks,positions=pad_sequence(positions,batch_first=True),labels=torch.stack(targets),
                 sample_ids=torch.tensor(ids),valid=torch.arange(width)[None,:]<torch.tensor([len(v) for v in vectors])[:,None])
    if repair[0] is not None:
        batch["repair"] = pad_sequence(repair,batch_first=True)
    if any(c is not None for c in context):
        sizes = torch.zeros(len(vectors),width,dtype=torch.long)
        indices,weights = [],[]
        for i,edge in enumerate(context):
            if edge is not None:
                n,index,mass = edge
                sizes[i,:len(n)] = n
                indices.append(index+i*width)
                weights.append(mass)
        batch.update(context_indices=torch.cat(indices),context_weights=torch.cat(weights),
                     context_offsets=F.pad(sizes.flatten().cumsum(0),(1,0)))
    return batch


def masked_mean(values, valid):
    return values.masked_fill(~valid[...,None],0.).sum(1)/valid.sum(1,keepdim=True)


class CrossBlockEncoder(nn.Module):
    def __init__(self, input_size, width=WIDTH):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(input_size,width),nn.LayerNorm(width))
        self.latents = nn.Parameter(torch.randn(LATENTS,width)*.02)
        self.read = nn.MultiheadAttention(width,ATTENTION_HEADS,batch_first=True)
        self.broadcast = nn.MultiheadAttention(width,ATTENTION_HEADS,batch_first=True)
        self.latent_norm = nn.LayerNorm(width)
        self.block_norm = nn.LayerNorm(width)
        self.register_buffer("frequency",10000**(-torch.arange(0,width,2)/width),persistent=False)

    def forward(self, batch):
        valid = batch["valid"]
        local = self.project(batch["blocks"].float())
        angles = batch["positions"][...,None]*self.frequency
        positions = torch.stack((angles.sin(),angles.cos()),dim=-1).flatten(-2)
        x = local+positions
        latent = self.latents[None].expand(len(x),-1,-1)
        hidden = self.latent_norm(latent+self.read(latent,x,x,key_padding_mask=~valid,need_weights=False)[0])
        contexts = self.block_norm(x+self.broadcast(x,hidden,hidden,need_weights=False)[0])
        return hidden[:,0],contexts,local


class ContextIndexer(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.query = nn.Linear(width,QUERY_HEADS*INDEX_WIDTH,bias=False)
        self.key = nn.Linear(width,INDEX_WIDTH,bias=False)

    def forward(self, g, contexts, batch):
        if "context_indices" in batch:
            related = F.embedding_bag(batch["context_indices"],contexts.flatten(0,1),batch["context_offsets"],
                                      mode="sum",per_sample_weights=batch["context_weights"],include_last_offset=True)
            contexts = contexts+related.view_as(contexts)
        q = self.query(g).reshape(-1,QUERY_HEADS,INDEX_WIDTH)
        k = self.key(contexts)
        score = torch.bmm(q,k.transpose(1,2))/math.sqrt(INDEX_WIDTH)
        return score.masked_fill(~batch["valid"][:,None,:],-torch.inf).softmax(-1).mean(1)


def repair_kl(p, target):
    return F.kl_div(p.clamp_min(1e-12).log(),target,reduction="none").sum()/target.any(1).sum().clamp_min(1)


class CoarseClassifier(nn.Module):
    def __init__(self, input_size, supervision, repair_weight=0., *, bypass=False):
        super().__init__()
        self.supervision,self.repair_weight,self.bypass = supervision,repair_weight,bypass
        self.encoder = CrossBlockEncoder(input_size)
        self.head = nn.Sequential(nn.LayerNorm(WIDTH),nn.Dropout(.1),nn.Linear(WIDTH,1))
        self.indexer = None if supervision=="global" else ContextIndexer(WIDTH)

    def forward(self, batch, *, details=False, profile=False):
        started = perf_counter()
        g,context,local = self.encoder(batch)
        if profile:
            torch.cuda.synchronize()
            encoded = perf_counter()
        if self.indexer is None:
            z = masked_mean(context,batch["valid"])
        else:
            p = self.indexer(g,context,batch)
            z = torch.bmm(p[:,None,:],local).squeeze(1)
        aux = z+g if self.bypass or self.indexer is None else z
        if self.supervision=="repair":
            aux = aux.detach()
        logits = self.head(aux).squeeze(-1)
        result = dict(logits=logits,probability=logits.sigmoid())
        if details:
            result.update(g=g,z=z)
            if self.indexer is not None:
                result["importance"] = p
        if "labels" in batch:
            bce = F.binary_cross_entropy_with_logits(logits,batch["labels"].float())
            parts = dict(train_bce=bce)
            loss = bce
            if self.repair_weight:
                kl = repair_kl(p,batch["repair"])
                parts["train_repair_kl"] = kl
                loss = loss+self.repair_weight*kl
            result.update(loss=loss,loss_components=parts)
        if profile:
            torch.cuda.synchronize()
            result["seconds"] = (encoded-started,perf_counter()-encoded)
        return result


class FusionClassifier(nn.Module):
    def __init__(self, fine, coarse, examples, tokenizer, *, uniform=False):
        super().__init__()
        self.fine,self.coarse = fine,coarse
        self.examples,self.tokenizer,self.uniform = examples,tokenizer,uniform
        self.joint = False
        self.uniform_cache = {}
        self.fusion = nn.Linear(2*WIDTH,fine.config.hidden_size,bias=False)
        nn.init.zeros_(self.fusion.weight)

    def layouts(self, batch, scores, *, uniform):
        layout = []
        ranking = scores.detach().cpu().numpy()
        for i,row in zip(batch["sample_ids"].tolist(),ranking,strict=True):
            source,candidates = self.examples[i]
            if uniform:
                if i not in self.uniform_cache:
                    self.uniform_cache[i] = uniform_layout(source)
                layout.append(self.uniform_cache[i])
            else:
                layout.append(candidates.retrieve(row[:len(candidates.centers)],source))
        return layout

    def fine_logits(self, ids, layouts, g, z):
        packed = []
        for i,x in zip(ids,layouts,strict=True):
            source = self.examples[i][0]
            fixed, length = pack_fixed(source.raw_ids,source.keep,x.spans,self.tokenizer)
            if length != occupied(x.spans)+2 or length != min(MAX_LENGTH,len(source.ids)+2):
                raise ValueError("Fine input does not match its recorded source spans or budget")
            packed.append((fixed,length))
        tokens = torch.stack([x[0][0] for x in packed]).to(g.device,dtype=torch.long)
        lengths = torch.tensor([x[1] for x in packed],device=g.device)
        mask = torch.arange(tokens.shape[1],device=g.device)[None,:]<lengths[:,None]
        cls = self.fine.encoder(input_ids=tokens,attention_mask=mask.long()).last_hidden_state[:,0]
        return self.fine.classifier((cls+self.fusion(torch.cat((g,z),dim=1)))[:,None])[:,0]

    def forward(self, batch, *, details=False):
        with nullcontext() if self.joint else torch.no_grad():
            coarse_input = batch if self.joint else {k:v for k,v in batch.items() if k!="labels"}
            coarse = self.coarse(coarse_input,details=True,profile=details)
        indexed = perf_counter()
        layouts = self.layouts(batch,coarse["importance"],uniform=self.uniform)
        selected = perf_counter()
        logits = self.fine_logits(batch["sample_ids"].tolist(),layouts,coarse["g"],coarse["z"])
        result = dict(logits=logits,probability=logits.sigmoid())
        if "labels" in batch:
            bce = F.binary_cross_entropy_with_logits(logits,batch["labels"].float())
            result.update(loss=bce,loss_components=dict(train_bce=bce))
            if self.joint:
                result["loss"] = bce+coarse["loss"]
                for k,v in coarse["loss_components"].items():
                    result["loss_components"]["aux_"+k] = v
        if details:
            torch.cuda.synchronize()
            result.update(layouts=layouts,importance=coarse["importance"],
                          seconds=(*coarse["seconds"],selected-indexed,perf_counter()-selected))
        return result
