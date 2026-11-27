"""Map full-source statement relations to additive attention scores."""
from bisect import bisect_left

import numpy as np
import torch
from transformers import RobertaModel

RELATION_TYPES = ("condition","declaration","assignment")


class AdditiveAttentionRoberta(RobertaModel):
    """Pass an already additive mask to the inherited eager attention layers."""

    def get_extended_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
        if attention_mask.ndim==4:
            if attention_mask.shape!=(input_shape[0],1,input_shape[1],input_shape[1]):
                raise ValueError("Unexpected additive self-attention mask shape")
            return attention_mask.to(dtype=dtype or self.dtype)
        return super().get_extended_attention_mask(attention_mask,input_shape,device=device,dtype=dtype)


def visible_relations(source, fixed_positions, links):
    """Map relations whose complete compacted intervals survive source-order packing."""
    packed = np.flatnonzero(fixed_positions>=0)
    original = fixed_positions[packed]
    compacted = [bisect_left(source.keep,int(pos)) for pos in original]
    if any(i>=len(source.keep) or source.keep[i]!=pos for i,pos in zip(compacted,original,strict=True)):
        raise ValueError("Packed positions do not match the original to compacted BPE mapping")
    endpoints = {}

    def project(span):
        if span not in endpoints:
            a,b = span
            first,last = bisect_left(compacted,a),bisect_left(compacted,b)
            value = None
            if b>a and last-first==b-a:
                start,end = int(packed[first]),int(packed[last-1])+1
                if end-start!=last-first:
                    raise ValueError("A complete source interval crossed a packed-code separator")
                value = (start,end)
            endpoints[span] = value
        return endpoints[span]

    result = set()
    for kind,statement,context in links:
        if kind not in RELATION_TYPES:
            continue
        left,right = project(statement),project(context)
        if left is not None and right is not None and left!=right:
            left,right = sorted((left,right))
            result.add((RELATION_TYPES.index(kind),*left,*right))
    return sorted(result)


def relation_bits(examples):
    """Deduplicate directed token pairs per type and add the opposite direction."""
    length = examples[0]["fixed"].shape[-1]
    bits = np.zeros((len(examples),length,length),dtype=np.uint8)
    for i,example in enumerate(examples):
        for kind,a,b,c,d in example["relations"]:
            bits[i,a:b,c:d] |= 1<<kind
            bits[i,c:d,a:b] |= 1<<kind
    return torch.from_numpy(bits)


def additive_relation_mask(bits, valid, weights):
    bias = sum(w*(bits.bitwise_and(1<<kind)!=0) for kind,w in enumerate(weights))
    return bias.masked_fill(~valid[:,None,:],torch.finfo(bias.dtype).min)[:,None]
