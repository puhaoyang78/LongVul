"""Straight-through window selection and source-coordinate features."""
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.roberta.modeling_roberta import create_position_ids_from_input_ids

from compaction import BUDGET, MAX_LENGTH, REGIONS, WIDTH

CHANNELS = 16
TEMPERATURE = 1.
POSITION_BINS = 64
DISTANCE_BINS = 33


def pack(bank_ids, source_positions, specs, choices, special):
    cls, sep, pad = special
    functions = torch.arange(len(choices), device=choices.device)[:, None]
    regions = torch.arange(REGIONS, device=choices.device)[None, :]
    ids = bank_ids[functions, regions, choices]
    originals = source_positions[functions, regions, choices]
    spans = specs[functions, regions, choices]
    keep = originals.ge(0)
    gaps = F.pad(spans[:, :-1, 1] != spans[:, 1:, 0], (0, 1))
    valid = torch.cat((keep, gaps.unsqueeze(-1)), dim=2).flatten(1)
    valid = F.pad(valid, (1, 1), value=True)
    slots = torch.arange(valid.shape[1], device=valid.device)
    order = (slots + (~valid) * len(slots)).argsort(1)[:, :MAX_LENGTH]
    ids = F.pad(F.pad(ids, (0, 1), value=sep).flatten(1), (1, 1), value=sep)
    ids[:, 0] = cls
    coded = ids.gather(1, order).masked_fill(~valid.gather(1, order), pad)
    original = F.pad(F.pad(originals, (0, 1), value=-1).flatten(1), (1, 1), value=-1)
    parts = (regions+1)[..., None].expand_as(originals).masked_fill(~keep, 0)
    parts = F.pad(F.pad(parts, (0, 1)).flatten(1), (1, 1))
    return coded, original.gather(1, order), parts.gather(1, order), valid.sum(1), spans, order


def position_features(original, parts, spans, raw):
    # 0 is reserved for selected special tokens or the unchanged short input.
    active = (parts > 0) & (raw[:, None] > BUDGET)
    relative = 1 + original.clamp_min(0).long() * (POSITION_BINS-1) // (raw[:, None]-1).clamp_min(1)
    start = parts.ne(F.pad(parts[:, :-1], (1, 0)))
    end = parts.ne(F.pad(parts[:, 1:], (0, 1)))
    boundary = 1 + start.long() + end.long()*2
    gaps = (spans[:, 1:, 2] - spans[:, :-1, 3]).clamp_min(0)
    gap_bins = 1 + (gaps.float()+1).log2().long().clamp(max=DISTANCE_BINS-2)
    gap_bins = F.pad(gap_bins, (2, 0)).gather(1, parts)
    return tuple(x.masked_fill(~active, 0) for x in (relative, parts, boundary, gap_bins))


class CompactionClassifier(nn.Module):
    def __init__(self, detector, tokenizer, *, select_windows, use_positions):
        super().__init__()
        self.detector = detector
        self.special = (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id)
        width = detector.config.hidden_size
        self.scorer, self.position = None, None
        # Do not consume any of the classifier's training or sampler RNG draws.
        with torch.random.fork_rng(devices=[]):
            if select_windows:
                self.conv = nn.Conv1d(width, CHANNELS, kernel_size=3, padding=1)
                self.scorer = nn.Sequential(nn.Linear(CHANNELS, CHANNELS), nn.GELU(), nn.Linear(CHANNELS, 1))
            if use_positions:
                self.position = nn.ModuleList(nn.Embedding(count, width, padding_idx=0)
                                              for count in (POSITION_BINS+1, REGIONS+1, 5, DISTANCE_BINS))
                for embedding in self.position:
                    nn.init.zeros_(embedding.weight)

    def forward(self, batch, *, details=False):
        table = self.detector.encoder.get_input_embeddings()
        ids, originals, parts = batch["fixed"].long().unbind(1)
        lengths, spans = batch["lengths"], batch["uniform"]
        choices = spans[..., 0].ge(0).long()-1
        residual = None
        if self.scorer is not None and "bank" in batch:
            bank_ids, positions, spec = batch["bank"]
            values = F.embedding(bank_ids, table.weight.detach())
            valid = positions.ge(0)
            values.masked_fill_(~valid.unsqueeze(-1), 0.)
            # The number of embeddings read is at most 4*8*127 per function.
            conv = F.gelu(self.conv(values.flatten(0, 2).transpose(1, 2).contiguous()))
            pooled = ((conv * valid.flatten(0, 2).unsqueeze(1)).sum(-1)
                      / valid.flatten(0, 2).sum(-1, keepdim=True).clamp_min(1))
            logits = self.scorer(pooled).view_as(spec[..., 0]).masked_fill(~valid.any(-1), -torch.inf)
            if self.training:
                weight = F.gumbel_softmax(logits, tau=TEMPERATURE, hard=True, dim=-1)
                chosen = weight.argmax(-1)
            else:
                chosen = logits.argmax(-1)
            new_ids, new_pos, new_parts, new_length, new_spans, order = pack(bank_ids, positions, spec, chosen, self.special)
            choose_index = batch["indices"]
            ids = ids.index_copy(0, choose_index, new_ids.long())
            originals = originals.index_copy(0, choose_index, new_pos.long())
            parts = parts.index_copy(0, choose_index, new_parts)
            lengths = lengths.index_copy(0, choose_index, new_length)
            spans = spans.index_copy(0, choose_index, new_spans)
            choices = choices.index_copy(0, choose_index, chosen)
            if self.training:
                # Keep the exact selected word embeddings in forward. Only this
                # zero-valued term supplies the straight-through gradient.
                delta = (weight-weight.detach()).unsqueeze(-2)
                delta = (delta @ values.flatten(-2)).squeeze(-2).reshape(-1, REGIONS, WIDTH, table.embedding_dim)
                delta = F.pad(F.pad(delta, (0, 0, 0, 1)).flatten(1, 2), (0, 0, 1, 1))
                residual = delta.gather(1, order[..., None].expand(-1, -1, table.embedding_dim))
                residual = residual.masked_fill(new_pos.lt(0)[..., None], 0.)

        attention = torch.arange(ids.shape[1], device=ids.device)[None, :].lt(lengths[:, None]).long()
        if self.position is None and residual is None:
            result = self.detector(ids, attention, None, None, labels=batch.get("labels"))
        else:
            embeddings = table(ids)
            if residual is not None:
                embeddings = embeddings.index_add(0, batch["indices"], residual)
            if self.position is not None:
                features = position_features(originals, parts, spans, batch["raw"])
                for table_entry, feature in zip(self.position, features, strict=True):
                    embeddings = embeddings + table_entry(feature)
            positions = create_position_ids_from_input_ids(ids, table.padding_idx)
            states = self.detector.encoder(inputs_embeds=embeddings, attention_mask=attention, position_ids=positions)[0]
            logits = self.detector.classifier(states[:, :1])[:, 0]
            result = dict(logits=logits, probability=logits.sigmoid())
            if "labels" in batch:
                result["loss"] = F.binary_cross_entropy_with_logits(logits, batch["labels"].float())
        if details:
            result.update(input_ids=ids, attention=attention, source_positions=originals,
                          input_lengths=lengths, spans=spans, choices=choices)
        return result
