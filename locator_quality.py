"""Contextual and FIM repair-region ranking on the historical 512-token inputs."""
import argparse
from bisect import bisect_right
import json
from itertools import islice
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from compaction import BUDGET
from oracle_noise import saved_retention
from paired import BACKBONES, DATA, ROOT, UNIX_BASE, UNIX_RESULTS, load_tokenizer, prime_source_rows
from predicted_diff import (RESULTS as PREDICTED_RESULTS, PredictedData, calibrate, load_sources, paired_annotations,
                            region_targets)
from region_locator import PARTS, Regions, region_metrics, score_regions, train_localizer
from repair_oracle import packed_edit_retention
from rgcl.artifacts import write_json

RESULTS = ROOT / "results" / "locator_quality_512"
WEIGHTS = ROOT / "checkpoints" / "locator_quality"
FIM_ID = "Qwen/Qwen2.5-Coder-0.5B"
FIM_REVISION = "8123ea2e9354afb7ffcc6c8641d1b2f5ecf18301"
FIM_MODEL = WEIGHTS / "qwen2.5-coder-0.5b"
FIM_TOKENS = ("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>")
FIM_SIDE = 64
FIM_BATCH = 64
FIM_LOSS_BATCH = 256
CONTEXT_BATCH = 24
GATE_MIN_RETENTION = .58
GATE_MIN_GAIN = .05


def whole_windows(source):
    """Cover each compacted candidate exactly once, with windows ending on a unit boundary."""
    ends = [r.end for r in source.regions]
    start = 0
    while start < len(ends):
        lo = source.regions[start].start
        stop = bisect_right(ends, lo+BUDGET, lo=start)
        if stop == start:
            raise ValueError("A historical region exceeds the encoder body budget")
        hi = ends[stop-1]
        yield lo, hi, source.regions[start:stop]
        start = stop


def load_context_encoder():
    return AutoModel.from_pretrained(UNIX_BASE, local_files_only=True, add_pooling_layer=False,
        attn_implementation="sdpa", dtype=torch.bfloat16).cuda().eval()


@torch.inference_mode()
def context_encoder_batches(model, tokenizer, windows):
    """Encode each source window once and expose local prefix sums for span pooling."""
    while batch := list(islice(windows, CONTEXT_BATCH)):
        width = max(hi-lo+2 for _,lo,hi,_ in batch)
        ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, (source, lo, hi, _) in enumerate(batch):
            values = [tokenizer.cls_token_id, *source.ids[lo:hi].tolist(), tokenizer.sep_token_id]
            ids[i, :len(values)] = torch.tensor(values)
            mask[i, :len(values)] = True
        hidden = model(input_ids=ids.cuda(), attention_mask=mask.cuda()).last_hidden_state
        yield batch, F.pad(hidden.float().cumsum(1), (0,0,1,0)), ids.numel()


def contextual_vectors(model, sources, tokenizer, split):
    regions = sum(len(s.regions) for s in sources)
    features = torch.empty((regions, PARTS, model.config.hidden_size), dtype=torch.float16)
    windows = ((s, lo, hi, units) for s in sources for lo, hi, units in whole_windows(s))
    cursor = calls = sequences = body = slots = 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    with torch.inference_mode():
        for batch, cumulative, positions in context_encoder_batches(model, tokenizer, windows):
            indices, breaks = [], []
            for i, (s, lo, hi, units) in enumerate(batch):
                for r in units:
                    indices.append(i)
                    breaks.append([1+r.start-lo+(r.end-r.start)*k//PARTS for k in range(PARTS+1)])
                body += hi-lo
            cuts = torch.tensor(breaks, device="cuda")
            batch_idx = torch.tensor(indices, device="cuda")[:, None]
            pooled = (cumulative[batch_idx, cuts[:, 1:]]-cumulative[batch_idx, cuts[:, :-1]])
            pooled /= (cuts[:, 1:]-cuts[:, :-1]).clamp_min(1)[..., None]
            features[cursor:cursor+len(indices)].copy_(pooled.to(dtype=features.dtype, device="cpu"))
            cursor += len(indices)
            calls += 1
            sequences += len(batch)
            slots += positions
            if calls % 500 == 0:
                print("contextual", split, cursor, "/", regions, "regions", flush=True)
    if cursor != regions or body != sum(len(s.ids) for s in sources):
        raise ValueError("The contextual encoder did not cover all historical BPE and regions")
    torch.cuda.synchronize()
    return features, dict(seconds=perf_counter()-started, regions=regions, encoder_calls=calls, windows=sequences,
                          body_bpe=body, padded_positions=slots, peak_allocated_bytes=torch.cuda.max_memory_allocated())


def fim_sequence(ids, a, b, markers):
    if not 0 <= a < b <= len(ids):
        raise ValueError("The masked region has no valid FIM-token overlap")
    prefix, suffix, middle = markers
    prompt = [prefix, *ids[max(0,a-FIM_SIDE):a], suffix, *ids[b:b+FIM_SIDE], middle]
    return prompt + ids[a:b], len(prompt)


def masked_regions(sources, tokenizer, markers):
    for start in range(0, len(sources), 32):
        batch = sources[start:start+32]
        encoded = tokenizer([s.code for s in batch], add_special_tokens=False, return_offsets_mapping=True,
                            return_attention_mask=False, verbose=False)
        for source, ids, offsets in zip(batch, encoded.input_ids, encoded.offset_mapping, strict=True):
            positions = np.asarray(offsets)
            lo = positions[:, 1].searchsorted([r.source_start for r in source.regions], side="right")
            hi = positions[:, 0].searchsorted([r.source_end for r in source.regions])
            for a,b in zip(lo, hi, strict=True):
                yield fim_sequence(ids, a, b, markers)


def fim_batch_loss(model, batch, pad):
    """Causal cross entropy on the teacher-forced middle. Extend logits only at target positions."""
    width = max(len(ids) for ids,_ in batch)
    padded = torch.full((len(batch), width), pad, dtype=torch.long)
    mask = torch.zeros_like(padded, dtype=torch.bool)
    batch_id, predicts, labels, sizes = [], [], [], []
    for i, (ids, begin) in enumerate(batch):
        padded[i, :len(ids)] = torch.tensor(ids)
        mask[i, :len(ids)] = True
        sizes.append(len(ids)-begin)
        batch_id.extend([i]*sizes[-1])
        predicts.extend(range(begin-1, len(ids)-1))
        labels.extend(ids[begin:])
    hidden = model.model(input_ids=padded.cuda(), attention_mask=mask.cuda(), use_cache=False).last_hidden_state
    index = torch.tensor(batch_id, device="cuda")
    positions = torch.tensor(predicts, device="cuda")
    targets = torch.tensor(labels, device="cuda")
    pieces = [F.cross_entropy(model.lm_head(hidden[bx, px]).float(), y, reduction="none")
              for bx,px,y in zip(index.split(FIM_LOSS_BATCH), positions.split(FIM_LOSS_BATCH),
                                 targets.split(FIM_LOSS_BATCH), strict=True)]
    losses = [seq.mean() for seq in torch.cat(pieces).split(sizes)]
    return torch.stack(losses).cpu().numpy(), padded.numel(), sum(sizes)


def load_fim_model():
    tokenizer = AutoTokenizer.from_pretrained(FIM_MODEL, local_files_only=True, use_fast=True)
    markers = tuple(tokenizer.convert_tokens_to_ids(t) for t in FIM_TOKENS)
    if len(set(markers)) != 3 or any(tokenizer.encode(t, add_special_tokens=False) != [i]
                                    for t,i in zip(FIM_TOKENS, markers, strict=True)):
        raise ValueError("The downloaded tokenizer is missing the three atomic FIM markers")
    model = AutoModelForCausalLM.from_pretrained(FIM_MODEL, local_files_only=True, use_safetensors=True,
            attn_implementation="sdpa", dtype=torch.bfloat16).cuda().eval()
    return model, tokenizer, markers


@torch.inference_mode()
def fim_scores(model, tokenizer, markers, sources, split):
    candidates = masked_regions(sources, tokenizer, markers)
    data = Regions(sources)
    scores = np.empty(len(data), dtype=np.float32)
    count = calls = tokens = padded = target = longest = 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    while batch := list(islice(candidates, FIM_BATCH)):
        p, slots, middle = fim_batch_loss(model, batch, tokenizer.pad_token_id)
        scores[count:count+len(p)] = p
        count += len(p)
        calls += 1
        tokens += sum(len(ids) for ids,_ in batch)
        padded += slots
        target += middle
        longest = max(longest, max(len(ids) for ids,_ in batch))
        if calls % 300 == 0:
            print("fim", split, count, "/", len(data), "regions", round(perf_counter()-started, 1), "s", flush=True)
    if count != len(data) or not np.isfinite(scores).all():
        raise ValueError("FIM loss was missing or nonfinite for at least one region")
    torch.cuda.synchronize()
    costs = dict(seconds=perf_counter()-started, regions=count, encoder_calls=calls, unpadded_positions=tokens,
                 padded_positions=padded, target_tokens=target, longest_input=longest,
                 peak_allocated_bytes=torch.cuda.max_memory_allocated())
    return data.regroup(scores), costs


def read_split(split, tokenizer):
    started = perf_counter()
    rows = prime_source_rows(DATA, split)
    sources = load_sources(rows, tokenizer)
    return rows, sources, perf_counter()-started


def layout(rows, sources, scores, tokenizer, policy, matched):
    return {name: PredictedData(rows, sources, scores, tokenizer, chosen)
            for name, chosen in (("valid_selected", policy), ("matched_127", matched))}


def measure(sources, scores, edits, inputs):
    labels = region_targets(sources, edits)
    long = [(p,y) for source,p,y in zip(sources, scores, labels, strict=True) if len(source.raw_ids) > BUDGET]
    return dict(regions=dict(overall=region_metrics(scores, labels, ks=(1,4,5,10)),
                             long=region_metrics(*zip(*long, strict=True), ks=(1,4,5,10))),
                inputs={name: dict(retention=packed_edit_retention(data, edits), packing_seconds=data.seconds)
                        for name,data in inputs.items()})


def evaluate_locator(name, tokenizer):
    root = RESULTS / name
    if (root / "metrics.json").is_file():
        return json.loads((root / "metrics.json").read_text())
    valid_rows, valid, valid_read = read_split("valid", tokenizer)
    valid_edits = paired_annotations(valid)
    model_stats = {}
    if name == "contextual":
        _, train, train_read = read_split("train", tokenizer)
        model = load_context_encoder()
        sources = dict(train=train, valid=valid)
        vectors, costs = {}, {}
        for split,samples in sources.items():
            vectors[split], costs[split] = contextual_vectors(model, samples, tokenizer, split)
        del model
        truth = dict(train=region_targets(train, paired_annotations(train)), valid=region_targets(valid, valid_edits))
        config = dict(backbone=BACKBONES["unixcoder"][0], revision=BACKBONES["unixcoder"][1], frozen_encoder=True,
                      input_window=512, batch_encoder=CONTEXT_BATCH, precision="bf16 SDPA, fp16 pooled features, fp32 head")
        localizer, values, model_stats = train_localizer(sources, truth, WEIGHTS / name, root,
                                                         vectors=vectors, feature_config=config)
        valid_scores = values["valid"]
        del train, sources, truth, vectors, values
        valid_cost = dict(**costs["valid"], head_seconds=model_stats["splits"]["valid"]["inference_seconds"])
        model_stats.update(train_feature_seconds=costs["train"]["seconds"], train_feature_cost=costs["train"],
                           train_read_seconds=train_read)
    elif name == "fim":
        model, fim_tok, markers = load_fim_model()
        config = dict(backbone=FIM_ID, revision=FIM_REVISION, trainable_parameters=0, input_context_per_side=FIM_SIDE,
                      markers=dict(zip(FIM_TOKENS, markers, strict=True)), batch=FIM_BATCH,
                      precision="bf16 SDPA, fp32 cross entropy", score="mean teacher-forced masked-region NLL")
        valid_scores, valid_cost = fim_scores(model, fim_tok, markers, valid, "valid")
    else:
        raise ValueError(f"Unknown locator {name}")

    historical = json.loads((PREDICTED_RESULTS / "policy.json").read_text())
    matched = {k: historical[k] for k in ("top_k", "anchor_budget")}
    policy = calibrate(valid, valid_scores, valid_edits, tokenizer)
    valid_inputs = layout(valid_rows, valid, valid_scores, tokenizer, policy, matched)
    validation = measure(valid, valid_scores, valid_edits, valid_inputs)
    uniform = saved_retention(UNIX_RESULTS / "uniform_sampling/valid.csv.gz", (valid_rows, valid, valid_edits))
    retention = validation["inputs"]["valid_selected"]["retention"]["long"]["actual_edits"]["fraction"]
    baseline = uniform["long"]["actual_edits"]["fraction"]
    gate = dict(split="valid", passed=retention >= GATE_MIN_RETENTION and retention-baseline >= GATE_MIN_GAIN,
                long_retention=retention, uniform_long_retention=baseline,
                min_long_retention=GATE_MIN_RETENTION, min_improvement=GATE_MIN_GAIN)
    write_json(root / "policy.json", dict(valid_selected=policy, matched_127=matched, gate=gate))
    print(name, "locked Valid policy", policy["top_k"], policy["anchor_budget"], "retention", round(retention, 4),
          "classification gate", gate["passed"], flush=True)
    del valid, valid_inputs, valid_edits, valid_scores

    test_rows, test, test_read = read_split("test", tokenizer)
    if name == "contextual":
        model = load_context_encoder()
        features, test_cost = contextual_vectors(model, test, tokenizer, "test")
        del model
        regions = Regions(test, vectors=features)
        values, seconds = score_regions(localizer, regions)
        test_scores = regions.regroup(values)
        test_cost["head_seconds"] = seconds
        del localizer, features, regions
    else:
        test_scores, test_cost = fim_scores(model, fim_tok, markers, test, "test")
    # All Test scores and both layouts are fixed before deriving any held-out diff.
    test_inputs = layout(test_rows, test, test_scores, tokenizer, policy, matched)
    result = dict(config=config, model_training=model_stats, valid=validation,
                  test=measure(test, test_scores, paired_annotations(test), test_inputs),
                  costs=dict(valid=dict(**valid_cost, source_seconds=valid_read), test=dict(**test_cost, source_seconds=test_read)),
                  gate=gate, policy=policy, matched_policy=matched)
    write_json(root / "metrics.json", result)
    print(name, "complete", result["test"]["regions"]["overall"]["ap"], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("methods", nargs="+", choices=("contextual", "fim"))
    args = parser.parse_args()
    if torch.cuda.device_count() != 1 or "4090" not in torch.cuda.get_device_name():
        raise ValueError("Select one RTX 4090 using CUDA_VISIBLE_DEVICES")
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    tokenizer = load_tokenizer("unixcoder", UNIX_BASE)
    for name in args.methods:
        evaluate_locator(name, tokenizer)


if __name__ == "__main__":
    main()
