"""Four source representations under one historical 512-position Uniform Sampling selector."""
import argparse
from collections import Counter
import csv
import gzip
import json
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from compaction import BUDGET, REGIONS, PackedData, line_boundaries
from context_selection import base_windows, describe, fixed_example, source_parser, uniform_fill, visible_source
from paired import DATA, ROOT, SEED, UNIX_BASE, UNIX_RESULTS, fit_512_inputs, load_tokenizer, prime_source_rows
from predicted_diff import HISTORICAL_REVISION, load_sources, paired_annotations
from repair_oracle import packed_edit_retention
from source_regions import LINE_SENSITIVE, PHYSICAL_LINE, PROTECTED_LINE, SPLICE_LINE, analyze, original_runs, simplify
from window_inputs import GROUPS
from rgcl.artifacts import write_json

METHODS = ("original", "simple", "lexical", "ast_aware")
RESULTS = ROOT / "results" / "compression_512"
WEIGHTS = ROOT / "checkpoints" / "compression_512"
FIELDS = ("encoding_runs", "compression_status", "removed_blank_lines", "removed_null_offsets", "syntax_error")


def lexical_form(code, parser, *, collect_nulls):
    nulls = []
    comments, literals, _, error = analyze(code, parser, null_statements=nulls if collect_nulls else None)
    if LINE_SENSITIVE.search(code):
        keep = np.arange(len(code), dtype=np.int32)
        return (code, keep, np.zeros(len(code), dtype=bool), dict(compression_status="line_sensitive",
                 removed_blank_lines=0, removed_null_offsets=[], syntax_error=error)), []
    text, origins, synthetic, status = simplify(code, comments, literals, drop_blank=not error)
    blank = sum(1 for _ in PHYSICAL_LINE.finditer(code))-sum(1 for _ in PHYSICAL_LINE.finditer(text))
    return (text, origins, synthetic, dict(compression_status=status, removed_blank_lines=blank,
            removed_null_offsets=[], syntax_error=error)), nulls if not error and status == "lexical" else []


def ast_form(code, base, nulls, parser):
    if not nulls or PROTECTED_LINE.search(code):
        return base
    text, origins, synthetic, info = base
    erased = np.isin(origins, nulls) & ~synthetic
    if not erased.any():
        return base
    compact = "".join(c for c, drop in zip(text, erased.tolist(), strict=True) if not drop)
    if SPLICE_LINE.search(compact) or parser.parse(compact.encode("utf-8")).root_node.has_error:
        return text, origins, synthetic, dict(info, compression_status="null_statement_reparse_veto")
    return compact, origins[~erased], synthetic[~erased], dict(info, compression_status="lexical_and_standalone_null",
                                                            removed_null_offsets=origins[erased].tolist())


def pack_uniform(ids, keep, boundaries, tokenizer):
    if len(keep) > BUDGET:
        base, limits = base_windows(boundaries)
        selected = uniform_fill(base, boundaries, limits, contiguous=False)
    else:
        selected = [(0, len(keep))] if len(keep) else []
    blank = torch.full((REGIONS, 4), -1, dtype=torch.int32)
    return dict(fixed_example(ids, keep, selected, tokenizer, blank), encoding_runs=selected)


def mapped_uniform(source, ids, offsets, form, tokenizer):
    text, origin, synthetic, info = form
    keep = range(len(ids))
    bounds = line_boundaries(text, offsets, keep, BUDGET) if len(ids) > BUDGET else [0, len(ids)]
    example = pack_uniform(ids, keep, bounds, tokenizer)
    mapped = SimpleNamespace(code=source.code, raw_offsets=source.offsets, offsets=offsets,
                              origin=origin, synthetic=synthetic)
    example.update(raw=len(source.raw_ids), original_runs=original_runs(mapped, example["encoding_runs"]), **info)
    return example


def prepare_profiles(rows, tokenizer, methods, parser):
    start = perf_counter()
    sources = load_sources(rows, tokenizer)
    costs = dict(shared_original_and_indentation_seconds=perf_counter()-start)
    examples = {m: [] for m in methods}
    for method in (m for m in methods if m in {"original", "simple"}):
        started = perf_counter()
        for source in sources:
            if method == "original":
                keep = range(len(source.raw_ids))
                bounds = line_boundaries(source.code, source.offsets, keep, BUDGET) if len(keep) > BUDGET else [0, len(keep)]
            else:
                keep, bounds = source.keep, source.boundaries
            example = pack_uniform(source.raw_ids, keep, bounds, tokenizer)
            examples[method].append(dict(example, compression_status=method, removed_blank_lines=0,
                                         removed_null_offsets=[], syntax_error=None))
        costs[f"{method}_layout_seconds"] = perf_counter()-started

    transformed = [m for m in methods if m in {"lexical", "ast_aware"}]
    if transformed:
        costs.update(shared_lexical_transform_seconds=0., ast_extra_transform_seconds=0.)
        costs.update({f"{m}_tokenize_and_pack_seconds": 0. for m in transformed})
        for begin in range(0, len(sources), 32):
            batch = sources[begin:begin+32]
            started = perf_counter()
            # The same raw parse and lexical form feed both representations.
            forms = [lexical_form(s.code, parser, collect_nulls="ast_aware" in transformed) for s in batch]
            costs["shared_lexical_transform_seconds"] += perf_counter()-started
            for m in transformed:
                started = perf_counter()
                variations = [form for form,_ in forms]
                if m == "ast_aware":
                    variations = [ast_form(s.code, form, nulls, parser) for s,(form,nulls) in zip(batch, forms, strict=True)]
                    costs["ast_extra_transform_seconds"] += perf_counter()-started
                started = perf_counter()
                encoded = tokenizer([form[0] for form in variations], add_special_tokens=False, return_offsets_mapping=True,
                                    return_attention_mask=False, verbose=False)
                for source, form, ids, offset in zip(batch, variations, encoded.input_ids, encoded.offset_mapping, strict=True):
                    offsets = np.asarray(offset, dtype=np.int32).reshape(-1,2)
                    if (offsets[:,0] >= offsets[:,1]).any():
                        raise ValueError("Invalid transformed BPE source offset")
                    examples[m].append(mapped_uniform(source, ids, offsets, form, tokenizer))
                costs[f"{m}_tokenize_and_pack_seconds"] += perf_counter()-started
    data = {method: PackedData(rows, supplied) for method,supplied in examples.items()}
    started = perf_counter()
    edits = paired_annotations(sources)
    costs["paired_diagnostic_seconds"] = perf_counter()-started
    return data, edits, costs


def compression_report(data):
    groups = {}
    for name, (lo,hi) in GROUPS.items():
        examples = [x for x in data.examples if lo <= x["raw"] <= hi]
        raw, compacted = sum(x["raw"] for x in examples), sum(x["compacted_bpe"] for x in examples)
        groups[name] = dict(functions=len(examples), original_bpe=raw, compacted_bpe=compacted,
                            bpe_saved_fraction=1-compacted/raw, fit=sum(x["compacted_bpe"] <= BUDGET for x in examples),
                            fit_fraction=sum(x["compacted_bpe"] <= BUDGET for x in examples)/len(examples),
                            removed_blank_lines=sum(x["removed_blank_lines"] for x in examples),
                            removed_standalone_null_statements=sum(len(x["removed_null_offsets"]) for x in examples),
                            statuses=dict(Counter(x["compression_status"] for x in examples)),
                            functions_with_syntax_error=sum(x["syntax_error"] is True for x in examples))
    return groups


def check_simple(data, split):
    with gzip.open(UNIX_RESULTS / "uniform_sampling" / f"{split}.csv.gz", "rt") as stream:
        for row, ex, meta in zip(csv.DictReader(stream), data.examples, data.metadata, strict=True):
            if (row["sample_key"] != meta["sample_key"] or int(row["original_bpe"]) != ex["raw"]
                    or int(row["input_length"]) != ex["length"] or int(row["compacted_bpe"]) != ex["compacted_bpe"]
                    or json.loads(row["source_runs"]) != [list(span) for span in visible_source(ex)]):
                raise ValueError(f"Simple failed historical Uniform Sampling parity in {split}")


def run(tokenizer, seed, methods):
    root = RESULTS / f"seed{seed}"
    parser = source_parser() if set(methods) & {"lexical", "ast_aware"} else None
    datasets = {m: {} for m in methods}
    edit_stats, profiles, costs, lexical_ast_changes = {}, {}, {}, {}
    for split in ("train", "valid", "test"):
        packed, edits, costs[split] = prepare_profiles(prime_source_rows(DATA, split), tokenizer, methods, parser)
        if "simple" in packed and split != "train":
            check_simple(packed["simple"], split)
        profiles[split] = {m: compression_report(data) for m,data in packed.items()}
        edit_stats[split] = {m: packed_edit_retention(data, edits) for m,data in packed.items()}
        if {"lexical","ast_aware"} <= packed.keys():
            lexical_ast_changes[split] = sum(not torch.equal(a["fixed"][0], b["fixed"][0])
                 for a,b in zip(packed["lexical"].examples, packed["ast_aware"].examples, strict=True))
        for m,data in packed.items():
            datasets[m][split] = data
        print("compression", split, {m: (p["overall"]["compacted_bpe"],p["overall"]["fit"]) for m,p in profiles[split].items()}, flush=True)
    write_json(root / "inputs.json", dict(profiles=profiles, edit_retention=edit_stats, timings=costs,
                                          lexical_ast_inputs_differ=lexical_ast_changes,
                                          test_collisions={m: describe(data["test"], None, None, paired=True)
                                                           for m,data in datasets.items()}))

    models = {}
    for method,data in datasets.items():
        if method == "simple" and seed == SEED:
            origin = UNIX_RESULTS / "uniform_sampling" / "metrics.json"
            models[method] = dict(**json.loads(origin.read_text()), reused=str(origin.relative_to(ROOT)))
        elif method == "ast_aware" and lexical_ast_changes and not any(lexical_ast_changes.values()):
            models[method] = dict(models["lexical"], reused="lexical; all Train, Valid and Test model inputs identical")
        else:
            models[method] = fit_512_inputs(tokenizer, data, root/method, WEIGHTS/f"seed{seed}"/method,
                method=method, seed=seed, configuration=dict(study="compression", sampler="9d95363 uniform sampling",
                historical_input_revision=HISTORICAL_REVISION, privileged_test=False), fields=FIELDS)
    write_json(root / "summary.json", dict(seed=seed, methods=models))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    args = parser.parse_args()
    if torch.cuda.device_count() != 1 or "4090" not in torch.cuda.get_device_name():
        raise ValueError("Select one RTX 4090 using CUDA_VISIBLE_DEVICES")
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    run(load_tokenizer("unixcoder", UNIX_BASE), args.seed, [m for m in METHODS if m in args.methods])


if __name__ == "__main__":
    main()
