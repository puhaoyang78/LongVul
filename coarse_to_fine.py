"""Repair-guided coarse-to-fine classification with center-first retrieval."""
import argparse
from bisect import bisect_left
import csv
from dataclasses import dataclass
import gc
import gzip
from importlib.metadata import version
import json
import math
from statistics import mean, stdev
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import set_seed

from coarse_encoding import (BLOCK, INDEX_WIDTH, QUERY_HEADS, BlockData, CoarseClassifier, FusionClassifier,
                             collate_blocks, collate_fine, encode_blocks, encoder_window_ends,
                             repair_block_distribution, repair_block_mass)
from coarse_retrieval import CONTEXT_TYPES, Layout, block_ends, lengths, retrieve, uncovered, uniform_layout
from compaction import BUDGET, MAX_LENGTH, REGIONS, PackedData, pack_fixed
from context_selection import (base_windows, contains, intersects, merge, occupied, prefer_context,
                               related_statements, source_parser, visible_source, write_prediction_rows)
from locator_quality import CONTEXT_BATCH, load_context_encoder
from paired import (AdamW, BACKBONES, BATCH_SIZE, DATA, ROOT, SEED, UNIX_BASE, UNIX_RESULTS, classifier_config,
                    fit_classifier, load_tokenizer, predict, prime_source_rows, to_gpu, with_recorded_names)
from predicted_diff import HISTORICAL_REVISION, fp32_classifier, load_sources, paired_annotations
from region_locator import region_metrics
from relation_attention import visible_relations
from repair_oracle import EDIT_TYPES, GAP_TYPES, grouped_edit_retention, overlap_bpe, raw_runs
from rgcl.artifacts import read_json, write_json
from rgcl.config import SEEDS
from rgcl.metrics import metrics, paired_metrics
from window_inputs import GROUPS

RESULTS = ROOT / "results" / "coarse_to_fine_center_512"
WEIGHTS = ROOT / "checkpoints" / RESULTS.name
ARCHIVE = ROOT / "results" / "coarse_to_fine_512"
ARCHIVE_WEIGHTS = ROOT / "checkpoints" / ARCHIVE.name
VARIABLE_CACHE = ROOT / "checkpoints" / "longvul_adaptive_512" / "blocks"
REPAIR_INDEX_WEIGHT = 1.
BUDGETS = (0,64,96,128)
METHODS = ("center_context","statement_context","global_fusion","weighted_fusion","selected_fusion","matched_uniform")


@dataclass(slots=True)
class SourceSplit:
    rows: list
    sources: list
    windows: list
    ends: list
    vectors: torch.Tensor
    encoding: dict
    source_seconds: float
    syntax_seconds: float = 0.
    cache_read_seconds: float = 0.
    syntax: list | None = None
    aligned: bool = False
    token_features: torch.Tensor | None = None


def source_inputs(split,tokenizer):
    started = perf_counter()
    rows = prime_source_rows(DATA,split)
    sources = load_sources(rows,tokenizer)
    windows = [encoder_window_ends(s) for s in sources]
    return rows,sources,windows,perf_counter()-started


def source_syntax(sources):
    parser = source_parser()
    output = []
    for s in sources:
        units = set()
        links,error = related_statements(s.code,s.offsets,s.keep,None,parser,units=units)
        output.append((sorted(units),sorted(links),error))
    return output


def cuts_for(sources, syntax, aligned):
    return [block_ends(len(source.ids), units,source.boundaries,BLOCK,aligned)
            for source,(units,_,_) in zip(sources,syntax,strict=True)] if aligned else [
                block_ends(len(source.ids),(),(),BLOCK,False) for source in sources]


def load_split(split,tokenizer,*,aligned=False,encoder=None,token_features=False):
    rows,sources,windows,source_time = source_inputs(split,tokenizer)
    syntax,syntax_time = None,0.
    if encoder is not None:
        if aligned:
            started = perf_counter()
            syntax = source_syntax(sources)
            syntax_time = perf_counter()-started
        ends = cuts_for(sources,syntax,aligned)
        states = (torch.empty((sum(len(s.ids) for s in sources),encoder.config.hidden_size),dtype=torch.float16)
                  if token_features else None)
        vectors, encoding = encode_blocks(encoder,sources,tokenizer,windows,ends,token_features=states)
        return SourceSplit(rows,sources,windows,ends,vectors,encoding,source_time,syntax_time,syntax=syntax,
                           aligned=aligned,token_features=states)

    members = [r["sample_key"] for r in rows]
    tokens = torch.from_numpy(np.concatenate([s.ids for s in sources]))
    started = perf_counter()
    cache = torch.load(ARCHIVE_WEIGHTS/"blocks"/f"{split}.pt",weights_only=True,mmap=True,map_location="cpu")
    config = dict(dataset="PrimeVul_original", split=split, members=members,
                  model=str(UNIX_BASE), revision=BACKBONES["unixcoder"][1], compression=HISTORICAL_REVISION,
                  block_size=BLOCK, windows=windows, batch=CONTEXT_BATCH, precision="BF16 SDPA / FP32 mean / FP16 cache")
    if cache["config"] != config or not torch.equal(cache["tokens"],tokens):
        raise ValueError(f"Archived {split} cache disagrees with source membership, encoder or tokens")
    encoding = cache["encoding"]
    # The prior variable-block experiment cached both relevant partitions for Train and Valid.
    if split in ("train","valid"):
        variable = torch.load(VARIABLE_CACHE/f"{split}.pt",weights_only=True,mmap=True,map_location="cpu")
        expected = dict(dataset="PrimeVul_original",split=split,members=members,model=BACKBONES["unixcoder"][1],
                        simple=HISTORICAL_REVISION,windows=windows,batch=CONTEXT_BATCH,
                        precision="BF16 SDPA / FP32 means / FP16 cache")
        if variable["config"] != expected or not torch.equal(variable["tokens"],tokens):
            raise ValueError(f"Variable-block cache does not match {split}")
        syntax = variable["syntax"]
    ends = cuts_for(sources,syntax,aligned)
    if aligned:
        if ends != variable["layouts"]["statement32"]:
            raise ValueError(f"Statement partition differs from cached {split} layout")
        vectors = variable["vectors"]["statement32"]
        # The original multi-layout timing includes the other five partitions.
        encoding = dict(variable["timing"]["encoding"], blocks=len(vectors),cached_vector_bytes=vectors.nbytes,
                        pooled_layouts=len(variable["vectors"]))
    else:
        vectors = cache["vectors"]
    if vectors.dtype != torch.float16 or not torch.isfinite(vectors).all() or len(vectors)!=sum(map(len,ends)):
        raise ValueError(f"Invalid block vectors for {split}")
    return SourceSplit(rows,sources,windows,ends,vectors,encoding,source_time,
                       syntax_time,perf_counter()-started,syntax,aligned)


def source_identity(split):
    return [a_end == b_end and np.array_equal(a.ids,b.ids)
            for a,b,a_end,b_end in zip(split.sources[::2],split.sources[1::2],
                                     split.ends[::2],split.ends[1::2],strict=True)]


def score_coarse(model, split, fusion):
    model.eval()
    token_features = split.token_features if model.indexer.matching != "block" else None
    loader = DataLoader(BlockData(split.rows,split.ends,split.vectors,token_features=token_features),
                        batch_size=BATCH_SIZE,collate_fn=collate_blocks)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    features, ranking = [], []
    key = {"original":"coarse","global":"global_state","weighted":"weighted"}[fusion]
    with torch.no_grad():
        for batch in loader:
            out = model(to_gpu(batch),details=True)
            features.append(out[key].cpu())
            counts = batch["valid"].sum(1).tolist()
            ranking.extend(row[:n].numpy().copy() for n,row in zip(counts,out["importance"].cpu(),strict=True))
    return torch.cat(features),ranking,dict(seconds=perf_counter()-started,
                                            peak_allocated_bytes=torch.cuda.max_memory_allocated())


def seed_root(root,seed):
    return root / f"seed{seed}"


def historical_index(seed):
    return read_json(seed_root(ARCHIVE,seed)/"policy.json")["selected"]


def run_config(method,seed):
    config = classifier_config("uniform_sampling")
    config.update(method=method,seed=seed,model_id=BACKBONES["unixcoder"][0],model_revision=BACKBONES["unixcoder"][1],
                  block_size=BLOCK,query_heads=QUERY_HEADS,index_dimension=INDEX_WIDTH,
                  full_encoder_precision="BF16 SDPA",full_encoder_window_batch=CONTEXT_BATCH,
                  tree_sitter=version("tree-sitter"),tree_sitter_cpp=version("tree-sitter-cpp"),
                  historical_input_revision=HISTORICAL_REVISION,split_functions=dict(train=8708,valid=1124,test=1128))
    return config


def fitted(config, selected, history, nvalid):
    return dict(config=config,selected_epoch=selected["epoch"],validation=selected["validation"],
                train_seconds=sum(row["training_seconds"] for row in history),
                validation_seconds=sum(row["validation_seconds"] for row in history),
                peak_train_allocated_bytes=max(row["peak_allocated_bytes"] for row in history),
                last_three_mean_train_loss=mean(row["train_loss"] for row in history[-3:]),
                one_class_valid_epochs=sum(row["validation"]["tp"]+row["validation"]["fp"] in (0,nvalid) for row in history))


def fit_repair_indexer(train,valid,seed,method,root,weights,*,matching="block"):
    width = train.vectors.shape[-1]
    if (root/method/"fit.json").is_file():
        return read_json(root/method/"fit.json")
    labels = [repair_block_distribution(source,ends,edits)
              for source,ends,edits in zip(train.sources,train.ends,paired_annotations(train.sources),strict=True)]
    for name,supervision,weight,token_match in (
        ("global","global",0.,"block"), (method,"joint",REPAIR_INDEX_WEIGHT,matching)
    ):
        area,location = root/name,weights/name
        if (area/"fit.json").is_file():
            continue
        use_tokens = token_match!="block"
        validation = BlockData(valid.rows,valid.ends,valid.vectors,token_features=valid.token_features if use_tokens else None)
        data = BlockData(train.rows,train.ends,train.vectors,labels if weight else None,
                         token_features=train.token_features if use_tokens else None)
        config = dict(run_config(name,seed),stage="coarse",block_layout="fixed32",
                      matching=token_match,supervision=supervision,repair_weight=weight,
                      precision="FP32 on cached FP16 states",initialization="fresh seeded weights",
                      loss="function BCE + labeled function mean KL" if weight else "function BCE")
        set_seed(seed)
        model = CoarseClassifier(width,supervision,weight,matching=token_match)
        if weight:
            config["initialization"] = "fresh Q/K; auxiliary head from newly trained same-seed global fit"
            update = model.load_state_dict(torch.load(weights/"global"/"best.pt",weights_only=True,map_location="cpu"),
                                           strict=False)
            if set(update.missing_keys)!={"indexer.query.weight","indexer.key.weight"} or update.unexpected_keys:
                raise ValueError("Newly trained auxiliary head did not match the joint model")
        best,history = fit_classifier(model.cuda(),data,validation,location,area,config,collator=collate_blocks)
        write_json(area/"fit.json",fitted(config,best,history,len(validation)))
        del model
        gc.collect()
    return read_json(root/method/"fit.json")


def fit_indexer(tokenizer,seed):
    root = seed_root(RESULTS,seed)/"statement_index"
    if (root/"joint"/"fit.json").is_file():
        return
    splits = [load_split(split,tokenizer,aligned=True) for split in ("train","valid")]
    train,valid = [BlockData(s.rows,s.ends,s.vectors) for s in splits]
    source = splits[0]
    masses = [repair_block_mass(s,e,a,(*EDIT_TYPES,*GAP_TYPES)) for s,e,a in
              zip(source.sources,source.ends,paired_annotations(source.sources),strict=True)]
    supervised = BlockData(source.rows,source.ends,source.vectors,[x/x.sum().clamp_min(1) for x in masses])
    weight = historical_index(seed)["weight"]
    base = seed_root(WEIGHTS,seed)/"statement_index"
    for mode,w,data in (("global",0.,train),("joint",weight,supervised)):
        location = root/mode
        if (location/"fit.json").is_file():
            continue
        config = dict(run_config(f"statement_index_{mode}",seed), block_layout="statement32",
                      stage="coarse",precision="FP32 on cached FP16 blocks",supervision=mode,repair_weight=w,
                      loss="function BCE + labeled function mean KL" if w else "function BCE",
                      historical_index=historical_index(seed)["name"])
        set_seed(seed)
        model = CoarseClassifier(source.vectors.shape[-1],mode,w)
        if mode == "joint":
            update = model.load_state_dict(torch.load(base/"global"/"best.pt",weights_only=True,map_location="cpu"),
                                           strict=False)
            if set(update.missing_keys) != {"indexer.query.weight","indexer.key.weight"} or update.unexpected_keys:
                raise ValueError("Statement indexer could not load its global warm start")
        best,history = fit_classifier(model.cuda(),data,valid,base/mode,location,config,collator=collate_blocks)
        write_json(location/"fit.json",fitted(config,best,history,len(valid)))
        del model
        gc.collect()


def load_index(seed,aligned):
    prior = historical_index(seed)
    path = (seed_root(WEIGHTS,seed)/"statement_index"/"joint" if aligned else
            seed_root(ARCHIVE_WEIGHTS,seed)/"coarse"/prior["name"])
    model = CoarseClassifier(768,"joint")
    model.load_state_dict(torch.load(path/"best.pt",weights_only=True,map_location="cpu"),strict=True)
    return model.cuda().eval().requires_grad_(False)


def fine_data(split,features,ranks,tokenizer,budget,*,uniform=False,importance_guided=False,with_relations=False):
    started = perf_counter()
    parser = source_parser() if budget and not uniform else None
    blank = torch.full((REGIONS,4),-1,dtype=torch.int32)
    if with_relations and split.syntax is None:
        split.syntax = source_syntax(split.sources)

    def examples():
        for i,(source,ends,f,scores) in enumerate(zip(split.sources,split.ends,features,ranks,strict=True)):
            layout = uniform_layout(source) if uniform else retrieve(scores,source,ends,budget,parser,
                                                                     importance_guided=importance_guided)
            fixed,n = pack_fixed(source.raw_ids,source.keep,layout.spans,tokenizer)
            if n != min(MAX_LENGTH,len(source.ids)+2):
                raise ValueError("Fine input does not fill the historical positional budget")
            example = dict(fixed=fixed,length=n,raw=len(source.raw_ids),compacted_bpe=len(source.ids),coarse=f,
                           uniform=blank,bank=None,layout=layout)
            if with_relations:
                example["relations"] = visible_relations(source,fixed[1].numpy(),split.syntax[i][1])
            yield example
    data = PackedData(split.rows,examples())
    return data,perf_counter()-started


def predict_fine(model,data):
    loader = DataLoader(data,batch_size=BATCH_SIZE,collate_fn=model.collator)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    scores,_ = predict(model,loader)
    return scores,dict(seconds=perf_counter()-started,windows=len(data),batch_calls=len(loader),
                       unpadded_positions=sum(x["length"] for x in data.examples),padded_positions=len(data)*MAX_LENGTH,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated())


def read_predictions(path,rows):
    with gzip.open(path,"rt") as file:
        saved = list(csv.DictReader(file))
    for row,record in zip(rows,saved,strict=True):
        if row["sample_key"]!=record["sample_key"] or row["target"]!=int(record["target"]):
            raise ValueError(f"Sample membership or label mismatch in {path}")
    return saved


def codebert_lengths(name,rows):
    reference = None
    for method in ("stagedvulbert","sea","vulnscout"):
        path = ROOT/"results"/method/f"{name}.csv.gz"
        current = [int(p["original_bpe"]) for p in read_predictions(path,rows)]
        if reference is not None and reference != current:
            raise ValueError(f"CodeBERT length definitions differ at {path}")
        reference = current
    if name == "test":
        counts = [sum(lo<=n<=hi for n in reference) for lo,hi in
                  (GROUPS[k] for k in ("short","511-1020","1021-2040","2041+"))]
        if counts != [305,242,235,346]:
            raise ValueError(f"Unexpected CodeBERT Test census {counts}")
    return reference


def pair_results(rows,scores,sizes):
    result = {}
    for group,(lo,hi) in GROUPS.items():
        ids = [i for a in range(0,len(rows),2) if lo<=max(sizes[a:a+2])<=hi for i in (a,a+1)]
        result[group] = paired_metrics([rows[i] for i in ids],[scores[i] for i in ids])
    return result


def classification(rows,scores,sizes):
    reports = {}
    for group,(lo,hi) in GROUPS.items():
        ids = [i for i,n in enumerate(sizes) if lo<=n<=hi]
        reports[group] = metrics([rows[i]["target"] for i in ids],[scores[i] for i in ids])
    return dict(groups=reports,pairs=pair_results(rows,scores,sizes))


def index_diagnostics(split,scores,edits,sizes):
    result = {}
    for kind,labels in [
            (label,[repair_block_mass(s,b,e,kinds).numpy()>0 for s,b,e in
                    zip(split.sources,split.ends,edits,strict=True)])
            for label,kinds in (("actual",EDIT_TYPES),*((k,(k,)) for k in GAP_TYPES))]:
        result[kind] = {}
        for group,(lo,hi) in GROUPS.items():
            ids = [i for i,n in enumerate(sizes) if lo<=n<=hi]
            predicted, y = [scores[i] for i in ids], [labels[i] for i in ids]
            positives,total = sum(int(v.sum()) for v in y),sum(map(len,y))
            result[kind][group] = (region_metrics(predicted,y,ks=(1,5,10,15)) if 0<positives<total else
                                  dict(functions=len(y),regions=total,positive_regions=positives,ap=None))
    return result


def layout_row(source,layout,syntax,edits,control):
    spans,centers = layout.spans,layout.centers
    units,links,error = syntax
    body = lengths(spans)
    reserved = merge([*centers,*layout.context])
    selected_statements = [u for u in units if intersects(u,centers)]
    touched = [u for u in units if intersects(u,spans)]
    actual = merge(x for kind in EDIT_TYPES for x in edits[kind])
    row = dict(body_bpe=body,center_bpe=lengths(centers),context_bpe=lengths(layout.context),fill_bpe=lengths(layout.fill),
               center_positions=occupied(centers),context_added_positions=occupied(reserved)-occupied(centers),
               fill_added_positions=occupied(spans)-occupied(reserved) if reserved else 0,
               passthrough_bpe=body if len(source.ids)<=BUDGET else 0,fragments=len(spans),
               special_positions=1+len(spans),internal_sep=len(spans)-1,
               fragment_lengths=[b-a for a,b in spans],longest_fragment=max(b-a for a,b in spans),
               center_edit_bpe=overlap_bpe(raw_runs(source.keep,centers),actual),
               context_edit_bpe=overlap_bpe(raw_runs(source.keep,layout.context),actual),
               fill_edit_bpe=overlap_bpe(raw_runs(source.keep,layout.fill),actual),
               selected_statements=len(selected_statements),complete_selected_statements=sum(contains(u,spans) for u in selected_statements),
               touched_statements=len(touched),complete_statements=sum(contains(u,spans) for u in touched),
               parse_error=error,shared_statement_links=0,shared_statement_context_count=0,shared_statement_control_context_count=0)
    if row["center_bpe"]+row["context_bpe"]+row["fill_bpe"]+row["passthrough_bpe"]!=body:
        raise ValueError("Budget attribution does not partition fine source tokens")
    for kind,statement,relation in links:
        if kind not in CONTEXT_TYPES or occupied(merge([statement,relation]))>BUDGET:
            continue
        if (contains(statement,spans) and contains(statement,control.spans) and
                (intersects(statement,centers) or intersects(statement,control.centers))):
            row["shared_statement_links"] += 1
            row["shared_statement_context_count"] += int(contains(relation,spans))
            row["shared_statement_control_context_count"] += int(contains(relation,control.spans))
    return row


def selection_reports(records,edits,sizes):
    retention = grouped_edit_retention((n,a,r["source_runs"]) for n,a,r in zip(sizes,edits,records,strict=True))
    summaries = {}
    for group,(lo,hi) in GROUPS.items():
        selected = [r for n,r in zip(sizes,records,strict=True) if lo<=n<=hi]
        pairs = [records[i] for i in range(0,len(records),2) if lo<=max(sizes[i:i+2])<=hi]
        collisions = sum(r["identical_fine_pair_input"] for r in pairs)
        s = dict(functions=len(selected),pairs=len(pairs),identical_fine_pair_inputs=collisions,
                 fine_pair_collision_rate=collisions/len(pairs) if pairs else None)
        if "identical_global_pair_input" in selected[0]:
            s["identical_complete_pair_inputs"] = sum(r["identical_fine_pair_input"] and
                                                       r["identical_global_pair_input"] for r in pairs)
            s["complete_pair_collision_rate"] = s["identical_complete_pair_inputs"]/len(pairs) if pairs else None
        if "selected_statements" in selected[0]:
            # Layout means and relation diagnostics exclude fully encoded short inputs.
            packed = [r for r in selected if r["compacted_bpe"]>BUDGET]
            s["functions_requiring_selection"] = len(packed)
            if packed:
                for key in ("body_bpe","center_bpe","context_bpe","fill_bpe","special_positions","internal_sep",
                            "fragments","longest_fragment","center_positions","context_added_positions",
                            "fill_added_positions"):
                    s[f"mean_{key}"] = mean(r[key] for r in packed)
                s["mean_fragment_bpe"] = mean(n for r in packed for n in r["fragment_lengths"])
                for key in ("center_edit_bpe","context_edit_bpe","fill_edit_bpe"):
                    s[key] = sum(r[key] for r in packed)
                for prefix,num,den in (
                        ("selected_statement_complete","complete_selected_statements","selected_statements"),
                        ("touched_statement_complete","complete_statements","touched_statements"),
                        ("shared_statement_context","shared_statement_context_count","shared_statement_links"),
                        ("shared_statement_control_context","shared_statement_control_context_count","shared_statement_links")):
                    hits,total = sum(r[num] for r in packed),sum(r[den] for r in packed)
                    s[prefix] = dict(retained=hits,total=total,fraction=hits/total if total else None)
                s["partial_parse_functions"] = sum(r["parse_error"] for r in packed)
        summaries[group] = s
    return dict(edit_retention=retention,layout=summaries)


def evaluate(split,data,scores,ranks,root,name,*,control,coarse_cost,packing_seconds,fine_cost,pipeline_seconds=None,
             privileged_retrieval=False):
    sources,rows = split.sources,split.rows
    raw = [len(s.raw_ids) for s in sources]
    codebert = codebert_lengths(name,rows)
    global_same = source_identity(split)
    ids = [ex["fixed"][0,:ex["length"]].tolist() for ex in data.examples]
    fine_same = [a==b for a,b in zip(ids[::2],ids[1::2],strict=True)]
    records = [dict(i, pair_key=r["pair_key"], codebert_bpe=cb, probability=p, prediction=int(p>.5),
                    compacted_bpe=len(s.ids),input_length=ex["length"],source_runs=visible_source(ex),
                    center_runs=ex["layout"].centers,context_runs=ex["layout"].context,fill_runs=ex["layout"].fill,
                    global_blocks=len(split.ends[j]),global_windows=len(split.windows[j]),
                    global_input_positions=len(s.ids)+2*len(split.windows[j]),
                    global_padded_positions=split.encoding["per_function_padded_positions"][j],
                    identical_global_pair_input=global_same[j//2],identical_fine_pair_input=fine_same[j//2])
               for j,(i,r,cb,p,s,ex) in enumerate(zip(data.metadata,rows,codebert,scores,sources,data.examples,strict=True))]
    root.mkdir(parents=True,exist_ok=True)
    write_prediction_rows(root/f"{name}.csv.gz",records)
    # For source-only runs this is the first access to paired edits during evaluation.
    edits = paired_annotations(sources)
    syntax = split.syntax if split.syntax is not None else source_syntax(sources)
    for record,source,ex,tree,edit,reference in zip(records,sources,data.examples,syntax,edits,control,strict=True):
        record.update(layout_row(source,ex["layout"],tree,edit,reference))
    write_prediction_rows(root/f"{name}.csv.gz",records)
    encoded = {k:v for k,v in split.encoding.items() if k!="per_function_padded_positions"}
    cost = dict(source_seconds=split.source_seconds,syntax_for_blocks_seconds=split.syntax_seconds,
                global_encoding=encoded,index_scoring=coarse_cost,retrieval_and_packing_seconds=packing_seconds,
                fine_encoding=fine_cost,cache_read_seconds=split.cache_read_seconds,
                measured_end_to_end_seconds=pipeline_seconds,
                peak_inference_allocated_bytes=max(encoded["peak_allocated_bytes"], coarse_cost["peak_allocated_bytes"],
                                                   fine_cost["peak_allocated_bytes"]),
                **{key:encoded[key]+fine_cost[key] for key in
                   ("windows","batch_calls","unpadded_positions","padded_positions")})
    indexer = {"unit":"statement-aligned blocks" if split.aligned else "fixed blocks","target_bpe":BLOCK}
    if privileged_retrieval:
        indexer["ranking_source"] = "privileged same-side repair annotations"
    else:
        indexer["unixcoder"] = index_diagnostics(split,ranks,edits,raw)
    return dict(unixcoder=classification(rows,scores,raw)|selection_reports(records,edits,raw),
                codebert=classification(rows,scores,codebert)|selection_reports(records,edits,codebert),
                indexer=indexer,cost=cost)


def zero_control(split,ranks):
    return [retrieve(p,s,e,0) for p,s,e in zip(ranks,split.sources,split.ends,strict=True)]


def archived_fine(tokenizer,seed):
    model = FusionClassifier(fp32_classifier(tokenizer).detector)
    path = seed_root(ARCHIVE_WEIGHTS,seed)/"joint"/"best.pt"
    model.load_state_dict(torch.load(path,weights_only=True,map_location="cpu"),strict=True)
    return model.cuda().eval().requires_grad_(False)


def budget_trials(tokenizer):
    policy_path = RESULTS/"budget_policy.json"
    if policy_path.is_file():
        return read_json(policy_path)
    split = load_split("valid",tokenizer)
    choices = {b:[] for b in BUDGETS}
    for seed in SEEDS:
        index = load_index(seed,False)
        features,ranks,index_cost = score_coarse(index,split,"original")
        fine = archived_fine(tokenizer,seed)
        root = seed_root(RESULTS,seed)/"budget_trials"
        control = zero_control(split,ranks)
        archived = read_predictions(seed_root(ARCHIVE,seed)/"joint"/"valid.csv.gz",split.rows)
        for budget in BUDGETS:
            location = root/str(budget)
            path = location/"metrics.json"
            if path.is_file():
                result = read_json(path)
            else:
                data,seconds = fine_data(split,features,ranks,tokenizer,budget)
                scores,cost = predict_fine(fine,data)
                if budget == 0:
                    for ex,p,saved in zip(data.examples,scores,archived,strict=True):
                        if (ex["layout"].spans != list(map(tuple,json.loads(saved["encoding_runs"]))) or
                                ex["length"] != int(saved["input_length"]) or
                                abs(p-float(saved["probability"])) > 1e-6 or (p>.5)!=bool(int(saved["prediction"]))):
                            raise ValueError(f"Seed {seed} failed historical layout or prediction parity")
                result = evaluate(split,data,scores,ranks,location,"valid",control=control,coarse_cost=index_cost,
                                  packing_seconds=seconds,fine_cost=cost)
                write_json(path,result)
            view = result["unixcoder"]
            layout = view["layout"]["long"]
            context = layout.get("shared_statement_context",layout.get("same_anchor_context"))
            control_context = layout.get("shared_statement_control_context",layout.get("same_anchor_control_context"))
            matched = context["fraction"]-control_context["fraction"]
            choices[budget].append(dict(seed=seed,valid_mcc=view["groups"]["overall"]["mcc"],
                                        long_mcc=view["groups"]["long"]["mcc"],
                                        long_real_edit_retention=view["edit_retention"]["long"]["actual_edits"]["fraction"],
                                        long_shared_statement_context_delta=matched))
            print("Valid budget",seed,budget,choices[budget][-1],flush=True)
        del index,fine
        gc.collect()
    averages = {b:{k:mean(row[k] for row in values) for k in values[0] if k!="seed"}
                for b,values in choices.items()}
    weights = dict(valid_mcc=2,long_real_edit_retention=1,long_shared_statement_context_delta=1)
    candidates = [b for b in BUDGETS if b]
    ranks = {b:sum(weight*(1+sum(averages[other][metric]>averages[b][metric] for other in candidates))
                   for metric,weight in weights.items()) for b in candidates}
    winner = min(candidates,key=lambda b:(ranks[b],-averages[b]["valid_mcc"],b))
    policy = dict(split="complete Valid",checkpoints="60688d2 frozen index and fine models",
                  seeds=list(SEEDS),rank_weights=weights,ranking=ranks,
                  tie_break="higher mean overall Valid MCC then smaller budget",trials=choices,means=averages,
                  zero_parity="historical spans, input length, probability tolerance 1e-6 and strict prediction",
                  selected_context_budget=winner)
    write_json(policy_path,policy)
    print("Frozen Valid selected context budget",winner,flush=True)
    return policy


def valid_winner(methods,policy_path):
    if policy_path.is_file():
        return read_json(policy_path)["selected"]
    provenance = {}
    for method,location in methods.items():
        fit = read_json(location/"metrics.json")
        provenance[method] = dict(selected_epoch=fit["selected_epoch"],valid_mcc=fit["validation"]["mcc"])
    chosen = max(methods,key=lambda method:provenance[method]["valid_mcc"])
    write_json(policy_path,dict(split="full Seed 42 Valid",metric="overall MCC",trials=provenance,selected=chosen,
                                tie_break=list(methods)))
    return chosen


def setup(method,seed):
    root = seed_root(RESULTS,SEED)
    original = ("center_context","statement_context")
    aligned = method == "statement_context"
    fusion = "original"
    if method in ("global_fusion","weighted_fusion","selected_fusion","matched_uniform"):
        retrieval = valid_winner({m:root/m for m in original},RESULTS/"retrieval_policy.json")
        aligned = retrieval=="statement_context"
        if method in ("global_fusion","weighted_fusion"):
            fusion = method.removesuffix("_fusion")
        else:
            winner = valid_winner({m:root/m for m in (retrieval,"global_fusion","weighted_fusion")},
                                  RESULTS/"fusion_policy.json")
            fusion = winner.removesuffix("_fusion") if winner.endswith("_fusion") else "original"
    if method == "selected_fusion" and seed==SEED:
        return None
    return aligned,fusion,method=="matched_uniform"


def fit_fine(tokenizer,seed,method):
    settings = setup(method,seed)
    if settings is None:
        return
    aligned,fusion,uniform = settings
    if method=="matched_uniform" and not aligned and fusion=="original":
        print("Matched uniform already exists at 60688d2 (same indexer, blocks and fusion)",flush=True)
        return
    context = 0 if uniform else read_json(RESULTS/"budget_policy.json")["selected_context_budget"]
    record = "final" if method=="selected_fusion" else method
    area = seed_root(RESULTS,seed)/record
    path = area/"metrics.json"
    config = dict(run_config(method,seed),stage="fine",block_layout="statement32" if aligned else "fixed32",
                  context_budget=context,center_budget=BUDGET-context,
                  selection_policy="historical Uniform Sampling" if uniform else "center-first then syntax then score fill",
                  indexer_checkpoint="statement_index/joint" if aligned else f"60688d2/{historical_index(seed)['name']}",
                  repair_weight=historical_index(seed)["weight"],coarse_frozen=True,fusion=fusion)
    if path.is_file():
        saved = read_json(path)["config"]
        if saved != with_recorded_names(config,saved):
            raise ValueError(f"Different experiment configuration at {path}")
        return
    if aligned:
        fit_indexer(tokenizer,seed)
    index = load_index(seed,aligned)
    return fit_frozen_fine(tokenizer,index,config,area,seed_root(WEIGHTS,seed)/record,uniform=uniform)


def fine_classifier(tokenizer,config):
    relation_attention = config.get("relation_attention",False)
    pretrained = fp32_classifier(tokenizer,additive_attention=relation_attention).detector
    return FusionClassifier(pretrained,config["fusion"],relation_attention=relation_attention).cuda()


def oracle_block_scores(split):
    """Score each function using its own side of the paired repair. Diagnostics only."""
    edits = paired_annotations(split.sources)
    return [repair_block_distribution(source,ends,annotation).numpy()
            for source,ends,annotation in zip(split.sources,split.ends,edits,strict=True)]


def prepare_fine_inputs(split,index,tokenizer,config,*,uniform=False):
    fusion = config["fusion"]
    features,ranks,scoring = score_coarse(index,split,"global" if fusion=="none" else fusion)
    if config.get("privileged_retrieval",False):
        started = perf_counter()
        ranks = oracle_block_scores(split)
        scoring["oracle_labeling_seconds"] = perf_counter()-started
    data,seconds = fine_data(split,features,ranks,tokenizer,config["context_budget"],uniform=uniform,
                             importance_guided=config.get("context_priority")=="importance_guided",
                             with_relations=config.get("relation_attention",False))
    return data,ranks,scoring,seconds


def fit_frozen_fine(tokenizer,index,config,area,weights,*,uniform=False,cached=None,prepared=None):
    method,seed = config["method"],config["seed"]
    aligned = config["block_layout"] == "statement32"
    path = area/"metrics.json"
    if path.is_file():
        result = read_json(path)
        if result["config"] != with_recorded_names(config,result["config"]):
            raise ValueError(f"Different experiment configuration at {path}")
        return result
    if any(p.requires_grad for p in index.parameters()):
        raise ValueError("Fine training requires a frozen indexer")
    datasets, stages = {}, {}
    relation_attention = config.get("relation_attention",False)
    privileged = config.get("privileged_retrieval",False)
    for name in ("train","valid"):
        split = cached[name] if cached is not None else load_split(name,tokenizer,aligned=aligned)
        if prepared is None:
            data,ranks,scoring,seconds = prepare_fine_inputs(split,index,tokenizer,config,uniform=uniform)
        else:
            data,ranks,scoring,seconds = prepared[name]
        stages[name] = dict(source_seconds=split.source_seconds,block_cache_read_seconds=split.cache_read_seconds,
                            index_scoring=scoring,retrieval_and_packing_seconds=seconds)
        if name=="train":
            datasets[name] = data
        else:
            validation = (split,data,ranks,scoring,seconds)
    set_seed(seed)
    model = fine_classifier(tokenizer,config)

    def record_bias(current,epoch):
        return {"relation_bias":current.relation_weights()} if relation_attention else {}

    best,history = fit_classifier(model,datasets["train"],data,weights,area,config,collator=model.collator,
                                  after_epoch=record_bias if relation_attention else None)
    del datasets
    result = fitted(config,best,history,len(data))
    result.update(preparation=stages,split={})
    split,data,ranks,scoring,seconds = validation
    scores,fine = predict_fine(model,data)
    result["split"]["valid"] = evaluate(split,data,scores,ranks,area,"valid",control=zero_control(split,ranks),
                                       coarse_cost=scoring,packing_seconds=seconds,fine_cost=fine,
                                       privileged_retrieval=privileged)
    if result["validation"] != result["split"]["valid"]["unixcoder"]["groups"]["overall"]:
        raise ValueError("Reloaded weights did not reproduce the selected Valid checkpoint")

    encoder = load_context_encoder().requires_grad_(False)
    torch.cuda.synchronize()
    started = perf_counter()
    split = load_split("test",tokenizer,aligned=aligned,encoder=encoder,
                       token_features=index.indexer.matching != "block")
    data,ranks,scoring,seconds = prepare_fine_inputs(split,index,tokenizer,config,uniform=uniform)
    scores,fine = predict_fine(model,data)
    elapsed = perf_counter()-started
    del encoder
    result["split"]["test"] = evaluate(split,data,scores,ranks,area,"test",control=zero_control(split,ranks),
                                      coarse_cost=scoring,packing_seconds=seconds,fine_cost=fine,pipeline_seconds=elapsed,
                                      privileged_retrieval=privileged)
    write_json(path,result)
    print(method,"complete",seed,"epoch",best["epoch"],"Test MCC",
          result["split"]["test"]["unixcoder"]["groups"]["overall"]["mcc"],flush=True)
    return result


def smoke(tokenizer):
    set_seed(SEED)
    rows = prime_source_rows(DATA,"train")
    longest = max(range(len(rows)),key=lambda i:len(rows[i]["func"]))//2*2
    shortest = min(range(0,len(rows),2),key=lambda i:max(len(rows[i]["func"]),len(rows[i+1]["func"])))
    rows = [*rows[longest:longest+2],*rows[shortest:shortest+2]]
    sources = load_sources(rows,tokenizer)
    windows = [encoder_window_ends(s) for s in sources]
    syntax = source_syntax(sources)
    encoder = load_context_encoder().requires_grad_(False)
    artifacts = {}
    for aligned in (False,True):
        ends = cuts_for(sources,syntax,aligned)
        bank,encoded = encode_blocks(encoder,sources,tokenizer,windows,ends)
        if encoded["body_bpe"]!=sum(len(s.ids) for s in sources) or max(map(len,windows))<=1:
            raise ValueError("Smoke must encode the entire extreme source")
        split = SourceSplit(rows,sources,windows,ends,bank,encoded,0.,syntax=syntax,aligned=aligned)
        masses = [repair_block_mass(s,e,a,(*EDIT_TYPES,*GAP_TYPES)) for s,e,a in
                  zip(sources,ends,paired_annotations(sources),strict=True)]
        train = BlockData(rows,ends,bank,[m/m.sum().clamp_min(1) for m in masses])
        index = load_index(SEED,False)
        index.requires_grad_(True).train()
        index.repair_weight = historical_index(SEED)["weight"]
        batch = to_gpu(collate_blocks([train[i] for i in range(len(train))]),labels=True)
        output = index(batch,details=True)
        grad = torch.autograd.grad(output["loss_components"]["train_bce"],index.indexer.parameters(),retain_graph=True)
        output["loss"].backward()
        bce_norm = sum(float(g.norm()) for g in grad)
        query = index.indexer.query.weight
        joint_norm = float(query.grad.norm())
        before = query.detach().clone()
        opt = AdamW(index.parameters(),lr=2e-5)
        opt.step()
        if bce_norm<=0 or joint_norm<=0 or torch.equal(query,before):
            raise ValueError("Task + Repair indexer did not receive gradients and update")
        features,ranks,_ = score_coarse(index.requires_grad_(False),split,"original")
        data,_ = fine_data(split,features,ranks,tokenizer,96)
        model = FusionClassifier(fp32_classifier(tokenizer).detector).cuda()
        batch = to_gpu(collate_fine(data.examples),labels=True)
        out = model(batch)
        out["loss"].backward()
        fusion_gradient = float(model.fusion.weight.grad.norm())
        if not math.isfinite(fusion_gradient) or fusion_gradient<=0 or features.requires_grad:
            raise ValueError("Fine encoder did not receive a valid gradient")
        AdamW(model.parameters(),lr=2e-5).step()
        # Changing partner only cannot affect this function's own packed input.
        single = SourceSplit(rows[:1],sources[:1],windows[:1],ends[:1],bank[:len(ends[0])],{},0.)
        direct,_ = fine_data(single,features[:1],ranks[:1],tokenizer,96)
        if not torch.equal(direct.examples[0]["fixed"],data.examples[0]["fixed"]):
            raise ValueError("Single-source and batch retrieval diverge")
        artifacts["statement32" if aligned else "fixed32"] = dict(encoding=encoded,
            training_bce_gradient=bce_norm,training_joint_query_gradient=joint_norm,fusion_gradient=fusion_gradient,
            fine_lengths=[ex["length"] for ex in data.examples],
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
        del model,index,opt,batch
        gc.collect()
    write_json(RESULTS/"smoke.json",dict(original_bpe=[len(s.raw_ids) for s in sources],
               compacted_bpe=[len(s.ids) for s in sources],windows=list(map(len,windows)),layouts=artifacts))
    print("GPU smoke passed",max(len(s.raw_ids) for s in sources),"original BPE",flush=True)


def baseline_layouts(method,sources,syntax,edits,saved,parser):
    for source,tree,edit,record in zip(sources,syntax,edits,saved,strict=True):
        raw = list(map(tuple,json.loads(record["source_runs"])))
        spans = merge((bisect_left(source.keep,a),bisect_left(source.keep,b)) for a,b in raw)
        if raw_runs(source.keep,spans)!=raw or int(record["input_length"])!=occupied(spans)+2:
            raise ValueError("Saved baseline positions do not match the Simple source mapping")
        centers,context = [],[]
        if len(source.ids)>BUDGET:
            if method=="historical_coarse_to_fine":
                ends = block_ends(len(source.ids),(),(),BLOCK,False)
                centers = merge((a,b) for a,b in zip([0,*ends[:-1]],ends) if contains((a,b),spans))
            else:
                centers = merge(base_windows(source.boundaries)[0])
                if method=="context_aware":
                    links,_ = related_statements(source.code,source.offsets,source.keep,centers,parser)
                    context = uncovered(prefer_context(centers,links),centers)
        reserved = merge([*centers,*context])
        if uncovered(reserved,spans):
            raise ValueError("Reconstructed parts disagree with the saved layout")
        layout = Layout(spans,centers,context,uncovered(spans,reserved) if reserved else [])
        yield layout_row(source,layout,tree,edit,layout)


def baseline_report(path,rows,*,codebert,baseline=None):
    saved = read_predictions(path/"test.csv.gz",rows)
    scores = [float(x["probability"]) for x in saved]
    sizes = [int(x["original_bpe"]) for x in saved]
    if any(int(p["prediction"])!=int(score>.5) for p,score in zip(saved,scores,strict=True)):
        raise ValueError(f"Historical fixed-threshold prediction differs at {path}")
    result = dict(codebert=classification(rows,scores,codebert))
    meta = read_json(path/"metrics.json")
    if baseline is not None:
        method,sources,syntax,edits,parser = baseline
        if sizes != [len(source.raw_ids) for source in sources]:
            raise ValueError(f"Saved UniXcoder lengths at {path} disagree with current source tokenization")
        result["unixcoder"] = classification(rows,scores,sizes)
        records = [dict(geometry,source_runs=json.loads(p["source_runs"]),compacted_bpe=int(p["compacted_bpe"]),
                        **({"identical_global_pair_input":p["identical_global_pair_input"]=="True"}
                           if "identical_global_pair_input" in p else {}),
                        identical_fine_pair_input=p.get("identical_fine_pair_input",p.get("identical_pair_input"))=="True")
                   for p,geometry in zip(saved,baseline_layouts(method,sources,syntax,edits,saved,parser),strict=True)]
        for key,values in (("unixcoder",sizes),("codebert",codebert)):
            result[key].update(selection_reports(records,edits,values))
    if "train_seconds" in meta:
        result["train_seconds"] = meta["train_seconds"]
    elif "training_seconds" in meta:
        result["train_seconds"] = meta["training_seconds"]
    split = meta.get("split",{}).get("test",{})
    result["peak_train_allocated_bytes"] = meta.get("peak_train_allocated_bytes")
    result["cost"] = meta.get("uncached_test_measurement",split.get("cost",{
        "inference_seconds":split.get("inference_seconds"),
        "preparation_seconds":meta.get("preparation_seconds",{}).get("test"),
        "peak_inference_allocated_bytes":split.get("peak_inference_allocated_bytes"),
        "model_tokens":split.get("model_tokens")}))
    return result


def summarize(tokenizer):
    rows = prime_source_rows(DATA,"test")
    sources = load_sources(rows,tokenizer)
    edits = paired_annotations(sources)
    syntax = source_syntax(sources)
    parser = source_parser()
    codebert = codebert_lengths("test",rows)
    policy = read_json(RESULTS/"fusion_policy.json")
    retrieval = read_json(RESULTS/"retrieval_policy.json")["selected"]
    verify_fusion_inputs(retrieval)
    history = {}
    for seed in SEEDS:
        root = seed_root(RESULTS,seed)
        history[seed] = {}
        for method,name in (("historical_coarse_to_fine","joint"),("historical_global_uniform","uniform"),
                            ("context_aware","context_aware")):
            path = seed_root(ARCHIVE,seed)/name if name!="context_aware" else UNIX_RESULTS/name
            if name=="context_aware" and seed!=SEED:
                path /= f"seed{seed}"
            history[seed][method] = baseline_report(path,rows,codebert=codebert,
                baseline=(method,sources,syntax,edits,parser))
        for method in ((*METHODS[:4], "matched_uniform") if seed==SEED else ("final","matched_uniform")):
            location = root/method/"metrics.json"
            if method=="matched_uniform" and not location.is_file():
                if setup(method,seed)!=(False,"original",True):
                    raise FileNotFoundError(location)
                continue
            value = read_json(location)
            history[seed][method] = value["split"]["test"]|{"train_seconds":value["train_seconds"],
                                         "peak_train_allocated_bytes":value["peak_train_allocated_bytes"],
                                         "selected_epoch":value["selected_epoch"],"valid_mcc":value["validation"]["mcc"]}
        history[seed]["simplified"] = history[seed][policy["selected"] if seed==SEED else "final"]
        history[seed]["matched_uniform"] = history[seed].get("matched_uniform",history[seed]["historical_global_uniform"])
    metrics_keys = ("mcc","f1","precision","recall","auc","auprc")
    averages = {}
    costs = {}
    for method in ("historical_coarse_to_fine","historical_global_uniform","context_aware","simplified","matched_uniform"):
        averages[method] = {}
        runs = [history[seed][method] for seed in SEEDS]
        costs[method] = dict(
            fine_training_seconds=seed_stats([run["train_seconds"] for run in runs]),
            peak_train_allocated_bytes=seed_stats([run["peak_train_allocated_bytes"] for run in runs]),
            test_pipeline_seconds=seed_stats([pipeline_seconds(run["cost"]) for run in runs]))
        for kind in ("unixcoder","codebert"):
            averages[method][kind] = {}
            for group in GROUPS:
                values = {}
                for field in metrics_keys+("p_c","pair_ranking_accuracy"):
                    section = "groups" if field in metrics_keys else "pairs"
                    values[field] = seed_stats([run[kind][section][group][field] for run in runs])
                for field in ("actual_edits",*GAP_TYPES):
                    values[f"retention_{field}"] = seed_stats(
                        [run[kind]["edit_retention"][group][field]["fraction"] for run in runs])
                layouts = [run[kind]["layout"][group] for run in runs]
                for field in layouts[0]:
                    if field.startswith("mean_") or field.endswith("_collision_rate"):
                        values[field] = seed_stats([layout[field] for layout in layouts])
                for field,archive_field in (("selected_statement_complete","anchor_statement_complete"),
                                            ("shared_statement_context","same_anchor_context"),
                                            ("touched_statement_complete","touched_statement_complete")):
                    retention = [layout.get(field,layout.get(archive_field)) for layout in layouts]
                    if all(retention):
                        values[field] = seed_stats([value["fraction"] for value in retention])
                averages[method][kind][group] = values
    external = {name:baseline_report(ROOT/"results"/name,rows,codebert=codebert)
                for name in ("stagedvulbert","sea","vulnscout")}
    write_json(RESULTS/"summary.json",dict(seeds=list(SEEDS),selected_fusion=policy,per_seed=history,
                                          three_seed=averages,three_seed_cost=costs,archived_external=external))
    print("Saved paired and function metrics under both length definitions",flush=True)


def verify_fusion_inputs(retrieval):
    location = seed_root(RESULTS,SEED)
    for name in ("valid","test"):
        rows = prime_source_rows(DATA,name)
        reference = read_predictions(location/retrieval/f"{name}.csv.gz",rows)
        for method in ("global_fusion","weighted_fusion"):
            other = read_predictions(location/method/f"{name}.csv.gz",rows)
            fields = ("input_length","source_runs","center_runs","context_runs","fill_runs","global_blocks","global_windows")
            if any(a[field]!=b[field] for a,b in zip(reference,other,strict=True) for field in fields):
                raise ValueError(f"{method} changed the fixed {name} fine layout")


def pipeline_seconds(cost):
    for field in ("measured_end_to_end_seconds","seconds","uncached_pipeline_seconds"):
        if cost.get(field) is not None:
            return cost[field]
    return cost["preparation_seconds"]+cost["inference_seconds"]


def seed_stats(values):
    return dict(mean=mean(values),sample_std=stdev(values),seeds=values) if None not in values else dict(
        mean=None,sample_std=None,seeds=values)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--stage",required=True,choices=("smoke","budget","index","fine","report"))
    cli.add_argument("--seed",type=int,choices=SEEDS,default=SEED)
    cli.add_argument("--methods",nargs="+",choices=METHODS)
    args = cli.parse_args()
    torch.set_num_threads(4)
    tokenizer = load_tokenizer("unixcoder",UNIX_BASE)
    if args.stage=="report":
        summarize(tokenizer)
        return
    if torch.cuda.device_count()!=1 or "4090" not in torch.cuda.get_device_name():
        raise ValueError("Select one RTX 4090 via CUDA_VISIBLE_DEVICES")
    torch.backends.cudnn.deterministic = True
    if args.stage=="smoke":
        smoke(tokenizer)
    elif args.stage=="budget":
        budget_trials(tokenizer)
    elif args.stage=="index":
        fit_indexer(tokenizer,args.seed)
    elif args.methods:
        for method in args.methods:
            fit_fine(tokenizer,args.seed,method)
    else:
        cli.error("--stage fine requires --methods")


if __name__=="__main__":
    main()
