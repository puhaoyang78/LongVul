"""Compare syntax-relation priorities under the same trained block indexer."""
import argparse
from time import perf_counter

import torch
from transformers import set_seed

import coarse_to_fine as pipeline
from coarse_encoding import (BlockData, CoarseClassifier, FusionClassifier, collate_blocks, collate_fine,
                             encode_blocks, repair_block_mass)
from coarse_retrieval import uncovered
from compaction import BUDGET, MAX_LENGTH
from locator_quality import load_context_encoder
from paired import AdamW, DATA, ROOT, SEED, UNIX_BASE, load_tokenizer, prime_source_rows, to_gpu
from predicted_diff import fp32_classifier, load_sources, paired_annotations
from repair_oracle import EDIT_TYPES, GAP_TYPES
from rgcl.artifacts import write_json
from rgcl.config import SEEDS

RESULTS = ROOT/"results"/"context_retrieval_512"
WEIGHTS = ROOT/"checkpoints"/RESULTS.name
METHODS = ("relation_based","importance_guided")
CONTEXT_BUDGET = 96
INDEX = "block_index"


def fit_seed(tokenizer,seed,methods=METHODS):
    area,weights = (pipeline.seed_root(parent,seed) for parent in (RESULTS,WEIGHTS))
    pending = [method for method in methods if not (area/method/"metrics.json").is_file()]
    if not pending:
        return
    splits = {name:pipeline.load_split(name,tokenizer) for name in ("train","valid")}
    train,valid = (splits[name] for name in ("train","valid"))
    index_fit = pipeline.fit_repair_indexer(train,valid,seed,INDEX,area/"coarse",weights/"coarse")
    index = load_index(seed,train.vectors.shape[-1])
    prepared = {method:{} for method in METHODS}
    verified = {}
    for name,split in splits.items():
        features,ranks,scoring = pipeline.score_coarse(index,split,"global")
        for method in METHODS:
            data,seconds = pipeline.fine_data(split,features,ranks,tokenizer,CONTEXT_BUDGET,
                                              importance_guided=method=="importance_guided")
            prepared[method][name] = (data,ranks,scoring,seconds)
        reference,guided = (prepared[m][name][0].examples for m in METHODS)
        counts = dict(functions=len(reference),same_selected_blocks=0,selected_functions=0,
                      changed_fine_inputs=0)
        for a,b in zip(reference,guided,strict=True):
            left,right = a["layout"],b["layout"]
            if left.centers != right.centers or not torch.equal(a["coarse"],b["coarse"]):
                raise ValueError("The comparison must share selected code blocks and global representation")
            counts["same_selected_blocks"] += 1
            counts["selected_functions"] += bool(left.centers)
            counts["changed_fine_inputs"] += int(not torch.equal(a["fixed"],b["fixed"]))
        verified[name] = counts
    if len(pending)==len(METHODS) or not (area/"shared_inputs.json").is_file():
        write_json(area/"shared_inputs.json",verified)
    for method in pending:
        config = dict(pipeline.run_config(method,seed),stage="fine",fusion="global",block_layout="fixed32",
                      context_budget=CONTEXT_BUDGET,center_budget=BUDGET-CONTEXT_BUDGET,
                      context_priority=method,indexer_checkpoint=f"coarse/{INDEX}/best.pt",
                      index_selected_epoch=index_fit["selected_epoch"],repair_weight=pipeline.REPAIR_INDEX_WEIGHT,
                      precision="FP32 Eager",loss="function BCE",
                      initialization="original pretrained UniXcoder; fresh Global Fusion and classification head")
        pipeline.fit_frozen_fine(tokenizer,index,config,area/method,weights/method,cached=splits,
                                 prepared=prepared[method])


def load_index(seed,width):
    location = pipeline.seed_root(WEIGHTS,seed)/"coarse"/INDEX/"best.pt"
    model = CoarseClassifier(width,"joint")
    model.load_state_dict(torch.load(location,weights_only=True,map_location="cpu"),strict=True)
    return model.cuda().eval().requires_grad_(False)


def smoke(tokenizer):
    set_seed(SEED)
    rows = prime_source_rows(DATA,"train")
    longest = max(range(len(rows)),key=lambda i:len(rows[i]["func"]))//2*2
    shortest = min(range(0,len(rows),2),key=lambda i:max(len(rows[i]["func"]),len(rows[i+1]["func"])))
    rows = [*rows[longest:longest+2],*rows[shortest:shortest+2]]
    sources = load_sources(rows,tokenizer)
    windows = [pipeline.encoder_window_ends(s) for s in sources]
    ends = pipeline.cuts_for(sources,None,False)
    encoder = load_context_encoder().requires_grad_(False)
    blocks,encoded = encode_blocks(encoder,sources,tokenizer,windows,ends)
    if max(map(len,windows))<2 or encoded["body_bpe"]!=sum(len(s.ids) for s in sources):
        raise ValueError("Incomplete extreme-length encoding")
    del encoder
    split = pipeline.SourceSplit(rows,sources,windows,ends,blocks,encoded,0.)
    targets = []
    for source,cuts,edits in zip(sources,ends,paired_annotations(sources),strict=True):
        mass = repair_block_mass(source,cuts,edits,(*EDIT_TYPES,*GAP_TYPES))
        targets.append(mass/mass.sum().clamp_min(1))
    data = BlockData(rows,ends,blocks,targets)
    index = CoarseClassifier(blocks.shape[-1],"joint",pipeline.REPAIR_INDEX_WEIGHT).cuda()
    inputs = to_gpu(collate_blocks([data[i] for i in range(len(data))]),labels=True)
    result = index(inputs)
    result["loss"].backward()
    gradient = float(index.indexer.query.weight.grad.norm())
    if not 0.<gradient<float("inf"):
        raise ValueError("Index Q/K gradient is missing or nonfinite")
    AdamW(index.parameters(),lr=2e-5).step()
    features,ranks,_ = pipeline.score_coarse(index.requires_grad_(False),split,"global")
    packs = [pipeline.fine_data(split,features,ranks,tokenizer,CONTEXT_BUDGET,
                               importance_guided=m=="importance_guided")[0] for m in METHODS]
    for a,b in zip(*(d.examples for d in packs),strict=True):
        if a["layout"].centers!=b["layout"].centers or a["length"]!=b["length"] or a["length"]>MAX_LENGTH:
            raise ValueError("Selection changed code blocks or positional budget")
        if uncovered(b["layout"].centers,b["layout"].spans):
            raise ValueError("Context selection discarded a selected code block")
    detector = FusionClassifier(fp32_classifier(tokenizer).detector,"global").cuda()
    checks = {}
    for name,pack in zip(METHODS,packs,strict=True):
        detector.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        loss = detector(to_gpu(collate_fine(pack.examples),labels=True))["loss"]
        loss.backward()
        fusion_gradient = float(detector.fusion.weight.grad.norm())
        if not 0.<fusion_gradient<float("inf"):
            raise ValueError("Fine gradient is missing or nonfinite")
        AdamW(detector.parameters(),lr=2e-5).step()
        checks[name] = dict(loss=float(loss.detach()),fusion_gradient=fusion_gradient,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated())
    write_json(RESULTS/"smoke.json",dict(original_bpe=[len(s.raw_ids) for s in sources],
               windows=list(map(len,windows)),body_bpe=encoded["body_bpe"],query_gradient=gradient,
               fine_lengths=[x["length"] for x in packs[0].examples],methods=checks))
    print("GPU smoke passed",max(len(s.raw_ids) for s in sources),"original BPE",flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage",choices=("smoke","run"))
    cli.add_argument("--seed",type=int,choices=SEEDS,default=SEED)
    cli.add_argument("--method",choices=METHODS)
    args = cli.parse_args()
    torch.set_num_threads(4)
    if torch.cuda.device_count()!=1 or "4090" not in torch.cuda.get_device_name():
        raise ValueError("Select one RTX 4090 via CUDA_VISIBLE_DEVICES")
    torch.backends.cudnn.deterministic = True
    tokenizer = load_tokenizer("unixcoder",UNIX_BASE)
    if args.stage=="smoke":
        smoke(tokenizer)
    else:
        fit_seed(tokenizer,args.seed,(args.method,) if args.method else METHODS)


if __name__=="__main__":
    main()
