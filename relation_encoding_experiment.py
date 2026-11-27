"""Matched experiments for additive statement relations and global classification features."""
import argparse

import torch
from transformers import set_seed

import coarse_to_fine as pipeline
from coarse_encoding import CoarseClassifier, encode_blocks
from compaction import BUDGET, MAX_LENGTH
from locator_quality import load_context_encoder
from paired import AdamW, DATA, ROOT, SEED, UNIX_BASE, load_tokenizer, prime_source_rows, to_gpu
from predicted_diff import load_sources
from relation_attention import RELATION_TYPES
from rgcl.artifacts import read_json, write_json
from rgcl.config import SEEDS

RESULTS = ROOT/"results"/"relation_encoding_512"
WEIGHTS = ROOT/"checkpoints"/RESULTS.name
METHODS = {
    "standard_fusion":("global",False),
    "standard_fine_only":("none",False),
    "relation_aware_fusion":("global",True),
    "relation_aware_fine_only":("none",True),
}
BASELINE = "standard_fusion"
CONTEXT_BUDGET = 96
INDEX = "block_index"


def load_index(seed,width):
    path = pipeline.seed_root(WEIGHTS,seed)/"coarse"/INDEX/"best.pt"
    index = CoarseClassifier(width,"joint")
    index.load_state_dict(torch.load(path,weights_only=True,map_location="cpu"),strict=True)
    return index.cuda().eval().requires_grad_(False)


def fit_seed(tokenizer,seed,methods):
    area,weights = (pipeline.seed_root(path,seed) for path in (RESULTS,WEIGHTS))
    pending = [method for method in methods if not (area/method/"metrics.json").is_file()]
    if not pending:
        return
    splits = {name:pipeline.load_split(name,tokenizer) for name in ("train","valid")}
    train,valid = (splits[name] for name in ("train","valid"))
    index_fit = pipeline.fit_repair_indexer(train,valid,seed,INDEX,area/"coarse",weights/"coarse")
    index = load_index(seed,train.vectors.shape[-1])
    prepared = {}
    for name,split in splits.items():
        features,ranks,scoring = pipeline.score_coarse(index,split,"global")
        data,seconds = pipeline.fine_data(split,features,ranks,tokenizer,CONTEXT_BUDGET,
                                          with_relations=any(METHODS[m][1] for m in pending))
        prepared[name] = (data,ranks,scoring,seconds)
    for method in pending:
        fusion,relations = METHODS[method]
        config = dict(pipeline.run_config(method,seed),stage="fine",fusion=fusion,block_layout="fixed32",
                      context_budget=CONTEXT_BUDGET,center_budget=BUDGET-CONTEXT_BUDGET,
                      context_priority="relation_based",indexer_checkpoint=f"coarse/{INDEX}/best.pt",
                      index_selected_epoch=index_fit["selected_epoch"],repair_weight=pipeline.REPAIR_INDEX_WEIGHT,
                      relation_attention=relations,relation_types=list(RELATION_TYPES) if relations else [],
                      bias_initialization="zero, three weights shared across heads and layers" if relations else None,
                      precision="FP32 Eager",loss="function BCE",
                      initialization="original pretrained UniXcoder and fresh classification head")
        pipeline.fit_frozen_fine(tokenizer,index,config,area/method,weights/method,cached=splits,prepared=prepared)


def choose():
    area = pipeline.seed_root(RESULTS,SEED)
    winner = pipeline.valid_winner({m:area/m for m in METHODS if m!=BASELINE},RESULTS/"method_policy.json")
    print(winner,flush=True)


def smoke(tokenizer):
    rows = prime_source_rows(DATA,"train")
    longest = max(range(len(rows)),key=lambda i:len(rows[i]["func"]))//2*2
    shortest = min(range(0,len(rows),2),key=lambda i:max(len(rows[i]["func"]),len(rows[i+1]["func"])))
    fixed_rows = [*rows[longest:longest+2],*rows[shortest:shortest+2]]
    candidates = [r for r in rows if 250<=len(r["func"])<=1000][:192]
    short_sources = load_sources(candidates,tokenizer)
    unseen = set(RELATION_TYPES)
    for row,source,(_,links,_) in zip(candidates,short_sources,pipeline.source_syntax(short_sources),strict=True):
        kinds = unseen & {kind for kind,_,_ in links}
        if kinds and len(source.ids)<=BUDGET:
            fixed_rows.append(row)
            unseen.difference_update(kinds)
        if not unseen:
            break
    if unseen:
        raise ValueError(f"Real sample does not exercise relation types {unseen}")
    sources = load_sources(fixed_rows,tokenizer)
    ends = pipeline.cuts_for(sources,None,False)
    windows = [pipeline.encoder_window_ends(s) for s in sources]
    encoder = load_context_encoder().requires_grad_(False)
    blocks,cost = encode_blocks(encoder,sources,tokenizer,windows,ends)
    if cost["body_bpe"]!=sum(len(s.ids) for s in sources):
        raise ValueError("Incomplete frozen encoding")
    del encoder
    split = pipeline.SourceSplit(fixed_rows,sources,windows,ends,blocks,cost,0.)
    set_seed(SEED)
    index = CoarseClassifier(blocks.shape[-1],"joint").cuda().eval()
    features,ranks,_ = pipeline.score_coarse(index,split,"global")
    data,_ = pipeline.fine_data(split,features,ranks,tokenizer,CONTEXT_BUDGET,with_relations=True)
    models = []
    for fusion,relations in METHODS.values():
        set_seed(SEED)
        models.append(pipeline.fine_classifier(tokenizer,dict(fusion=fusion,relation_attention=relations)).eval())
    del index
    initial_rng = torch.get_rng_state()
    logits = []
    for model in models:
        calls = []
        with model.fine.encoder.register_forward_hook(lambda *args: calls.append(1)), torch.inference_mode():
            result = model(to_gpu(model.collator(data.examples)),details=True)
        if len(calls)!=1:
            raise ValueError("A fine batch required multiple encoder calls")
        logits.append(result["logits"])
    for result in logits:
        torch.testing.assert_close(logits[0],result,atol=1e-6,rtol=1e-6)
    if not torch.equal(torch.get_rng_state(),initial_rng):
        raise ValueError("Evaluation changed CPU initialization/shuffle RNG")
    model = models[2]
    del models
    torch.cuda.reset_peak_memory_stats()
    model.train()
    inputs = to_gpu(model.collator(data.examples),labels=True)
    positions = inputs["fixed"][:,1]>=0
    if (inputs["relations"] & (~(positions[:,None,:] & positions[:,:,None])).to(torch.uint8)*7).any():
        raise ValueError("A relation includes a padding or special position")
    loss = model(inputs)["loss"]
    loss.backward()
    gradient = model.relation_bias.grad.detach().cpu()
    if not gradient.isfinite().all() or not (gradient!=0).all():
        raise ValueError("A trainable relation bias did not receive BCE gradient")
    AdamW(model.parameters(),lr=2e-5).step()
    if not model.relation_bias.detach().ne(0).all():
        raise ValueError("An optimizer step did not update a relation bias")
    if max(x["length"] for x in data.examples)!=MAX_LENGTH:
        raise ValueError("Unexpected fine positional budget")
    write_json(RESULTS/"smoke.json",dict(
        max_original_bpe=max(len(s.raw_ids) for s in sources),windows=list(map(len,windows)),
        fine_lengths=[x["length"] for x in data.examples],relation_counts=[len(x["relations"]) for x in data.examples],
        max_zero_bias_logit_error=max(float((x-logits[0]).abs().max()) for x in logits),
        bias_gradients=dict(zip(RELATION_TYPES,gradient.tolist(),strict=True)),
        bias_after_step=model.relation_weights(),peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    print("GPU relation smoke passed",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("smoke","run","choose"))
    parser.add_argument("--seed",type=int,choices=SEEDS,default=SEED)
    parser.add_argument("--methods",choices=METHODS,nargs="+")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.stage=="choose":
        choose()
        return
    if torch.cuda.device_count()!=1 or "4090" not in torch.cuda.get_device_name():
        raise ValueError("Select one RTX 4090 via CUDA_VISIBLE_DEVICES")
    torch.backends.cudnn.deterministic = True
    tokenizer = load_tokenizer("unixcoder",UNIX_BASE)
    if args.stage=="smoke":
        smoke(tokenizer)
    else:
        if not args.methods:
            parser.error("run requires --methods")
        if args.seed!=SEED:
            selected = read_json(RESULTS/"method_policy.json")["selected"]
            if not set(args.methods)<={BASELINE,"standard_fine_only",selected}:
                parser.error("supplementary seeds use the standard controls or the Valid-selected method")
        fit_seed(tokenizer,args.seed,args.methods)


if __name__=="__main__":
    main()
