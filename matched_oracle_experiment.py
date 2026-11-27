"""Privileged Oracle retrieval with the published fine input budget and ranking policy."""
import argparse

import numpy as np
import torch
from transformers import set_seed

import coarse_to_fine as pipeline
from coarse_encoding import encode_blocks
from compaction import BUDGET, MAX_LENGTH
from context_selection import occupied
from locator_quality import load_context_encoder
from paired import AdamW, DATA, ROOT, SEED, UNIX_BASE, load_tokenizer, prime_source_rows, to_gpu
from predicted_diff import load_sources
from relation_encoding_experiment import BASELINE, RESULTS as PREDICTED_RESULTS, load_index
from rgcl.artifacts import read_json, write_json
from rgcl.config import SEEDS

RESULTS = ROOT/"results"/"matched_oracle_512"
WEIGHTS = ROOT/"checkpoints"/RESULTS.name
METHODS = {"oracle_fusion":"global","oracle_fine_only":"none"}


def predicted_config(seed):
    path = pipeline.seed_root(PREDICTED_RESULTS,seed)/BASELINE/"config.json"
    config = read_json(path)
    expected = dict(block_layout="fixed32",block_size=32,context_priority="relation_based",context_budget=96,
                    center_budget=BUDGET-96,length=MAX_LENGTH,fusion="global",relation_attention=False,
                    repair_weight=pipeline.REPAIR_INDEX_WEIGHT,seed=seed)
    if any(config[k]!=v for k,v in expected.items()):
        raise ValueError("Predicted reference differs from the fixed matched-Oracle protocol")
    return config


def oracle_config(seed,method):
    return dict(predicted_config(seed),method=method,fusion=METHODS[method],
                privileged_retrieval=True,privileged_splits=["train","valid","test"],
                block_ranking="same-side actual edit and empty-side proxy BPE mass, normalized",
                equal_score_order="source order via stable descending sort",
                indexer_checkpoint_source=f"relation_encoding_512/seed{seed}")


def score_counts(ranks):
    positive = [p[p>0] for p in ranks]
    return dict(functions=len(ranks),no_effective_score=sum(len(p)==0 for p in positive),
                total_blocks=sum(map(len,ranks)),zero_score_blocks=sum(len(p)-len(q) for p,q in zip(ranks,positive,strict=True)),
                functions_with_positive_ties=sum(len(np.unique(p))<len(p) for p in positive))


def fit_seed(tokenizer,seed,methods):
    area,weights = (pipeline.seed_root(root,seed) for root in (RESULTS,WEIGHTS))
    pending = [method for method in methods if not (area/method/"metrics.json").is_file()]
    if not pending:
        return
    splits = {name:pipeline.load_split(name,tokenizer) for name in ("train","valid")}
    index = load_index(seed,splits["train"].vectors.shape[-1])
    # Both classifiers train on the same privileged inputs for this seed.
    prep_config = oracle_config(seed,"oracle_fusion")
    prepared = {name:pipeline.prepare_fine_inputs(split,index,tokenizer,prep_config)
                for name,split in splits.items()}
    for method in pending:
        config = oracle_config(seed,method)
        path = area/method
        write_json(path/"training_repair_scores.json",{name:score_counts(out[1]) for name,out in prepared.items()})
        result = pipeline.fit_frozen_fine(tokenizer,index,config,path,weights/method,cached=splits,prepared=prepared)
        if not result["config"]["privileged_retrieval"]:
            raise ValueError("Oracle output is missing its privileged information label")


def smoke(tokenizer):
    rows = prime_source_rows(DATA,"train")
    longest = max(range(len(rows)),key=lambda i:len(rows[i]["func"]))//2*2
    shortest = min(range(0,len(rows),2),key=lambda i:max(len(rows[i]["func"]),len(rows[i+1]["func"])))
    rows = [*rows[longest:longest+2],*rows[shortest:shortest+2]]
    sources = load_sources(rows,tokenizer)
    windows = [pipeline.encoder_window_ends(s) for s in sources]
    ends = pipeline.cuts_for(sources,None,False)
    encoder = load_context_encoder().requires_grad_(False)
    blocks,encoded = encode_blocks(encoder,sources,tokenizer,windows,ends)
    if encoded["body_bpe"]!=sum(len(s.ids) for s in sources):
        raise ValueError("Frozen encoding did not cover the complete functions")
    del encoder
    split = pipeline.SourceSplit(rows,sources,windows,ends,blocks,encoded,0.)
    index = load_index(SEED,blocks.shape[-1])
    pred,_p,_,_ = pipeline.prepare_fine_inputs(split,index,tokenizer,predicted_config(SEED))
    oracle,scores,_,_ = pipeline.prepare_fine_inputs(split,index,tokenizer,oracle_config(SEED,"oracle_fusion"))
    for a,b,source in zip(pred.examples,oracle.examples,sources,strict=True):
        torch.testing.assert_close(a["coarse"],b["coarse"],atol=0.,rtol=0.)
        if a["length"]!=b["length"] or b["length"]!=min(MAX_LENGTH,len(source.ids)+2):
            raise ValueError("Prediction and Oracle used different position limits")
        if occupied(b["layout"].centers) > BUDGET-96:
            raise ValueError("Oracle used a larger center budget")
        packed_ids,positions = b["fixed"][:2].numpy()
        valid = positions>=0
        np.testing.assert_array_equal(packed_ids[valid],source.raw_ids[positions[valid]])
    if all(torch.equal(a["fixed"],b["fixed"]) for a,b in zip(pred.examples,oracle.examples,strict=True)):
        raise ValueError("Real smoke inputs did not exercise score replacement")
    validation = {}
    for method in METHODS:
        set_seed(SEED)
        model = pipeline.fine_classifier(tokenizer,oracle_config(SEED,method)).train()
        batch = to_gpu(model.collator(oracle.examples),labels=True)
        torch.cuda.reset_peak_memory_stats()
        calls = []
        with model.fine.encoder.register_forward_hook(lambda *args:calls.append(1)):
            output = model(batch)
        output["loss"].backward()
        parameter = model.fine.classifier.out_proj.weight
        before = parameter.detach().clone()
        AdamW(model.parameters(),lr=2e-5).step()
        if len(calls)!=1 or torch.equal(before,parameter):
            raise ValueError("Fine encoding did not perform one forward and update the classifier")
        validation[method] = dict(loss=float(output["loss"].detach()),fine_encoder_calls=len(calls),
                                  peak_allocated_bytes=torch.cuda.max_memory_allocated())
        del model,batch,output
    write_json(RESULTS/"smoke.json",dict(privileged_retrieval=True,original_bpe=[len(s.raw_ids) for s in sources],
               windows=list(map(len,windows)),fine_lengths=[x["length"] for x in oracle.examples],
               exact_global_state_match=True,repair_scores=score_counts(scores),methods=validation))
    print("Matched Oracle GPU smoke passed",max(len(s.raw_ids) for s in sources),"original BPE",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("smoke","run"))
    parser.add_argument("--seed",type=int,choices=SEEDS,default=SEED)
    parser.add_argument("--methods",choices=METHODS,nargs="+",default=tuple(METHODS))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.device_count()!=1 or "4090" not in torch.cuda.get_device_name():
        raise ValueError("Select one RTX 4090 via CUDA_VISIBLE_DEVICES")
    tokenizer = load_tokenizer("unixcoder",UNIX_BASE)
    if args.stage=="smoke":
        smoke(tokenizer)
    else:
        fit_seed(tokenizer,args.seed,args.methods)


if __name__=="__main__":
    main()
