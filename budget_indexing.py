"""Select index checkpoints by deployed retrieval and train with budget-matched ranking."""
import argparse
from dataclasses import replace
from itertools import accumulate, islice
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
from transformers import set_seed

import coarse_to_fine as pipeline
from coarse_encoding import (BlockData, CoarseClassifier, collate_blocks, encode_blocks, repair_block_distribution,
                             window_block_requests)
from coarse_retrieval import retrieve
from context_selection import source_parser
from locator_quality import CONTEXT_BATCH, context_window_inputs, load_context_encoder
from matched_oracle_experiment import predicted_config
from paired import AdamW, DATA, ROOT, SEED, UNIX_BASE, load_tokenizer, prime_source_rows, to_gpu
from predicted_diff import load_sources, paired_annotations
from rgcl.artifacts import read_json, write_json
from rgcl.config import SEEDS

RESULTS = ROOT/"results"/"budget_indexing_512"
WEIGHTS = ROOT/"checkpoints"/RESULTS.name
RETRIEVAL, RANKING, ADAPTED, DIAGNOSTIC = "retrieval_selection", "budget_ranking", "adapted_encoding", "adapted_index_frozen_fusion"
JOINT, ADAPTED_JOINT, JOINT_DIAGNOSTIC = "joint_supervision", "adapted_joint_supervision", "adapted_joint_frozen_fusion"
BASE_METHODS = (RETRIEVAL,RANKING,ADAPTED,DIAGNOSTIC)
METHODS = (*BASE_METHODS,JOINT,ADAPTED_JOINT,JOINT_DIAGNOSTIC)
FUSION_DIAGNOSTICS = {DIAGNOSTIC:ADAPTED,JOINT_DIAGNOSTIC:ADAPTED_JOINT}
ADAPTIVE_METHODS = (ADAPTED,ADAPTED_JOINT)
SUPERVISION = {RETRIEVAL:(1.,0.),RANKING:(0.,1.),JOINT:(1.,.1)}
CONTEXT_BUDGET = 96
WINDOW_BATCH = CONTEXT_BATCH
FROZEN_LAYERS = 10


class IndexData(BlockData):
    def __init__(self, split, targets):
        super().__init__(split.rows,split.ends,split.vectors,targets)
        self.body_lengths = [len(s.ids) for s in split.sources]

    def __getitem__(self,index):
        return super().__getitem__(index),index,self.body_lengths[index]


def collate_index(examples):
    vectors,ids,lengths = zip(*examples,strict=True)
    return dict(collate_blocks(vectors),source_indices=torch.tensor(ids),body_lengths=torch.tensor(lengths))


class AdaptiveEncoder(nn.Module):
    def __init__(self,tokenizer):
        super().__init__()
        self.encoder = load_context_encoder().requires_grad_(False)
        self.encoder.encoder.layer[FROZEN_LAYERS:].float().requires_grad_(True)
        self.config = self.encoder.config
        self.tokenizer = tokenizer

    def train(self,mode=True):
        super().train(mode)
        self.encoder.embeddings.eval()
        self.encoder.encoder.layer[:FROZEN_LAYERS].eval()
        return self

    def forward(self,**inputs):
        with torch.autocast("cuda",dtype=torch.bfloat16):
            return self.encoder(**inputs)

    def block_batch(self,split,indices):
        blocks = [split.ends[i] for i in indices]
        offsets = list(accumulate(map(len,blocks),initial=0))
        windows = ((split.sources[i],lo,hi,owner) for owner,i in enumerate(indices)
                   for lo,hi in zip([0,*split.windows[i][:-1]],split.windows[i]))
        samples = []
        sizes = torch.tensor([b-a for ends in blocks for a,b in zip([0,*ends[:-1]],ends)],
                             dtype=torch.int32,device="cuda")
        counts = torch.zeros(len(sizes),dtype=torch.int32)
        while batch := list(islice(windows,WINDOW_BATCH)):
            ids,mask = context_window_inputs(self.tokenizer,batch)
            requests = window_block_requests(batch,blocks,offsets)
            counts.index_add_(0,requests[0],(requests[3]-requests[2]).int())
            requests = requests.cuda()
            values = checkpoint(self.window_pool,ids.cuda(),mask.cuda(),requests,use_reentrant=False)
            samples.append((requests[0],values))
        if not torch.equal(counts,sizes.cpu()):
            raise ValueError("Adaptive window pooling lost or duplicated source tokens")
        owners,values = zip(*samples,strict=True)
        summed = values[0].new_zeros((len(sizes),self.config.hidden_size)).index_add(
            0,torch.cat(owners),torch.cat(values))
        means = (summed/sizes[:,None]).half()
        return pad_sequence(means.split(tuple(map(len,blocks))),batch_first=True)

    def window_pool(self,ids,mask,requests):
        hidden = self(input_ids=ids,attention_mask=mask).last_hidden_state
        prefix = F.pad(hidden.float().cumsum(1),(0,0,1,0))
        _,row,a,b = requests
        return prefix[row,b]-prefix[row,a]


class AdaptiveIndex(nn.Module):
    def __init__(self,tokenizer,index,train):
        super().__init__()
        self.encoder = AdaptiveEncoder(tokenizer)
        self.index = index
        self.source = train

    def forward(self,batch):
        blocks = self.encoder.block_batch(self.source,batch["source_indices"].tolist())
        return self.index(dict(batch,blocks=blocks))


def adaptive_split(encoder,split,tokenizer):
    vectors,cost = encode_blocks(encoder.eval(),split.sources,tokenizer,split.windows,split.ends)
    return replace(split,vectors=vectors,encoding=cost,cache_read_seconds=0.)


def validation_pass(model,split,tokenizer,edits,*,adapted):
    model.eval()
    active = adaptive_split(model.encoder,split,tokenizer) if adapted else split
    scores = []
    _,ranks,_ = pipeline.score_coarse(model.index if adapted else model,active,"global",probabilities=scores)
    parser = source_parser()
    start = perf_counter()
    layouts = (retrieve(rank,s,ends,CONTEXT_BUDGET,parser)
               for rank,s,ends in zip(ranks,active.sources,active.ends,strict=True))
    rows = list(pipeline.retrieval_quality_rows(active,layouts,edits))
    result = pipeline.retrieval_quality([r for s,r in zip(active.sources,rows,strict=True) if len(s.raw_ids)>510])
    result.update(length_group="original UniXcoder BPE >510",context_budget=CONTEXT_BUDGET,
                  retrieval_and_quality_seconds=perf_counter()-start)
    if adapted:
        result["encoding"] = active.encoding
    return scores,result


def paths(seed):
    return tuple(pipeline.seed_root(root,seed) for root in (RESULTS,WEIGHTS))


def load_auxiliary(seed,model):
    _,weights = paths(seed)
    updated = model.load_state_dict(torch.load(weights/"auxiliary_global"/"best.pt",weights_only=True,
                                              map_location="cpu"),strict=False)
    if set(updated.missing_keys)!={"indexer.query.weight","indexer.key.weight"} or updated.unexpected_keys:
        raise ValueError("Fresh auxiliary head is incompatible with the block index")


def fit_auxiliary(seed,splits):
    area,weights = paths(seed)
    area,weights = area/"auxiliary_global",weights/"auxiliary_global"
    if (area/"fit.json").is_file():
        return
    train,valid = (BlockData(s.rows,s.ends,s.vectors) for s in splits.values())
    config = dict(pipeline.run_config("auxiliary_global",seed),stage="coarse",block_layout="fixed32",
                  precision="FP32 on cached FP16 states",initialization="fresh seeded weights")
    set_seed(seed)
    model = CoarseClassifier(768,"global").cuda()
    best,history = pipeline.fit_classifier(model,train,valid,weights,area,config,collator=collate_blocks)
    write_json(area/"fit.json",pipeline.fitted(config,best,history,len(valid)))


def supervision_gradients(model,data,seed):
    index = model.index if isinstance(model,AdaptiveIndex) else model
    params = (index.indexer.query.weight,index.indexer.key.weight)
    model.eval()
    rng = np.random.default_rng(seed)
    groups = {name:[i for i,n in enumerate(data.body_lengths) if lo<=n<=hi]
              for name,(lo,hi) in {"511-1020":(511,1020),"1021-2040":(1021,2040),
                                   "2041+":(2041,float("inf"))}.items()}
    weights = dict(train_bce=1.,train_repair_kl=index.repair_weight,train_repair_ranking=index.ranking_weight)
    results = {}
    for group,indices in groups.items():
        batches = []
        for picks in rng.choice(indices,32,replace=False).reshape(4,8):
            batch = collate_index([data[i] for i in picks])
            report = model(to_gpu(batch,labels=True))
            terms = {}
            for term,value in report["loss_components"].items():
                grad = torch.autograd.grad(value,params,retain_graph=True)
                norm = torch.linalg.vector_norm(torch.stack([g.detach().norm() for g in grad])).item()
                terms[term] = dict(loss=value.item(),gradient_norm=norm,weight=weights[term],
                                   weighted_gradient_norm=norm*weights[term])
            batches.append(dict(train_indices=picks.tolist(),terms=terms))
        results[group] = batches
    return results


def fit_index(seed,method,supervision,splits,tokenizer):
    if method not in (*SUPERVISION,*ADAPTIVE_METHODS) or supervision not in SUPERVISION:
        raise ValueError("Unknown index training design")
    area,weights = paths(seed)
    area,weights = area/method/"index",weights/method/"index"
    if (area/"fit.json").is_file():
        return read_json(area/"fit.json")
    fit_auxiliary(seed,splits)
    train,valid = splits["train"],splits["valid"]
    start = perf_counter()
    annotations = {name:paired_annotations(s.sources) for name,s in splits.items()}
    targets = [repair_block_distribution(source,ends,edits)
               for source,ends,edits in zip(train.sources,train.ends,annotations["train"],strict=True)]
    annotation_seconds = perf_counter()-start
    adapted = method in ADAPTIVE_METHODS
    kl,ranking = SUPERVISION[supervision]
    config = dict(pipeline.run_config(method,seed),stage="coarse",block_layout="fixed32",
                  context_budget=CONTEXT_BUDGET,repair_weight=1.,
                  repair_supervision="kl_and_budget_ranking" if kl and ranking else "budget_ranking" if ranking else "kl",
                  loss="function BCE"+(" + labeled function mean KL" if kl else "")+
                       (" + budget matched Softplus ranking on log probability" if ranking else ""),
                  selection="highest full-budget valid retention of actual edit BPE (original UniXcoder >510), earliest tie",
                  precision="BF16 SDPA on encoder; FP32 index on FP16 pooled blocks" if adapted else
                            "FP32 on cached FP16 blocks",
                  trainable_encoder_layers=[10,11] if adapted else [],encoder_window_micro_batch=WINDOW_BATCH if adapted else None,
                  initialization="fresh Q/K and same-run global auxiliary fit; original pretrained UniXcoder")
    if supervision==JOINT:
        config["ranking_weight"] = ranking
    set_seed(seed)
    index = CoarseClassifier(768,"joint",kl,ranking_budget=CONTEXT_BUDGET if ranking else None,ranking_weight=ranking)
    load_auxiliary(seed,index)
    model = AdaptiveIndex(tokenizer,index,train) if adapted else index
    data = IndexData(train,targets)
    val_data = IndexData(valid,None)
    # Validation computes coverage from the identical full inference layout after scoring current-code inputs.
    def validate(current):
        return validation_pass(current,valid,tokenizer,annotations["valid"],adapted=adapted)

    gradients = area/"supervision_gradients.json"
    if supervision==JOINT and not gradients.is_file():
        write_json(gradients,dict(mode="dropout off, fixed Train batches",initial=supervision_gradients(model.cuda(),data,seed)))
    best,history = pipeline.fit_classifier(model.cuda(),data,val_data,weights,area,config,
                                          collator=collate_index,validate=validate)
    if supervision==JOINT:
        write_json(gradients,dict(read_json(gradients),selected=supervision_gradients(model,data,seed)))
    result = pipeline.fitted(config,best,history,len(val_data)) | dict(
        selected_retrieval=best["retrieval_validation"],annotation_seconds=annotation_seconds)
    write_json(area/"fit.json",result)
    return result


def fitted_index(seed,method,splits,tokenizer):
    _,weights = paths(seed)
    index = CoarseClassifier(768,"joint")
    model = AdaptiveIndex(tokenizer,index,splits["train"]) if method in ADAPTIVE_METHODS else index
    state = torch.load(weights/method/"index"/"best.pt",weights_only=True,map_location="cpu")
    model.load_state_dict(state)
    model.cuda().eval().requires_grad_(False)
    return model


def fit_fine(seed,method,index,encoder,splits,tokenizer,*,frozen_splits=None):
    area,weights = paths(seed)
    config = dict(predicted_config(seed),method=method,indexer_checkpoint="index/best.pt",
                  coarse_representation="original pretrained" if encoder is None else "task-adapted last two layers",
                  fusion_representation="original pretrained" if frozen_splits is not None or encoder is None else "task-adapted",
                  index_selected_epoch=read_json(area/FUSION_DIAGNOSTICS.get(method,method)/"index"/"fit.json")["selected_epoch"])
    prepared = {name:pipeline.prepare_fine_inputs(s,index,tokenizer,config,
                global_vectors=frozen_splits[name].vectors if frozen_splits is not None else None)
                for name,s in splits.items()}
    return pipeline.fit_frozen_fine(tokenizer,index,config,area/method/"fine",weights/method/"fine",
                                    cached=splits,prepared=prepared,coarse_encoder=encoder,
                                    pretrained_global=frozen_splits is not None)


def load_training_splits(tokenizer):
    return {s:pipeline.load_split(s,tokenizer) for s in ("train","valid")}


def run(seed,methods,tokenizer,*,splits=None):
    if splits is None:
        splits = load_training_splits(tokenizer)
    for method in methods:
        area,_ = paths(seed)
        if (area/method/"fine"/"metrics.json").is_file():
            continue
        print(f"Seed {seed}: {method}",flush=True)
        index_method = FUSION_DIAGNOSTICS.get(method,method)
        if index_method==ADAPTED:
            configs = [read_json(RESULTS/f"seed{SEED}"/label/"index"/"fit.json") for label in (RETRIEVAL,RANKING)]
            supervision = max(configs,key=lambda x:x["selected_retrieval"]["retention"])["config"]["method"]
            write_json(RESULTS/"adaptation_policy.json",dict(seed=SEED,supervision=supervision,
                       criterion="Valid original UniXcoder >510 actual edit BPE retrieval",
                       valid_retention={r["config"]["method"]:r["selected_retrieval"]["retention"] for r in configs}))
        else:
            supervision = JOINT if index_method==ADAPTED_JOINT else method
        fit_index(seed,index_method,supervision,splits,tokenizer)
        model = fitted_index(seed,index_method,splits,tokenizer)
        if index_method in ADAPTIVE_METHODS:
            adapted = {n:adaptive_split(model.encoder,s,tokenizer) for n,s in splits.items()}
            fit_fine(seed,method,model.index,model.encoder,adapted,tokenizer,
                     frozen_splits=splits if method in FUSION_DIAGNOSTICS else None)
        else:
            fit_fine(seed,method,model,None,splits,tokenizer)
        del model


def joint_policy():
    area,_ = paths(SEED)
    scores = {m:read_json(area/m/"index"/"fit.json")["selected_retrieval"]["retention"] for m in (RETRIEVAL,JOINT)}
    policy = dict(seed=SEED,valid_retention=scores,train_adapted_joint=scores[JOINT]>scores[RETRIEVAL],
                  criterion="strict improvement over KL on Valid original UniXcoder >510 actual edit BPE retrieval")
    write_json(RESULTS/"joint_policy.json",policy)
    return policy


def complete_joint(tokenizer,*,splits=None):
    area,_ = paths(SEED)
    if not (area/JOINT/"fine"/"metrics.json").is_file():
        if splits is None:
            splits = load_training_splits(tokenizer)
        run(SEED,(JOINT,),tokenizer,splits=splits)
    if joint_policy()["train_adapted_joint"]:
        pending = tuple(m for m in (ADAPTED_JOINT,JOINT_DIAGNOSTIC) if not (area/m/"fine"/"metrics.json").is_file())
        if pending:
            run(SEED,pending,tokenizer,splits=splits)


def winning_method():
    methods = (RETRIEVAL,RANKING,ADAPTED,JOINT)
    if read_json(RESULTS/"joint_policy.json")["train_adapted_joint"]:
        methods = (*methods,ADAPTED_JOINT)
    scores = {m:read_json(RESULTS/f"seed{SEED}"/m/"fine"/"metrics.json")["validation"]["mcc"]
              for m in methods}
    winner = max(scores,key=scores.get)
    write_json(RESULTS/"selection.json",dict(seed=SEED,criterion="full downstream Valid MCC",valid_mcc=scores,
                                           selected=winner))
    return winner


def smoke(tokenizer):
    set_seed(SEED)
    rows = prime_source_rows(DATA,"train")
    i = max(range(len(rows)),key=lambda i:len(rows[i]["func"]))
    rows = rows[i//2*2:i//2*2+2]
    sources = load_sources(rows,tokenizer)
    windows = [pipeline.encoder_window_ends(s) for s in sources]
    ends = pipeline.cuts_for(sources,None,False)
    original = load_context_encoder().requires_grad_(False)
    vectors,cost = encode_blocks(original,sources,tokenizer,windows,ends)
    del original
    split = pipeline.SourceSplit(rows,sources,windows,ends,vectors,cost,0.)
    annotations = paired_annotations(sources)
    targets = [repair_block_distribution(s,e,a) for s,e,a in zip(sources,ends,annotations,strict=True)]
    data = IndexData(split,targets)
    index = CoarseClassifier(768,"joint",ranking_budget=CONTEXT_BUDGET,ranking_weight=1.)
    model = AdaptiveIndex(tokenizer,index,split).cuda().train()
    frozen = [p for p in model.encoder.parameters() if not p.requires_grad]
    layers = model.encoder.encoder.encoder.layer
    assert not any(p.requires_grad for p in model.encoder.encoder.embeddings.parameters())
    assert not any(p.requires_grad for p in layers[:FROZEN_LAYERS].parameters())
    assert all(p.requires_grad for p in layers[FROZEN_LAYERS:].parameters())
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    batch = to_gpu(collate_index([data[j] for j in range(len(data))]),labels=True)
    result = model(batch)
    result["loss"].backward()
    params = {"query":model.index.indexer.query.weight,"key":model.index.indexer.key.weight,
              "layer10":layers[10].attention.self.query.weight,"layer11":layers[11].attention.self.query.weight}
    gradients = {name:float(p.grad.norm()) for name,p in params.items()}
    if any(p.grad is not None for p in frozen) or not all(0<v<float("inf") for v in gradients.values()):
        raise ValueError("Expected gradients only in trainable index and encoder layers")
    before = {name:p.detach().clone() for name,p in params.items()}
    AdamW(model.parameters(),lr=2e-5).step()
    if any(torch.equal(p,before[n]) for n,p in params.items()):
        raise ValueError("The adaptive smoke optimizer did not update all checked parameters")
    torch.cuda.synchronize()
    training = dict(seconds=perf_counter()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    gradients=gradients,loss=float(result["loss"].detach()))
    split = adaptive_split(model.encoder,split,tokenizer)
    features,ranks,_ = pipeline.score_coarse(model.index,split,"global")
    fine,_ = pipeline.fine_data(split,features,ranks,tokenizer,CONTEXT_BUDGET)
    if split.encoding["body_bpe"]!=sum(len(s.ids) for s in sources) or any(x["length"]!=512 for x in fine.examples):
        raise ValueError("Adaptive encoding did not preserve full coverage and fine input budget")
    write_json(RESULTS/"smoke.json",dict(original_bpe=[len(s.raw_ids) for s in sources],training=training,
                                       encoding=split.encoding,fine_lengths=[x["length"] for x in fine.examples]))
    print("Adaptive index GPU smoke passed",training,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("smoke","run","choose"))
    parser.add_argument("--seed",type=int,choices=SEEDS,default=SEED)
    parser.add_argument("--methods",nargs="+",choices=METHODS)
    args = parser.parse_args()
    if args.stage=="choose":
        print(winning_method())
        return
    if torch.cuda.device_count()!=1 or "4090" not in torch.cuda.get_device_name():
        raise ValueError("Select one RTX 4090")
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    tokenizer = load_tokenizer("unixcoder",UNIX_BASE)
    if args.stage=="smoke":
        smoke(tokenizer)
    elif args.seed==SEED:
        if args.methods:
            run(args.seed,args.methods,tokenizer)
        else:
            splits = load_training_splits(tokenizer)
            run(args.seed,BASE_METHODS,tokenizer,splits=splits)
            complete_joint(tokenizer,splits=splits)
    else:
        complete_joint(tokenizer)
        winner = winning_method()
        if args.methods and args.methods!=[winner]:
            parser.error("Supplemental seeds run the Valid-selected main method")
        run(args.seed,[winner],tokenizer)


if __name__=="__main__":
    main()
