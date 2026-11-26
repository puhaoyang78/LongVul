"""Frozen Token and block probes. Oracle metrics use privileged paired edits."""
import argparse
import json
from itertools import accumulate
from math import sqrt
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn.functional import binary_cross_entropy_with_logits, layer_norm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaConfig, set_seed
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

import coarse_to_fine as center
from coarse_encoding import BlockData, collate_blocks
from coarse_retrieval import uniform_layout
from context_selection import merge, visible_source, write_prediction_rows
from diagnostics.next_stage import baseline_path, part_shares
from locator_quality import context_encoder_batches, load_context_encoder
from paired import BATCH_SIZE, ROOT, SEED, UNIX_BASE, fit_classifier, load_tokenizer, predict, to_gpu
from predicted_diff import paired_annotations
from repair_oracle import prepare_side
from rgcl.artifacts import read_json, write_json
from window_inputs import GROUPS

OUT = ROOT/"results"/"representation_diagnosis"
WEIGHTS = ROOT/"checkpoints"/OUT.name
TOKEN_METHODS = ("token","token_position","token_no_global")
BLOCK_METHODS = ("dense","uniform","predicted","oracle")


def shuffled_blocks(batch,generator):
    blocks,valid = batch["blocks"],batch["valid"]
    perm = torch.arange(blocks.shape[1]).expand(len(blocks),-1).clone()
    for row,n in zip(perm,valid.sum(1).tolist(),strict=True):
        row[:n] = torch.randperm(n,generator=generator)
    gpu = perm.to(blocks.device)
    shuffled = blocks.gather(1,gpu[:,:,None].expand_as(blocks))
    return dict(batch,blocks=shuffled),gpu.argsort(1)


@torch.inference_mode()
def permutation_check(split,index,codebert):
    rng = torch.Generator().manual_seed(SEED)
    records,offset = [],0
    loader = DataLoader(BlockData(split.rows,split.ends,split.vectors),batch_size=BATCH_SIZE,collate_fn=collate_blocks)
    for cpu in loader:
        batch = to_gpu(cpu)
        replaced,inverse = shuffled_blocks(batch,rng)
        a,b = (index(x,details=True) for x in (batch,replaced))
        g = (a["global_state"]-b["global_state"]).abs().amax(1)
        difference = (a["importance"]-b["importance"].gather(1,inverse)).abs().amax(1)
        score = (a["probability"]-b["probability"]).abs()
        for j,(gerr,perr,error) in enumerate(zip(g.tolist(),difference.tolist(),score.tolist(),strict=True)):
            row = split.rows[offset+j]
            records.append(dict(sample_key=row["sample_key"],target=row["target"],codebert_bpe=codebert[offset+j],
                blocks=int(cpu["valid"][j].sum()),probability=float(a["probability"][j]),global_abs_delta=gerr,
                index_abs_delta=perr,aux_probability_abs_delta=error,
                threshold_flip=int((a["probability"][j]>.5)!=(b["probability"][j]>.5))))
        offset += len(cpu["valid"])
    write_prediction_rows(OUT/"block_permutation_valid.csv.gz",records)
    fields = ("global_abs_delta","index_abs_delta","aux_probability_abs_delta")
    return {group:dict(functions=len(selected),**{f"max_{k}":max(r[k] for r in selected) for k in fields},
                       threshold_flips=sum(r["threshold_flip"] for r in selected))
            for group,(lo,hi) in GROUPS.items()
            if (selected:=[r for r in records if lo<=r["codebert_bpe"]<=hi])}


def compact_positions(source,fixed):
    """Return original and compacted body coordinates for the exact packed layout."""
    raw = fixed[1].numpy()
    body = raw>=0
    positions = raw[body]
    keep = np.asarray(source.keep)
    compact = np.searchsorted(keep,positions)
    if (not len(positions) or np.any(np.diff(positions)<=0) or np.any(compact>=len(keep))
            or not np.array_equal(keep[compact],positions)
            or not np.array_equal(source.ids[compact],fixed[0].numpy()[body])):
        raise ValueError("A selected Token does not match its raw position and Simple source ID")
    return positions,compact


def compact_ranges(indices):
    cuts = np.flatnonzero(np.diff(indices)>1)+1
    return [(int(chunk[0]),int(chunk[-1])+1) for chunk in np.split(indices,cuts) if len(chunk)]


def selected_tokens(split,layout,tokenizer,encoder):
    raw_and_compact = [compact_positions(src,ex["fixed"]) for src,ex in zip(split.sources,layout.examples,strict=True)]
    positions,compact = zip(*raw_and_compact,strict=True)
    offsets = list(accumulate(map(len,compact),initial=0))
    width = encoder.config.hidden_size
    bank = torch.empty((offsets[-1],width),dtype=torch.float16)
    counts = np.zeros(len(compact),dtype=np.int32)
    windows = ((source,lo,hi,i) for i,(source,ends) in enumerate(zip(split.sources,split.windows,strict=True))
               for lo,hi in zip([0,*ends[:-1]],ends))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    num_windows=calls=padded=0
    for batch,hidden,slots in context_encoder_batches(encoder,tokenizer,windows,prefix_sum=False):
        rows,inside,destination = [],[],[]
        for row,(_,lo,hi,i) in enumerate(batch):
            a,b = np.searchsorted(compact[i],(lo,hi))
            if int(a)!=counts[i]:
                raise ValueError("Selected Token windows skipped or repeated source positions")
            counts[i] = b
            rows.extend([row]*(b-a))
            inside.extend((compact[i][a:b]-lo+1).tolist())
            destination.extend(range(offsets[i]+a,offsets[i]+b))
        if rows:
            row,position = torch.as_tensor(rows,device=hidden.device),torch.as_tensor(inside,device=hidden.device)
            bank[torch.as_tensor(destination)] = hidden[row,position].to(torch.float16).cpu()
        num_windows += len(batch)
        calls += 1
        padded += slots
    if counts.tolist()!=list(map(len,positions)) or num_windows!=sum(map(len,split.windows)):
        raise ValueError("Did not recover all selected body Tokens from the original windows")
    torch.cuda.synchronize()
    cost = dict(seconds=perf_counter()-started,frozen_encoder_windows=num_windows,plm_calls=calls,
                actual_encoder_positions=sum(len(s.ids) for s in split.sources)+2*num_windows,
                padded_encoder_positions=padded,retained_tokens=int(sum(counts)),bank_bytes=bank.nbytes,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),token_cache_on_disk_bytes=0)
    return bank,[torch.from_numpy(raw.copy()).float() for raw in positions],offsets,cost


class FeatureData(Dataset):
    def __init__(self,split,features,offsets,coarse,selection,positions=None):
        self.split,self.features,self.offsets,self.coarse = split,features,offsets,coarse
        self.selection,self.positions = selection,positions
        self.targets = torch.tensor([row["target"] for row in split.rows])

    def __len__(self):
        return len(self.targets)

    def __getitem__(self,i):
        lo,hi = self.offsets[i:i+2]
        values = self.features[lo:hi]
        position = self.positions[i] if self.positions is not None else torch.zeros(len(values))
        if self.selection is None:
            mass = torch.ones(len(values))
        else:
            indices,mass = self.selection[i]
            values,position = values[indices],position[indices]
        return values,mass,position,self.coarse[i],self.targets[i]


def collate_features(batch):
    values,mass,position,coarse,labels = zip(*batch,strict=True)
    return dict(features=pad_sequence(values,batch_first=True),mass=pad_sequence(mass,batch_first=True),
                positions=pad_sequence(position,batch_first=True),coarse=torch.stack(coarse),labels=torch.stack(labels))


class AttentionProbe(nn.Module):
    def __init__(self,config,*,global_fusion,position):
        super().__init__()
        width = config.hidden_size
        self.query = nn.Parameter(torch.empty(width))
        nn.init.normal_(self.query)
        self.classifier = RobertaClassificationHead(config)
        for layer in self.classifier.modules():
            if isinstance(layer,nn.Linear):
                nn.init.normal_(layer.weight,std=config.initializer_range)
                nn.init.zeros_(layer.bias)
        self.fusion = nn.Linear(2*width,width,bias=False) if global_fusion else None
        if self.fusion is not None:
            nn.init.zeros_(self.fusion.weight)
        self.use_position = position
        self.register_buffer("frequencies",10000.**(-torch.arange(0,width,2)/width))

    def forward(self,batch,*,details=False):
        values = batch["features"].float()
        mass = batch["mass"]
        if self.use_position:
            angles = batch["positions"][:,:,None]*self.frequencies
            values = values+.1*torch.stack((angles.sin(),angles.cos()),dim=-1).flatten(-2)
        keys = layer_norm(values,(values.shape[-1],))
        scores = keys@self.query/sqrt(values.shape[-1])+mass.log()
        p = scores.softmax(1)
        pooled = torch.bmm(p[:,None,:],values).squeeze(1)
        if self.fusion is not None:
            pooled = pooled+self.fusion(batch["coarse"])
        logits = self.classifier(pooled[:,None]).squeeze(-1)
        record = dict(logits=logits,probability=logits.sigmoid())
        if "labels" in batch:
            record["loss"] = binary_cross_entropy_with_logits(logits,batch["labels"].float())
        return record


def block_selection(spans,ends):
    full = part_shares(spans,ends)
    indices = np.flatnonzero(full>0)
    if not len(indices):
        raise ValueError("No covered blocks for this layout")
    return torch.from_numpy(indices),torch.from_numpy(full[indices]).float()


def block_inputs(split,layout,coarse,tokenizer):
    offsets = list(accumulate(map(len,split.ends),initial=0))
    data = {"dense":FeatureData(split,split.vectors,offsets,coarse,None)}
    geometry = {"dense":None}
    for method in ("uniform","predicted","oracle"):
        if method=="oracle":
            # The only member allowed to inspect actual counterpart edits.
            edits = paired_annotations(split.sources)
            spans = [compact_ranges(compact_positions(source,prepare_side(
                (source.code,source.raw_ids,source.offsets),tokenizer,edit,with_context=False,parser=None)["fixed"])[1])
                     for source,edit in zip(split.sources,edits,strict=True)]
        elif method=="uniform":
            spans = [uniform_layout(source).spans for source in split.sources]
        else:
            spans = [ex["layout"].spans for ex in layout.examples]
        selection = [block_selection(parts,ends) for parts,ends in zip(spans,split.ends,strict=True)]
        data[method] = FeatureData(split,split.vectors,offsets,coarse,selection)
        geometry[method] = selection
    return data,geometry


def fitted_probe(name,datasets,codebert,configs,*,extra_cost):
    root,weights = OUT/name,WEIGHTS/name
    if (root/"metrics.json").is_file():
        return read_json(root/"metrics.json")
    positions = name=="token_position"
    global_fusion = name in ("token","token_position")
    config = dict(center.run_config(name,SEED),stage="frozen representation diagnostic",global_fusion=global_fusion,
                  position="sinusoid on original BPE with scale 0.1 after frozen encoding" if positions else None,
                  heads=1,oracle_privileged=name=="oracle",log_fraction_prior=True,frozen_dtype="FP16 cache",
                  warm_start=False)
    set_seed(SEED)
    model = AttentionProbe(configs,global_fusion=global_fusion,position=positions).cuda()
    train,valid = (datasets[key] for key in ("train","valid"))
    best,history = fit_classifier(model,train,valid,weights,root,config,collator=collate_features)
    result = center.fitted(config,best,history,len(valid))
    result.update(split={},extraction=extra_cost)
    for key in ("valid","test"):
        data = datasets[key]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        loader = DataLoader(data,batch_size=BATCH_SIZE,collate_fn=collate_features)
        p,_ = predict(model,loader)
        torch.cuda.synchronize()
        seconds = perf_counter()-start
        peak = torch.cuda.max_memory_allocated()
        raw = [len(source.raw_ids) for source in data.split.sources]
        result["split"][key] = dict(unixcoder=center.classification(data.split.rows,p,raw),
             codebert=center.classification(data.split.rows,p,codebert[key]),probe_seconds=seconds,
             probe_peak_allocated_bytes=peak,
             attended_vectors=sum(data.offsets[i+1]-data.offsets[i] if data.selection is None else len(data.selection[i][0])
                                  for i in range(len(data))))
        if key=="valid" and result["split"][key]["unixcoder"]["groups"]["overall"]!=best["validation"]:
            raise ValueError("Reloaded diagnostic head failed Valid checkpoint replay")
        rows = [dict(sample_key=r["sample_key"],target=r["target"],original_bpe=n,codebert_bpe=c,
                    probability=score,prediction=int(score>.5))
                for r,n,c,score in zip(data.split.rows,raw,codebert[key],p,strict=True)]
        write_prediction_rows(root/f"{key}.csv.gz",rows)
    write_json(root/"metrics.json",result)
    print("Representation probe",name,"epoch",best["epoch"],"Test MCC",
          result["split"]["test"]["unixcoder"]["groups"]["overall"]["mcc"],flush=True)
    return result


def coverage_rows(split,layouts,geometry,token_positions,codebert):
    rows = []
    for i,(row,source,ends,positions,n) in enumerate(zip(split.rows,split.sources,split.ends,token_positions,
                                                       codebert,strict=True)):
        values = dict(sample_key=row["sample_key"],target=row["target"],codebert_bpe=n,original_bpe=len(source.raw_ids),
                      compacted_bpe=len(source.ids),selected_raw_token_bpe=len(positions),blocks=len(ends),
                      selected_token_raw_runs=visible_source(layouts.examples[i]))
        for method,selection in geometry.items():
            if selection is None:
                continue
            indices,mass = selection[i]
            values[f"{method}_block_indices"] = indices.tolist()
            values[f"{method}_block_fractions"] = mass.tolist()
        rows.append(values)
    return rows


def compare_all():
    if (OUT/"summary.json").is_file():
        return
    OUT.mkdir(parents=True,exist_ok=True)
    tokenizer = load_tokenizer("unixcoder",UNIX_BASE)
    config = RobertaConfig.from_pretrained(UNIX_BASE,local_files_only=True,num_labels=1)
    index = center.load_index(SEED,False)
    encoder = load_context_encoder().requires_grad_(False)
    fits,codebert,costs = {name:{} for name in (*TOKEN_METHODS,*BLOCK_METHODS)},{},{}
    perm = {}
    for key in ("train","valid","test"):
        started = perf_counter()
        split = center.load_split(key,tokenizer)
        loaded = perf_counter()-started
        coarse,ranks,score_cost = center.score_coarse(index,split,"global")
        layouts,layout_seconds = center.fine_data(split,coarse,ranks,tokenizer,96)
        bank,positions,offsets,extraction = selected_tokens(split,layouts,tokenizer,encoder)
        data = FeatureData(split,bank,offsets,coarse,None,positions)
        for name in TOKEN_METHODS:
            fits[name][key] = data
        began = perf_counter()
        blocks,geometry = block_inputs(split,layouts,coarse,tokenizer)
        block_layout_seconds = perf_counter()-began
        for name,dataset in blocks.items():
            fits[name][key] = dataset
        costs[key] = dict(load_source_and_cache_seconds=loaded,index_scoring=score_cost,layout_seconds=layout_seconds,
                          frozen_token_extraction=extraction,block_layout_and_oracle_seconds=block_layout_seconds,
                          historical_block_encoding_seconds=split.encoding["seconds"])
        if key in ("valid","test"):
            codebert[key] = center.codebert_lengths(key,split.rows)
            saved = center.read_predictions(baseline_path("longvul",SEED)/f"{key}.csv.gz",split.rows)
            for row,ex in zip(saved,layouts.examples,strict=True):
                if [list(v) for v in visible_source(ex)]!=json.loads(row["source_runs"]):
                    raise ValueError(f"Frozen Token layout differs from published LongVul on {key}")
            rows = coverage_rows(split,layouts,geometry,positions,codebert[key])
            write_prediction_rows(OUT/f"coverage_{key}.csv.gz",rows)
        if key=="valid":
            perm = permutation_check(split,index,codebert[key])
        print("Frozen feature datasets prepared",key,extraction,flush=True)
    del encoder,index
    write_json(OUT/"preparation.json",costs)
    write_json(OUT/"permutation.json",dict(scope="complete Valid; identical existing vectors and masked padding",
               restored_by_identity=True,order="randomized valid block rows",
               groups=perm,packing="strict increasing raw BPE across retained body positions"))
    results = {name:fitted_probe(name,data,codebert,config,extra_cost=costs) for name,data in fits.items()}
    reference = read_json(baseline_path("longvul",SEED)/"metrics.json")
    write_json(OUT/"summary.json",dict(seed=SEED,protocol="same full splits, 16 epochs, Valid MCC, >0.5",
               current_longvul=reference["split"]["test"],probes=results,
               privileged_oracle="standalone diagnostic with real paired diffs; no other head accesses these inputs",
               information_scope="Token and block vectors include context from the original frozen windows",
               block_dense_scope="fixed frozen representation and one Query attention readout",
               cost_scope="the Token encoding and cached block banks are separate measured stages; summed costs are not a cold single-pass deployment measurement",
               shared_token_cache_ram_bytes=sum(x["frozen_token_extraction"]["bank_bytes"] for x in costs.values()),
               persisted_token_cache_bytes=0))
    print("All representation comparisons complete",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.device_count()!=1 or "4090" not in torch.cuda.get_device_name():
        raise ValueError("Select one RTX 4090")
    compare_all()


if __name__=="__main__":
    main()
