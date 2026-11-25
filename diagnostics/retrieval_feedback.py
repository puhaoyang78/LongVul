"""Replay the saved first Train feedback round with frozen weights."""
import argparse
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.stats import spearmanr
from torch.nn import functional as F

from coarse_encoding import collate_blocks
from coarse_retrieval import lengths
from coarse_to_fine import RESULTS, block_data, changed_blocks, corpus, plans
from context_selection import occupied, write_prediction_rows
from diagnostics.coarse_to_fine import LAYOUT, OUT, SEEDS, encode, legacy_data, legacy_module, load_pair, pack, trace
from diagnostics.retrieval_budget import actual_edits, distribution, group_ids
from paired import BATCH_SIZE, UNIX_BASE, load_tokenizer
from predicted_diff import paired_annotations
from repair_oracle import overlap_bpe
from rgcl.artifacts import read_json, write_json


def contrast(values,ends,removed,added):
    ids,delta = changed_blocks(ends,removed,added)
    return float(np.array(delta)@np.log(values[ids].clip(1e-12))) if ids else 0.


def compare(rows,score,benefit):
    x,y = np.asarray([r[score] for r in rows]),np.asarray([r[benefit] for r in rows])
    active = (x!=0)&(y!=0)
    both = [np.mean(x[y>0]>0) if (y>0).any() else None,np.mean(x[y<0]<0) if (y<0).any() else None]
    rho = spearmanr(x,y).statistic if len(x)>1 and np.ptp(x) and np.ptp(y) else None
    return dict(comparisons=len(rows),sign_comparable=int(active.sum()),spearman=None if rho is None else float(rho),
                sign_agreement=float(np.mean(np.sign(x[active])==np.sign(y[active]))) if active.any() else None,
                balanced_sign_agreement=float(np.mean(both)) if all(r is not None for r in both) else None,
                loss_lower_with_replacement=float(np.mean(y>0)) if len(y) else None,
                always_keep_original=float(np.mean(y[active]<0)) if active.any() else None,
                score_ties=int(np.sum(x==0)),loss_ties=int(np.sum(y==0)),
                score_prefers_replacement=float(np.mean(x>0)) if len(x) else None)


@torch.inference_mode()
def replay(seed,data,new_data,old_data,joint,annotations,tokenizer,archive):
    start = perf_counter()
    rng = np.random.default_rng(np.random.SeedSequence([seed,9]))
    current, old, old_coarse = load_pair(seed,list(zip(data.sources,joint,strict=True)),archive,tokenizer,warm=True)
    fresh = trace(current.coarse,new_data,collate_blocks)
    previous = trace(old_coarse,old_data,archive.collate_blocks,archive=archive)
    old_coarse_features = previous["coarse"]
    edits = [actual_edits(s,a)[1] for s,a in zip(data.sources,annotations,strict=True)]
    def losses(ids,layouts):
        g,z = fresh["g"][ids].cuda(),fresh["z"][ids].cuda()
        logits = current.fine_logits(ids,layouts,g,z)
        cls = encode(old.fine,[pack(data.sources[i],layout,tokenizer) for i,layout in zip(ids,layouts,strict=True)])
        archived = old.fine.classifier((cls+old.fusion(old_coarse_features[ids].cuda()))[:,None])[:,0]
        y = torch.tensor([data.rows[i]["target"] for i in ids],device="cuda",dtype=torch.float32)
        return tuple(F.binary_cross_entropy_with_logits(x,y,reduction="none").cpu().numpy() for x in (logits,archived))

    rows = []
    for lo in range(0,len(data.rows),BATCH_SIZE):
        hi = min(lo+BATCH_SIZE,len(data.rows))
        baseline,alternatives = {},[[],[]]
        for i in range(lo,hi):
            candidate,source,p = joint[i],data.sources[i],fresh["p"][i]
            layout = candidate.retrieve(p,source)
            for k,(b,removed,added) in enumerate(candidate.alternatives(layout,p,source,rng)):
                baseline[i]=layout
                alternatives[k].append((i,b,removed,added))
        if not baseline:
            continue
        members = sorted(baseline)
        original = [{i:v for i,v in zip(members,x,strict=True)}
                    for x in losses(members,[baseline[i] for i in members])]
        for k,group in enumerate(alternatives):
            if not group:
                continue
            ids = [i for i,*_ in group]
            observed, archived = losses(ids,[b for _,b,_,_ in group])
            for n,(i,b,removed,added) in enumerate(group):
                a = baseline[i]
                if occupied(a.spans)!=occupied(b.spans) or lengths(a.spans)!=lengths(b.spans):
                    raise ValueError("Replay compared unmatched fine input budgets")
                old_ends = [*range(archive.BLOCK,len(data.sources[i].ids),archive.BLOCK),len(data.sources[i].ids)]
                rows.append(dict(seed=seed,sample_key=data.rows[i]["sample_key"],pair_key=data.rows[i]["pair_key"],
                     original_bpe=len(data.sources[i].raw_ids),proposal_number=k+1,changed_bpe=lengths(added),
                     new_score_delta=contrast(fresh["p"][i],[h for _,h in joint[i].centers],removed,added),
                     old_score_delta=contrast(previous["p"][i],old_ends,removed,added),
                     repair_delta=overlap_bpe(added,edits[i])-overlap_bpe(removed,edits[i]),
                     new_bce_reduction=float(original[0][i]-observed[n]),
                     old_bce_reduction=float(original[1][i]-archived[n])))
    published = read_json(RESULTS/f"seed{seed}"/"full"/"feedback"/"epoch9.json")["generation"]
    difference = float(np.median([abs(r["new_bce_reduction"]) for r in rows])-published["abs_delta_median"])
    if len(rows)!=published["comparisons"] or abs(difference)>3e-6 or len(
            {r["sample_key"] for r in rows})!=published["comparable_functions"]:
        raise ValueError(f"Feedback replay failed its saved round-9 check: {len(rows)} substitutions, delta error {difference}")
    report = {}
    for g,ids in group_ids([r["original_bpe"] for r in rows]).items():
        selected = [rows[i] for i in ids]
        report[g] = dict(by_pair={f"{score}_vs_{benefit}":compare(selected,score,benefit)
                                  for score in ("new_score_delta","old_score_delta","repair_delta")
                                  for benefit in ("new_bce_reduction","old_bce_reduction")},
                         new_bce_reduction=distribution(r["new_bce_reduction"] for r in selected),
                         old_bce_reduction=distribution(r["old_bce_reduction"] for r in selected))
    later = read_json(RESULTS/f"seed{seed}"/"full"/"feedback"/"epoch13.json")["generation"]
    return rows,dict(seed=seed,comparisons=len(rows),published_epoch9=published,
                     median_absolute_bce_delta_error=difference,results=report,
                     epoch13_saved_aggregates=later,seconds=perf_counter()-start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds",type=int,nargs="+",choices=SEEDS,default=list(SEEDS))
    parser.add_argument("--out",type=Path,default=OUT)
    args = parser.parse_args()
    if torch.cuda.device_count()!=1:
        raise ValueError("Select one GPU with CUDA_VISIBLE_DEVICES")
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True
    tok,archive = load_tokenizer("unixcoder",UNIX_BASE),legacy_module()
    data = corpus("train",tok,[LAYOUT])
    joint = plans(data,LAYOUT,True)
    new_data,old_data = block_data(data,LAYOUT,candidates=joint),legacy_data(data,"train",archive)
    annotations = paired_annotations(data.sources)
    args.out.mkdir(parents=True,exist_ok=True)
    for seed in args.seeds:
        print("Train replay",seed,flush=True)
        rows,report = replay(seed,data,new_data,old_data,joint,annotations,tok,archive)
        write_prediction_rows(args.out/f"train_seed{seed}_feedback.csv.gz",rows)
        write_json(args.out/f"train_seed{seed}_feedback.json",report)
        print(seed,report["comparisons"],"comparisons",round(report["seconds"],2),"s",flush=True)


if __name__=="__main__":
    main()
