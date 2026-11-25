"""Frozen-checkpoint diagnostics for the current and archived LongVul models."""
import argparse
from collections import defaultdict
import csv
import gc
import gzip
import json
from pathlib import Path
import subprocess
from time import perf_counter
from types import ModuleType

import numpy as np
import torch
from scipy.special import expit, logit
from scipy.stats import rankdata
from torch.nn import functional as F
from torch.utils.data import DataLoader

from coarse_encoding import CoarseClassifier, FusionClassifier, collate_blocks, masked_mean
from coarse_retrieval import Candidates, Layout, uniform_layout
from coarse_to_fine import ARCHIVE, RESULTS, WEIGHTS, block_data, changed_blocks, context_quality, corpus, plans
from compaction import BUDGET, MAX_LENGTH, pack_fixed
from context_selection import intersects, write_prediction_rows
from diagnostics.retrieval_budget import (
    CONTEXT_CAP, actual_edits, budget_row, capped_candidates, center_only_with_fixed_anchors,
    distribution, geometry, group_ids, high_score_blockers, project_scores, summarize_budgets,
)
from paired import BATCH_SIZE, ROOT, UNIX_BASE, load_tokenizer, to_gpu
from predicted_diff import fp32_classifier, paired_annotations
from region_locator import region_metrics
from repair_oracle import grouped_edit_retention, overlap_bpe, raw_runs
from rgcl.artifacts import read_json, write_json
from rgcl.metrics import metrics, paired_metrics
from window_inputs import GROUPS

ARCHIVED_COMMIT = "60688d25cd334e4202cefaa373a310341bc7ec16"
CURRENT_COMMIT = "3a82c39b01e3df0383b2400d56b73ff946b3acfe"
OLD_WEIGHTS = ROOT/"checkpoints"/ARCHIVE.name
LAYOUT = "statement128"
OUT = ROOT/"results"/"coarse_to_fine_diagnosis"
SEEDS = (42,1234,123456)
NEW, OLD = "joint128", "archived32"
BASELINES = {"current":NEW,"archived":OLD}


def legacy_module():
    path = f"{ARCHIVED_COMMIT}:coarse_encoding.py"
    source = subprocess.check_output(["git","show",path],cwd=ROOT)
    archive = ModuleType("longvul_archived_coarse_encoding")
    exec(compile(source,path,"exec"),archive.__dict__)
    return archive


def legacy_data(data, split, archive):
    cache = torch.load(OLD_WEIGHTS/"blocks"/f"{split}.pt",map_location="cpu",weights_only=True,mmap=True)
    members = [r["sample_key"] for r in data.rows]
    if cache["config"]["members"]!=members or cache["config"]["windows"]!=data.windows:
        raise ValueError("Archived block cache differs in membership or encoder windows")
    if not torch.equal(cache["tokens"],torch.from_numpy(np.concatenate([s.ids for s in data.sources]))):
        raise ValueError("Archived block cache differs in compacted tokens")
    return archive.BlockData(data.rows,data.sources,cache["vectors"])


def load_pair(seed, examples, archive, tokenizer, *, warm=False):
    new_root = RESULTS/f"seed{seed}"/"full"
    old_root = ARCHIVE/f"seed{seed}"/"joint"
    cfg = read_json(new_root/"config.json")
    current = FusionClassifier(fp32_classifier(tokenizer).detector,
                               CoarseClassifier(768,"joint",cfg["repair_weight"]),examples,tokenizer)
    if warm:
        # Locally produced checkpoint holds epoch-8 weights plus optimizer and RNG state.
        cp = torch.load(WEIGHTS/f"seed{seed}"/"context_warm"/"last.pt",map_location="cpu",weights_only=False,mmap=True)
        if cp["config"]["seed"]!=seed or cp["loop"][-1]["epoch"]!=8:
            raise ValueError("Replay requires the committed epoch-8 warm state")
        current.load_state_dict(cp["model"],strict=True)
        del cp
    else:
        current.load_state_dict(torch.load(WEIGHTS/f"seed{seed}"/"full"/"best.pt",
                                            map_location="cpu",weights_only=True,mmap=True),strict=True)
    archived = archive.FusionClassifier(fp32_classifier(tokenizer).detector)
    archived.load_state_dict(torch.load(OLD_WEIGHTS/f"seed{seed}"/"joint"/"best.pt",
                                        map_location="cpu",weights_only=True,mmap=True),strict=True)
    name = read_json(old_root/"config.json")["index_name"]
    old_coarse = archive.CoarseClassifier(768,"joint")
    old_coarse.load_state_dict(torch.load(OLD_WEIGHTS/f"seed{seed}"/"coarse"/name/"best.pt",
                                          map_location="cpu",weights_only=True,mmap=True),strict=True)
    return tuple(m.cuda().eval().requires_grad_(False) for m in (current,archived,old_coarse))


@torch.inference_mode()
def trace(model, data, collate, *, archive=None):
    """Observe Q and K from the frozen production forward, and verify the head average."""
    captured, output = {}, defaultdict(list)
    def capture(name):
        def hook(_module, _inputs, result):
            captured[name] = result
        return hook
    handles = [model.indexer.query.register_forward_hook(capture("q")),
               model.indexer.key.register_forward_hook(capture("k"))]
    if archive is None:
        handles.append(model.encoder.register_forward_hook(capture("encoder")))
    max_error = 0.
    try:
        for cpu in DataLoader(data,batch_size=BATCH_SIZE,collate_fn=collate):
            x = to_gpu(cpu)
            result = model(x,details=True)
            q = captured["q"].reshape(len(x["valid"]),4,-1)
            k = captured["k"]
            logits = torch.bmm(q,k.transpose(1,2))/q.shape[-1]**.5
            heads = logits.masked_fill(~x["valid"][:,None,:],-torch.inf).softmax(-1)
            p = result["importance"]
            max_error = max(max_error,float((heads.mean(1)-p).abs().max()))
            if archive is None:
                g,z,local = result["g"],result["z"],captured["encoder"][2]
            else:
                local = x["blocks"].float()
                g = archive.global_pool(local,x["valid"])
                z = torch.bmm(p[:,None,:],local).squeeze(1)
                output["coarse"].append(result["coarse"].cpu())
            output["g"].append(g.cpu())
            output["z"].append(z.cpu())
            output["uniform"].append(masked_mean(local,x["valid"]).cpu())
            for n,value,per_head in zip(cpu["valid"].sum(1).tolist(),p.cpu().numpy(),heads.cpu().numpy(),strict=True):
                output["p"].append(value[:n].copy())
                output["heads"].append(per_head[:,:n].copy())
    finally:
        for handle in handles:
            handle.remove()
    if max_error>2e-7:
        raise ValueError(f"Head probes changed or misread index scores: {max_error}")
    return {k:v if k in ("p","heads") else torch.cat(v) for k,v in output.items()}


def old_layout(scores,source,plan,archive):
    spans = tuple(archive.retrieve(scores,len(source.ids)))
    if len(source.ids)<=BUDGET:
        return Layout(spans)
    # Archived retrieval can clip its last block. Count that code separately as remainder.
    centers = tuple(i for i,(a,b) in enumerate(plan.centers) if any(lo<=a and b<=hi for lo,hi in spans))
    return Layout(spans,centers)


def make_layouts(data, joint, plain, fixed, cap, current, archived, archive, *, core_only):
    variants = defaultdict(list)
    for i,(source,new,old) in enumerate(zip(data.sources,current["p"],archived["p"],strict=True)):
        bundled = joint[i].retrieve(new,source)
        variants[NEW].append(bundled)
        variants[OLD].append(old_layout(old,source,fixed[i],archive))
        variants["center128"].append(plain[i].retrieve(new,source))
        variants["uniform"].append(uniform_layout(source))
        if not core_only:
            variants["cap64"].append(cap[i].retrieve(new,source))
            variants["same_centers"].append(center_only_with_fixed_anchors(joint[i],source,bundled))
            variants["old_score_center128"].append(
                plain[i].retrieve(project_scores(old,fixed[i].centers,plain[i].centers),source))
            variants["new_score_fixed32"].append(
                old_layout(project_scores(new,plain[i].centers,fixed[i].centers),source,fixed[i],archive))
    return dict(variants)


def pearson(x,y):
    x,y = np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    a,b = x-x.mean(),y-y.mean()
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/denom) if denom else np.nan


def index_summary(data, traces, grids, edits, groups):
    report = {}
    head_pairs = [(a,b) for a in range(4) for b in range(a+1,4)]
    for name,x in traces.items():
        rows, actual_blocks = [], []
        grids_ = grids[name]
        for i,(source,p,heads,plan) in enumerate(zip(data.sources,x["p"],x["heads"],grids_,strict=True)):
            edges = np.array([0,*[b for _,b in plan.centers]])
            width = np.diff(edges)
            mass = np.array([overlap_bpe([span],edits[i]) for span in plan.centers])
            actual_blocks.append(mass>0)
            rank = np.argsort(-p,kind="stable")
            high = rank[:4]
            hot = [plan.centers[j] for j in high]
            source_mid = np.array([(source.keep[a]+source.keep[b-1])*.5/len(source.raw_ids) for a,b in plan.centers])
            pr = rankdata(p)
            entropy, corr, jaccard = [np.nan]*4, [np.nan]*6, [np.nan]*6
            if len(p)>1:
                entropy = -(heads*np.log(heads.clip(1e-20))).sum(1)/np.log(len(p))
                ranked = rankdata(heads,axis=1)
                corr = [pearson(ranked[a],ranked[b]) for a,b in head_pairs]
                if len(p)>4:
                    top = [set(np.argsort(-h,kind="stable")[:4]) for h in heads]
                    jaccard = [len(top[a]&top[b])/len(top[a]|top[b]) for a,b in head_pairs]
            links = defaultdict(lambda:[0,0])
            for kind,anchor,context in data.syntax[i][1]:
                if intersects(anchor,hot):
                    links[kind][0] += 1
                    links[kind][1] += overlap_bpe([context],edits[i])>0
            rows.append(dict(entropy=entropy, corr=corr, jaccard=jaccard,
                             score_width=pearson(pr,rankdata(width)),score_position=pearson(pr,rankdata(source_mid)),
                             score_edit_density=pearson(pr,rankdata(mass/width)),
                             top_four_bpe=int(width[high].sum()),top_four_edit=int(mass[high].sum()),
                             top_four_hit=int((mass[high]>0).sum()),top_four_blocks=len(high),
                             edit_bpe=int(mass.sum()),links=dict(links)))
        by_group = {}
        for g,ids in groups.items():
            selected = [rows[i] for i in ids]
            s = lambda field:sum(r[field] for r in selected)
            truth = [actual_blocks[i] for i in ids]
            eligible = truth and sum(int(a.sum()) for a in truth) and any(not a.all() for a in truth)
            links = defaultdict(lambda:[0,0])
            for row in selected:
                for k,(count,hit) in row["links"].items():
                    links[k][0] += count
                    links[k][1] += hit
            by_group[g] = dict(functions=len(ids), head_entropy=[
                distribution(r["entropy"][h] for r in selected) for h in range(4)],
                head_spearman={f"{a+1}/{b+1}":distribution(r["corr"][j] for r in selected)
                               for j,(a,b) in enumerate(head_pairs)},
                head_top4_jaccard={f"{a+1}/{b+1}":distribution(r["jaccard"][j] for r in selected)
                                   for j,(a,b) in enumerate(head_pairs)},
                same_function_spearman={k:distribution(r[k] for r in selected)
                    for k in ("score_width","score_position","score_edit_density")},
                top4={k:s("top_four_"+k) for k in ("bpe","edit","hit","blocks")},
                top4_link_counts={k:dict(relations=v[0],relation_targets_with_edit=v[1]) for k,v in links.items()},
                block_metrics=region_metrics([x["p"][i] for i in ids],truth,ks=(1,4,10,15)) if eligible else None)
        report[name] = dict(unit="32-BPE fixed blocks" if name=="archived" else "128-target statement blocks",
                            by_group=by_group)
    return report


def summarize_layouts(data, variants, annotations, edits, grids, trace_, groups, *, same_anchors):
    report = {}
    for name,layouts in variants.items():
        plan = grids[name]
        rows = [budget_row(source,layout,candidate,edits[i],data.syntax[i][0]) for i,(source,layout,candidate)
                in enumerate(zip(data.sources,layouts,plan,strict=True))]
        raw = [len(s.raw_ids) for s in data.sources]
        retention = grouped_edit_retention((n,d,raw_runs(s.keep,l.spans)) for n,d,s,l in
                                            zip(raw,annotations,data.sources,layouts,strict=True))
        report[name] = dict(budget={g:summarize_budgets(rows,ids) for g,ids in groups.items()},edit_retention=retention)
        if same_anchors:
            report[name]["same_anchor_vs_joint128"] = context_quality(data,layouts,variants[NEW])
    blockers = [high_score_blockers(s,p,l,j,c,m,edits[i]) for i,(s,p,l,j,c,m) in enumerate(zip(
        data.sources,trace_["p"],variants[NEW],grids[NEW],grids["center128"],variants["center128"],strict=True))]
    report["geometry"] = {k:{g:geometry(data,grids[k],ids) for g,ids in groups.items()} for k in (OLD,NEW)}
    report["raw_top4_blockers"] = {g:{key:sum(blockers[i].get(key,0) for i in ids) for key in (
        "high","missed","terminal_cost_blocked","displaced_vs_center_only","oversized_high","displaced_edit_bpe")}
        for g,ids in groups.items()}
    return report


def pack(source, layout, tokenizer):
    ids,n = pack_fixed(source.raw_ids,source.keep,layout.spans,tokenizer)
    if n!=min(MAX_LENGTH,len(source.ids)+2):
        raise ValueError("An intervention failed to fill the same fine budget")
    return ids[0],n


@torch.inference_mode()
def encode(fine, packed):
    result = []
    for start in range(0,len(packed),BATCH_SIZE):
        ids,n = zip(*packed[start:start+BATCH_SIZE],strict=True)
        tokens = torch.stack(ids).cuda().long()
        mask = torch.arange(tokens.shape[1],device="cuda")[None,:] < torch.tensor(n,device="cuda")[:,None]
        result.append(fine.encoder(input_ids=tokens,attention_mask=mask.long()).last_hidden_state[:,0])
    return torch.cat(result)


def read_predictions(path, data):
    with gzip.open(path,"rt") as handle:
        rows = list(csv.DictReader(handle))
    for row,ref,source in zip(rows,data.rows,data.sources,strict=True):
        if (row["sample_key"]!=ref["sample_key"] or int(row["target"])!=ref["target"]
                or int(row["original_bpe"])!=len(source.raw_ids)):
            raise ValueError(f"Misaligned archived predictions in {path}")
    return rows


def parity(path, data, spans, probabilities):
    saved = read_predictions(path,data)
    mismatches = sum([list(s) for s in l.spans]!=json.loads(r["encoding_runs"])
                     for l,r in zip(spans,saved,strict=True))
    values = np.array([float(r["probability"]) for r in saved])
    diff = float(np.max(np.abs(values-probabilities)))
    flipped = int(np.sum((values>.5)!=(probabilities>.5)))
    if mismatches or diff>3e-6 or flipped:
        raise ValueError(f"Baseline parity failed in {path}: {mismatches} layouts, {diff:.3g}, {flipped} decisions")
    return dict(layout_changes=mismatches,max_absolute_probability_error=diff,decision_changes=flipped)


def current_parts(model,g,z,z_uniform):
    wg,wz = model.fusion.weight.split(g.shape[1],dim=1)
    full = model.fusion(torch.cat((g,z),dim=1))
    a,b = F.linear(g,wg),F.linear(z,wz)
    torch.testing.assert_close(a+b,full,atol=2e-6,rtol=1e-5)
    return full,a,b,model.fusion(torch.cat((g,z_uniform),dim=1))


def representation_gradients(model, g,z,f,targets, projected):
    with torch.enable_grad():
        g,z,f = [x.detach().clone().requires_grad_(True) for x in (g,z,f)]
        mixed = model.fusion(torch.cat((g,z),dim=1))+f
        logits = model.fine.classifier(mixed[:,None])[:,0]
        total = F.binary_cross_entropy_with_logits(logits,targets,reduction="sum")
        gg,gz,gf = torch.autograd.grad(total,(g,z,f))
    parts = dict(g=projected[0],z=projected[1],fine=f.detach())
    report = {f"norm_{name}":part.norm(dim=1) for name,part in parts.items()}
    for a,b in (("g","z"),("g","fine"),("z","fine")):
        report[f"cos_{a}_{b}"] = F.cosine_similarity(parts[a],parts[b],dim=1)
    for name,grad in zip(parts,(gg,gz,gf),strict=True):
        report[f"bce_grad_{name}"] = grad.norm(dim=1)
        # Derivative with respect to a gain on this summand, at gain 1.
        report[f"bce_abs_gain_grad_{name}"] = (gf*parts[name]).sum(1).abs()
    return report


def inference(model, old_coarse, trace_, data, variants, tokenizer, version):
    output, norms = defaultdict(list), defaultdict(list)
    native = BASELINES[version]
    for lo in range(0,len(data.rows),BATCH_SIZE):
        hi = min(lo+BATCH_SIZE,len(data.rows))
        fine_by_mode,shared = {},{}
        for kind,layouts in variants.items():
            key = tuple(l.spans for l in layouts[lo:hi])
            if key not in shared:
                shared[key] = encode(model.fine,[pack(data.sources[i],layouts[i],tokenizer) for i in range(lo,hi)])
            fine_by_mode[kind] = shared[key]
        with torch.no_grad():
            g,z,z_uniform = (trace_[k][lo:hi].cuda() for k in ("g","z","uniform"))
            if version=="current":
                full,a,b,uniform = current_parts(model,g,z,z_uniform)
            else:
                full = model.fusion(trace_["coarse"][lo:hi].cuda())
                uniform = model.fusion(old_coarse.representation(torch.cat((g,z_uniform),dim=1)))
            classify = lambda mixed:model.fine.classifier(mixed[:,None])[:,0].sigmoid().cpu().numpy()
            for kind,cls in fine_by_mode.items():
                output[kind].append(classify(cls+full))
            fine = fine_by_mode[native]
            output["repr_fine_only"].append(classify(fine))
            output["repr_coarse_only"].append(classify(full))
            output["repr_uniform_weights"].append(classify(fine+uniform))
            if version=="current":
                output["repr_mask_g"].append(classify(fine+b))
                output["repr_mask_z"].append(classify(fine+a))
            else:
                norms["norm_archived_coarse"].append(full.norm(dim=1).cpu().numpy())
                norms["norm_fine"].append(fine.norm(dim=1).cpu().numpy())
                norms["cos_archived_coarse_fine"].append(F.cosine_similarity(full,fine,dim=1).cpu().numpy())
        if version=="current":
            y = torch.tensor([r["target"] for r in data.rows[lo:hi]],device="cuda",dtype=torch.float32)
            for k,v in representation_gradients(model,g,z,fine,y,(a,b)).items():
                norms[k].append(v.cpu().numpy())
    return ({k:np.concatenate(v) for k,v in output.items()},
            {k:np.concatenate(v) for k,v in norms.items()})


def confusion(y,p,t):
    pos = p>t
    return int(np.sum((y==1)&pos)),int(np.sum((y==0)&~pos)),int(np.sum((y==0)&pos)),int(np.sum((y==1)&~pos))


def mcc_counts(tp,tn,fp,fn):
    arrays = np.broadcast_arrays(*[np.asarray(x,dtype=np.float64) for x in (tp,tn,fp,fn)])
    tp,tn,fp,fn = arrays
    den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return np.divide(tp*tn-fp*fn,den,where=den>0,out=np.zeros_like(den))


def valid_threshold(y,p):
    values = np.unique(np.r_[0.,p,1.])
    thresholds = np.unique(np.r_[0.,.5,1.,(values[1:]+values[:-1])/2])
    order = np.argsort(p,kind="stable")
    below = np.searchsorted(p[order],thresholds,side="right")
    fn = np.r_[0,y[order].cumsum()][below]
    tp = y.sum()-fn
    tn = below-fn
    fp = len(y)-y.sum()-tn
    mcc = mcc_counts(tp,tn,fp,fn)
    best = np.lexsort((thresholds,np.abs(thresholds-.5),-mcc))[0]
    return float(thresholds[best])


def threshold_metrics(y,p,t):
    tp,tn,fp,fn = confusion(y,p,t)
    return dict(threshold=t,mcc=float(mcc_counts(tp,tn,fp,fn)),
                f1=2*tp/max(1,2*tp+fp+fn),pred_positive=(tp+fp)/len(y),
                tp=tp,tn=tn,fp=fp,fn=fn)


def comparison_report(data, values, native, groups, valid=None):
    y = np.array([r["target"] for r in data.rows])
    lengths = np.array([len(s.raw_ids) for s in data.sources])
    pair_ids = group_ids(lengths.reshape(-1,2).max(1))
    logits = {k:logit(v.astype(float).clip(1e-10,1-1e-10)) for k,v in values.items()}
    baseline = values[native]
    report = {}
    for kind,probs in values.items():
        threshold = valid_threshold(y,probs) if valid is None else valid[kind]["valid_threshold"]
        shift = float(np.mean(logits[kind]-logits[native])) if valid is None else valid[kind]["valid_global_logit_shift"]
        shifted_baseline = expit(logits[native]+shift)
        grouped = {}
        for g,ids in groups.items():
            yy,pp = y[ids],probs[ids]
            pids = np.stack((2*pair_ids[g],2*pair_ids[g]+1),axis=1).flatten()
            pair_probs = probs[pids]
            delta = pp-baseline[ids]
            diff = logits[kind][ids]-logits[native][ids]
            item = metrics(yy.tolist(),pp.tolist())
            item.update(pred_positive=float((pp>.5).mean()),brier=float(np.mean((pp-yy)**2)),
                        bce=float(np.mean(np.logaddexp(0,logits[kind][ids])-yy*logits[kind][ids])),
                        score_distribution={str(c):distribution(pp[yy==c]) for c in (0,1)},
                        probability_delta={str(c):distribution(delta[yy==c]) for c in (0,1)},
                        abs_probability_delta=distribution(np.abs(delta)),logit_delta=distribution(diff),
                        logit_residual_after_valid_shift=distribution(diff-shift),
                        ranking_spearman=distribution([pearson(rankdata(pp),rankdata(baseline[ids]))]),
                        decision_changes=int(np.sum((pp>.5)!=(baseline[ids]>.5))),
                        valid_threshold_application=threshold_metrics(yy,pp,threshold),
                        baseline_with_valid_global_shift=threshold_metrics(yy,shifted_baseline[ids],.5))
            if len(pids):
                item["paired"] = paired_metrics([data.rows[i] for i in pids],pair_probs.tolist())
                item["paired"]["probability_margin"] = distribution(pair_probs[::2]-pair_probs[1::2])
                item["paired"]["p_c_valid_threshold"] = float(np.mean((pair_probs[::2]>threshold)&(pair_probs[1::2]<=threshold)))
            grouped[g] = item
        report[kind] = dict(valid_threshold=threshold,valid_global_logit_shift=shift, groups=grouped)
    return report


def paired_bootstrap(data, base, intervention, groups):
    rng = np.random.default_rng(42)
    y = np.array([r["target"] for r in data.rows])
    # Resample official pairs and then restrict functions to each length group.
    picked = rng.integers(len(y)//2,size=(1000,len(y)//2))
    out = {}
    for g in ("overall","long","2041+"):
        scope = np.zeros(len(y),dtype=bool)
        scope[groups[g]] = True
        ro = []
        for probs in (base,intervention):
            pred = probs>.5
            counts = [((y==c)&(pred==p)&scope).reshape(-1,2).sum(1)[picked].sum(1)
                      for c,p in ((1,1),(0,0),(0,1),(1,0))]
            ro.append(mcc_counts(*counts))
        out[g] = np.quantile(ro[1]-ro[0],[.025,.5,.975]).tolist()
    return dict(replicates=1000,pair_bootstrap_percentiles=out)


def history(seed):
    a,b = ARCHIVE/f"seed{seed}", RESULTS/f"seed{seed}"
    methods = {"archived_"+m:a/m for m in ("joint","uniform","task","repair")}
    methods.update({"current_"+m:b/m for m in ("full","uniform","context","balanced","block_context")})
    out = {}
    for name,root in methods.items():
        if not (root/"epochs.json").is_file():
            continue
        fit = read_json(root/"metrics.json")
        epochs = read_json(root/"epochs.json")
        out[name] = dict(best_epoch=fit["selected_epoch"],epochs=[
            dict(epoch=e["epoch"],train_bce=e.get("train_bce",e["train_loss"]),train_objective=e["train_loss"],
                 valid_mcc=e["validation"]["mcc"],valid_auc=e["validation"]["auc"],valid_ap=e["validation"]["auprc"],
                 valid_pred_positive=(e["validation"]["tp"]+e["validation"]["fp"])/e["validation"]["samples"])
            for e in epochs])
    return out


def run(seed,split,data,tokenizer,archive,datasets,joint,plain,fixed,cap,out):
    started = perf_counter()
    examples = list(zip(data.sources,joint,strict=True))
    current,archived,coarse_old = load_pair(seed,examples,archive,tokenizer)
    traced = {"current":trace(current.coarse,datasets[0],collate_blocks),
              "archived":trace(coarse_old,datasets[1],archive.collate_blocks,archive=archive)}
    variants = make_layouts(data,joint,plain,fixed,cap,traced["current"],traced["archived"],archive,core_only=split=="train")
    groups = group_ids([len(s.raw_ids) for s in data.sources])
    grids = {NEW:joint,OLD:fixed,"center128":plain,"uniform":plain,"cap64":cap,
             "same_centers":plain,"old_score_center128":plain,"new_score_fixed32":fixed}
    result = dict(seed=seed,split=split,functions=len(data.rows),
                  commits=dict(current=CURRENT_COMMIT,archived=ARCHIVED_COMMIT),context_increment_cap=CONTEXT_CAP)
    if split!="train":
        predicted, report = {}, {}
        valid = read_json(out/f"valid_seed{seed}.json")["models"] if split=="test" else None
        for v,model in (("current",current),("archived",archived)):
            print(split,seed,v,"frozen layout and representation forward",flush=True)
            scores,norms = inference(model,coarse_old,traced[v],data,variants,tokenizer,v)
            root = (RESULTS/f"seed{seed}"/"full") if v=="current" else (ARCHIVE/f"seed{seed}"/"joint")
            native = BASELINES[v]
            check = {native:parity(root/f"{split}.csv.gz",data,variants[native],scores[native]),
                     "uniform":parity(root/"same_checkpoint_uniform"/f"{split}.csv.gz",data,variants["uniform"],scores["uniform"])}
            metrics_ = comparison_report(data,scores,native,groups,None if valid is None else valid[v]["metrics"])
            report[v] = dict(native=native,archived_parity=check,metrics=metrics_,
                             representations={k:{g:distribution(xs[ids]) for g,ids in groups.items()} for k,xs in norms.items()},
                             bootstrap={mode:paired_bootstrap(data,scores[native],scores[mode],groups) for mode in (
                                 "center128","uniform",OLD if v=="current" else NEW) if mode!=native} if seed==42 else {})
            predicted[v] = {**{f"p_{m}":vs for m,vs in scores.items()}, **norms}
        rows = []
        for v,x in predicted.items():
            native_prob = x["p_"+BASELINES[v]]
            for i,(row,source) in enumerate(zip(data.rows,data.sources,strict=True)):
                record = dict(version=v,sample_key=row["sample_key"],pair_key=row["pair_key"],target=row["target"],
                             raw_bpe=len(source.raw_ids),compacted_bpe=len(source.ids))
                record["values"] = {k:float(series[i]) for k,series in x.items()}
                record["probability_change"] = {k[2:]:float(series[i]-native_prob[i]) for k,series in x.items()
                                                if k.startswith("p_") and k!="p_"+BASELINES[v]}
                rows.append(record)
        write_prediction_rows(out/f"{split}_seed{seed}_predictions.csv.gz",rows)
        result["models"] = report
        if split=="valid":
            result["training_history"] = history(seed)
    # Use true paired edits only after frozen Test layouts and predictions have been saved.
    annotations = paired_annotations(data.sources)
    edits = [actual_edits(s,a)[1] for s,a in zip(data.sources,annotations,strict=True)]
    print(split,seed,"edit and score audit",flush=True)
    result["layouts"] = summarize_layouts(data,variants,annotations,edits,grids,traced["current"],groups,
                                          same_anchors=split!="train")
    result["indexers"] = index_summary(data,traced,{"current":joint,"archived":fixed},edits,groups)
    result["seconds"] = perf_counter()-started
    del current,archived,coarse_old
    gc.collect()
    write_json(out/f"{split}_seed{seed}.json",result)
    print(split,seed,"finished",round(result["seconds"],2),"s",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split",choices=("train","valid","test"))
    parser.add_argument("--seeds",type=int,nargs="+",choices=SEEDS,default=list(SEEDS))
    parser.add_argument("--out",type=Path,default=OUT)
    args = parser.parse_args()
    if args.split=="test" and any(not (args.out/f"valid_seed{s}.json").exists() for s in args.seeds):
        raise ValueError("Freeze and verify every Valid configuration before running Test")
    if torch.cuda.device_count()!=1:
        raise ValueError("Select one GPU with CUDA_VISIBLE_DEVICES")
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    args.out.mkdir(parents=True,exist_ok=True)
    tok,archive = load_tokenizer("unixcoder",UNIX_BASE),legacy_module()
    started = perf_counter()
    data = corpus(args.split,tok,[LAYOUT])
    joint,plain = plans(data,LAYOUT,True),plans(data,LAYOUT,False)
    old = legacy_data(data,args.split,archive)
    fixed = [Candidates([*range(archive.BLOCK,len(s.ids),archive.BLOCK),len(s.ids)],(),(),contextual=False)
             for s in data.sources]
    cap = ([] if args.split=="train" else
           [capped_candidates(j,u,r) for j,(u,r,_e) in zip(joint,data.syntax,strict=True)])
    datasets = block_data(data,LAYOUT,candidates=joint),old
    print(args.split,"sources and caches verified",len(data.rows),round(perf_counter()-started,2),"s",flush=True)
    for seed in args.seeds:
        run(seed,args.split,data,tok,archive,datasets,joint,plain,fixed,cap,args.out)


if __name__=="__main__":
    main()
