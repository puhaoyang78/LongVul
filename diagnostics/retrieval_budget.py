"""Read-only budget, geometry, and whole-candidate Oracle diagnostics."""
from bisect import bisect_left
from collections import Counter, defaultdict
from time import perf_counter

import numpy as np

from coarse_retrieval import Candidates, LAYOUTS, Layout, PRIORITY, block_ends, lengths, part_overlap, uncovered
from compaction import BUDGET, MAX_LENGTH
from context_selection import contains, intersects, merge, occupied, uniform_fill
from repair_oracle import EDIT_TYPES, compact_edits, overlap_bpe
from window_inputs import GROUPS

CONTEXT_CAP = 64
ORACLE_TIME_LIMIT = .30


def distribution(values):
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0}
    q = np.quantile(a, [.05,.25,.5,.75,.95])
    return dict(n=len(a), mean=float(a.mean()), min=float(a.min()), max=float(a.max()),
                **dict(zip(("p05","p25","median","p75","p95"), q.tolist(), strict=True)))


def group_ids(raw_lengths):
    raw = np.asarray(raw_lengths)
    return {g:np.flatnonzero((lo<=raw)&(raw<=hi)) for g,(lo,hi) in GROUPS.items()}


def actual_edits(source, annotations):
    raw = merge(a for kind in EDIT_TYPES for a in annotations[kind])
    mapping = compact_edits(source.keep, annotations)
    compact = merge(a for kind in EDIT_TYPES for a in mapping[kind])
    return raw, compact


def capped_candidates(joint, units, links):
    """Keep accepted whole relations under a cap on incremental positions per bundle."""
    ends = [hi for _,hi in joint.centers]
    relations = defaultdict(set)
    for a,b in units:
        for i,lo,hi in part_overlap(a,b,ends):
            if (lo,hi)!=(a,b) and contains((a,b),joint.bundle[i]):
                relations[i].add(("completion",a,b))
    for kind,anchor,context in links:
        if kind not in PRIORITY:
            continue
        for i,_,_ in part_overlap(*anchor,ends):
            if contains(context,joint.bundle[i]):
                relations[i].add((kind,*context))
    capped = Candidates(ends,(),(),contextual=False)
    capped.contextual = True
    for i,center in enumerate(joint.centers):
        lo,hi = center
        ordered = sorted(relations[i],key=lambda r:(PRIORITY[r[0]],max(0,lo-r[2],r[1]-hi),r[1],r[2]))
        spans = [center]
        for _,a,b in ordered:
            expanded = merge([*spans,(a,b)])
            if occupied(expanded)-(hi-lo)<=CONTEXT_CAP:
                spans = expanded
        if uncovered(spans,joint.bundle[i]):
            raise ValueError("Capped candidate is not contained in the original")
        capped.bundle[i] = tuple(spans)
    capped.starts,capped.stops = np.asarray([s for bundle in capped.bundle for s in bundle],dtype=np.int32).T
    capped.offsets = np.r_[0,np.cumsum([len(b) for b in capped.bundle])[:-1]]
    return capped


def project_scores(values, old, new):
    """Transport probability mass uniformly within each block over the shared BPE axis."""
    ends = [b for _,b in old]
    if len(values)!=len(old) or old[-1][1]!=new[-1][1]:
        raise ValueError("Score partitions must cover the same source")
    density = np.asarray(values,dtype=np.float64)/np.diff([0,*ends])
    out = np.array([sum(density[i]*(b-a) for i,a,b in part_overlap(lo,hi,ends)) for lo,hi in new])
    if not np.isclose(out.sum(), np.sum(values,dtype=np.float64), rtol=1e-10, atol=1e-10):
        raise ValueError("Transport did not preserve score mass")
    return out


def center_only_with_fixed_anchors(joint, source, selected):
    if not selected.centers:
        return selected
    spans = [joint.centers[i] for i in selected.centers]
    return Layout(tuple(uniform_fill(spans,source.boundaries,[],contiguous=True)),selected.centers)


def budget_row(source, layout, plan, compact_edit, units):
    take = len(source.ids)>BUDGET
    centers = merge(plan.centers[i] for i in layout.centers)
    selected = merge(s for i in layout.centers for s in plan.bundle[i])
    context = uncovered(selected,centers)
    fill = uncovered(layout.spans,selected) if take else []
    body = lengths(layout.spans)
    marks = 2+max(0,len(layout.spans)-1)
    if uncovered(selected,layout.spans) or body+marks!=min(MAX_LENGTH,len(source.ids)+2):
        raise ValueError("Layout violates the budget or drops a selected candidate")
    if lengths(centers)+lengths(context)+lengths(fill)+(body if not take else 0)!=body:
        raise ValueError("Layout budget attribution is not a disjoint partition")
    touched = [u for u in units if intersects(u,layout.spans)]
    anchors = [u for u in units if intersects(u,centers)]
    return dict(needed=int(take), blocks=len(plan.centers), centers=len(layout.centers), fragments=len(layout.spans),
                center_bpe=lengths(centers), context_bpe=lengths(context), fill_bpe=lengths(fill),
                passthrough_bpe=body if not take else 0, special=marks, internal_sep=marks-2,
                center_edit=overlap_bpe(centers,compact_edit), context_edit=overlap_bpe(context,compact_edit),
                fill_edit=overlap_bpe(fill,compact_edit), visible_edit=overlap_bpe(layout.spans,compact_edit),
                touched=len(touched), complete=sum(contains(u,layout.spans) for u in touched),
                anchored=len(anchors), anchors_complete=sum(contains(u,layout.spans) for u in anchors),
                longest_fragment=max(b-a for a,b in layout.spans),
                average_fragment=body/len(layout.spans),
                gap_widths=[a-b for (_,b),(a,_) in zip(layout.spans,layout.spans[1:])])


def high_score_blockers(source, scores, selected, joint, plain, plain_layout, compact_edits):
    # The raw-score top four have roughly one 128-BPE fine-budget worth of centers.
    if len(source.ids)<=BUDGET:
        return {}
    high = np.argsort(-scores,kind="stable")[:4]
    keep = merge(s for i in selected.centers for s in joint.bundle[i])
    room = BUDGET-occupied(keep)
    marginal = joint.marginal(keep,separators=True)
    bare = plain.marginal(keep,separators=True)
    missed = [int(i) for i in high if not contains(joint.centers[i],selected.spans)]
    only_bare = [i for i in missed if bare[i]<=room<marginal[i]]
    displaced = [i for i in missed if contains(joint.centers[i],plain_layout.spans)]
    return dict(high=len(high), missed=len(missed), terminal_cost_blocked=len(only_bare),
                displaced_vs_center_only=len(displaced), oversized_high=sum(occupied(joint.bundle[i])>MAX_LENGTH//2 for i in high),
                displaced_edit_bpe=sum(overlap_bpe([joint.centers[i]],compact_edits) for i in displaced))


def geometry(data, plan_rows, group):
    sizes = [hi-lo for i in group for lo,hi in plan_rows[i].centers]
    eligible = [i for i in group if len(data.sources[i].ids)>BUDGET]
    size = [occupied(bundle) for i in eligible for bundle in plan_rows[i].bundle]
    ctx = [lengths(bundle)-(hi-lo) for i in eligible
           for bundle,(lo,hi) in zip(plan_rows[i].bundle,plan_rows[i].centers,strict=True)]
    over = sum(n>MAX_LENGTH//2 for n in size)
    return dict(functions=len(group), need_retrieval=len(eligible), block_lengths=distribution(sizes),
                blocks_per_function=distribution(len(plan_rows[i].centers) for i in group),
                standalone_bundle_positions=distribution(size), context_bpe_per_bundle=distribution(ctx),
                over_half=over, candidate_denominator=len(size), over_half_rate=over/len(size) if size else None)


def summarize_budgets(rows, indices):
    subset = [rows[i] for i in indices if rows[i]["needed"]]
    out = dict(need_retrieval=len(subset), functions=len(indices),
               passthrough=sum(not rows[i]["needed"] for i in indices))
    if not subset:
        return out
    for key in subset[0]:
        if key in ("needed","gap_widths"):
            continue
        out[key] = distribution(r[key] for r in subset)
    s = lambda key: sum(r[key] for r in subset)
    out.update(statement_complete_rate=s("complete")/max(1,s("touched")),
               anchor_complete_rate=s("anchors_complete")/max(1,s("anchored")),
               gap_widths=distribution(w for r in subset for w in r["gap_widths"]),
               edit_totals={k:s(k+"_edit") for k in ("center","context","fill","visible")})
    return out


def oracle_bounds(bundles, edits):
    """Maximize edit BPE over unions of whole candidates, with exact SEP accounting."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix

    available = lengths(edits)
    if not available:
        return 0,0,"trivial"
    useful = list(dict.fromkeys(tuple(b) for b in bundles if overlap_bpe(b,edits)))
    if not useful:
        return 0,0,"trivial"
    covering = merge(span for bundle in useful for span in bundle)
    ceiling = overlap_bpe(covering,edits)
    lower = max(overlap_bpe(b,edits) for b in useful)
    if lower==ceiling or occupied(covering)<=BUDGET:
        return ceiling,ceiling,"trivial"
    # y is the union of chosen candidates x. A rise in y starts a source run.
    cuts = sorted({v for b in useful for span in b for v in span})
    atoms = list(zip(cuts,cuts[1:]))
    k,n = len(useful),len(atoms)
    width = k+2*n
    incident = [[] for _ in atoms]
    row,col,val,lo,hi = [],[],[],[],[]
    def bound(variables,coeffs,minimum=-np.inf,maximum=np.inf):
        variables = list(variables)
        row.extend([len(lo)]*len(variables))
        col.extend(variables)
        val.extend(coeffs)
        lo.append(minimum)
        hi.append(maximum)
    for i,bundle in enumerate(useful):
        covered = [j for a,b in bundle for j in range(bisect_left(cuts,a),bisect_left(cuts,b))]
        for j in covered:
            incident[j].append(i)
        bound([i,*[k+j for j in covered]],[-len(covered),*[1]*len(covered)],minimum=0)
    for j,indices in enumerate(incident):
        bound([k+j,*indices],[1,*[-1]*len(indices)],maximum=0)
        prior = [k+j-1] if j else []
        bound([k+j,k+n+j,*prior],[1,-1,*[-1]*len(prior)],maximum=0)
    bound(range(k,width),[b-a for a,b in atoms]+[1]*n,maximum=BUDGET+1)
    constraint = LinearConstraint(coo_matrix((val,(row,col)),shape=(len(lo),width)).tocsc(),lo,hi)
    objective = np.zeros(width)
    objective[k:k+n] = [-overlap_bpe([span],edits) for span in atoms]
    result = milp(objective,integrality=np.ones(width),bounds=Bounds(0,1),constraints=constraint,
                  options={"time_limit":ORACLE_TIME_LIMIT,"mip_rel_gap":0.})
    if result.x is not None:
        selected = merge(span for b,on in zip(useful,result.x[:k],strict=True) if on>.5 for span in b)
        if occupied(selected)>BUDGET:
            raise ValueError("Oracle returned an infeasible union")
        lower = max(lower,overlap_bpe(selected,edits))
    if result.status==0:
        return lower,lower,"optimal"
    if result.status!=1:
        raise RuntimeError(f"Unexpected Oracle solver status {result.status}: {result.message}")
    dual = result.get("mip_dual_bound")
    upper = min(ceiling,max(lower,int(np.floor(-dual+1e-6)))) if dual is not None and np.isfinite(dual) else ceiling
    return lower,upper,"bounded" if lower<upper else "certified"


def oracle_report(data, annotations, members):
    """Aligned and fixed partitions use the same source, syntax, BPE and budget."""
    started = perf_counter()
    lengths_raw = [len(data.sources[i].raw_ids) for i in members]
    groups = group_ids(lengths_raw)
    report = {}
    for name, (target, aligned) in LAYOUTS.items():
        rows = {"independent":[], "bundled":[]}
        for i in members:
            source = data.sources[i]
            units,edges,_ = data.syntax[i]
            raw,comp = actual_edits(source,annotations[i])
            ends = block_ends(len(source.ids),units,source.boundaries,target,aligned)
            if name in data.layouts and ends!=data.layouts[name][i]:
                raise ValueError(f"Recreated {name} cuts differ from cache")
            for mode in rows:
                if len(source.ids)<=BUDGET:
                    low=high=lengths(comp)
                    status="passthrough"
                else:
                    plan = Candidates(ends,units,edges,contextual=mode=="bundled")
                    low,high,status = oracle_bounds(plan.bundle,comp)
                rows[mode].append((lengths(raw),lengths(comp),low,high,status))
        for mode, table in rows.items():
            groups_out = {}
            for g, ids in groups.items():
                subset = [table[i] for i in ids]
                raw,comp,low,high = (sum(x[j] for x in subset) for j in range(4))
                groups_out[g] = dict(functions=len(ids),raw_edit_bpe=raw,compacted_edit_bpe=comp,
                    best_feasible_bpe=low,certified_upper_bpe=high, lower=low/raw if raw else None,
                    upper=high/raw if raw else None,status=dict(Counter(x[4] for x in subset)))
            report[f"{name}/{mode}"] = groups_out
        print("oracle",name,round(perf_counter()-started,2),"s",flush=True)
    return dict(results=report,functions=len(members),solver_time_limit=ORACLE_TIME_LIMIT,seconds=perf_counter()-started)


def main():
    import argparse
    from pathlib import Path

    import torch

    from coarse_to_fine import corpus
    from paired import ROOT, UNIX_BASE, load_tokenizer
    from predicted_diff import paired_annotations
    from rgcl.artifacts import write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("splits",nargs="+",choices=("train","valid","test"))
    parser.add_argument("--out",type=Path,default=ROOT/"results"/"coarse_to_fine_diagnosis")
    args = parser.parse_args()
    torch.set_num_threads(4)
    tokenizer = load_tokenizer("unixcoder", UNIX_BASE)
    for split in args.splits:
        if split=="test" and not (args.out/"oracle_valid.json").is_file():
            raise ValueError("Complete the fixed Valid Oracle evaluation before Test")
        data = corpus(split,tokenizer,["statement128"])
        pairs = np.arange(len(data.rows)//2)
        if split=="train":
            lengths_raw = np.asarray([len(s.raw_ids) for s in data.sources]).reshape(-1,2).max(1)
            rng = np.random.default_rng(42)
            strata = ("short","511-1020","1021-2040","2041+")
            pairs = np.sort(np.concatenate([rng.choice(ids,min(64,len(ids)),replace=False)
                             for g,ids in group_ids(lengths_raw).items() if g in strata]))
        ids = np.stack((pairs*2,pairs*2+1),axis=1).flatten().tolist()
        # Only sampled complete pairs are diffed. Source membership of the cache is validated above.
        observed = paired_annotations([data.sources[i] for i in ids])
        report = oracle_report(data,dict(zip(ids,observed,strict=True)),ids)
        report["sampling"] = "64 pairs per raw max-side length stratum, seed 42" if split=="train" else "complete split"
        report["sample_keys"] = [data.rows[i]["sample_key"] for i in ids] if split=="train" else None
        write_json(args.out/f"oracle_{split}.json",report)


if __name__=="__main__":
    main()
