import unittest
from types import SimpleNamespace

import numpy as np
import torch

from coarse_encoding import (INDEX_WIDTH, QUERY_HEADS, BlockData, CoarseClassifier, ContextIndexer,
                             block_mass, collate_blocks, repair_block_distribution, repair_block_mass)
from coarse_retrieval import Layout, block_ends, lengths, part_overlap, retrieve, selected_statement_scores, uncovered
from context_selection import contains, merge, occupied, prefer_context, source_parser
from diagnostics.context_retrieval import group_counts, related_context_counts, score_bands
from repair_oracle import EDIT_TYPES, GAP_TYPES


class BlockEncodingTest(unittest.TestCase):
    def test_oracle_uses_the_training_distribution_without_inventing_labels(self):
        source = SimpleNamespace(keep=list(range(64)))
        edits = {kind:[] for kind in (*EDIT_TYPES,*GAP_TYPES)}
        self.assertEqual(repair_block_distribution(source,[32,64],edits).tolist(),[0.,0.])
        edits["replace"] = [(0,3)]
        edits["insert_gap"] = [(2,4),(33,37)]
        self.assertEqual(repair_block_distribution(source,[32,64],edits).tolist(),[.5,.5])

    def test_statement_boundaries_take_priority_and_preserve_the_tail(self):
        length, target = 165, 32
        statements = [(0,38),(20,32),(38,64),(64,90),(90,150),(150,165)]
        cuts = block_ends(length,statements,list(range(0,length,10))+[length],target,True)
        self.assertEqual(cuts,[38,64,90,120,150,165])
        self.assertEqual(block_ends(length,statements,[],target,False),[32,64,96,128,160,165])
        self.assertEqual(list(part_overlap(35,68,cuts)),[(0,35,38),(1,38,64),(2,64,68)])
        windows = [(0,45),(45,130),(130,165)]
        per_block = torch.stack([block_mass(cuts,[span]) for span in windows]).sum(0)
        self.assertEqual(per_block.tolist(),[38.,26.,26.,30.,30.,15.])

    def test_repair_mass_deduplicates_edits_and_preserves_empty_side_signal(self):
        source = SimpleNamespace(keep=list(range(67)))
        edits = {kind:[] for kind in (*EDIT_TYPES,*GAP_TYPES)}
        edits.update(replace=[(30,35)],insert_gap=[(33,40)],delete_gap=[(63,67)])
        cuts = [32,64,67]
        self.assertEqual(repair_block_mass(source,cuts,edits,(*EDIT_TYPES,*GAP_TYPES)).tolist(),[2.,9.,3.])
        self.assertEqual(repair_block_mass(source,cuts,edits,EDIT_TYPES).tolist(),[2.,3.,0.])
        self.assertEqual(repair_block_mass(source,cuts,edits,("delete_gap",)).tolist(),[0.,1.,3.])


class RetrievalTest(unittest.TestCase):
    def test_all_zero_scores_follow_the_existing_stable_source_order(self):
        source = SimpleNamespace(ids=range(1024))
        ends = block_ends(len(source.ids),[],[],32,False)
        layout = retrieve(np.zeros(len(ends)),source,ends,0)
        self.assertEqual(layout.spans,[(0,510)])
        self.assertEqual(layout.centers,[(0,480)])

    def test_relation_audit_distinguishes_retrieved_from_filled_context(self):
        source = SimpleNamespace(keep=list(range(0,96,2)))
        centers = [(0,16),(24,32)]
        links = [("condition",(2,4),(20,24)),("assignment",(24,30),(36,40))]
        layouts = [Layout(merge([*centers,(20,24),(36,40)]),centers,[(20,24)],[(36,40)]),
                   Layout(merge([*centers,(36,40)]),centers,[(36,40)],[])]
        counts = related_context_counts(source,[16,24,32,48],[.1,.8,.5,.6],layouts,links)
        report = group_counts([counts],[96])["overall"]
        self.assertEqual(score_bands({(2,4):.2,(24,30):.2}),{(2,4):"tied",(24,30):"tied"})
        self.assertEqual(report["importance"]["lower"]["relation_based_after_context_rate"],1.)
        self.assertEqual(report["importance"]["upper"]["relation_based_after_context_rate"],0.)
        self.assertEqual(report["importance"]["upper"]["relation_based_in_fine_input_rate"],1.)
        self.assertEqual(report["distance"]["33-128"]["candidates"],1)
        self.assertEqual(report["distance"]["1-32"]["candidates"],1)
        self.assertEqual(report["relation"]["condition"]["candidates"],1)

    def test_statement_score_uses_maximum_over_selected_blocks_only(self):
        links = [("condition",(60,70),(0,8)),("assignment",(80,110),(19,28))]
        ends = [32,64,96,128,160]
        ranking = [.8,.4,.2,.9,.1]
        self.assertEqual(selected_statement_scores(ranking,ends,[(32,96)],links),
                         {(60,70):.4,(80,110):.2})

    def test_importance_changes_context_priority_and_sums_shared_relations(self):
        blocks = [(64,96),(128,160)]
        low,high = (70,85),(130,150)
        left,right = (10,16),(170,176)
        links = [("condition",low,left),("declaration",high,right)]
        budget = occupied(blocks)+7
        reference = prefer_context(blocks,links,budget=budget)
        guided = prefer_context(blocks,links,budget=budget,statement_scores={low:.1,high:.9})
        self.assertEqual(reference,merge([*blocks,left]))
        self.assertEqual(guided,merge([*blocks,right]))
        self.assertEqual(occupied(guided),budget)
        shared = prefer_context(blocks,[*links,("assignment",high,left)],budget=budget,
                                statement_scores={low:.1,high:.9})
        self.assertEqual(shared,reference)

    def test_zero_budget_matches_historical_partial_block_and_short_passthrough(self):
        source = SimpleNamespace(ids=range(1600))
        ends = block_ends(len(source.ids),[],[],32,False)
        scores = -np.arange(len(ends),dtype=float)
        scores[-1] = 1.
        pick = retrieve(scores,source,ends,0)
        self.assertEqual(pick.spans,[(0,477),(1568,1600)])
        self.assertEqual(pick.centers,[(0,448),(1568,1600)])
        self.assertEqual(pick.fill,[(448,477)])
        self.assertEqual(pick.context,[])
        short = SimpleNamespace(ids=range(45))
        self.assertEqual(retrieve(scores[:2],short,[32,45],128).spans,[(0,45)])

    def test_context_keeps_centers_and_charges_shared_declaration_once(self):
        declaration = "int value = 1;"
        calls = "use(value);\n"*100
        code = f"void f() {{\n{declaration}\n{calls}}}\n"
        source = SimpleNamespace(code=code,ids=range(len(code)),keep=list(range(len(code))),
                                 offsets=[(i,i+1) for i in range(len(code))])
        ends = block_ends(len(code),[],[],32,False)
        scores = np.zeros(len(ends))
        scores[10:14] = [3,4,2,1]
        pick = retrieve(scores,source,ends,64,source_parser())
        start = code.index(declaration)
        statement = (start,start+len(declaration))
        self.assertTrue(contains(statement,pick.spans))
        self.assertTrue(contains((320,448),pick.centers))
        # The declaration overlaps the first center in this initial ranking.
        scores[:8] = -1
        pick = retrieve(scores,source,ends,64,source_parser())
        self.assertTrue(contains(statement,pick.context))
        self.assertEqual(sum(b-a for a,b in pick.context if (a,b)==statement),len(declaration))
        self.assertFalse(uncovered(pick.centers,pick.spans))
        self.assertEqual(occupied(pick.spans),510)
        self.assertEqual(lengths(pick.centers)+lengths(pick.context)+lengths(pick.fill),lengths(pick.spans))

    def test_exact_budget_for_variable_blocks_and_extreme_length(self):
        rng = np.random.default_rng(42)
        for n in (511,543,590,1024,278200):
            source = SimpleNamespace(ids=range(n))
            ends = block_ends(n,[],list(range(0,n,29)),32,True)
            scores = rng.random(len(ends))
            pick = retrieve(scores,source,ends,0)
            self.assertEqual(pick.spans,merge(pick.spans))
            self.assertEqual(occupied(pick.spans),510)
            self.assertTrue(all(0<=a<b<=n for a,b in pick.spans))
            self.assertFalse(uncovered(pick.centers,pick.spans))


class IndexerTest(unittest.TestCase):
    def test_token_matching_matches_per_head_reference_and_masks_tail(self):
        torch.manual_seed(42)
        ends = [[32,35],[9],[32,64,65]]
        width = 8
        values = torch.randn(109,width)
        blocks = torch.randn(6,width)
        data = BlockData([{"target":1}]*3,ends,blocks,token_features=values)
        batch = collate_blocks([data[i] for i in range(3)])
        self.assertEqual(batch["block_sizes"].tolist(),[32,3,9,32,32,1])
        self.assertEqual(batch["tokens"].shape,(6*32,width))
        self.assertEqual(batch["owners"].tolist(),[0,0,1,2,2,2])
        g = torch.randn(3,2*width)
        for mode in ("token_mean","token_logmeanexp"):
            model = ContextIndexer(width,mode)
            keys = model.key(torch.nn.functional.layer_norm(values,(width,)))
            queries = model.query(torch.nn.functional.layer_norm(g,(2*width,))).reshape(3,QUERY_HEADS,INDEX_WIDTH)
            expected = torch.zeros(3,QUERY_HEADS,3)
            offset = 0
            for i,cut in enumerate(ends):
                logits = queries[i] @ keys[offset:offset+cut[-1]].T / INDEX_WIDTH**.5
                part = logits.split([b-a for a,b in zip([0,*cut[:-1]],cut)],dim=1)
                agg = [x.mean(1) if mode=="token_mean" else x.logsumexp(1)-np.log(x.shape[1]) for x in part]
                expected[i,:,:len(agg)] = torch.stack(agg,1).softmax(1)
                offset += cut[-1]
            output = model(batch["blocks"],batch["valid"],g,per_head=True,token_batch=batch)
            torch.testing.assert_close(output,expected)
            tail = torch.arange(32)[None,:] >= batch["block_sizes"][:,None]
            perturbed = dict(batch,tokens=batch["tokens"].clone())
            perturbed["tokens"][tail.flatten()] = 100*torch.randn_like(perturbed["tokens"][tail.flatten()])
            changed = model(batch["blocks"],batch["valid"],g,per_head=True,token_batch=perturbed)
            torch.testing.assert_close(changed,output,atol=0.,rtol=0.)
            output[:,0,0].sum().backward()
            self.assertTrue(all(p.grad.isfinite().all() and p.grad.norm()>0 for p in model.parameters()))

    def test_logmeanexp_has_no_length_bias_at_equal_scores_and_emphasizes_peaks(self):
        values = torch.zeros(2,32,4)
        values[1] = torch.tensor([1.,-1.,1.,-1.])
        values[0,0] = values[1,0]
        values[0,1] = -values[1,0]
        for mode in ("token_mean","token_logmeanexp"):
            model = ContextIndexer(4,mode)
            with torch.no_grad():
                model.query.weight.zero_()
                model.query.weight[:,0] = 1.
                model.key.weight.zero_()
                model.key.weight[:,0] = 1.
            batch = dict(tokens=values.reshape(-1,4),block_sizes=torch.tensor([2,32]),owners=torch.tensor([0,0]))
            valid = torch.ones(1,2,dtype=torch.bool)
            g = torch.tensor([[1.,-1.,1.,-1.,1.,-1.,1.,-1.]])
            blocks = torch.zeros(1,2,4)
            equal = dict(batch,tokens=values[1].repeat(2,1))
            torch.testing.assert_close(model(blocks,valid,g,token_batch=equal),torch.full((1,2),.5))
            novel = values.clone()
            novel[1] = 0.
            score = model(blocks,valid,g,token_batch=dict(batch,tokens=novel.reshape(-1,4)))
            if mode=="token_mean":
                torch.testing.assert_close(score,torch.full((1,2),.5))
            else:
                self.assertGreater(score[0,0].item(),.9)

    def test_weighted_only_auxiliary_has_no_direct_global_path(self):
        torch.manual_seed(42)
        g = torch.randn(2,16,requires_grad=True)
        z = torch.randn(2,8,requires_grad=True)
        index = CoarseClassifier(8,"joint",bypass=False).eval()
        other = CoarseClassifier(8,"joint").eval()
        _,prediction = index.auxiliary(g,z)
        _,changed = index.auxiliary(g+10,z)
        torch.testing.assert_close(prediction,changed,rtol=0.,atol=0.)
        self.assertIsNone(torch.autograd.grad(prediction.sum(),g,allow_unused=True)[0])
        self.assertIsNotNone(torch.autograd.grad(other.auxiliary(g,z)[1].sum(),g)[0])
        self.assertEqual(index.representation[1].out_features,other.representation[1].out_features)

    def test_frozen_and_joint_phases_keep_the_same_scoring_contract(self):
        torch.manual_seed(42)
        model = CoarseClassifier(8,"joint",bypass=False).eval()
        batch = dict(blocks=torch.randn(2,4,8),valid=torch.tensor([[1,1,1,1],[1,1,0,0]]).bool(),
                     labels=torch.tensor([0,1]),repair=torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.]]))
        model.indexer.requires_grad_(False)
        warm = model(batch,details=True,inspect=True)
        warm["loss"].backward()
        self.assertTrue(all(p.grad is None for p in model.indexer.parameters()))
        self.assertGreater(float(model.head[-1].weight.grad.norm()),0.)
        torch.testing.assert_close(warm["importance"],warm["head_importance"].mean(1))
        model.indexer.requires_grad_(True)
        model.repair_weight = .5
        trained = model(batch,details=True)
        torch.testing.assert_close(warm["importance"],trained["importance"],rtol=0.,atol=0.)
        for key in ("train_bce","train_repair_kl"):
            gradients = torch.autograd.grad(trained["loss_components"][key],model.indexer.parameters(),
                                            retain_graph=True)
            self.assertTrue(all(float(g.norm())>0. for g in gradients))

    def test_task_and_repair_gradients_with_padding_invariance(self):
        torch.manual_seed(42)
        model = CoarseClassifier(8,"joint",1.).eval()
        blocks = torch.randn(2,4,8)
        valid = torch.tensor([[True,True,True,True],[True,True,False,False]])
        batch = dict(blocks=blocks,valid=valid,labels=torch.tensor([1,0]),
                     repair=torch.tensor([[0.,.6,.4,0.],[0.,0.,0.,0.]]))
        output = model(batch,details=True)
        p = output["importance"]
        torch.testing.assert_close(p.sum(-1),torch.ones(2))
        torch.testing.assert_close(p[~valid],torch.zeros_like(p[~valid]))
        for key in ("train_bce","train_repair_kl"):
            grads = torch.autograd.grad(output["loss_components"][key],model.indexer.parameters(),retain_graph=True)
            self.assertTrue(all(torch.isfinite(g).all() and g.norm()>0 for g in grads))
        changed = blocks.clone()
        changed[~valid] = 42.
        other = model(dict(blocks=changed,valid=valid),details=True)
        for key in ("importance","coarse","weighted","global_state"):
            torch.testing.assert_close(other[key],output[key])


if __name__ == "__main__":
    unittest.main()
