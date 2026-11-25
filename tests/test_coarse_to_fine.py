import unittest
from types import SimpleNamespace

import numpy as np
import torch

from coarse_encoding import CoarseClassifier, block_mass, repair_block_mass
from coarse_retrieval import Candidates, block_ends, lengths, part_overlap, uncovered
from coarse_to_fine import changed_blocks, ranking_loss
from context_selection import contains, merge, occupied
from repair_oracle import EDIT_TYPES, GAP_TYPES


class BlockEncodingTest(unittest.TestCase):
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
    def test_completion_and_link_precedence_with_exact_marginal_cost(self):
        n = 1600
        ends = block_ends(n,[],[],64,False)
        unit = (113,137)
        links = {("condition",unit,(551,578)),("declaration",unit,(180,194)),
                 ("assignment",unit,(800,1300)),("assignment",unit,(270,281))}
        plan = Candidates(ends,[unit],links,contextual=True)
        self.assertTrue(contains(unit,plan.bundle[1]))
        self.assertTrue(contains((551,578),plan.bundle[1]))
        self.assertTrue(contains((180,194),plan.bundle[1]))
        self.assertTrue(contains((270,281),plan.bundle[1]))
        self.assertEqual(plan.omitted["assignment"],2)
        for i in (1,2):
            self.assertAlmostEqual(sum(w for a,_,w in plan.edges if a==i),1.0)
        rng = np.random.default_rng(42)
        for _ in range(50):
            starts = rng.integers(0,n-60,10)
            selected = merge([(a,a+int(width)) for a,width in zip(starts,rng.integers(1,60,10))])
            actual = [merge([*selected,*bundle]) for bundle in plan.bundle]
            for separators,count in ((False,lengths),(True,occupied)):
                expected = [count(spans)-count(selected) for spans in actual]
                np.testing.assert_array_equal(plan.marginal(selected,separators=separators),expected)

    def test_budget_and_bounded_matching_for_realistic_lengths(self):
        rng = np.random.default_rng(42)
        for n in (13,510,511,543,590,1024,278200):
            source = SimpleNamespace(ids=range(n),boundaries=[*range(0,n,26),n])
            ends = block_ends(n,[],[],64,False)
            plan = Candidates(ends,[],[],contextual=False)
            scores = rng.random(len(ends))
            selected = plan.retrieve(scores,source)
            self.assertEqual(list(selected.spans),merge(selected.spans))
            self.assertEqual(occupied(selected.spans),min(n,510))
            self.assertTrue(all(0<=a<b<=n for a,b in selected.spans))
            self.assertEqual(selected,plan.retrieve(scores,source))
            if n<=510:
                self.assertEqual(selected.spans,((0,n),))
            replacements = plan.alternatives(selected,scores,source,rng)
            self.assertLessEqual(len(replacements),2)
            if n==278200:
                self.assertTrue(replacements)
            for other,removed,added in replacements:
                self.assertEqual(occupied(selected.spans),occupied(other.spans))
                self.assertEqual(lengths(selected.spans),lengths(other.spans))
                self.assertEqual(removed,uncovered(selected.spans,other.spans))
                self.assertEqual(added,uncovered(other.spans,selected.spans))
                self.assertEqual(len(set(selected.centers)-set(other.centers)),1)
                self.assertEqual(len(set(other.centers)-set(selected.centers)),1)
                for center in set(selected.centers)&set(other.centers):
                    self.assertTrue(all(contains(span,other.spans) for span in plan.bundle[center]))


class IndexerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch = dict(blocks=torch.randn(2,4,8),positions=torch.arange(4)[None,:].expand(2,-1)*64.,
                          valid=torch.tensor([[True,True,True,True],[True,True,False,False]]),
                          labels=torch.tensor([1,0]),repair=torch.tensor([[0.,.6,.4,0.],[0.,0.,0.,0.]]))

    def test_task_gradient_and_source_positions_without_padding_signal(self):
        model = CoarseClassifier(8,"task").eval()
        report = model(self.batch,details=True)
        p = report["importance"]
        torch.testing.assert_close(p.sum(-1),torch.ones(2))
        torch.testing.assert_close(p[1,2:],torch.zeros(2),atol=0,rtol=0)
        altered = self.batch["blocks"].clone()
        altered[1,2:] *= 10000
        padding = model(dict(self.batch,blocks=altered),details=True)
        torch.testing.assert_close(p,padding["importance"])
        reordered = model(dict(self.batch,positions=self.batch["positions"].flip(1)),details=True)
        self.assertFalse(torch.allclose(p,reordered["importance"]))
        report["loss"].backward()
        self.assertGreater(model.indexer.query.weight.grad.norm().item(),0.)
        self.assertGreater(model.encoder.broadcast.in_proj_weight.grad.norm().item(),0.)

    def test_auxiliary_logits_require_indexed_local_evidence(self):
        model = CoarseClassifier(8,"task").eval()
        torch.nn.init.zeros_(model.indexer.query.weight)
        p = model(self.batch,details=True)
        # With uniform fixed scores, changing only the global reader cannot affect the auxiliary logits.
        with torch.no_grad():
            model.encoder.latents.add_(torch.randn_like(model.encoder.latents))
        q = model(self.batch,details=True)
        self.assertFalse(torch.allclose(p["g"],q["g"]))
        torch.testing.assert_close(p["logits"],q["logits"])

    def test_repair_only_detaches_task_loss_and_ignores_empty_target_in_kl(self):
        model = CoarseClassifier(8,"repair",1.)
        report = model(self.batch,details=True)
        bce = report["loss_components"]["train_bce"]
        kl = report["loss_components"]["train_repair_kl"]
        q = self.batch["repair"][0,1:3]
        p = report["importance"][0,1:3]
        torch.testing.assert_close(kl,(q*(q.log()-p.log())).sum())
        params = tuple(model.indexer.parameters())
        self.assertEqual(torch.autograd.grad(bce,params,allow_unused=True,retain_graph=True),(None,None))
        report["loss"].backward()
        self.assertGreater(model.indexer.key.weight.grad.norm().item(),0.)
        self.assertGreater(model.head[-1].weight.grad.norm().item(),0.)

    def test_ranking_updates_all_changed_blocks_using_actual_loss_direction(self):
        ids,sign = changed_blocks([30,60,90,120],[(20,40)],[(70,95)])
        self.assertEqual(ids,[0,1,2,3])
        torch.testing.assert_close(torch.tensor(sign),torch.tensor([-.5,-.5,.8,.2]))
        p = torch.tensor([[.2,.3,.4,.1]],requires_grad=True)
        loss = ranking_loss(p,[9],{9:[(.4,ids,sign)]})
        expected = .4*torch.nn.functional.softplus((p.log()*torch.tensor(sign)).sum())
        torch.testing.assert_close(loss,expected)
        loss.backward()
        self.assertTrue((p.grad[0,:2]<0).all())
        self.assertTrue((p.grad[0,2:]>0).all())
