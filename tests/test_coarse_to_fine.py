import unittest
from types import SimpleNamespace

import numpy as np
import torch

from coarse_encoding import CoarseClassifier, global_pool, repair_block_mass, retrieve, window_blocks
from context_selection import merge, occupied
from repair_oracle import EDIT_TYPES, GAP_TYPES


class BlockEncodingTest(unittest.TestCase):
    def test_window_cut_does_not_create_new_block(self):
        self.assertEqual(window_blocks(480,510,10), [(25,480,510)])
        self.assertEqual(window_blocks(510,521,10), [(25,510,512),(26,512,521)])

    def test_mean_and_max_exclude_padding(self):
        blocks = torch.tensor([[[1.,-2.],[3.,-4.],[900.,800.]]])
        valid = torch.tensor([[True,True,False]])
        torch.testing.assert_close(global_pool(blocks,valid),torch.tensor([[2.,-3.,3.,-2.]]))

    def test_repair_mass_deduplicates_edits_and_preserves_empty_side_signal(self):
        source = SimpleNamespace(ids=list(range(67)),keep=list(range(67)))
        edits = {kind:[] for kind in (*EDIT_TYPES,*GAP_TYPES)}
        edits.update(replace=[(30,35)],insert_gap=[(33,40)],delete_gap=[(63,67)])
        self.assertEqual(repair_block_mass(source,edits,(*EDIT_TYPES,*GAP_TYPES)).tolist(),[2.,9.,3.])
        self.assertEqual(repair_block_mass(source,edits,EDIT_TYPES).tolist(),[2.,3.,0.])
        self.assertEqual(repair_block_mass(source,edits,("delete_gap",)).tolist(),[0.,1.,3.])

    def test_retrieval_accounts_for_separators_and_short_inputs(self):
        rng = np.random.default_rng(42)
        for n in (13, 510, 511, 543, 590, 1024, 278200):
            scores = rng.random((n+31)//32)
            spans = retrieve(scores,n)
            self.assertEqual(spans,merge(spans))
            self.assertEqual(occupied(spans),min(n,510))
            self.assertTrue(all(0<=a<b<=n for a,b in spans))
            self.assertEqual(spans,retrieve(scores,n))

    def test_remaining_budget_joins_an_adjacent_block(self):
        scores = np.array([3.,0.,2.,1.])
        spans = retrieve(scores,111,budget=33)
        self.assertEqual(spans,[(0,33)])


class IndexerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.batch = dict(blocks=torch.randn(2,4,8), valid=torch.tensor([[True,True,True,True],[True,True,False,False]]),
                          labels=torch.tensor([1,0]),repair=torch.tensor([[0.,.6,.4,0.],[0.,0.,0.,0.]]))

    def test_task_supervision_and_masks(self):
        model = CoarseClassifier(8,"task")
        report = model(self.batch,details=True)
        p = report["importance"]
        torch.testing.assert_close(p.sum(-1),torch.ones(2))
        self.assertTrue(torch.equal(p[1,2:],torch.zeros(2)))
        report["loss"].backward()
        self.assertGreater(model.indexer.query.weight.grad.norm().item(),0.)

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
        self.assertGreater(model.head[1].weight.grad.norm().item(),0.)
