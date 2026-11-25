import unittest
from types import SimpleNamespace

import numpy as np
import torch

from coarse_encoding import CoarseClassifier, block_mass, repair_block_mass
from coarse_retrieval import block_ends, lengths, part_overlap, retrieve, uncovered
from context_selection import contains, merge, occupied, source_parser
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
