import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from context_selection import occupied
from diagnostics.next_stage import auxiliary_weights, outside_layout, threshold_row, valid_threshold
from diagnostics.representations import AttentionProbe, compact_positions, shuffled_blocks
from coarse_encoding import CoarseClassifier
from paired import LOSVERAdamW
from transformers import RobertaConfig
from window_inputs import seed_path


class DiagnosticTests(unittest.TestCase):
    def test_baseline_root_preserves_existing_seed42(self):
        path = Path("results/sea")
        self.assertEqual(seed_path(path,42),path)
        self.assertEqual(seed_path(path,1234),path/"seed1234")
        self.assertEqual(seed_path(path,123456),path/"seed123456")

    def test_outside_layout_exactly_fills_positions_without_overlap(self):
        selected = outside_layout([(5,7)],11,6)
        self.assertEqual(occupied(selected),6)
        self.assertTrue(all(hi<=5 or lo>=7 for lo,hi in selected))
        self.assertIsNone(outside_layout([(3,6)],8,8))
        self.assertIsNone(outside_layout([(1,2),(3,4)],5,4))

    def test_index_intervention_uses_proportional_coverage(self):
        layout = SimpleNamespace(centers=[(0,4)],spans=[(0,5),(8,10)])
        uniform = SimpleNamespace(spans=[(3,7)])
        p = np.array([.2,.3,.5])
        weights,geometry = auxiliary_weights(layout,uniform,[4,8,12],p)
        np.testing.assert_allclose(weights["visible"],[.2,.075,.25])
        np.testing.assert_allclose(weights["visible"]+weights["discarded"],p)
        np.testing.assert_allclose(weights["uniform_layout"],[.05,.225,0])
        np.testing.assert_allclose(geometry,[1/3,.2,.525])

    def test_valid_threshold_is_optimal_with_tied_probabilities(self):
        y = [0,0,1,1,1,0]
        p = [.2,.2,.4,.4,.75,.8]
        found = valid_threshold(y,p)
        actual = threshold_row(y,p,found)["mcc"]
        candidates = sorted({0.,.5,1.,*p})
        self.assertAlmostEqual(actual,max(threshold_row(y,p,t)["mcc"] for t in candidates))
        self.assertEqual(valid_threshold([0,1],[.42,.42]),.5)

    def test_losver_first_update_matches_epsilon_before_bias_correction(self):
        param = torch.nn.Parameter(torch.tensor([1.,-2.],dtype=torch.float64))
        grad = torch.tensor([1e-6,-3e-6],dtype=torch.float64)
        opt = LOSVERAdamW([param],lr=2e-5,eps=1e-8)
        param.grad = grad
        expected = param.detach()-2e-5*math.sqrt(.001)/.1*(.1*grad)/((.001*grad.square()).sqrt()+1e-8)
        opt.step()
        torch.testing.assert_close(param,expected,rtol=0.,atol=1e-13)
        self.assertEqual(opt.state[param]["step"],1)

    def test_reordered_blocks_keep_per_identity_index_scores(self):
        torch.manual_seed(42)
        model = CoarseClassifier(8,"joint").eval()
        batch = dict(blocks=torch.randn(2,7,8),valid=torch.arange(7)[None,:]<torch.tensor([[7],[4]]))
        swapped,inverse = shuffled_blocks(batch,torch.Generator().manual_seed(42))
        original,changed = (model(x,details=True) for x in (batch,swapped))
        torch.testing.assert_close(original["importance"],changed["importance"].gather(1,inverse))
        torch.testing.assert_close(original["global_state"],changed["global_state"])
        torch.testing.assert_close(original["probability"],changed["probability"])

    def test_selected_tokens_check_identity_across_removed_indentation(self):
        source = SimpleNamespace(keep=[0,3,4,8],ids=np.array([7,8,9,10]))
        fixed = torch.tensor([[0,7,9,2,10,2,1],[-1,0,4,-1,8,-1,-1],[0,1,1,0,2,0,0]])
        original,compact = compact_positions(source,fixed)
        np.testing.assert_array_equal(original,[0,4,8])
        np.testing.assert_array_equal(compact,[0,2,3])
        fixed[0,2] = 11
        with self.assertRaises(ValueError):
            compact_positions(source,fixed)

    def test_light_heads_share_initialization_and_ignore_padding(self):
        config = RobertaConfig(hidden_size=8,num_labels=1)
        torch.manual_seed(42)
        base = AttentionProbe(config,global_fusion=True,position=False).eval()
        torch.manual_seed(42)
        positioned = AttentionProbe(config,global_fusion=True,position=True).eval()
        for a,b in zip(base.parameters(),positioned.parameters(),strict=True):
            torch.testing.assert_close(a,b,rtol=0.,atol=0.)
        values = torch.randn(1,3,8)
        batch = dict(features=values,mass=torch.tensor([[1.,.5,0.]]),
                     positions=torch.tensor([[10.,55.,0.]]),coarse=torch.randn(1,16))
        p = base(batch)["probability"]
        values[:,-1] = 100
        torch.testing.assert_close(p,base(batch)["probability"],rtol=0.,atol=0.)


if __name__ == "__main__":
    unittest.main()
