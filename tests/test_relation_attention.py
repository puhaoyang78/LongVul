import unittest
from types import SimpleNamespace

import numpy as np
import torch
from transformers import RobertaConfig, RobertaModel

from coarse_encoding import FusionClassifier
from relation_attention import AdditiveAttentionRoberta, additive_relation_mask, relation_bits, visible_relations


class RelationAttentionTest(unittest.TestCase):
    def test_fine_only_preserves_initialization_rng_and_excludes_global_features(self):
        config = RobertaConfig(vocab_size=16,hidden_size=12,intermediate_size=16,num_attention_heads=3,
                               num_hidden_layers=1,max_position_embeddings=20)
        config._attn_implementation = "eager"
        fine = torch.nn.Module()
        fine.config = config
        fine.encoder = RobertaModel(config,add_pooling_layer=False)
        fine.classifier = torch.nn.Linear(config.hidden_size,1)
        torch.manual_seed(42)
        fused = FusionClassifier(fine,"global").eval()
        rng = torch.get_rng_state()
        torch.manual_seed(42)
        only = FusionClassifier(fine,"none").eval()
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertIsNone(only.fusion)
        ids = torch.tensor([[[0,3,4,2,1]]])
        batch = dict(fixed=ids,lengths=torch.tensor([4]),coarse=torch.zeros(1,24))
        changed = dict(batch,coarse=torch.ones(1,24))
        with torch.inference_mode():
            torch.testing.assert_close(only(batch)["logits"],only(changed)["logits"],atol=0.,rtol=0.)
            fused.fusion.weight.fill_(.1)
            self.assertFalse(torch.equal(fused(batch)["logits"],fused(changed)["logits"]))

    def test_mapping_requires_both_complete_source_intervals(self):
        source = SimpleNamespace(keep=[1,2,4,5,6,9,10,11,13,14])
        positions = np.array([-1,1,2,4,-1,9,10,11,-1,-1])
        links = [("condition",(0,3),(5,8)),("condition",(0,3),(5,8)),("condition",(5,8),(0,3)),
                 ("declaration",(1,3),(5,7)),("assignment",(2,4),(5,7)),
                 ("completion",(0,3),(0,3)),("neighbor",(1,3),(5,8))]
        mapped = visible_relations(source,positions,links)
        self.assertEqual(mapped,[(0,1,4,5,8),(1,2,4,5,7)])
        bits = relation_bits([dict(fixed=torch.zeros(3,10),relations=mapped+mapped)])
        torch.testing.assert_close(bits,bits.transpose(1,2),atol=0.,rtol=0.)
        self.assertEqual(int(bits[0,1,7]),1)
        self.assertEqual(int(bits[0,2,6]),3)
        missing = torch.from_numpy(positions<0)
        self.assertFalse(bits[0,missing].any())
        self.assertFalse(bits[0,:,missing].any())
        w = torch.tensor([.3,-.2,.1],requires_grad=True)
        mask = additive_relation_mask(bits,torch.tensor([[1,1,1,1,1,1,1,1,1,0]]).bool(),w)
        self.assertAlmostEqual(float(mask[0,0,2,6].detach()),.1,places=6)
        self.assertEqual(mask[0,0,2,9].item(),torch.finfo(torch.float32).min)
        mask[0,0,2,6].backward()
        self.assertEqual(w.grad.tolist(),[1.,1.,0.])

    def test_eager_zero_bias_and_absent_relations_match_native_roberta(self):
        config = RobertaConfig(vocab_size=16,hidden_size=12,intermediate_size=16,num_attention_heads=3,
                               num_hidden_layers=2,max_position_embeddings=20)
        config._attn_implementation = "eager"
        baseline = RobertaModel(config,add_pooling_layer=False).eval()
        model = AdditiveAttentionRoberta(config,add_pooling_layer=False).eval()
        model.load_state_dict(baseline.state_dict(),strict=True)
        ids = torch.tensor([[0,3,4,7,2,1],[0,4,7,2,1,1]])
        valid = ids.ne(1)
        bits = torch.zeros(2,6,6,dtype=torch.uint8)
        bits[0,1,3] = bits[0,3,1] = 1
        with torch.inference_mode():
            native = baseline(ids,attention_mask=valid.long(),output_attentions=True)
            zero = model(ids,attention_mask=additive_relation_mask(bits,valid,torch.zeros(3)),
                         output_attentions=True)
            changed = model(ids,attention_mask=additive_relation_mask(bits,valid,torch.tensor([1.,2.,3.])),
                            output_attentions=True)
        torch.testing.assert_close(zero.last_hidden_state,native.last_hidden_state,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(changed.last_hidden_state[1],native.last_hidden_state[1],atol=1e-6,rtol=1e-6)
        self.assertTrue((changed.attentions[0][0,:,1,3]>native.attentions[0][0,:,1,3]).all())
        self.assertFalse(changed.attentions[0][...,5].any())
