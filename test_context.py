"""Checks for the context slot, source partition, and BCE gradient path."""
import unittest

import numpy as np
import torch
from transformers import RobertaConfig, RobertaModel
from context import BINS, ContextClassifier, bags, context_data, source_view
from dataset import tokenizer
from rgcl.model import Detector


class ContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = tokenizer()
        cls.code = "int f(int a){\n" + "    a += 2;\n" * 140 + "    return a;\n}"
        cls.rows = [dict(idx=i, target=i % 2, func=c) for i,c in enumerate(("int f(){return 3;}", cls.code))]
        cls.noise = np.full(len(cls.tok), 1/len(cls.tok))

    def test_frequency_weighted_bags(self):
        ids = [4, 4, 9, 4, 6] * 20
        missing = np.asarray([0, 1, 2, 4, 39, 40, 51, 99])
        tokens, weights, sizes, mass = bags(ids, missing, 10, None, [])
        offsets = torch.cat((torch.zeros(1, dtype=torch.long), sizes.cumsum(0)))
        table = torch.arange(30, dtype=torch.float32).reshape(10, 3)
        pooled = torch.nn.functional.embedding_bag(tokens, table, offsets, mode="sum",
                                                    per_sample_weights=weights, include_last_offset=True)
        for region in range(BINS):
            positions = missing[missing * BINS // len(ids) == region]
            self.assertEqual(len(positions), mass[region].item())
            reference = table[[ids[p] for p in positions]].mean(0) if len(positions) else torch.zeros(3)
            torch.testing.assert_close(pooled[region], reference)

    def test_partition_and_controls(self):
        enc = self.tok(self.code, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        ids = enc["input_ids"]
        positions, missing = source_view(self.code, ids, enc["offset_mapping"])
        visible = [i for i in positions if i is not None]
        self.assertEqual(sorted([*visible, *missing]), list(range(len(ids))))
        views = [context_data(self.rows, m, self.tok, False, p) for m,p in (
            ("context_real", None), ("context_random", self.noise), ("context_empty", None))]
        for data,_ in views:
            self.assertTrue(np.array_equal(data[0], views[0][0][0]))
            self.assertLessEqual(data[1].max(), 512)
        real, random = [next(iter(v[1])) for v in views[:2]]
        self.assertTrue(torch.equal(real[-1], random[-1]))
        self.assertEqual(real[-1][1].sum().item(),len(missing))
        self.assertFalse(torch.equal(real[5], random[5]))
        self.assertEqual(real[4].tolist(), [0, views[0][0][1][1]-2])

    def test_forward_and_gradients(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        cfg = RobertaConfig(vocab_size=len(self.tok), hidden_size=32, num_hidden_layers=1,
                            num_attention_heads=4, intermediate_size=64, max_position_embeddings=514)
        cfg._attn_implementation = "eager"
        base = Detector(RobertaModel(cfg, add_pooling_layer=False), cfg, "plm_classifier", 1.).eval()
        rng = torch.get_rng_state()
        model = ContextClassifier(base).eval()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        data, loader = context_data(self.rows, "context_real", self.tok, False)
        ids, mask, y, *extra = next(iter(loader))
        calls = []
        hook = model.encoder.register_forward_hook(lambda *args: calls.append(1))
        out = model(ids, mask, *extra, labels=y)
        hook.remove()
        self.assertEqual(calls,[1])
        reference = base(ids, mask, None, None)["logits"]
        torch.testing.assert_close(reference[:1], out["logits"][:1])
        out["loss"].backward()
        self.assertGreater(model.project.weight.grad.norm().item(),0)
        # Add a unique token in an omitted bag to test its embedding gradient
        # even when a function repeats the same few lexical items throughout.
        changed = extra[2].clone()
        changed[0] = 50000
        model.zero_grad()
        model(ids, mask, extra[0], extra[1], changed, *extra[3:], labels=y)["loss"].backward()
        self.assertGreater(model.encoder.get_input_embeddings().weight.grad[50000].norm().item(),0)
        _, only_short = context_data(self.rows[:1], "context_real", self.tok, False)
        x, mask, _, *rest = next(iter(only_short))
        torch.testing.assert_close(model(x, mask, *rest)["logits"], base(x, mask, None, None)["logits"])


if __name__ == "__main__":
    unittest.main()
