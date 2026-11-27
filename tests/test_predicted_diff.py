from types import SimpleNamespace
import unittest

import numpy as np
import torch

from compaction import BUDGET
from context_selection import merge
from predicted_diff import PredictedData, calibrate, compact_source, prepare_example, region_targets
from repair_oracle import EDIT_TYPES, GAP_TYPES, prepare_side


TOKENIZER = SimpleNamespace(cls_token_id=0, sep_token_id=2, pad_token_id=1)


def character_source(code):
    return compact_source(code, list(range(3, len(code)+3)), [(i, i+1) for i in range(len(code))])


class PredictedDiffTest(unittest.TestCase):
    def setUp(self):
        self.source = character_source("int f() {\n" + "    x += y + z;\n"*70 + "}\n")
        self.scores = np.arange(len(self.source.regions), dtype=float)

    def test_same_regions_match_oracle_layout(self):
        chosen = prepare_example(self.source, self.scores, TOKENIZER, 4, BUDGET//2)
        edits = dict.fromkeys((*EDIT_TYPES, *GAP_TYPES), ())
        edits["replace"] = chosen["predicted_anchor_runs"]
        original = prepare_side((self.source.code, self.source.raw_ids, self.source.offsets), TOKENIZER,
                                edits, with_context=False, parser=None)
        self.assertEqual(original["oracle_anchor_runs"], chosen["predicted_anchor_runs"])
        self.assertTrue(torch.equal(original["fixed"], chosen["fixed"]))

    def test_cap_preserves_uniform_space_and_predicted_regions(self):
        example = prepare_example(self.source, self.scores, TOKENIZER, 25, 127)
        self.assertLessEqual(example["predicted_anchor_positions"], 127)
        self.assertEqual(example["length"], 512)
        retained = set(example["fixed"][1, :512].tolist())
        self.assertTrue(all(i in retained for a, b in example["predicted_anchor_runs"] for i in range(a, b)))
        with self.assertRaisesRegex(ValueError, "region policy"):
            prepare_example(self.source, self.scores, TOKENIZER, 25, BUDGET)

    def test_source_only_input_and_short_passthrough(self):
        source = character_source("int f() {\n    return 5;\n}\n")
        rows = [dict(idx=i, sample_key=str(i), target=i) for i in range(2)]
        scores = np.ones(len(source.regions))
        data = PredictedData(rows, [source, source], [scores, 1-scores], TOKENIZER,
                             dict(top_k=8, anchor_budget=255))
        self.assertTrue(torch.equal(data[0]["fixed"], data[1]["fixed"]))
        self.assertEqual(data[0]["fixed"][0, 1:data[0]["length"]-1].tolist(), source.raw_ids.tolist())
        self.assertEqual(data[0]["predicted_anchor_positions"], 0)

    def test_validation_tie_prefers_less_predictive_budget(self):
        edits = dict.fromkeys((*EDIT_TYPES, *GAP_TYPES), ())
        edits["replace"] = [(len(self.source.raw_ids)-2, len(self.source.raw_ids)-1)]
        decision = calibrate([self.source], [self.scores], [edits], TOKENIZER)
        self.assertEqual(decision["split"], "valid")
        self.assertEqual((decision["top_k"], decision["anchor_budget"]), (4, 127))
        self.assertTrue(all(row["retained_bpe"] == 1 for row in decision["trials"]))

    def test_weak_labels_project_to_historical_compacted_lines(self):
        edits = dict.fromkeys((*EDIT_TYPES, *GAP_TYPES), ())
        edits["replace"] = [(self.source.code.index("x"), self.source.code.index("x")+1)]
        edit = merge(edits["replace"])
        labels = region_targets([self.source], [edits])[0]
        expected = [any(a <= p < b for a, b in edit for p in self.source.keep[r.start:r.end])
                    for r in self.source.regions]
        self.assertEqual(labels.tolist(), expected)
        self.assertEqual(int(labels.sum()), 1)
