import unittest

from compaction import BUDGET
from context_selection import intersects
from locator_quality import FIM_SIDE, fim_sequence, whole_windows
from oracle_noise import RATES, crossing_report, matched_slots, noise_data, oracle_regions
from repair_oracle import EDIT_TYPES, GAP_TYPES
from test_predicted_diff import TOKENIZER, character_source


class OracleNoiseTest(unittest.TestCase):
    def test_reserved_alternatives_are_deterministic_disjoint_and_width_matched(self):
        anchors = [(13, 16), (21, 30), (55, 56)]
        edited = [(11, 18), (20, 41), (53, 57)]
        slots = matched_slots(110, anchors, edited, "42/valid/23")
        self.assertEqual(slots, matched_slots(110, anchors, edited, "42/valid/23"))
        used = []
        for (a,b), (lo,hi), _ in slots:
            self.assertEqual(b-a, hi-lo)
            self.assertFalse(intersects((lo, hi), edited))
            self.assertFalse(intersects((lo-1, hi+1), anchors+used))
            used.append((lo, hi))

    def test_no_fabricated_alternative_when_edit_fills_function(self):
        slots = matched_slots(28, [(7, 11)], [(0, 28)], "42/train/3")
        self.assertIsNone(slots[0][1])

    def test_noise_uses_same_packed_cost_for_all_five_rates(self):
        source = character_source("int f() {\n" + "    x += y + z;\n"*80 + "}\n")
        rows = [dict(idx=1, sample_key="valid:1", target=1)]
        edits = dict.fromkeys((*EDIT_TYPES, *GAP_TYPES), ())
        edits["replace"] = [(100, 115), (350, 380), (800, 815)]
        policy = dict(top_k=4, anchor_budget=127)
        anchors, edited = oracle_regions(source, edits, policy)
        slots = matched_slots(len(source.ids), anchors, edited, "42/valid/1")
        order = sorted((priority, 0, j) for j, (_,_,priority) in enumerate(slots))
        source_plan = rows, [source], [edits], [slots], order, 0.
        attempted = 0
        for rate in RATES:
            data, stats = noise_data(source_plan, TOKENIZER, rate)
            self.assertEqual(data[0]["length"], 512)
            self.assertLessEqual(data[0]["anchor_positions"], policy["anchor_budget"])
            if rate == 0:
                cost = data[0]["anchor_positions"]
            self.assertEqual(data[0]["anchor_positions"], cost)
            self.assertGreaterEqual(stats["attempted_regions"], attempted)
            attempted = stats["attempted_regions"]
        self.assertEqual(stats["replaced_regions"], len(anchors))

    def test_curve_does_not_extrapolate_outside_observed_points(self):
        counts = crossing_report([(.6, .14), (.5, .09), (.4, .11)])
        self.assertEqual(counts["0.1"]["minimum_observed_retention"], .4)
        self.assertEqual(len(counts["0.1"]["interpolated_crossings"]), 2)
        self.assertIsNone(counts["0.15"]["minimum_observed_retention"])
        self.assertEqual(counts["0.15"]["interpolated_crossings"], [])


class LocatorQualityTest(unittest.TestCase):
    def test_context_windows_cover_whole_historical_units(self):
        source = character_source("int f() {\n" + "    x += y + z;\n"*200 + "}\n")
        windows = list(whole_windows(source))
        self.assertGreater(len(windows), 1)
        self.assertEqual([r for _,_,units in windows for r in units], source.regions)
        self.assertTrue(all(0 < b-a <= BUDGET for a,b,_ in windows))
        self.assertEqual(sum(b-a for a,b,_ in windows), len(source.ids))

    def test_fim_masks_middle_and_bounds_both_sides(self):
        seq, begin = fim_sequence(list(range(240)), 117, 125, (1001,1002,1003))
        self.assertEqual(seq[:begin], [1001, *range(117-FIM_SIDE,117), 1002, *range(125,125+FIM_SIDE), 1003])
        self.assertEqual(seq[begin:], list(range(117,125)))
        with self.assertRaisesRegex(ValueError, "no valid FIM-token"):
            fim_sequence([], 0, 0, (1,2,3))
