import unittest
from types import SimpleNamespace

import numpy as np

from compression_study import ast_form, lexical_form, mapped_uniform
from context_selection import contains, source_parser
from repair_oracle import EDIT_TYPES, GAP_TYPES
from source_regions import Region
from syntax_region_study import Candidate, Plan, choose, context_group_windows, target_scores
from test_predicted_diff import TOKENIZER, character_source


class CompressionStudyTest(unittest.TestCase):
    def setUp(self):
        self.parser = source_parser()

    def test_blank_lines_only_outside_multiline_literals(self):
        text = 'int f() {\n  char const *x = R"(left\n\n  right)";\n\n  return 0;\n}\n'
        form, _ = lexical_form(text, self.parser, collect_nulls=False)
        self.assertIn('R"(left\n\n  right)"', form[0])
        self.assertEqual(form[-1]["removed_blank_lines"], 1)

    def test_directives_cr_and_line_queries_are_protected(self):
        text = 'int f() {\r#define X  1\r\r return X;\r}\r'
        form, _ = lexical_form(text, self.parser, collect_nulls=False)
        self.assertIn("#define X  1\r\r", form[0])
        observed = 'int f() {\n\n  return __builtin_COLUMN();\n}\n'
        form, _ = lexical_form(observed, self.parser, collect_nulls=True)
        self.assertEqual(form[0], observed)
        self.assertEqual(form[-1]["compression_status"], "line_sensitive")

    def test_ast_only_drops_plain_compound_null_statements(self):
        text = "int f(int n) { ; while(n--) ; n: ; f(({ ; 2; })); { ; } return n; }"
        base, nulls = lexical_form(text, self.parser, collect_nulls=True)
        form = ast_form(text, base, nulls, self.parser)
        self.assertEqual(form[-1]["removed_null_offsets"], [text.index(";"), text.index("{ ; }")+2])
        self.assertIn("while(n--) ; n: ;", form[0])
        self.assertIn("f(({ ; 2; }));", form[0])

    def test_removed_original_tokens_do_not_count_as_visible(self):
        source = character_source("int f() {\n  ; /* note */\n\n return 1;\n}\n")
        base, nulls = lexical_form(source.code, self.parser, collect_nulls=True)
        final = ast_form(source.code, base, nulls, self.parser)
        ids = list(range(20, 20+len(final[0])))
        offsets = np.column_stack((np.arange(len(ids)), np.arange(1, len(ids)+1)))
        example = mapped_uniform(source, ids, offsets, final, TOKENIZER)
        self.assertEqual(example["raw"], len(source.code))
        self.assertLess(example["compacted_bpe"], example["raw"])
        comment = source.code.index("note")
        for deleted in (comment, *final[-1]["removed_null_offsets"]):
            self.assertFalse(contains((deleted, deleted+1), example["original_runs"]))


class SyntaxStudyTest(unittest.TestCase):
    def test_contextual_pools_cover_disjoint_spans_across_encoder_windows(self):
        source = SimpleNamespace(ids=np.arange(540), regions=[Region(0,500,0,500,"line"),Region(500,540,500,540,"line")])
        candidate = Candidate(((20,22),(497,503)), (), 0)
        plan = Plan([], [], [candidate], (), 1, False)
        windows = list(context_group_windows([source], [plan]))
        self.assertEqual([(lo,hi) for _,lo,hi,_ in windows], [(0,500),(500,540)])
        tokens = [[] for _ in range(4)]
        for _,lo,hi,pools in windows:
            for part,a,b in pools:
                self.assertTrue(lo <= a < b <= hi)
                tokens[part].extend(range(a,b))
        self.assertEqual(tokens, [[20,21],[497,498],[499,500],[501,502]])

    def test_density_excludes_base_and_deduplicates_positive_sources(self):
        source = SimpleNamespace(keep=list(range(10)))
        candidate = Candidate(((1,6),(8,9)), ((3,6),(8,9)), 5)
        plan = Plan([(0,3)], [], [candidate], (), 1, False)
        annotations = dict.fromkeys((*EDIT_TYPES, *GAP_TYPES), ())
        annotations.update(replace=[(2,5)], insert_gap=[(4,6)])
        self.assertAlmostEqual(target_scores(source, plan, annotations)[0], 3/5)

    def test_selection_preserves_base_and_deduplicates_overlap(self):
        c1 = Candidate(((4,8),), ((5,8),), 3)
        c2 = Candidate(((6,10),), ((6,10),), 5)
        plan = Plan([(0,5)], [], [c1,c2], (), 2, False)
        kept,count = choose(plan, np.array([.4,.3]), 2)
        self.assertEqual((kept,count), ([(0,10)],2))
        self.assertEqual(choose(plan, np.zeros(2), 2, positive_only=True), (plan.base,0))
