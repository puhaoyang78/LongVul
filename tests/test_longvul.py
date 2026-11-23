from types import SimpleNamespace
import unittest

import numpy as np

from budget_selection import available_graph, choose, completion_units, optimize
from compaction import pack_fixed
from context_selection import contains, intersects, occupied, source_parser, uniform_fill
from dependence import physical_lines, source_position
from repair_oracle import line_edits, map_edits
from source_regions import Region, analyze, regions_for, simplify


class ContinuousFillTest(unittest.TestCase):
    def test_completes_selected_ast_unit_before_seeding(self):
        units = [(0, 7), (7, 15), (15, 24)]

        def unit_at(position):
            return next((a, b) for a, b in units if a <= position < b)

        self.assertEqual(uniform_fill([(9, 11)], [0, 6, 12, 18, 24], [(0, 12), (12, 24)],
                                      budget=8, unit_at=unit_at), [(7, 15)])

    def test_prioritizes_multiline_ast_and_falls_back_to_original_line(self):
        code = "int f(int a, int b) {\r\n  int c = a\r\n           ? a : b;\r\n  return c;\r\n}\r\n"
        _, _, statements, parse_error = analyze(code, source_parser())
        self.assertFalse(parse_error)
        ids = np.arange(len(code))
        offsets = np.column_stack((ids, ids+1))
        lines = physical_lines(code)
        source = SimpleNamespace(code=code, ids=ids, source_offsets=offsets, parse_error=parse_error,
                                 regions=regions_for(statements, offsets, offsets, lines, 1022))
        cuts, unit_at = completion_units(source)
        self.assertEqual(cuts, lines)
        self.assertEqual(unit_at(code.index("?")), (code.index("int c"), code.index("b;")+2))
        self.assertEqual(unit_at(code.index("}")), (code.index("}"), len(code)))

    def test_budget_counts_merge_and_separator_at_full_occupancy(self):
        spans = [(2, 4), (5, 8)]
        self.assertEqual(occupied(spans), 6)
        filled = uniform_fill(spans, [0, 2, 4, 5, 8, 14], [(0, 14)], budget=6)
        self.assertEqual(filled, [(2, 8)])
        self.assertEqual(occupied(filled), 6)

    def test_residue_remains_adjacent(self):
        filled = uniform_fill([(40, 100)], [0, 40, 100, 200], [(0, 200)], budget=63)
        self.assertEqual(filled, [(40, 103)])

    def test_covers_unrepresented_source_bands(self):
        bands = [(0, 40), (40, 80), (80, 120), (120, 160)]
        filled = uniform_fill([(5, 10)], list(range(0, 161, 5)), bands, budget=42)
        self.assertTrue(all(intersects(band, filled) for band in bands))
        self.assertLessEqual(len(filled), len(bands))
        self.assertEqual(occupied(filled), 42)

    def test_preserves_prefill_and_source_order_under_1024_positions(self):
        size = 4096
        cuts = list(range(0, size+1, 16))
        bands = [(0, 1024), (1024, 2048), (2048, 3072), (3072, 4096)]
        retained = [(250, 255), (720, 745), (1000, 1050), (2200, 2320), (3600, 3655)]
        filled = uniform_fill(retained, cuts, bands, budget=1022, fragment_limit=12)
        self.assertTrue(all(contains(span, filled) for span in retained))
        self.assertLessEqual(len(filled), len(retained))
        ids = list(range(3, size+3))
        tokenizer = SimpleNamespace(cls_token_id=0, sep_token_id=2, pad_token_id=1)
        packed, length = pack_fixed(ids, range(size), filled, tokenizer, max_length=1024)
        self.assertEqual(length, 1024)
        submitted_ids, positions = packed[:2, :length].tolist()
        expected = [i for a, b in filled for i in range(a, b)]
        self.assertEqual([p for p in positions if p >= 0], expected)
        self.assertEqual([t for t in submitted_ids if t >= 3], [ids[i] for i in expected])
        self.assertEqual(submitted_ids.count(tokenizer.sep_token_id), len(filled))

    def test_archived_centered_policy(self):
        filled = uniform_fill([(0, 5)], list(range(0, 101, 5)), [(0, 50), (50, 100)],
                              budget=21, contiguous=False)
        self.assertEqual(filled, [(0, 5), (25, 34), (70, 75)])


class SourceAndDependenciesTest(unittest.TestCase):
    def test_simplification_preserves_literal_newlines_and_origin(self):
        code = 'int f() {\n  const char *s = R"(//literal/*)";\n  return 1/**/+2; // note\n}\n'
        comments, literals, _, _ = analyze(code, source_parser())
        simple, origin, synthetic, status = simplify(code, comments, literals)
        self.assertEqual(status, "lexical")
        self.assertIn('R"(//literal/*)"', simple)
        self.assertIn("return 1 +2;", simple)
        self.assertNotIn("note", simple)
        self.assertEqual(code.count("\n"), simple.count("\n"))
        self.assertTrue(all(simple[i] == code[p] for i, p in enumerate(origin) if not synthetic[i]))

    def test_both_sides_of_disjoint_diff_blocks(self):
        left = ["start\n", "old\n", "keep\n", "before\n", "end\n"]
        right = ["start\n", "keep\n", "after\n", "end\n", "inserted\n"]
        diff = line_edits(left, right)
        self.assertEqual([edit[0] for edit in diff], ["delete", "replace", "insert"])

        def map_text(lines, side):
            text = "".join(lines)
            return map_edits(lines, [(i, i+1) for i in range(len(text))], diff, side), text

        old, old_text = map_text(left, 0)
        new, new_text = map_text(right, 1)
        self.assertEqual([old_text[a:b] for a, b in old["delete"]], ["old\n"])
        self.assertEqual([old_text[a:b] for a, b in old["replace"]], ["before\n"])
        self.assertEqual([new_text[a:b] for a, b in new["insert"]], ["inserted\n"])
        self.assertEqual([old_text[a:b] for a, b in old["insert_gap"]], ["end\n"])
        self.assertEqual([new_text[a:b] for a, b in new["delete_gap"]], ["start\nkeep\n"])

    def test_joern_column_is_utf16_and_lines_are_physical(self):
        code = "a😀z;\r\nx = 3;\r\n"
        lines = physical_lines(code)
        self.assertEqual(source_position(code, lines, 1, 4), code.index("z"))
        self.assertEqual(source_position(code, lines, 2, 1), code.index("x"))

    def test_missing_graph_and_missing_relation_are_reported_separately(self):
        graph = dict(status="ok", edges={"REACHING_DEF": [(0, 1)], "CDG": []})
        self.assertEqual(available_graph(graph, (1, 0)), (graph["edges"], "available"))
        self.assertEqual(available_graph(graph, (1, 1)), (None, "no_effective_dependence"))
        failed = dict(status="method_position_mismatch", edges=graph["edges"])
        self.assertEqual(available_graph(failed, (1, 1)), (None, "parse_or_graph_failed"))

    def test_upstream_edge_choice_accounts_for_joined_fragments(self):
        units = [(0, 2), (2, 3), (3, 5), (5, 6), (6, 8)]
        regions = [Region(a, b, a, b, "statement") for a, b in units]
        edges = {"REACHING_DEF": [(2, 4)], "CDG": [(0, 4)]}
        self.assertEqual(optimize(regions, [(6, 8)], {4}, edges, (2, 1), 5), ([(3, 8)], 1))

    def test_source_only_selection_rejects_oracle_edits(self):
        with self.assertRaisesRegex(ValueError, "privileged Oracle"):
            choose(None, "longvul", 1022, edits={})


if __name__ == "__main__":
    unittest.main()
