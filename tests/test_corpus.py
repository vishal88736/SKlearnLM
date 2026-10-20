import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_corpus import BT, EOS, assign_splits, clean_layout, code_fence, extract_page


class CorpusTests(unittest.TestCase):
    def test_code_and_math_are_preserved(self):
        code = 'if True:\n    value = "a  b"\n\n\n    print(value)\n'
        raw = ('<article class="bd-article"><h1>Example</h1>'
               '<p>Use <code>foo_bar</code>.</p><pre>' + code + '</pre>'
               r'<div class="math">\[x = y^2\]</div>'
               '<dt><strong>alpha</strong><span class="classifier">float</span></dt>'
               '<div class="sk-top-container">estimator widget noise</div></article>')
        row = extract_page(raw, "modules/example.html")
        self.assertIn(code, row["text"])
        self.assertIn(BT + "foo_bar" + BT, row["text"])
        self.assertIn(r"\[x = y^2\]", row["text"])
        self.assertIn("**alpha** : float", row["text"])
        self.assertNotIn("widget noise", row["text"])

    def test_backticks_inside_code_cannot_close_the_fence(self):
        code = 'text = "' + BT * 3 + '"\n'
        fenced = code_fence(code, "python")
        self.assertTrue(fenced.startswith(BT * 4 + "python\n"))
        self.assertEqual(clean_layout(fenced), fenced)

    def test_table_columns_and_landing_content_survive(self):
        raw = ('<main><div class="sk-landing-container"><h1>Guide</h1>'
               '<table><tr><th>Name</th><th>Type</th></tr>'
               '<tr><td>alpha</td><td>float</td></tr></table></div></main>')
        row = extract_page(raw, "index.html")
        self.assertIn("| Name | Type |", row["text"])
        self.assertIn("| alpha | float |", row["text"])

    def test_splits_are_reproducible_and_keep_topic_families_together(self):
        rows = [{"group_id": f"topic/{index}", "category": "api_reference",
                 "text": "x" * (40 + index), "split": None}
                for index in range(30) for _ in range(2)]
        first, second = copy.deepcopy(rows), copy.deepcopy(list(reversed(rows)))
        assign_splits(first, seed=42)
        assign_splits(second, seed=42)
        assignments = {row["group_id"]: row["split"] for row in first}
        self.assertEqual(assignments, {row["group_id"]: row["split"] for row in second})
        self.assertEqual(set(assignments.values()), {"train", "validation", "test"})
        for group in assignments:
            self.assertEqual(len({row["split"] for row in first if row["group_id"] == group}), 1)


if __name__ == "__main__":
    unittest.main()
