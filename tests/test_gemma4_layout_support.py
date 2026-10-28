"""Keep the layout correction independent of test labels and source identity."""

import unittest

from anomaly_eval.gemma4.layout_support import branch, source_metrics
from anomaly_eval.gemma4.protocol import BASELINE, POLICIES


class LayoutSupportTests(unittest.TestCase):
    def test_fallback_depends_only_on_layout_coverage_and_view_count(self):
        row = {"views": ["full", "last"], "source": "x", "target": 0}
        sheet = POLICIES[-1]
        self.assertEqual(branch(row, sheet, 0), "baseline")
        self.assertEqual(
            branch({**row, "source": "y", "target": 1}, sheet, 0), "baseline"
        )
        self.assertEqual(branch(row, sheet, 1), "optimized")
        self.assertEqual(branch({**row, "views": ["single"]}, sheet, 0), "optimized")
        self.assertEqual(branch(row, BASELINE, 0), "optimized")

    def test_mixed_source_uses_branch_cutoffs_but_keeps_raw_ranking_ap(self):
        rows = [
            {"id": "a", "source": "mixed", "target": 1, "group_id": "1"},
            {"id": "b", "source": "mixed", "target": 0, "group_id": "2"},
        ]
        scores = {"baseline": {"a": 0.2, "b": 99}, "optimized": {"a": -99, "b": 0.8}}
        routes = {"a": "baseline", "b": "optimized"}
        cutoffs = {"baseline": {"p": 0.1}, "optimized": {"p": 1.0}}
        value = source_metrics(rows, scores, routes, cutoffs)["mixed"]["p"]
        self.assertEqual((value["tp"], value["fp"], value["tn"]), (1, 0, 1))
        self.assertEqual(value["f1"], 1)
        self.assertEqual(value["ap"], 0.5)
        self.assertIsNone(value["threshold"])
        self.assertEqual(value["route_counts"], {"baseline": 1, "optimized": 1})


if __name__ == "__main__":
    unittest.main()
