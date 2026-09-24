import ast
import unittest
from pathlib import Path

from scripts.benchmark_reporting_order import SITE_ORDER, external_row_sort_key, site_sort_key


class ReportingOrderTests(unittest.TestCase):
    def test_known_sites_follow_paper_order(self):
        self.assertEqual(sorted(reversed(SITE_ORDER), key=site_sort_key), list(SITE_ORDER))

    def test_unknown_sites_follow_known_sites(self):
        self.assertEqual(
            sorted(["zzz", "Nextcloud", "aaa", "PetClinic"], key=site_sort_key),
            ["PetClinic", "Nextcloud", "aaa", "zzz"],
        )

    def test_site_precedes_algorithm(self):
        rows = [
            {"site": "github", "algorithm": "qexplore", "seed": "seed1"},
            {"site": "petclinic", "algorithm": "webrled-official", "seed": "seed1"},
            {"site": "petclinic", "algorithm": "qexplore", "seed": "seed2"},
            {"site": "petclinic", "algorithm": "qexplore", "seed": "seed1"},
        ]
        ordered = sorted(rows, key=external_row_sort_key)
        self.assertEqual(ordered, [rows[3], rows[2], rows[1], rows[0]])
        self.assertEqual(rows[0]["site"], "github")

    def test_matches_frozen_unified_summary_order(self):
        root = Path(__file__).resolve().parents[1]
        tree = ast.parse((root / "summarize_runs.py").read_text(encoding="utf-8-sig"))
        site_assignment = next(
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "SITE_ORDER" for t in node.targets)
        )
        self.assertEqual(tuple(ast.literal_eval(site_assignment.value)), SITE_ORDER)


if __name__ == "__main__":
    unittest.main()
