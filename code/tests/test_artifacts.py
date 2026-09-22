"""Check protected outputs and graph-level statistical units without writing files."""
import unittest
from unittest.mock import patch

from gcqaoa.artifacts import analyze
from gcqaoa.provenance import CANONICAL_DATA, CODE_ROOT, PAPER_CONFIG, run_id, safe_output


class ArtifactTests(unittest.TestCase):
    def test_every_configuration_preserves_released_evidence(self):
        execution = "study-" + "1" * 20
        for name in ("smoke", "pilot", "paper"):
            config = CODE_ROOT / "configs" / f"{name}.json"
            with self.subTest(name=name), self.assertRaises(ValueError):
                safe_output(config, CANONICAL_DATA)
            self.assertEqual(safe_output(config, execution_id=execution),
                             CODE_ROOT / "results/executions" / execution / "study.json.gz")
        with self.assertRaises(ValueError):
            safe_output(PAPER_CONFIG, CODE_ROOT.parent / "unrequested.json.gz")
        with self.assertRaises(ValueError):
            safe_output(PAPER_CONFIG, CODE_ROOT / "results/aggregate/study.json.gz")
        with self.assertRaises(ValueError):
            safe_output(PAPER_CONFIG, execution_id="../../escape")

    def test_run_ids_distinguish_execution_and_selection(self):
        run = {"graph": "test_graph", "depth": 2, "replicate": 1, "budget": 100, "method": "a"}
        self.assertEqual(run_id("study-123", run), "study-123:test_graph:p2:r1:b100:a")
        self.assertNotEqual(run_id("study-123", run), run_id("study-456", run))
        self.assertNotEqual(run_id("study-123", run), run_id("study-123", {**run, "method": "b"}))

    def test_aggregate_and_paired_intervals_use_graph_means(self):
        runs = []
        for method, values in (("a", [("x", -.1), ("y", -.9), ("y", -.9), ("y", -.9)]),
                               ("b", [("x", -.4), ("y", -.4), ("y", -.4), ("y", -.4)])):
            for graph, objective in values:
                runs.append({"graph": graph, "method": method, "depth": 1, "budget": 100,
                             "objective": objective, "expected_ratio": -objective,
                             "signed_reference_gap": objective + 1, "hitting_shots": None,
                             "shots": 64, "calls": 2, "jobs": 1, "seconds": .01,
                             "validation": {"shots": 16, "calls": 1, "jobs": 1}})
        data = {"execution_run_id": "study-test", "runs": runs,
                "config": {"depths": [1], "budgets": [100], "methods": ["a", "b"]}}
        captured = {}
        with patch("gcqaoa.artifacts.write_csv", side_effect=lambda path, rows: captured.update({path.name: rows})):
            outputs = analyze(data, CANONICAL_DATA)
        self.assertTrue(all(path.parent == CANONICAL_DATA.parent / "aggregate" for path in outputs))
        summary = next(row for row in captured["summary.csv"] if row["method"] == "a")
        self.assertEqual(summary["n_graphs"], 2)
        self.assertEqual(summary["n_runs"], 4)
        self.assertAlmostEqual(summary["objective_mean"], -.5)
        self.assertEqual(summary["total_shots_mean"], 80)
        difference = next(row for row in captured["paired_comparisons.csv"] if row["metric"] == "objective")
        self.assertAlmostEqual(difference["mean_difference"], -.1)
        self.assertEqual(difference["difference"], "a_minus_b")


if __name__ == "__main__":
    unittest.main()
