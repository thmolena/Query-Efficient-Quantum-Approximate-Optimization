"""Independent checks of pooling, finite-look certification, and replay."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np
from numpy.testing import assert_allclose

from gcqaoa import followup
from gcqaoa.qaoa import MaxCutQAOA


def settings():
    return {"radius": 0.3, "model_shots": 8, "min_radius": 0.025,
            "max_radius": 0.8, "baseline_shots": 8, "max_trials": 2,
            "acceptance_looks": [8, 32], "alpha": 0.05, "eta": 0.1}


class MomentTests(unittest.TestCase):
    def test_pooled_variance_matches_unbiased_individual_observations(self):
        samples = [np.array([-1., -0.5, 0., 0.]), np.array([-1., -1., -0.5]),
                   np.array([0., -0.1, -0.4, -0.8, -1.])]
        batches = [{"shots": len(values), "mean": float(values.mean()),
                    "variance_of_mean": float(values.var(ddof=1) / len(values))} for values in samples]
        n, mean, variance = followup.pool_moments(batches)
        combined = np.concatenate(samples)
        self.assertEqual(n, len(combined))
        assert_allclose([mean, variance], [combined.mean(), combined.var(ddof=1)], atol=1e-15)

    def test_between_batch_variance_is_not_lost(self):
        n, mean, variance = followup.pool_moments([
            {"shots": 4, "mean": -1., "variance_of_mean": 0.},
            {"shots": 4, "mean": 0., "variance_of_mean": 0.}])
        self.assertEqual(n, 8)
        self.assertEqual(mean, -0.5)
        self.assertAlmostEqual(variance, 2 / 7)

    def test_bound_matches_primary_theorem_and_union_allocation(self):
        n, variance, alpha, horizon, looks = 1024, 0.04, 0.05, 16, 4
        delta = alpha / (2 * 2 * horizon * looks)
        expected = np.sqrt(2 * variance * np.log(2 / delta) / n) + 7 * np.log(2 / delta) / (3 * (n - 1))
        self.assertAlmostEqual(followup.bernstein_radius(n, variance, alpha, horizon, looks), expected)
        self.assertGreater(followup.bernstein_radius(n, 0, alpha, horizon, looks), 0)

    def test_invalid_moments_and_bounds_are_rejected(self):
        for bad in [[], [{"shots": 1, "mean": 0., "variance_of_mean": 0.}],
                    [{"shots": 2, "mean": 0., "variance_of_mean": -1.}]]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                followup.pool_moments(bad)
        with self.assertRaises(ValueError):
            followup.bernstein_radius(1, 0.1, 0.05, 1, 1)


class OptimizerTests(unittest.TestCase):
    def test_certified_acceptance_updates_incumbent(self):
        class BoundedDeterministicCircuit:
            def sample(self, theta, shots, rng):
                return float(-0.5 - 0.4 * np.tanh(theta[0])), 0.0
        cfg = {**settings(), "max_trials": 1, "model_shots": 128,
               "acceptance_looks": [512, 2048]}
        result = followup.bernstein_optimize(BoundedDeterministicCircuit(), [0., 0.], np.eye(2),
                                             cfg, 9, 8192)
        self.assertEqual(result["decisions"][0]["decision"], "accepted")
        self.assertEqual(len(result["incumbents"]), 2)
        self.assertGreater(result["theta"][0], 0.25)
        self.assertGreaterEqual(result["decisions"][0]["lower_decrease"],
                                cfg["eta"] * result["decisions"][0]["predicted"])

    def test_initialization_baseline_never_queries_oracle(self):
        class ForbiddenCircuit:
            def sample(self, *args):
                raise AssertionError("Initialization-only baseline queried objective")
        result = followup.run_optimizer(ForbiddenCircuit(), [0.2, 0.1], np.eye(2), settings(),
                                        "initialization_only", 123, 256)
        self.assertEqual(result["shots"], 0)
        self.assertEqual(result["evaluations"], [])
        assert_allclose(result["theta"], [0.2, 0.1])

    def test_fresh_samples_and_thresholds_explain_every_decision(self):
        circuit = MaxCutQAOA(nx.path_graph(3))
        cfg = settings()
        result = followup.bernstein_optimize(circuit, [0.1, 0.05], np.eye(2), cfg, 77, 512)
        self.assertGreater(len(result["decisions"]), 0)
        for decision in result["decisions"]:
            observations = [row for row in result["evaluations"]
                            if row["stage"] == "acceptance" and row["iteration"] == decision["iteration"]]
            self.assertEqual(sum(row["shots"] for row in observations),
                             2 * decision["acceptance_shots_per_point"])
            for interval in decision["intervals"]:
                cumulative = interval["shots_per_point"]
                chosen, total = [], 0
                for index in range(0, len(observations), 2):
                    if total == cumulative:
                        break
                    chosen.extend(observations[index:index + 2])
                    total += observations[index]["shots"]
                moments = [followup.pool_moments(chosen[side::2]) for side in (0, 1)]
                radii = [followup.bernstein_radius(n, var, cfg["alpha"], cfg["max_trials"],
                                                 len(cfg["acceptance_looks"])) for n, mean, var in moments]
                assert_allclose(interval["sample_variances"], [part[2] for part in moments])
                assert_allclose(interval["radii"], radii)
            threshold = cfg["eta"] * decision["predicted"]
            if decision["decision"] == "accepted":
                self.assertGreaterEqual(decision["lower_decrease"], threshold)
            elif decision["decision"] == "rejected":
                self.assertLess(decision["upper_decrease"], threshold)
            else:
                self.assertLess(decision["lower_decrease"], threshold)
                self.assertGreaterEqual(decision["upper_decrease"], threshold)

    def test_real_circuit_replay_detects_observation_tampering(self):
        circuit = MaxCutQAOA(nx.cycle_graph(4))
        result = followup.bernstein_optimize(circuit, [0.1, 0.05], np.eye(2), settings(), 3, 256)
        replayed = followup.bernstein_optimize(circuit, [0.1, 0.05], np.eye(2), settings(), 3, 256)
        followup.compare_results(replayed, result, {"checks": 0})
        result["evaluations"][0]["mean"] += 0.01
        with self.assertRaisesRegex(ValueError, "mismatch"):
            followup.compare_results(replayed, result, {"checks": 0})


class ProtocolTests(unittest.TestCase):
    def test_all_panels_and_splits_are_nonisomorphic(self):
        config = json.loads(followup.FOLLOWUP_CONFIG.read_text())
        rows = followup.instances(config)
        self.assertEqual(len(rows), 40)
        self.assertEqual(sum(row["split"] == "test" for row in rows), 16)
        for index, row in enumerate(rows):
            for previous in rows[:index]:
                if row["n"] == previous["n"]:
                    self.assertFalse(nx.is_isomorphic(followup.graph_from(row), followup.graph_from(previous)))

    def test_hyperparameter_selection_ignores_test_and_exact_values(self):
        config = {"depths": [2], "tuning_candidates": [{"radius": 0.1}, {"radius": 0.3}]}
        runs = [{"phase": "validation", "depth": 2, "candidate": index,
                 "objective": -1.0 if index == 0 else 0.0,
                 "validation": {"mean": -0.5 if index == 0 else -0.7}} for index in (0, 1)]
        runs.append({"phase": "test", "depth": 2, "candidate": 0,
                     "validation": {"mean": -1000}})
        selected, _ = followup.select_settings(config, runs)
        self.assertEqual(selected, {"2": {"radius": 0.3}})

    def test_small_complete_bundle_and_semantic_replay(self):
        config = {"protocol_version": 1, "panel_seeds": [971], "families": ["erdos_renyi"],
                  "bank_sizes": [6], "validation_sizes": [8], "test_sizes": [10],
                  "depths": [1], "repetitions": 1, "budget": 128, "reference_starts": 1,
                  "reference_maxiter": 3, "validation_shots": 8, "target_gaps": [0.01, 0.02],
                  "methods": ["initialization_only", "bernstein_iso", "spsa"],
                  "tuning_methods": ["bernstein_iso"],
                  "tuning_candidates": [{"model_shots": 8, "radius": 0.3}],
                  "optimizer": settings()}
        with tempfile.TemporaryDirectory(dir=followup.CODE_ROOT / "results") as directory:
            path = Path(directory)
            cfg = path / "config.json"
            cfg.write_text(json.dumps(config))
            with patch.object(followup, "FOLLOWUP_CONFIG", cfg):
                data = followup.run_study(path / "followup.json.gz")
                report = followup.verify_followup(data, replay=True)
                self.assertEqual(report["replayed_runs"], 4)
                self.assertGreater(report["reference_calls"], 0)
                broken = copy.deepcopy(data)
                broken["runs"][-1]["evaluations"][0]["mean"] += 0.1
                broken["payload_sha256"] = followup.content_hash({key: value for key, value in broken.items()
                                                                 if key != "payload_sha256"})
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    followup.verify_followup(broken, replay=True)
                with self.assertRaisesRegex(ValueError, "exists"):
                    followup.run_study(path / "followup.json.gz")


if __name__ == "__main__":
    unittest.main()
