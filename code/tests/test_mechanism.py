"""Independent checks of bias/variance prediction and online replay accounting."""
import itertools
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from gcqaoa import mechanism
from gcqaoa.search import fit_model


class VariancePredictionTests(unittest.TestCase):
    def test_physical_gradient_map_matches_independent_weighted_qr(self):
        design = np.array([[1., 0., 0.], [1., 1., 0.], [1., 0., -1.]])
        shape = np.array([[1.2, 0.3], [0.1, 0.8]])
        means, radius = np.array([-0.4, -0.6, -0.3]), 0.07
        coefficient, _ = fit_model(design, means, [17, 101, 53])
        independently_fitted = np.linalg.solve(shape.T, coefficient[1:]) / radius
        assert_allclose(mechanism.gradient_map(design, shape, radius) @ means,
                        independently_fitted, atol=2e-14)

    def test_mse_identity_matches_exhaustive_binomial_outcomes(self):
        design = np.array([[1., 0., 0.], [1., 1., 0.], [1., 0., -1.]])
        shape, radius = np.array([[1.1, 0.2], [0.2, 0.9]]), 0.3
        probabilities = np.array([0.4, 0.6, 0.7])
        means, variances = -probabilities, probabilities * (1 - probabilities)
        mapping = mechanism.gradient_map(design, shape, radius)
        truth = np.array([0.3, -0.2])
        prediction = mechanism.variance_prediction(mapping, means, variances, truth, shots=2)
        total_mse, sampling_mse, weight_sum = 0., 0., 0.
        for counts in itertools.product(range(3), repeat=3):
            counts = np.asarray(counts)
            weight = np.prod(np.array([1., 2., 1.])[counts] * probabilities ** counts *
                             (1 - probabilities) ** (2 - counts))
            gradient = mapping @ (-counts / 2)
            total_mse += weight * np.sum((gradient - truth) ** 2)
            sampling_mse += weight * np.sum((gradient - mapping @ means) ** 2)
            weight_sum += weight
        self.assertAlmostEqual(weight_sum, 1.)
        self.assertAlmostEqual(total_mse, prediction["total_mse"])
        self.assertAlmostEqual(sampling_mse, prediction["sampling_mse"])

    def test_radius_squared_schedule_preserves_propagated_variance_at_fixed_point_variance(self):
        design, shape = np.array([[1., 0., 0.], [1., 1., 0.], [1., 0., -1.]]), np.eye(2)
        values = []
        for radius, shots in zip([.025, .05, .1, .2, .4], [4096, 1024, 256, 64, 16]):
            mapping = mechanism.gradient_map(design, shape, radius)
            values.append(mechanism.variance_prediction(mapping, [-.5] * 3, [.1] * 3,
                                                       [0., 0.], shots)["sampling_mse"])
        assert_allclose(values, [values[0]] * len(values), atol=1e-15)


class OnlinePowerTests(unittest.TestCase):
    def revision(self):
        looks = [512, 2048, 8192, 16384, 65536]
        observations, previous = [], 0
        for count in looks:
            observations.append({"shots_per_point": count,
                "batches": [{"shots": count - previous, "mean": mean, "variance_of_mean": .04 / (count - previous)}
                            for mean in [-.5, -.503]]})
            previous = count
        return {"config": {"power_looks": looks, "power_horizon": 16,
                            "power_alpha": .05, "optimizer": {"eta": .1}},
                "model_diagnostics": [{"id": 0, "graph": "g", "depth": 2, "shape_kind": "shape",
                    "radius": .1, "model_shots": 1024, "predicted": .01,
                    "true_decrease": .003, "acceptance_margin": .002}],
                "power_samples": [{"proposal_id": 0, "replicate": 0, "observations": observations}], "runs": []}

    def config(self):
        return {"budget": 65536, "online_looks": [512, 2048, 8192, 16384],
                "online_pair_budgets": [1024, 4096, 16384, 32768]}

    def test_online_and_extended_replays_keep_same_proposals_with_distinct_J(self):
        online, _ = mechanism.online_power(self.config(), self.revision())
        extended, _ = mechanism.online_power(self.config(), self.revision(), extended=True)
        self.assertEqual(len(online), 8)
        self.assertEqual(len(extended), 10)
        self.assertEqual({row["J"] for row in online}, {4})
        self.assertEqual({row["J"] for row in extended}, {5})
        self.assertEqual({row["proposal_id"] for row in online + extended}, {0})
        self.assertEqual(max(row["endpoint_pair_budget"] for row in online), 32768)
        self.assertEqual(max(row["endpoint_pair_budget"] for row in extended), 131072)
        for row in online + extended:
            self.assertLessEqual(row["mean_acceptance_shots"], row["endpoint_pair_budget"])

    def test_partial_pair_budget_never_spends_an_unaffordable_atomic_look(self):
        config = {**self.config(), "budget": 5120 + 3000, "online_pair_budgets": [32768]}
        records, _ = mechanism.online_power(config, self.revision())
        for record in records:
            self.assertEqual(record["affordable_looks"], [512])
            self.assertEqual(record["cap_per_endpoint"], 512)
            self.assertEqual(record["mean_acceptance_shots"], 1024)


class ProtocolTests(unittest.TestCase):
    def test_four_cells_have_identical_graph_depth_seed_and_replication(self):
        config = {"depths": [2, 3], "attribution_repetitions": 2, "seed": 123,
                  "attribution_cells": [[sampling, stopping] for sampling in ["fixed", "radius"]
                                        for stopping in ["stop", "continue"]]}
        specs = mechanism.allocation_specs(config, {"instances": [{"id": "g", "split": "test"}]})
        for index in range(0, len(specs), 4):
            group = specs[index:index + 4]
            self.assertEqual(len({row["seed"] for row in group}), 1)
            self.assertEqual(len({(row["graph"], row["depth"], row["replicate"]) for row in group}), 1)
            self.assertEqual(len({row["method"] for row in group}), 4)

    def test_full_replay_refuses_changed_live_numerical_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source.py").write_text("changed")
            data = {"study": "mechanism", "schema_version": 1, "status": "complete",
                    "provenance": {"source_text": {"source.py": "original"},
                                   "source_sha256": {"source.py": hashlib.sha256(b"original").hexdigest()}}}
            data["payload_sha256"] = mechanism.content_hash(data)
            with patch.object(mechanism, "CODE_ROOT", root), patch.object(mechanism, "SOURCES", ("source.py",)):
                with self.assertRaisesRegex(ValueError, "requires the recorded numerical sources"):
                    mechanism.verify_mechanism(data, replay=True)


if __name__ == "__main__":
    unittest.main()
