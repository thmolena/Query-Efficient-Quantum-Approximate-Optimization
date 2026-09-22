"""Independent geometry, interpolation, and finite-shot accounting checks."""

import itertools
import unittest

import networkx as nx
import numpy as np
from numpy.testing import assert_allclose

from gcqaoa.qaoa import MaxCutQAOA, normalized_shortest_path_matrix
from gcqaoa.search import (
    BudgetExhausted,
    ShotOracle,
    align,
    descriptor,
    design_points,
    fit_model,
    graph_prior,
    optimize,
    periods,
    structural_score,
    wrap,
)


class DeterministicSingleEdge:
    """Analytic bounded objective; no simulator or sampling code is reused."""

    @staticmethod
    def objective(theta):
        gamma, beta = theta
        return -0.5 * (1 + np.sin(gamma) * np.sin(4 * beta))

    def sample(self, theta, shots, rng):
        return self.objective(theta), 0.0


def optimizer_settings():
    return {
        "radius": 0.18,
        "max_radius": 0.3,
        "min_radius": 0.01,
        "model_shots": 128,
        "baseline_shots": 128,
        "max_trials": 8,
        "acceptance_looks": [4096, 16384, 65536],
        "alpha": 0.05,
        "eta": 0.1,
    }


class StructuralGeometryTests(unittest.TestCase):
    def test_coupling_marginals_and_direct_four_index_distortion(self):
        for left, right in (
            (nx.path_graph(3), nx.cycle_graph(5)),
            (nx.star_graph(3), nx.path_graph(5)),
            (nx.complete_graph(3), nx.complete_graph(4)),
        ):
            D = normalized_shortest_path_matrix(left)
            E = normalized_shortest_path_matrix(right)
            score, details = structural_score(D, E)
            coupling = np.asarray(details["coupling"])
            n, m = len(D), len(E)
            with self.subTest(n=n, m=m, edges=tuple(left.edges)):
                self.assertTrue(np.isfinite(coupling).all())
                self.assertTrue((coupling >= 0).all())
                assert_allclose(coupling.sum(axis=1), np.full(n, 1 / n), atol=1e-10)
                assert_allclose(coupling.sum(axis=0), np.full(m, 1 / m), atol=1e-10)
                direct = sum(
                    (D[i, k] - E[j, ell]) ** 2 * coupling[i, j] * coupling[k, ell]
                    for i, j, k, ell in itertools.product(range(n), range(m), range(n), range(m))
                )
                self.assertAlmostEqual(score, direct, 12)

    def test_structural_score_is_invariant_to_node_order(self):
        D = normalized_shortest_path_matrix(nx.path_graph(4))
        E = normalized_shortest_path_matrix(nx.star_graph(4))
        score, _ = structural_score(D, E)
        a, b = [2, 0, 3, 1], [4, 1, 0, 3, 2]
        permuted, _ = structural_score(D[np.ix_(a, a)], E[np.ix_(b, b)])
        self.assertAlmostEqual(score, permuted, 11)

    def test_angle_alignment_preserves_depth_one_and_two_objectives(self):
        rng = np.random.default_rng(8192)
        for graph in (nx.path_graph(4), nx.complete_graph(3), nx.cycle_graph(5)):
            circuit = MaxCutQAOA(graph)
            for depth in (1, 2):
                for _ in range(8):
                    theta = rng.uniform(-8, 8, 2 * depth)
                    anchor = rng.uniform(-12, 12, 2 * depth)
                    expected = circuit.objective(theta)
                    shifts = periods(depth) * rng.integers(-4, 5, 2 * depth)
                    with self.subTest(graph=tuple(graph.edges), depth=depth):
                        for candidate in (wrap(theta), align(theta, anchor), -theta, theta + shifts):
                            self.assertAlmostEqual(circuit.objective(candidate), expected, 12)
                        aligned = align(theta, anchor)
                        self.assertTrue(np.all(abs(aligned - anchor) <= periods(depth) / 2 + 1e-12))

    def test_prior_has_unit_volume_and_bounded_condition(self):
        graphs = [nx.path_graph(5), nx.cycle_graph(5), nx.star_graph(4), nx.complete_graph(5)]
        angles = [
            [0.8, 0.4, 0.22, 0.13],
            [0.98, 0.44, 0.25, 0.16],
            [1.15, 0.35, 0.35, 0.01],
            [2.9, 2.8, 0.62, 0.64],
        ]
        bank = [
            {"descriptor": descriptor(graph).tolist(), "theta": theta,
             "structure": normalized_shortest_path_matrix(graph).tolist()}
            for graph, theta in zip(graphs, angles)
        ]
        for kind in ("uniform", "descriptor", "gw"):
            anchor, shape, details = graph_prior(graphs[0], bank, kind=kind)
            with self.subTest(kind=kind):
                assert_allclose(shape, shape.T, atol=1e-13)
                self.assertTrue(np.all(np.linalg.eigvalsh(shape) > 0))
                self.assertAlmostEqual(np.linalg.det(shape), 1.0, 12)
                self.assertLessEqual(np.linalg.cond(shape @ shape.T), 4 + 1e-11)
                weights = np.asarray(details["weights"])
                self.assertAlmostEqual(weights.sum(), 1.0, 13)
                self.assertTrue(np.all(weights >= 0))
                assert_allclose(anchor, bank[details["anchor"]]["theta"])
                for weight, row in zip(weights, bank):
                    if weight > 0:
                        self.assertLessEqual(np.linalg.norm(align(row["theta"], anchor) - anchor), 1 + 1e-12)


class LocalModelTests(unittest.TestCase):
    def test_repaired_design_recovers_exact_affine_model(self):
        rng = np.random.default_rng(808)
        for dimension in (2, 4, 8):
            points, design = design_points(dimension, rng)
            truth = rng.normal(size=dimension + 1)
            values = truth[0] + points @ truth[1:]
            estimates, condition = fit_model(design, values, np.arange(2, dimension + 3) ** 2)
            with self.subTest(dimension=dimension):
                assert_allclose(estimates, truth, atol=2e-13)
                self.assertTrue(np.isfinite(condition))
                self.assertTrue(np.all(np.linalg.norm(points, axis=1) <= 1 + 1e-12))

    def test_weighted_qr_matches_independent_svd_least_squares(self):
        design = np.array([[1, -1], [1, 0], [1, 1], [1, 2]], dtype=float)
        measurements = np.array([2.1, 0.2, 1.3, -0.4])
        shots = np.array([2, 8, 32, 128])
        expected, _, _, _ = np.linalg.lstsq(
            design * np.sqrt(shots[:, None]), measurements * np.sqrt(shots), rcond=None
        )
        actual, _ = fit_model(design, measurements, shots)
        assert_allclose(actual, expected, atol=1e-13)
        assert_allclose(design.T @ (shots * (measurements - design @ actual)), 0, atol=2e-12)


class ResourceAccountingTests(unittest.TestCase):
    def test_shot_oracle_records_every_evaluation_and_batch(self):
        oracle = ShotOracle(DeterministicSingleEdge(), 30, np.random.default_rng(4))
        points = [[0.3, 0.1], [0.5, 0.2]]
        values = oracle.batch(points, [4, 8], "model")
        oracle.batch([[0.8, 0.3]], 6, "acceptance")
        assert_allclose(values, [DeterministicSingleEdge.objective(x) for x in points])
        self.assertEqual((oracle.calls, oracle.jobs, oracle.shots), (3, 2, 18))
        self.assertEqual([row["call"] for row in oracle.records], [1, 2, 3])
        self.assertEqual([row["job"] for row in oracle.records], [1, 1, 2])
        self.assertEqual([row["cumulative_shots"] for row in oracle.records], [4, 12, 18])
        self.assertEqual([row["stage"] for row in oracle.records], ["model", "model", "acceptance"])
        self.assertEqual(sum(row["shots"] for row in oracle.records), oracle.shots)

    def test_budget_rejection_is_atomic_and_exact_limit_is_allowed(self):
        oracle = ShotOracle(DeterministicSingleEdge(), 12, np.random.default_rng(4))
        points = [[0.3, 0.1], [0.5, 0.2]]
        oracle.batch(points, [4, 8], "model")
        before = (oracle.calls, oracle.jobs, oracle.shots, list(oracle.records))
        with self.assertRaises(BudgetExhausted):
            oracle.batch(points, [2, 2], "acceptance")
        self.assertEqual((oracle.calls, oracle.jobs, oracle.shots, oracle.records), before)

    def test_invalid_shot_counts_do_not_silently_change_accounting(self):
        points = [[0.3, 0.1], [0.5, 0.2]]
        for counts in (3.5, True, [4], [4, 8, 12], [4, 3.5], [4, 1]):
            oracle = ShotOracle(DeterministicSingleEdge(), 100, np.random.default_rng(4))
            with self.subTest(counts=counts):
                with self.assertRaises(ValueError):
                    oracle.batch(points, counts, "model")
                self.assertEqual((oracle.calls, oracle.jobs, oracle.shots, oracle.records), (0, 0, 0, []))

    def test_accepted_deterministic_steps_improve_exact_objective(self):
        settings = optimizer_settings()
        circuit = DeterministicSingleEdge()
        result = optimize(circuit, [0.3, 0.15], np.eye(2), settings, "trust", 7, 1000000)
        accepted = [row for row in result["decisions"] if row["decision"] == "accepted"]
        self.assertGreater(len(accepted), 0)
        self.assertTrue(any(row["decision"] == "rejected" for row in result["decisions"]))
        for row in accepted:
            actual_decrease = circuit.objective(row["center"]) - circuit.objective(row["trial"])
            self.assertGreater(actual_decrease, 0)
            self.assertGreaterEqual(actual_decrease + 1e-13, settings["eta"] * row["predicted"])
            self.assertGreaterEqual(row["lower_decrease"], settings["eta"] * row["predicted"])
        self.assertEqual(len(result["incumbents"]), 1 + len(accepted))
        self.assertEqual(len(result["decisions"]), settings["max_trials"])
        self.assertNotEqual(result["stop"], "budget")
        self.assertLess(result["shots"], 1000000)

    def test_finite_shot_methods_obey_budget_and_reproduce_decisions(self):
        settings = optimizer_settings()
        settings.update(max_trials=3, acceptance_looks=[64, 256], model_shots=32)
        circuit = MaxCutQAOA(nx.path_graph(3))
        for method in ("trust", "cobyla", "spsa", "fixed_shots", "fixed_radius", "no_repair"):
            result = optimize(circuit, [0.3, 0.15], np.eye(2), settings, method, 923, 1700)
            repeated = optimize(circuit, [0.3, 0.15], np.eye(2), settings, method, 923, 1700)
            with self.subTest(method=method):
                self.assertLessEqual(result["shots"], 1700)
                records = result["evaluations"]
                self.assertEqual(result["calls"], len(records))
                self.assertEqual(result["jobs"], len({row["job"] for row in records}))
                self.assertEqual(result["shots"], sum(row["shots"] for row in records))
                assert_allclose([row["cumulative_shots"] for row in records], np.cumsum([row["shots"] for row in records]))
                for key in ("theta", "shots", "calls", "jobs", "decisions", "incumbents"):
                    self.assertEqual(result[key], repeated[key])


if __name__ == "__main__":
    unittest.main()
