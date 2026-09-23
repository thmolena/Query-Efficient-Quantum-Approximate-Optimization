"""Independent geometry, interpolation, and finite-shot accounting checks."""

import gzip
import itertools
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

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
    refine,
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
    def test_saturated_local_scores_tie_in_bank_order_with_documented_tolerance(self):
        graph = nx.path_graph(3)
        bank = [{'descriptor': descriptor(graph).tolist(), 'theta': [0.2+k*.1, 0.1]}
                for k in range(3)]
        anchor, _, details = graph_prior(graph, bank, 'local',
                                        scores=[1., 1.-5e-15, 1.+2e-15])
        assert_allclose(anchor, bank[0]['theta'])
        self.assertEqual(details['anchor'], 0)
        self.assertEqual(details['tied_donors'], [0, 1, 2])
        self.assertEqual(details['tie_tolerance'], 1e-12)
        _, _, informative = graph_prior(graph, bank, 'local', scores=[1., 1.-1e-8, 1.])
        self.assertEqual(informative['anchor'], 1)
        with self.assertRaises(ValueError):
            graph_prior(graph, bank, 'local')

    def test_default_prior_is_descriptor_and_rejects_invalid_scores(self):
        graphs = [nx.path_graph(4), nx.complete_graph(4)]
        bank = [{'descriptor': descriptor(graph).tolist(), 'theta': [.2+k*.1, .1]}
                for k, graph in enumerate(graphs)]
        default = graph_prior(graphs[1], bank)
        explicit = graph_prior(graphs[1], bank, 'descriptor')
        assert_allclose(default[0], explicit[0])
        self.assertEqual(default[2]['anchor'], 1)
        for scores in ([0.], [0., np.nan], [[0.], [1.]]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                graph_prior(graphs[0], bank, 'local', scores=scores)

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


class RevisedRefinementTests(unittest.TestCase):
    @staticmethod
    def deterministic_fields(value):
        if isinstance(value, dict):
            return {key: RevisedRefinementTests.deterministic_fields(part)
                    for key, part in value.items() if key != 'seconds'}
        if isinstance(value, list):
            return [RevisedRefinementTests.deterministic_fields(part) for part in value]
        return value

    def test_default_policy_matches_immutable_precontrol_source(self):
        """Compare every deterministic field with the already released source."""
        path = Path(__file__).resolve().parents[1]/'results'/'refinement.json.gz'
        with gzip.open(path, 'rt') as stream:
            bundle = json.load(stream)
        source = bundle['provenance']['source_text']['src/gcqaoa/search.py']
        frozen = types.ModuleType('gcqaoa._frozen_refinement_search')
        frozen.__package__ = 'gcqaoa'
        with patch.dict(sys.modules, {frozen.__name__: frozen}):
            exec(compile(source, '<released refinement search>', 'exec'), frozen.__dict__)
        cases = [
            ({**optimizer_settings(), 'model_shots': 8, 'acceptance_looks': [32]}, 10000),
            ({**optimizer_settings(), 'model_shots': 8, 'acceptance_looks': [32]}, 87),
            ({**optimizer_settings(), 'max_trials': 2, 'acceptance_looks': [65536]}, 1000000),
        ]
        for bound in ('hoeffding', 'bernstein'):
            for settings, budget in cases:
                for circuit in (DeterministicSingleEdge(), MaxCutQAOA(nx.path_graph(3))):
                    with self.subTest(bound=bound, settings=settings, circuit=type(circuit).__name__):
                        arguments = (circuit, [.3, .15], np.eye(2), settings, 7, budget)
                        expected = frozen.refine(*arguments, bound=bound)
                        actual = refine(*arguments, bound=bound)
                        self.assertEqual(self.deterministic_fields(actual), self.deterministic_fields(expected))

    def test_four_controls_share_first_trial_and_pure_continuation_holds_radius(self):
        settings = {**optimizer_settings(), 'model_shots': 8,
                    'acceptance_looks': [32], 'max_trials': 3}
        results = {(sampling, unresolved): refine(DeterministicSingleEdge(), [.3, .15],
                    np.eye(2), settings, 7, 1000, model_sampling=sampling, unresolved_policy=unresolved)
                   for sampling, unresolved in itertools.product(('radius', 'fixed'), ('stop', 'continue'))}
        first = results[('radius', 'stop')]
        for (sampling, unresolved), result in results.items():
            with self.subTest(sampling=sampling, unresolved=unresolved):
                self.assertEqual(result['models'][0], first['models'][0])
                self.assertEqual(result['decisions'][0], first['decisions'][0])
                self.assertEqual(self.deterministic_fields(result['evaluations'][:5]),
                                 self.deterministic_fields(first['evaluations']))
                self.assertEqual(result['theta'], first['theta'])
                self.assertEqual(result['incumbents'], first['incumbents'])
                self.assertEqual(result['final_radius'], settings['radius'])
                if unresolved == 'stop':
                    self.assertEqual(len(result['models']), 1)
                    self.assertEqual(result['shots'], 88)
                else:
                    self.assertEqual([row['radius'] for row in result['models']], [settings['radius']]*3)
                    self.assertEqual([row['decision'] for row in result['decisions']], ['unresolved']*3)
                    self.assertEqual(result['stop_reason'], 'trial_horizon')
                    self.assertEqual(result['shots'], 264)
                    self.assertEqual([row['iteration'] for row in result['models']], [0, 1, 2])
                    self.assertEqual((result['calls'], result['jobs']), (15, 6))
                    self.assertEqual([row['job'] for row in result['evaluations']
                                      if row['stage'] == 'model'], [1]*3+[3]*3+[5]*3)
        # With no radius transition, the sampling factor has no effect at all.
        for unresolved in ('stop', 'continue'):
            a = {key: value for key, value in self.deterministic_fields(results[('radius', unresolved)]).items()
                 if key != 'policy'}
            b = {key: value for key, value in self.deterministic_fields(results[('fixed', unresolved)]).items()
                 if key != 'policy'}
            self.assertEqual(a, b)

    def test_continuation_retains_first_look_reservation_and_budget_atomicity(self):
        settings = {**optimizer_settings(), 'model_shots': 8, 'acceptance_looks': [32, 128]}
        for sampling in ('radius', 'fixed'):
            result = refine(DeterministicSingleEdge(), [.3, .15], np.eye(2), settings, 7, 200,
                            model_sampling=sampling, unresolved_policy='continue')
            with self.subTest(sampling=sampling):
                self.assertEqual(result['shots'], 176)
                self.assertEqual(len(result['models']), 2)
                self.assertEqual([row['acceptance_shots_per_point'] for row in result['decisions']], [32, 32])
                self.assertEqual(result['stop_reason'], 'model_and_first_look_unaffordable')
                self.assertEqual(result['last_allocation']['remaining_shots'], 24)
                self.assertEqual(result['last_allocation']['first_look_shots'], 64)
                self.assertEqual(result['final_radius'], settings['radius'])
                self.assertEqual(sum(row['shots'] for row in result['evaluations']), 176)

    def test_four_controls_match_finite_shot_streams_until_policy_divergence(self):
        settings = {**optimizer_settings(), 'model_shots': 8,
                    'acceptance_looks': [32], 'max_trials': 3}
        circuit = MaxCutQAOA(nx.path_graph(3))
        results = {(sampling, unresolved): refine(circuit, [.3, .15], np.eye(2), settings,
                    923, 1000, bound='bernstein', model_sampling=sampling, unresolved_policy=unresolved)
                   for sampling, unresolved in itertools.product(('radius', 'fixed'), ('stop', 'continue'))}
        first = results[('radius', 'stop')]
        self.assertEqual(first['decisions'][0]['decision'], 'unresolved')
        for controls, result in results.items():
            with self.subTest(controls=controls):
                self.assertEqual(result['decisions'][0], first['decisions'][0])
                self.assertEqual(self.deterministic_fields(result['evaluations'][:5]),
                                 self.deterministic_fields(first['evaluations']))
        for unresolved in ('stop', 'continue'):
            self.assertEqual(self.deterministic_fields(results[('radius', unresolved)]['evaluations']),
                             self.deterministic_fields(results[('fixed', unresolved)]['evaluations']))

    def test_fixed_sampling_remains_fixed_when_certified_decisions_change_radius(self):
        class StationaryQuadratic:
            def sample(self, theta, shots, rng):
                return -.75+.25*theta[0]**2+.5*theta[1]**2, 0.

        settings = {**optimizer_settings(), 'radius': .4, 'min_radius': .1,
                    'max_radius': .4, 'model_shots': 8, 'acceptance_looks': [65536]}
        for unresolved in ('stop', 'continue'):
            fixed = refine(StationaryQuadratic(), [0., 0.], np.eye(2), settings, 11, 1000000,
                           bound='bernstein', model_sampling='fixed', unresolved_policy=unresolved)
            radius = refine(StationaryQuadratic(), [0., 0.], np.eye(2), settings, 11, 1000000,
                            bound='bernstein', model_sampling='radius', unresolved_policy=unresolved)
            with self.subTest(unresolved=unresolved):
                self.assertEqual([row['radius'] for row in fixed['models']], [.4, .2, .1])
                self.assertEqual([row['model_shots_per_point'] for row in fixed['models']], [8, 8, 8])
                self.assertEqual([row['model_shots_per_point'] for row in radius['models']], [8, 32, 128])
                self.assertEqual([row['decision'] for row in fixed['decisions']], ['rejected']*3)
                self.assertEqual(fixed['stop'], 'radius_limit')
                self.assertEqual(fixed['decisions'][0], radius['decisions'][0])
        settings = {**optimizer_settings(), 'max_trials': 2, 'acceptance_looks': [65536]}
        fixed = refine(DeterministicSingleEdge(), [.3, .15], np.eye(2), settings, 7, 1000000,
                       bound='bernstein', model_sampling='fixed')
        self.assertEqual(fixed['decisions'][0]['decision'], 'accepted')
        self.assertEqual([row['model_shots_per_point'] for row in fixed['models']], [128, 128])
        self.assertEqual(fixed['models'][1]['radius'], 1.5*settings['radius'])

    def test_invalid_control_values_fail_before_sampling(self):
        class NeverSample:
            def sample(self, theta, shots, rng):
                raise AssertionError('Invalid controls must fail before sampling')

        for controls in ({'model_sampling': 'adaptive'}, {'unresolved_policy': 'contract'},
                         {'model_sampling': None}, {'unresolved_policy': False}):
            with self.subTest(controls=controls), self.assertRaises(ValueError):
                refine(NeverSample(), [.3, .15], np.eye(2), optimizer_settings(), 7, 10000, **controls)

    def test_reserve_first_comparison_before_spending_model_shots(self):
        settings = {**optimizer_settings(), 'model_shots': 8, 'acceptance_looks': [32]}
        required = 3*8+2*32
        result = refine(DeterministicSingleEdge(), [.3, .15], np.eye(2), settings,
                        7, required-1)
        self.assertEqual((result['shots'], result['calls'], result['jobs']), (0, 0, 0))
        self.assertEqual(result['stop'], 'resolution_limited')
        self.assertEqual(result['stop_reason'], 'model_and_first_look_unaffordable')
        self.assertEqual(result['evaluations'], [])
        self.assertEqual(result['decisions'], [])
        self.assertEqual(result['last_allocation']['model_shots'], 24)
        exact = refine(DeterministicSingleEdge(), [.3, .15], np.eye(2), settings, 7, required)
        self.assertEqual(exact['shots'], required)
        self.assertEqual(exact['decisions'][0]['acceptance_shots_per_point'], 32)

    def test_unresolved_evidence_stops_without_radius_contraction(self):
        settings = {**optimizer_settings(), 'model_shots': 8, 'acceptance_looks': [32]}
        for bound in ('hoeffding', 'bernstein'):
            result = refine(DeterministicSingleEdge(), [.3, .15], np.eye(2), settings,
                            7, 1000000, bound=bound)
            with self.subTest(bound=bound):
                self.assertEqual(result['stop'], 'resolution_limited')
                self.assertEqual(result['stop_reason'], 'acceptance_look_cap')
                self.assertEqual([row['decision'] for row in result['decisions']], ['unresolved'])
                self.assertEqual(result['final_radius'], settings['radius'])
                self.assertEqual(len(result['incumbents']), 1)
                self.assertEqual(result['theta'], result['incumbents'][0]['theta'])
                self.assertEqual(result['shots'], 88)

    def test_unaffordable_later_look_keeps_previous_incumbent_and_interval(self):
        settings = {**optimizer_settings(), 'model_shots': 8, 'acceptance_looks': [32, 128]}
        result = refine(DeterministicSingleEdge(), [.3, .15], np.eye(2), settings, 7, 100)
        self.assertEqual(result['stop_reason'], 'next_acceptance_look_unaffordable')
        self.assertEqual(result['shots'], 88)
        self.assertEqual(len(result['decisions'][0]['intervals']), 1)
        self.assertIsNotNone(result['decisions'][0]['lower_decrease'])
        self.assertEqual(result['final_radius'], settings['radius'])

    def test_certified_rejection_alone_contracts_and_quadruples_model_shots(self):
        class StationaryQuadratic:
            def sample(self, theta, shots, rng):
                return -.75+.25*theta[0]**2+.5*theta[1]**2, 0.

        settings = {**optimizer_settings(), 'radius': .4, 'min_radius': .1,
                    'max_radius': .4, 'model_shots': 8, 'max_trials': 4,
                    'acceptance_looks': [65536]}
        result = refine(StationaryQuadratic(), [0., 0.], np.eye(2), settings,
                        11, 1000000, bound='bernstein')
        self.assertEqual([row['decision'] for row in result['decisions']], ['rejected']*3)
        self.assertEqual([row['radius'] for row in result['decisions']], [.4, .2, .1])
        self.assertEqual([row['model_shots_per_point'] for row in result['models']], [8, 32, 128])
        self.assertEqual([row['model_shots'] for row in result['decisions']], [24, 96, 384])
        self.assertEqual(result['stop'], 'radius_limit')
        self.assertEqual(result['theta'], result['incumbents'][0]['theta'])
        for row in result['decisions']:
            self.assertLess(row['upper_decrease'], settings['eta']*row['predicted'])
            costs = [entry['shots'] for entry in result['evaluations']
                     if entry['stage'] == 'model' and entry['iteration'] == row['iteration']]
            self.assertEqual(sum(costs), row['model_shots'])

    def test_radius_increase_uses_the_declared_inverse_square_allocation(self):
        settings = {**optimizer_settings(), 'model_shots': 128, 'max_trials': 2,
                    'acceptance_looks': [65536]}
        events = []
        result = refine(DeterministicSingleEdge(), [.3, .15], np.eye(2), settings,
                        7, 1000000, observer=lambda event, row: events.append((event, dict(row))),
                        bound='bernstein')
        self.assertEqual(result['decisions'][0]['decision'], 'accepted')
        self.assertEqual(result['models'][1]['model_shots_per_point'], 57)
        for decision in result['decisions']:
            if decision['decision'] == 'accepted':
                decrease = (DeterministicSingleEdge.objective(decision['center'])
                            - DeterministicSingleEdge.objective(decision['trial']))
                self.assertGreaterEqual(decrease, settings['eta']*decision['predicted'])
                self.assertGreaterEqual(decision['lower_decrease'], settings['eta']*decision['predicted'])
        self.assertEqual([row for event, row in events if event == 'incumbent'], result['incumbents'])
        self.assertEqual([row for event, row in events if event == 'decision'], result['decisions'])
        self.assertEqual(sum(row['shots'] for row in result['evaluations']), result['shots'])

    def test_rejection_does_not_spend_an_unaffordable_next_model(self):
        class StationaryQuadratic:
            def sample(self, theta, shots, rng):
                return -.75+.25*theta[0]**2+.5*theta[1]**2, 0.

        settings = {**optimizer_settings(), 'radius': .4, 'min_radius': .1,
                    'max_radius': .4, 'model_shots': 8, 'acceptance_looks': [65536]}
        first_cost = 3*8+2*65536
        next_required = 3*32+2*65536
        result = refine(StationaryQuadratic(), [0., 0.], np.eye(2), settings,
                        11, first_cost+next_required-1, bound='bernstein')
        self.assertEqual(result['stop_reason'], 'model_and_first_look_unaffordable')
        self.assertEqual(result['shots'], first_cost)
        self.assertEqual(result['final_radius'], .2)
        self.assertEqual(len(result['models']), 1)
        self.assertEqual(result['last_allocation']['model_shots_per_point'], 32)
        self.assertEqual(result['decisions'][0]['decision'], 'rejected')

    def test_qaoa_replays_exactly_and_saved_design_reconstructs_the_model(self):
        settings = {**optimizer_settings(), 'model_shots': 32, 'acceptance_looks': [64, 256]}
        circuit = MaxCutQAOA(nx.path_graph(3))
        for bound in ('hoeffding', 'bernstein'):
            result = refine(circuit, [.3, .15], np.eye(2), settings, 923, 1700, bound=bound)
            repeated = refine(circuit, [.3, .15], np.eye(2), settings, 923, 1700, bound=bound)
            with self.subTest(bound=bound):
                for key in ('theta', 'shots', 'calls', 'jobs', 'stop', 'stop_reason',
                            'decisions', 'models', 'incumbents', 'final_radius', 'last_allocation'):
                    self.assertEqual(result[key], repeated[key])
                for model in result['models']:
                    points = np.array(model['design_points'])
                    design = np.c_[np.ones(len(points)), points]
                    evaluations = [row for row in result['evaluations']
                                   if row['stage'] == 'model' and row['iteration'] == model['iteration']]
                    coefficients = np.linalg.lstsq(design, [row['mean'] for row in evaluations],
                                                   rcond=None)[0]
                    assert_allclose(model['coefficients'], coefficients, atol=1e-13)
                    locations = [wrap(np.array(model['center'])+model['radius']*point) for point in points]
                    assert_allclose([row['theta'] for row in evaluations], locations, atol=1e-13)

    def test_fresh_endpoint_stream_and_independently_reconstructed_bernstein_moments(self):
        class RecordedBernoulli:
            def __init__(self):
                self.draws, self.streams = [], []

            def objective(self, theta):
                raise AssertionError('Exact objectives must not enter decisions')

            def sample(self, theta, shots, rng):
                observations = -rng.binomial(1, .5+.1*np.sin(theta[0]), shots).astype(float)
                self.draws.append(observations)
                self.streams.append(rng)
                return observations.mean(), observations.var(ddof=1)/shots

        circuit = RecordedBernoulli()
        settings = {**optimizer_settings(), 'model_shots': 16, 'acceptance_looks': [8, 24]}
        result = refine(circuit, [.3, .15], np.eye(2), settings, 76, 10000, bound='bernstein')
        self.assertEqual(len(result['decisions']), 1)
        self.assertIsNot(circuit.streams[0], circuit.streams[3])
        self.assertTrue(all(stream is circuit.streams[0] for stream in circuit.streams[:3]))
        self.assertTrue(all(stream is circuit.streams[3] for stream in circuit.streams[3:]))
        intervals = result['decisions'][0]['intervals']
        self.assertEqual(len(intervals), 2)
        for index, interval in enumerate(intervals):
            raw = [np.concatenate([circuit.draws[3+2*k+side] for k in range(index+1)])
                   for side in (0, 1)]
            means = [values.mean() for values in raw]
            variances = [values.var(ddof=1) for values in raw]
            count = len(raw[0])
            logarithm = np.log(8*settings['max_trials']*2/settings['alpha'])
            radii = [np.sqrt(2*variance*logarithm/count)+7*logarithm/(3*(count-1))
                     for variance in variances]
            assert_allclose(interval['means'], means, atol=1e-14)
            assert_allclose(interval['sample_variances'], variances, atol=1e-14)
            assert_allclose(interval['radii'], radii, atol=1e-14)
            self.assertAlmostEqual(interval['lower'], means[0]-means[1]-sum(radii))
            self.assertAlmostEqual(interval['upper'], means[0]-means[1]+sum(radii))

    def test_flat_model_is_explicit_and_all_model_spending_is_recorded(self):
        class Flat:
            def sample(self, theta, shots, rng):
                return -.5, 0.

        result = refine(Flat(), [.3, .15], np.eye(2), optimizer_settings(), 7, 100000)
        self.assertEqual(result['stop'], 'resolution_limited')
        self.assertEqual(result['stop_reason'], 'flat_model')
        self.assertEqual(result['decisions'], [])
        self.assertEqual(result['shots'], 384)
        self.assertEqual(result['models'][0]['model_shots'], result['shots'])
        self.assertEqual(len(result['incumbents']), 1)

    def test_invalid_adaptive_schedule_fails_before_any_measurement(self):
        class NeverSample:
            def sample(self, theta, shots, rng):
                raise AssertionError('Invalid settings must fail before sampling')

        for changes in ({'acceptance_looks': []}, {'acceptance_looks': [4, 5]},
                        {'acceptance_looks': [8, 4]}, {'model_shots': 2.5},
                        {'radius': 0.}, {'alpha': 1.}, {'min_radius': .2}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                refine(NeverSample(), [.3, .15], np.eye(2),
                       {**optimizer_settings(), **changes}, 7, 10000)


if __name__ == "__main__":
    unittest.main()
