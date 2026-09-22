"""Independent dense-operator and sampling checks for the QAOA simulator."""

import itertools
import unittest

import networkx as nx
import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import expm

from gcqaoa.qaoa import (
    MaxCutQAOA,
    exact_maxcut,
    local_signature,
    local_tv,
    normalized_shortest_path_matrix,
)


def dense_reference(graph, theta):
    """Build full Pauli operators without calling simulator internals."""
    nodes = tuple(graph.nodes)
    n = len(nodes)
    identity = np.eye(2)
    pauli_x = np.array([[0.0, 1.0], [1.0, 0.0]])
    pauli_z = np.diag([1.0, -1.0])

    def on_qubit(operator, qubit):
        result = np.ones((1, 1))
        for j in reversed(range(n)):
            result = np.kron(result, operator if j == qubit else identity)
        return result

    cost = np.zeros((2**n, 2**n))
    for u, v in graph.edges:
        cost += (
            np.eye(2**n)
            - on_qubit(pauli_z, nodes.index(u)) @ on_qubit(pauli_z, nodes.index(v))
        ) / 2
    mixer = sum(on_qubit(pauli_x, j) for j in range(n))
    state = np.ones(2**n, dtype=complex) / np.sqrt(2**n)
    p = len(theta) // 2
    for layer in range(p):
        state = expm(-1j * theta[layer] * cost) @ state
        state = expm(-1j * theta[p + layer] * mixer) @ state
    objective = -float(np.real(np.vdot(state, cost @ state))) / graph.number_of_edges()
    return state, objective


class QAOATests(unittest.TestCase):
    def test_dense_operator_reference_depths_one_and_two(self):
        for graph in (nx.path_graph(3), nx.cycle_graph(4), nx.complete_graph(3)):
            simulator = MaxCutQAOA(graph)
            for theta in ([0.37, 0.19], [0.17, -0.61, 0.39, 0.24]):
                with self.subTest(edges=tuple(graph.edges), theta=theta):
                    expected_state, expected_objective = dense_reference(graph, theta)
                    actual_state = simulator.statevector(theta)
                    assert_allclose(actual_state, expected_state, atol=2e-14)
                    self.assertAlmostEqual(float(np.linalg.norm(actual_state)), 1.0, 13)
                    self.assertAlmostEqual(simulator.objective(theta), expected_objective, 13)

    def test_single_edge_analytic_sign_and_values(self):
        simulator = MaxCutQAOA(nx.path_graph(2))
        for gamma, beta in itertools.product([0, 0.4, np.pi / 2], [0, 0.2, np.pi / 8]):
            expected = -(1 + np.sin(gamma) * np.sin(4 * beta)) / 2
            self.assertAlmostEqual(simulator.objective([gamma, beta]), expected, 13)
        self.assertAlmostEqual(simulator.objective([np.pi / 2, np.pi / 8]), -1, 13)
        self.assertAlmostEqual(simulator.objective([-np.pi / 2, np.pi / 8]), 0, 13)

    def test_cut_bit_order_and_arbitrary_node_labels(self):
        graph = nx.Graph()
        graph.add_nodes_from(["a", "b", "c"])
        graph.add_edge("a", "c")
        simulator = MaxCutQAOA(graph)
        assert_allclose(simulator.cut_values, [0, 1, 0, 1, 1, 0, 1, 0])
        graph.add_edge("a", "b")
        self.assertEqual(simulator.m, 1)

    def test_distribution_matches_exact_expectation(self):
        simulator = MaxCutQAOA(nx.cycle_graph(5))
        theta = [0.7, -0.2, 0.25, 0.4]
        outcomes, probabilities = simulator.distribution(theta)
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, 14)
        self.assertTrue(np.all(probabilities >= 0))
        self.assertAlmostEqual(float(outcomes @ probabilities), simulator.objective(theta), 14)
        self.assertEqual(probabilities[0], 0.0)  # An odd cycle cannot cut every edge.

    def test_sampling_reproducibility_and_variance_estimate(self):
        simulator = MaxCutQAOA(nx.path_graph(2))
        theta = [0.4, 0.2]
        shots = 128
        rng = np.random.default_rng(78123)
        samples = np.array([simulator.sample(theta, shots, rng) for _ in range(2500)])
        exact_mean = simulator.objective(theta)
        success_probability = -exact_mean
        exact_variance = success_probability * (1 - success_probability) / shots
        self.assertLess(abs(samples[:, 0].mean() - exact_mean), 5 * np.sqrt(exact_variance / len(samples)))
        self.assertLess(abs(samples[:, 1].mean() - exact_variance), 0.05 * exact_variance)
        self.assertLess(abs(samples[:, 0].var(ddof=1) - exact_variance), 0.12 * exact_variance)
        self.assertEqual(
            simulator.sample(theta, shots, np.random.default_rng(12)),
            simulator.sample(theta, shots, np.random.default_rng(12)),
        )

    def test_exact_maxcut_against_independent_partitions(self):
        for graph in (nx.path_graph(4), nx.cycle_graph(5), nx.complete_graph(5)):
            nodes = tuple(graph.nodes)
            values = []
            for bits in itertools.product((0, 1), repeat=len(nodes)):
                assignment = dict(zip(nodes, bits))
                values.append(sum(assignment[u] != assignment[v] for u, v in graph.edges))
            value, (left, right) = exact_maxcut(graph)
            self.assertEqual(value, max(values))
            self.assertEqual(set(left) | set(right), set(nodes))
            self.assertFalse(set(left) & set(right))
            self.assertEqual(nx.cut_size(graph, left, right), value)

    def test_shortest_paths_and_disconnected_policy(self):
        assert_allclose(
            normalized_shortest_path_matrix(nx.path_graph(3)),
            [[0, 0.5, 1], [0.5, 0, 0.5], [1, 0.5, 0]],
        )
        graph = nx.disjoint_union(nx.path_graph(2), nx.path_graph(2))
        self.assertAlmostEqual(MaxCutQAOA(graph).objective([0, 0]), -0.5)
        with self.assertRaisesRegex(ValueError, "connected"):
            normalized_shortest_path_matrix(graph)

    def test_input_validation(self):
        for graph in (nx.Graph(), nx.empty_graph(3), nx.DiGraph([(0, 1)]), nx.MultiGraph([(0, 1)])):
            with self.subTest(graph=graph), self.assertRaises(ValueError):
                MaxCutQAOA(graph)
        with self.assertRaisesRegex(ValueError, "self-loops"):
            MaxCutQAOA(nx.Graph([(0, 0)]))
        graph = nx.Graph()
        graph.add_edge(0, 1, weight=2)
        with self.assertRaisesRegex(ValueError, "unweighted"):
            MaxCutQAOA(graph)
        simulator = MaxCutQAOA(nx.path_graph(2))
        for theta in ([], [0], [[0, 0]], [np.nan, 0], [np.inf, 0]):
            with self.subTest(theta=theta), self.assertRaises(ValueError):
                simulator.objective(theta)
        for shots in (0, 1, 1.5, True):
            with self.subTest(shots=shots), self.assertRaises(ValueError):
                simulator.sample([0, 0], shots, np.random.default_rng(0))
        with self.assertRaises(TypeError):
            simulator.sample([0, 0], 2, None)

    def test_local_types_ignore_labels_and_root_endpoint_order(self):
        graph = nx.path_graph(6)
        relabeled = nx.relabel_nodes(graph, {j: f"node-{5-j}" for j in graph})
        relabeled = nx.Graph(list(reversed(list(relabeled.edges))))
        for p in (0, 1, 2):
            self.assertEqual(local_tv(graph, relabeled, p), 0.0)
            signature = local_signature(graph, p)
            self.assertEqual(sum(count for _, count in signature), graph.number_of_edges())
            self.assertTrue(all(sum(nx.get_node_attributes(rep, "root").values()) == 2 for rep, _ in signature))

    def test_local_types_identify_shared_motifs_exactly(self):
        # At p=1, every edge of either cycle has the same rooted four-node path.
        self.assertEqual(local_tv(nx.cycle_graph(6), nx.cycle_graph(8), 1), 0.0)
        self.assertEqual(local_tv(nx.cycle_graph(6), nx.cycle_graph(8), 2), 1.0)
        # Root-edge position separates internal and terminal path edges.
        self.assertAlmostEqual(local_tv(nx.path_graph(4), nx.path_graph(5), 1), 1 / 6)
        self.assertEqual(local_tv(nx.complete_graph(3), nx.path_graph(3), 0), 0.0)

    def test_local_tv_bounds_exact_objective_discrepancies(self):
        graphs = [
            nx.path_graph(4), nx.path_graph(5), nx.cycle_graph(4),
            nx.cycle_graph(6), nx.cycle_graph(8), nx.complete_graph(4),
            nx.star_graph(4),
        ]
        simulators = [MaxCutQAOA(graph) for graph in graphs]
        rng = np.random.default_rng(173)
        for p in (1, 2):
            angles = rng.uniform(-np.pi, np.pi, size=(12, 2 * p))
            objective_values = [[sim.objective(theta) for theta in angles] for sim in simulators]
            for i, j in itertools.combinations(range(len(graphs)), 2):
                bound = local_tv(graphs[i], graphs[j], p)
                self.assertAlmostEqual(bound, local_tv(graphs[j], graphs[i], p))
                discrepancy = np.max(np.abs(np.asarray(objective_values[i]) - objective_values[j]))
                self.assertLessEqual(discrepancy, bound + 2e-14)

    def test_local_signature_input_validation(self):
        graph = nx.path_graph(3)
        for p in (-1, 0.5, True):
            with self.subTest(p=p), self.assertRaisesRegex(ValueError, "nonnegative integer"):
                local_signature(graph, p)
        with self.assertRaisesRegex(ValueError, "at least one edge"):
            local_tv(graph, nx.empty_graph(2), 1)


if __name__ == "__main__":
    unittest.main()
