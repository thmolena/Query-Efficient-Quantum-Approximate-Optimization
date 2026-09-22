"""Exact, unweighted MaxCut QAOA and finite-shot objective observations.

The objective is minus the expected cut divided by the number of edges.
Angles are ordered ``(gamma_1, ..., gamma_p, beta_1, ..., beta_p)``. Each
layer applies ``exp(-i gamma C)`` followed by ``exp(-i beta sum_j X_j)``
to an initial uniform superposition. Node ``nodes[j]`` is bit ``j`` of a
computational-basis index. This convention also fixes the returned cuts.

The state-vector implementation has exponential storage and is intended
for small, reproducible numerical experiments, not large-graph simulation.
"""

from __future__ import annotations

from numbers import Integral

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_graph(graph: nx.Graph) -> tuple:
    """Validate a nonempty simple, undirected, unweighted MaxCut instance."""
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a networkx Graph")
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("MaxCut requires a simple undirected graph")
    if graph.number_of_edges() == 0:
        raise ValueError("graph must contain at least one edge")
    if nx.number_of_selfloops(graph):
        raise ValueError("self-loops are not supported")
    if any(data.get("weight", 1) != 1 for _, _, data in graph.edges(data=True)):
        raise ValueError("only unweighted graphs (edge weight one) are supported")
    return tuple(graph.nodes)


def _enumerated_cuts(
    graph: nx.Graph, max_qubits: int = 24
) -> tuple[tuple, NDArray[np.int64]]:
    nodes = _validate_graph(graph)
    if not isinstance(max_qubits, Integral) or isinstance(max_qubits, bool):
        raise ValueError("max_qubits must be a positive integer")
    if max_qubits < 1 or len(nodes) > max_qubits:
        raise ValueError(f"exact enumeration is limited to {max_qubits} qubits")
    node_index = {node: j for j, node in enumerate(nodes)}
    basis = np.arange(1 << len(nodes), dtype=np.uint64)
    cuts = np.zeros(basis.size, dtype=np.int64)
    for u, v in graph.edges:
        cuts += (((basis >> node_index[u]) ^ (basis >> node_index[v])) & 1).astype(
            np.int64
        )
    return nodes, cuts


class MaxCutQAOA:
    """Simulate standard QAOA for a fixed simple, unweighted graph.

    Disconnected graphs are valid QAOA instances. The separate structural
    distance helper requires connected graphs. The graph is copied at
    construction so later changes to the caller's graph cannot alter the
    simulated instance.
    """

    def __init__(self, graph: nx.Graph, *, max_qubits: int = 24) -> None:
        self.nodes, self.cut_values = _enumerated_cuts(graph, max_qubits)
        self.graph = nx.freeze(graph.copy())
        self.n = len(self.nodes)
        self.m = self.graph.number_of_edges()
        self.normalized_costs = -self.cut_values.astype(float) / self.m
        self.cut_values.flags.writeable = False
        self.normalized_costs.flags.writeable = False

    def statevector(self, theta: ArrayLike) -> NDArray[np.complex128]:
        """Return the final state in computational-basis index order."""
        angles = np.asarray(theta, dtype=float)
        if angles.ndim != 1 or angles.size == 0 or angles.size % 2:
            raise ValueError("theta must be a nonempty vector of 2p angles")
        if not np.all(np.isfinite(angles)):
            raise ValueError("theta must contain finite angles")
        depth = angles.size // 2
        state = np.full(1 << self.n, 2.0 ** (-self.n / 2), dtype=np.complex128)
        for gamma, beta in zip(angles[:depth], angles[depth:]):
            state *= np.exp(-1j * gamma * self.cut_values)
            cosine, sine = np.cos(beta), np.sin(beta)
            for qubit in range(self.n):
                blocks = state.reshape(-1, 2, 1 << qubit)
                zero = blocks[:, 0, :].copy()
                one = blocks[:, 1, :].copy()
                blocks[:, 0, :] = cosine * zero - 1j * sine * one
                blocks[:, 1, :] = cosine * one - 1j * sine * zero
        return state

    def probabilities(self, theta: ArrayLike) -> NDArray[np.float64]:
        """Return computational-basis probabilities, correcting roundoff."""
        probabilities = np.abs(self.statevector(theta)) ** 2
        return probabilities / probabilities.sum()

    def objective(self, theta: ArrayLike) -> float:
        """Return the exact objective, minus expected cut divided by edges."""
        return float(self.normalized_costs @ self.probabilities(theta))

    def distribution(
        self, theta: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return objective outcomes and their aggregated probabilities.

        Outcomes are sorted from -1 to 0, with zero-probability outcomes
        retained. A shot samples one of these values, not the expectation.
        """
        weights = np.bincount(
            self.cut_values, weights=self.probabilities(theta), minlength=self.m + 1
        )
        outcomes = -np.arange(self.m, -1, -1, dtype=float) / self.m
        return outcomes, weights[::-1].copy()

    def sample(
        self, theta: ArrayLike, shots: int, rng: np.random.Generator
    ) -> tuple[float, float]:
        """Return a shot mean and an unbiased estimated variance of that mean.

        At least two shots are needed to estimate the variance. The second
        return value is the sample variance divided by ``shots``, not the
        variance of individual outcomes. Randomness is supplied explicitly
        by the caller; this method never uses NumPy's global random state.
        """
        if not isinstance(shots, Integral) or isinstance(shots, bool) or shots < 2:
            raise ValueError("shots must be an integer of at least two")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        outcomes, probabilities = self.distribution(theta)
        counts = rng.multinomial(shots, probabilities)
        mean = float(counts @ outcomes / shots)
        centered_sum = float(counts @ ((outcomes - mean) ** 2))
        return mean, centered_sum / (shots * (shots - 1))


def exact_maxcut(
    graph: nx.Graph, *, max_qubits: int = 24
) -> tuple[int, tuple[tuple, tuple]]:
    """Enumerate all bitstrings and return a maximum cut and its partition.

    Equal maxima are resolved by choosing the first computational-basis
    index. Original graph node labels are preserved in the two partitions.
    """
    nodes, cuts = _enumerated_cuts(graph, max_qubits)
    best = int(np.argmax(cuts))
    zero = tuple(node for j, node in enumerate(nodes) if not ((best >> j) & 1))
    one = tuple(node for j, node in enumerate(nodes) if (best >> j) & 1)
    return int(cuts[best]), (zero, one)


def normalized_shortest_path_matrix(graph: nx.Graph) -> NDArray[np.float64]:
    """Return unweighted shortest-path distances divided by graph diameter.

    Rows and columns use graph node insertion order, as in ``MaxCutQAOA``.
    The matrix is symmetric, has a zero diagonal and entries in [0, 1].
    Disconnected and edgeless graphs are rejected because this normalization
    would have an infinite or zero denominator.
    """
    nodes = _validate_graph(graph)
    if not nx.is_connected(graph):
        raise ValueError("normalized shortest paths require a connected graph")
    node_index = {node: j for j, node in enumerate(nodes)}
    distances = np.zeros((len(nodes), len(nodes)), dtype=float)
    for source, lengths in nx.all_pairs_shortest_path_length(graph):
        for target, length in lengths.items():
            distances[node_index[source], node_index[target]] = length
    return distances / distances.max()


def _edge_neighborhoods(graph: nx.Graph, p: int):
    """Yield induced radius-p neighborhoods with unordered root endpoints."""
    _validate_graph(graph)
    if not isinstance(p, Integral) or isinstance(p, bool) or p < 0:
        raise ValueError("p must be a nonnegative integer")
    for u, v in graph.edges:
        nodes = set(nx.single_source_shortest_path_length(graph, u, cutoff=p))
        nodes.update(nx.single_source_shortest_path_length(graph, v, cutoff=p))
        neighborhood = graph.subgraph(nodes).copy()
        nx.set_node_attributes(neighborhood, {node: node in (u, v) for node in nodes}, "root")
        yield neighborhood


def _rooted_isomorphic(first: nx.Graph, second: nx.Graph) -> bool:
    # These invariants only reject candidates. Equivalence always requires an
    # exact isomorphism test, including the unordered two-endpoint root marks.
    if len(first) != len(second) or first.number_of_edges() != second.number_of_edges():
        return False
    first_degrees = sorted((data["root"], first.degree(node)) for node, data in first.nodes(data=True))
    second_degrees = sorted((data["root"], second.degree(node)) for node, data in second.nodes(data=True))
    if first_degrees != second_degrees:
        return False
    return nx.is_isomorphic(
        first, second, node_match=nx.algorithms.isomorphism.categorical_node_match("root", False)
    )


def local_signature(graph: nx.Graph, p: int) -> tuple[tuple[nx.Graph, int], ...]:
    """Count exact rooted edge-neighborhood isomorphism classes.

    Each returned pair contains a representative induced radius-p graph and
    its integer multiplicity among the original graph's edges. Both endpoints
    of the distinguished edge have node attribute ``root=True``; all other
    nodes have ``root=False``. Endpoint order is immaterial.

    Representatives retain arbitrary original node labels and are not
    canonical encodings. Compare two graphs with :func:`local_tv`, which
    aligns these classes by exact rooted isomorphism, rather than comparing
    representative labels or hashes. Worst-case classification cost includes
    quadratically many graph-isomorphism tests; this implementation targets
    the small graphs used by the exact simulator.
    """
    representatives = []
    counts = []
    for neighborhood in _edge_neighborhoods(graph, p):
        for index, representative in enumerate(representatives):
            if _rooted_isomorphic(neighborhood, representative):
                counts[index] += 1
                break
        else:
            representatives.append(nx.freeze(neighborhood))
            counts.append(1)
    return tuple(zip(representatives, counts))


def local_tv(graph1: nx.Graph, graph2: nx.Graph, p: int) -> float:
    """Return total variation between exact depth-p edge-type frequencies.

    For the standard unweighted MaxCut circuit at depth p, each normalized
    edge contribution lies in [0, 1] and depends only on this rooted induced
    neighborhood. Consequently, at every shared angle vector,
    ``abs(f_graph1(theta) - f_graph2(theta)) <= local_tv(graph1, graph2, p)``.
    The bound is a sufficient structural comparator and may be conservative.
    Computing it uses graph operations only and consumes no quantum shots.
    """
    signature1 = local_signature(graph1, p)
    signature2 = local_signature(graph2, p)
    representatives = [representative for representative, _ in signature1]
    first = [count / graph1.number_of_edges() for _, count in signature1]
    second = [0.0] * len(representatives)
    for neighborhood, count in signature2:
        mass = count / graph2.number_of_edges()
        for index, representative in enumerate(representatives):
            if _rooted_isomorphic(neighborhood, representative):
                second[index] += mass
                break
        else:
            representatives.append(neighborhood)
            first.append(0.0)
            second.append(mass)
    return float(np.sum(np.abs(np.asarray(first) - np.asarray(second))) / 2)
