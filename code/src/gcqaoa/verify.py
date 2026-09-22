"""Independently audit the saved study, without rerunning any optimizer.

Checks source/configuration identity, split disjointness, full experimental
coverage, every shot ledger, reference selection, exact objective diagnostics,
and acceptance decisions reconstructed from the recorded endpoint samples.
No data or reports are written. A failed check returns a nonzero exit status.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

import networkx as nx
import numpy as np

from .qaoa import MaxCutQAOA


from .provenance import CODE_ROOT, CANONICAL_DATA, run_id, verify_source_identity

ROOT = CODE_ROOT
PRIOR_KINDS = {"uniform", "descriptor", "gw", "local", "shuffled"}
ISOTROPIC = {
    "uniform_iso", "random_iso", "descriptor_iso", "gw_iso", "local_iso",
    "cobyla", "spsa",
}


def integer(value, minimum=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def periods(p):
    return np.r_[np.full(p, 2 * np.pi), np.full(p, np.pi / 2)]


def wrap(theta):
    theta = np.asarray(theta, dtype=float)
    period = periods(len(theta) // 2)
    return (theta + period / 2) % period - period / 2


class Audit:
    def __init__(self):
        self.failures = []
        self.checks = 0
        self.objective_cache = {}
        self.circuits = {}
        self.accepted = 0
        self.nonpositive_acceptances = 0
        self.insufficient_true_decreases = 0
        self.reference_failures = 0
        self.models_reconstructed = 0
        self.ambiguous_models = 0

    def check(self, condition, label):
        self.checks += 1
        if not condition:
            self.failures.append(label)
        return bool(condition)

    def close(self, actual, expected, label, atol=2e-9):
        return self.check(
            np.shape(actual) == np.shape(expected)
            and np.allclose(actual, expected, rtol=2e-9, atol=atol), label
        )

    def angles(self, theta, p, label):
        if not self.check(isinstance(theta, list) and len(theta) == 2 * p, label + " length"):
            raise ValueError(label + ": invalid angle dimension")
        if not self.check(all(finite(x) for x in theta), label + " finite"):
            raise ValueError(label + ": nonfinite angles")
        self.close(theta, wrap(theta), label + " canonical periods", atol=2e-12)

    def same_angles(self, actual, expected, label):
        self.close(wrap(np.asarray(actual) - np.asarray(expected)), np.zeros(len(expected)), label)

    def objective(self, graph_id, theta):
        key = (graph_id, tuple(theta))
        if key not in self.objective_cache:
            self.objective_cache[key] = self.circuits[graph_id].objective(theta)
        return self.objective_cache[key]

    def observation(self, record, graph_id, p, label):
        self.check(integer(record["shots"], 2), label + " integer shots >= 2")
        shots = record["shots"]
        mean, variance = record["mean"], record["variance_of_mean"]
        self.check(finite(mean) and -1 - 1e-12 <= mean <= 1e-12, label + " bounded mean")
        self.check(finite(variance) and -1e-15 <= variance <= 1 / (4 * (shots - 1)) + 1e-12,
                   label + " variance of mean")
        # The total observed integer cut is recoverable from a normalized mean.
        total_cut = -mean * shots * self.circuits[graph_id].m
        self.close(total_cut, round(total_cut), label + " integer cut total", atol=2e-7)
        if "theta" in record:
            self.angles(record["theta"], p, label + " theta")
        if "seconds" in record:
            self.check(finite(record["seconds"]) and record["seconds"] >= 0, label + " time")


def graph_from(row):
    graph = nx.Graph()
    graph.add_nodes_from(range(row["n"]))
    graph.add_edges_from(row["edges"])
    return graph


def independent_maxcut(graph):
    nodes = tuple(graph.nodes)
    best = 0
    for bits in itertools.product((False, True), repeat=len(nodes)):
        side = dict(zip(nodes, bits))
        best = max(best, sum(side[u] != side[v] for u, v in graph.edges))
    return best


def audit_instances(audit, data, config):
    expected = []
    metadata = {}
    for split in ("bank", "test"):
        for family in config["families"]:
            for n in config[split + "_sizes"]:
                for rep in range(config[split + "_per_family_size"]):
                    graph_id = f"{split}_{family}_{n}_{rep}"
                    expected.append(graph_id)
                    metadata[graph_id] = (split, family, n)
    audit.check([row["id"] for row in data["instances"]] == expected,
                "complete ordered instance product without duplicate IDs")
    instances = {row["id"]: row for row in data["instances"]}
    graphs = {}
    for graph_id, row in instances.items():
        audit.check((row["split"], row["family"], row["n"]) == metadata[graph_id], graph_id + " metadata")
        audit.check(integer(row["seed"], 1), graph_id + " generation seed")
        edges = row["edges"]
        audit.check(all(len(edge) == 2 and all(integer(v) and v < row["n"] for v in edge)
                        and edge[0] < edge[1] for edge in edges), graph_id + " simple labeled edges")
        audit.check(len({tuple(edge) for edge in edges}) == len(edges), graph_id + " unique edges")
        graph = graph_from(row)
        graphs[graph_id] = graph
        audit.check(nx.is_connected(graph) and nx.number_of_selfloops(graph) == 0, graph_id + " connected simple graph")
        audit.close(row["maxcut"], independent_maxcut(graph), graph_id + " independent exact MaxCut")
        distances = np.asarray(nx.floyd_warshall_numpy(graph), dtype=float)
        audit.close(row["structure"], distances / distances.max(), graph_id + " diameter-normalized distances")
        degrees = np.asarray([degree for _, degree in graph.degree()], dtype=float)
        descriptor = np.r_[degrees.mean(), degrees.std(), np.quantile(degrees, [.25, .5, .75]),
                           np.mean(list(nx.clustering(graph).values()))]
        audit.close(row["descriptor"], descriptor, graph_id + " descriptor")
        audit.circuits[graph_id] = MaxCutQAOA(graph)
    for first, second in itertools.combinations(instances, 2):
        if len(graphs[first]) == len(graphs[second]):
            audit.check(not nx.is_isomorphic(graphs[first], graphs[second]),
                        first + " / " + second + " nonisomorphic and split-disjoint")
    print(f"Instances: {len(instances)} connected graphs; exact cuts and isomorphism disjointness checked.")
    return instances


def audit_references_and_priors(audit, data, config, instances):
    expected_refs = {f"{graph_id}:p{p}" for graph_id in instances for p in config["depths"]}
    audit.check(set(data["references"]) == expected_refs, "complete reference graph/depth product")
    for key, reference in data["references"].items():
        graph_id, depth = key.rsplit(":p", 1)
        p = int(depth)
        starts = reference["starts"]
        audit.check(len(starts) == config["reference_starts"], key + " reference start count")
        for index, start in enumerate(starts):
            label = f"{key} reference start {index}"
            audit.angles(start["theta"], p, label)
            audit.close(start["objective"], audit.objective(graph_id, start["theta"]), label + " exact objective")
            audit.check(integer(start["calls"], 1), label + " exact call count")
            audit.check(isinstance(start["success"], bool), label + " termination status")
            audit.reference_failures += not start["success"]
        best = min(starts, key=lambda row: row["objective"])
        audit.close(reference["objective"], best["objective"], key + " best reference objective")
        audit.same_angles(reference["theta"], best["theta"], key + " selected reference angles")
        audit.check(reference["exact_calls"] == sum(start["calls"] for start in starts), key + " reference call ledger")
        audit.check(finite(reference["seconds"]) and reference["seconds"] >= 0, key + " reference time")
    bank = [row for row in instances.values() if row["split"] == "bank"]
    tests = [row for row in instances.values() if row["split"] == "test"]
    expected_priors = {f"{row['id']}:p{p}" for row in tests for p in config["depths"]}
    audit.check(set(data["priors"]) == expected_priors, "complete test-only prior graph/depth product")
    for key, priors in data["priors"].items():
        graph_id, depth = key.rsplit(":p", 1)
        p = int(depth)
        audit.check(set(priors) == PRIOR_KINDS, key + " complete prior kinds")
        for kind, prior in priors.items():
            label = key + " " + kind + " prior"
            audit.angles(prior["theta"], p, label)
            weights = np.asarray(prior["weights"])
            audit.check(weights.shape == (len(bank),) and np.isfinite(weights).all()
                        and (weights >= 0).all(), label + " finite nonnegative bank weights")
            audit.close(weights.sum(), 1, label + " normalized weights")
            scores = np.asarray(prior["scores"])
            audit.check(scores.shape == (len(bank),) and np.isfinite(scores).all()
                        and (scores >= -1e-12).all(), label + " finite structural scores")
            anchor = prior["anchor"]
            audit.check(integer(anchor) and anchor < len(bank), label + " bank-only anchor index")
            expected = data["references"][f"{bank[anchor]['id']}:p{p}"]["theta"]
            audit.same_angles(prior["theta"], expected, label + " bank-only anchor angles")
            audit.close(prior["initial_objective"], audit.objective(graph_id, prior["theta"]), label + " initial objective")
            shape = np.asarray(prior["shape"])
            audit.check(shape.shape == (2 * p, 2 * p) and np.isfinite(shape).all(), label + " shape dimensions")
            audit.close(shape, shape.T, label + " symmetric shape")
            audit.check(np.linalg.eigvalsh(shape).min() > 0 and np.linalg.cond(shape) <= 2 + 1e-8,
                        label + " bounded positive shape")
            audit.close(np.linalg.det(shape), 1, label + " unit determinant")
        # Verify every saved structural coupling, without calling its solver.
        D = np.asarray(instances[graph_id]["structure"])
        audit.check(len(priors["gw"]["solver"]) == len(bank), key + " coupling count")
        for index, (reference, solver) in enumerate(zip(bank, priors["gw"]["solver"])):
            E = np.asarray(reference["structure"])
            coupling = np.asarray(solver["coupling"])
            label = f"{key} GW coupling {index}"
            audit.check(coupling.shape == (len(D), len(E)) and np.isfinite(coupling).all()
                        and (coupling >= 0).all(), label + " feasibility")
            audit.close(coupling.sum(axis=1), np.full(len(D), 1 / len(D)), label + " row marginals", atol=1e-8)
            audit.close(coupling.sum(axis=0), np.full(len(E), 1 / len(E)), label + " column marginals", atol=1e-8)
            distortion = ((D * D).mean(axis=1)[:, None]
                          + (E * E).mean(axis=1)[None, :] - 2 * D @ coupling @ E.T)
            audit.close(priors["gw"]["scores"][index], max(0.0, float(np.sum(distortion * coupling))), label + " distortion")
    print(f"References: {len(expected_refs)} best-of-{config['reference_starts']} labels; bank-only priors and GW couplings checked.")


def expected_initial(run, data, config):
    method, p = run["method"], run["depth"]
    kind = "gw" if method.startswith("gw") else "descriptor"
    if method in {"uniform_iso", "random_iso"}:
        kind = "uniform"
    elif method == "shuffled_gw":
        kind = "shuffled"
    elif method == "local_iso":
        kind = "local"
    prior = data["priors"][f"{run['graph']}:p{p}"][kind]
    initial, shape = np.asarray(prior["theta"]), np.asarray(prior["shape"])
    if method in ISOTROPIC:
        shape = np.eye(2 * p)
    if method == "random_iso":
        rng = np.random.default_rng(run["seed"] + 88)
        initial = np.r_[rng.uniform(-np.pi, np.pi, p), rng.uniform(-np.pi / 4, np.pi / 4, p)]
    elif method == "bad_prior":
        initial = wrap(initial + np.r_[np.full(p, 1.2), np.full(p, .35)])
    return initial, shape


def model_coordinates(records, center, shape, radius):
    """Recover unit-ball model coordinates, accounting for angle periods.

    Model designs contain the origin or unit vectors. Enumerating possible
    lifts avoids assuming that an ellipsoid fits inside an angle chart.
    Ambiguous lifts are reported instead of silently picking a branch.
    """
    dimension = len(center)
    period = periods(dimension // 2)
    maximum = radius * np.linalg.norm(shape, axis=1)
    points = []
    for record in records:
        delta = wrap(np.asarray(record["theta"]) - center)
        shifts = [range(int(np.ceil((-maximum[j] - delta[j]) / period[j] - 1e-9)),
                        int(np.floor((maximum[j] - delta[j]) / period[j] + 1e-9)) + 1)
                  for j in range(dimension)]
        candidates = []
        for shift in itertools.product(*shifts):
            point = np.linalg.solve(shape, delta + np.asarray(shift) * period) / radius
            norm = np.linalg.norm(point)
            if min(norm, abs(norm - 1)) < 1e-7:
                candidates.append(point)
        if len(candidates) != 1:
            return None
        points.append(candidates[0])
    return np.c_[np.ones(len(points)), np.asarray(points)]


def audit_decisions(audit, run, jobs, initial, shape, config, label):
    settings = config["optimizer"]
    method, p = run["method"], run["depth"]
    expected_incumbents = [(0, wrap(initial))]
    if method in {"cobyla", "spsa"}:
        audit.check(not run["decisions"], label + " baseline has no trust-region decisions")
        best = float("inf")
        for job in jobs:
            allowed = {"cobyla"} if method == "cobyla" else {"spsa-gradient", "spsa-incumbent"}
            audit.check(job[0]["stage"] in allowed, label + " baseline stage")
            audit.check(len(job) == (2 if job[0]["stage"] == "spsa-gradient" else 1), label + " baseline batch size")
            audit.check(all(record["shots"] == settings["baseline_shots"] for record in job), label + " baseline shot allocation")
            if job[0]["stage"] in {"cobyla", "spsa-incumbent"} and job[0]["mean"] < best:
                best = job[0]["mean"]
                expected_incumbents.append((job[0]["cumulative_shots"], job[0]["theta"]))
        return expected_incumbents
    looks = [settings["acceptance_looks"][-1]] if method == "fixed_shots" else settings["acceptance_looks"]
    logterm = np.log(4 * settings["max_trials"] * len(looks) / settings["alpha"])
    center, radius, position = wrap(initial), settings["radius"], 0
    audit.check(len(run["decisions"]) <= settings["max_trials"], label + " prespecified trial horizon")
    for index, decision in enumerate(run["decisions"]):
        tag = f"{label} decision {index}"
        audit.check(decision["iteration"] == index, tag + " iteration")
        audit.close(decision["radius"], radius, tag + " radius schedule")
        audit.same_angles(decision["center"], center, tag + " center")
        audit.angles(decision["trial"], p, tag + " trial")
        prediction = decision["predicted"]
        audit.check(finite(prediction) and prediction > 0, tag + " positive predicted decrease")
        audit.check(finite(decision["condition"]) and decision["condition"] >= 1 - 1e-9, tag + " finite design condition")
        model = jobs[position]
        position += 1
        audit.check(len(model) == 2 * p + 1 and all(record["stage"] == "model" for record in model), tag + " model batch")
        audit.check(all(record["shots"] == settings["model_shots"] for record in model), tag + " model allocation")
        design = model_coordinates(model, center, shape, radius)
        if design is None:
            audit.ambiguous_models += 1
        else:
            audit.models_reconstructed += 1
            audit.check(np.linalg.matrix_rank(design) == 2 * p + 1, tag + " full-rank model")
            coefficients = np.linalg.lstsq(design, [record["mean"] for record in model], rcond=None)[0]
            gradient = coefficients[1:]
            audit.close(prediction, np.linalg.norm(gradient), tag + " independent least-squares prediction", atol=1e-7)
            audit.close(decision["condition"], np.linalg.cond(design), tag + " design condition", atol=1e-6)
            trial = wrap(center - radius * shape @ gradient / np.linalg.norm(gradient))
            audit.same_angles(decision["trial"], trial, tag + " model trial")
        sums, cumulative = np.zeros(2), 0
        lower = upper = None
        expected_decision = "unresolved"
        look_index = 0
        last_shots = model[-1]["cumulative_shots"]
        while position < len(jobs) and jobs[position][-1]["cumulative_shots"] <= decision["shots"]:
            job = jobs[position]
            position += 1
            audit.check(expected_decision == "unresolved", tag + " stop at first resolved look")
            audit.check(len(job) == 2 and all(record["stage"] == "acceptance" for record in job), tag + " fresh endpoint pair")
            expected_count = looks[look_index] - cumulative
            audit.check(all(record["shots"] == expected_count for record in job), tag + " prespecified cumulative look")
            audit.same_angles(job[0]["theta"], center, tag + " current endpoint")
            audit.same_angles(job[1]["theta"], decision["trial"], tag + " trial endpoint")
            sums += expected_count * np.asarray([record["mean"] for record in job])
            cumulative = looks[look_index]
            look_index += 1
            decrease = (sums[0] - sums[1]) / cumulative
            half_width = 2 * np.sqrt(logterm / (2 * cumulative))
            lower, upper = decrease - half_width, decrease + half_width
            if lower >= settings["eta"] * prediction:
                expected_decision = "accepted"
            elif upper < settings["eta"] * prediction:
                expected_decision = "rejected"
            last_shots = job[-1]["cumulative_shots"]
        audit.check(decision["shots"] == last_shots, tag + " decision shot boundary")
        audit.check(decision["acceptance_shots_per_point"] == cumulative, tag + " cumulative endpoint allocation")
        if lower is None:
            audit.check(decision["lower_decrease"] is None and decision["upper_decrease"] is None, tag + " unobserved interval")
        else:
            audit.close(decision["lower_decrease"], lower, tag + " reconstructed confidence lower bound")
            audit.close(decision["upper_decrease"], upper, tag + " reconstructed confidence upper bound")
        audit.check(decision["decision"] == expected_decision, tag + " confidence decision")
        if expected_decision == "accepted":
            audit.accepted += 1
            true_decrease = (audit.objective(run["graph"], center)
                             - audit.objective(run["graph"], decision["trial"]))
            audit.nonpositive_acceptances += true_decrease <= 0
            audit.insufficient_true_decreases += true_decrease + 2e-12 < settings["eta"] * prediction
            center = np.asarray(decision["trial"])
            expected_incumbents.append((decision["shots"], center))
            if method != "fixed_radius":
                radius = min(settings["max_radius"], 1.5 * radius)
        elif method != "fixed_radius":
            radius = max(settings["min_radius"], .5 * radius)
    # A final fitted flat model may terminate before a decision is logged.
    remaining = jobs[position:]
    audit.check(not remaining or (run["stop"] == "flat_model" and len(remaining) == 1
                                  and all(row["stage"] == "model" for row in remaining[0])),
                label + " every oracle batch assigned to a decision or flat-model termination")
    return expected_incumbents


def audit_run(audit, run, data, config, instances, test_indices):
    graph_id, p, method = run["graph"], run["depth"], run["method"]
    label = f"{graph_id}/p{p}/{method}/r{run['replicate']}/b{run['budget']}"
    seed = config["seed"] + 100000 * test_indices[graph_id] + 1000 * p + run["replicate"]
    audit.check(run["seed"] == seed, label + " frozen seed")
    initial, shape = expected_initial(run, data, config)
    audit.angles(run["theta"], p, label + " final theta")
    audit.close(run["initial_objective"], audit.objective(graph_id, initial), label + " exact initial objective")
    exact = audit.objective(graph_id, run["theta"])
    audit.close(run["objective"], exact, label + " exact final objective")
    reference = data["references"][f"{graph_id}:p{p}"]["objective"]
    audit.close(run["reference_objective"], reference, label + " reference identity")
    audit.close(run["signed_reference_gap"], exact - reference, label + " signed reference gap")
    ratio = -exact * audit.circuits[graph_id].m / instances[graph_id]["maxcut"]
    audit.close(run["expected_ratio"], ratio, label + " expected approximation ratio")
    audit.check(-1e-10 <= ratio <= 1 + 1e-10, label + " approximation ratio bounds")
    audit.check(finite(run["seconds"]) and run["seconds"] >= 0, label + " elapsed time")
    logs = run["evaluations"]
    cumulative, jobs = 0, []
    for index, record in enumerate(logs):
        tag = f"{label} call {index + 1}"
        audit.observation(record, graph_id, p, tag)
        cumulative += record["shots"]
        audit.check(record["call"] == index + 1, tag + " sequential call ID")
        audit.check(record["cumulative_shots"] == cumulative, tag + " cumulative shot ledger")
        if not jobs or record["job"] != jobs[-1][0]["job"]:
            audit.check(record["job"] == len(jobs) + 1, tag + " sequential job ID")
            jobs.append([])
        jobs[-1].append(record)
        audit.check(record["stage"] == jobs[-1][0]["stage"], tag + " common batch stage")
    audit.check(run["shots"] == cumulative and integer(run["shots"]) and cumulative <= run["budget"], label + " budget ledger")
    audit.check(run["calls"] == len(logs), label + " oracle call ledger")
    audit.check(run["jobs"] == len(jobs), label + " submitted job ledger")
    expected_incumbents = audit_decisions(audit, run, jobs, initial, shape, config, label)
    incumbents = run["incumbents"]
    audit.check(len(incumbents) == len(expected_incumbents), label + " incumbent update count")
    for index, (incumbent, (shots, theta)) in enumerate(zip(incumbents, expected_incumbents)):
        tag = f"{label} incumbent {index}"
        audit.check(incumbent["shots"] == shots, tag + " update shot boundary")
        audit.angles(incumbent["theta"], p, tag + " theta")
        audit.same_angles(incumbent["theta"], theta, tag + " update parameters")
        audit.close(incumbent["objective"], audit.objective(graph_id, incumbent["theta"]), tag + " exact diagnostic")
    audit.same_angles(run["theta"], incumbents[-1]["theta"], label + " returned incumbent")
    hits = [incumbent["shots"] for incumbent in incumbents
            if incumbent["objective"] <= reference + config["target_gap"]]
    audit.check(run["hitting_shots"] == (min(hits) if hits else None), label + " incumbent hitting time")
    validation = run["validation"]
    audit.observation(validation, graph_id, p, label + " independent validation")
    audit.check(validation["shots"] == config["validation_shots"]
                and validation["calls"] == 1 and validation["jobs"] == 1, label + " separate validation ledger")
    # Reproduce only the held-out measurement; no optimizer is called.
    validation_rng = np.random.default_rng(seed + run["budget"] + 9000000)
    validation_mean, validation_variance = audit.circuits[graph_id].sample(
        run["theta"], config["validation_shots"], validation_rng)
    audit.close(validation["mean"], validation_mean, label + " independent validation seed")
    audit.close(validation["variance_of_mean"], validation_variance, label + " independent validation variance")


def verify(path):
    started = time.perf_counter()
    audit = Audit()
    with gzip.open(path, "rt") as stream:
        data = json.load(stream, parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Nonfinite JSON value: " + value)))
    provenance = verify_source_identity(data)
    config = provenance["config"]
    audit.check(data["config"] == config, "saved and verified configuration identity")
    audit.check(bool(provenance["verified_sha256"]), "complete frozen source hash set")
    for name, expected in provenance["verified_sha256"].items():
        audit.check(data["source_sha256"].get(name) == expected, name + " source SHA256")
    audit.check(all(isinstance(data["environment"].get(name), str) and data["environment"][name]
                    for name in ("python", "numpy", "scipy", "networkx", "machine", "system")),
                "complete execution environment metadata")
    audit.check(finite(data["seconds"]) and data["seconds"] >= 0, "total elapsed time")
    print(f"Provenance: configuration and frozen sources verified against {provenance['mode']}; environment metadata checked.")
    identity = {key: data[key] for key in ("config_path", "source_sha256", "started_utc",
                                         "source_commit", "source_dirty", "working_tree_dirty")}
    expected_identity = "study-" + hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]
    audit.check(data.get("execution_run_id") == expected_identity, "derivable execution ID")
    audit.check(data.get("source_commit") is None or len(data["source_commit"]) == 40, "source commit metadata")
    audit.check(data.get("source_dirty") in (True, False, None), "source dirty-state metadata")
    print(f"Source commit: {data['source_commit']}; frozen source dirty: {data['source_dirty']}.")
    instances = audit_instances(audit, data, config)
    audit_references_and_priors(audit, data, config, instances)
    tests = [row for row in instances.values() if row["split"] == "test"]
    test_indices = {row["id"]: index for index, row in enumerate(tests)}
    expected = set(itertools.product(test_indices, config["depths"], range(config["repetitions"]),
                                     config["budgets"], config["methods"]))
    actual = Counter((run["graph"], run["depth"], run["replicate"], run["budget"], run["method"])
                     for run in data["runs"])
    audit.check(set(actual) == expected and all(count == 1 for count in actual.values()),
                "complete graph/depth/replicate/budget/method product exactly once")
    for index, run in enumerate(data["runs"]):
        try:
            audit.check(run.get("run_id") == run_id(data["execution_run_id"], run), f"run {index} identity")
            audit_run(audit, run, data, config, instances, test_indices)
        except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError, np.linalg.LinAlgError) as error:
            audit.check(False, f"run {index} malformed or unreplayable: {error}")
        if (index + 1) % 500 == 0:
            print(f"Audited {index + 1}/{len(data['runs'])} runs.", flush=True)
    print(f"Runs: {len(data['runs'])}/{len(expected)}; all raw shot ledgers, final diagnostics, and hitting times checked.")
    print(f"Models: {audit.models_reconstructed} independently reconstructed; {audit.ambiguous_models} ambiguous angle lifts.")
    print(f"Acceptance: {audit.accepted} accepted; {audit.nonpositive_acceptances} nonpositive exact decreases; "
          f"{audit.insufficient_true_decreases} below the exact eta threshold.")
    print(f"Reference diagnostics: {audit.reference_failures} starts reported unsuccessful termination; "
          "saved labels are best-known values, not certified QAOA optima.")
    print(f"Exact objective diagnostics: {len(audit.objective_cache)} distinct parameter vectors. "
          "Acceptance diagnostics do not certify stationarity.")
    if audit.failures:
        print(f"FAILED: {len(audit.failures)} of {audit.checks} checks failed.", file=sys.stderr)
        for failure in audit.failures[:40]:
            print("  " + failure, file=sys.stderr)
        if len(audit.failures) > 40:
            print(f"  ... {len(audit.failures) - 40} further failures.", file=sys.stderr)
        return 1
    print(f"PASS: {audit.checks} checks in {time.perf_counter() - started:.1f} seconds.")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=CANONICAL_DATA,
                        help="saved compressed JSON study (default: code/results/study.json.gz)")
    arguments = parser.parse_args(argv)
    if not arguments.data.is_file():
        parser.error(f"study data does not exist: {arguments.data}; run experiment.py first")
    try:
        return verify(arguments.data)
    except (OSError, KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as error:
        print(f"FAILED: cannot audit study: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
