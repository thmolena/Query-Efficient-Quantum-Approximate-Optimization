"""Independent-bank follow-up with an auditable variance-aware comparator.

The empirical Bernstein interval is Theorem 4 of Maurer and Pontil (2009),
arXiv:0907.3740, applied to both tails at both endpoints and a declared finite
set of trials/looks. This is an independent comparator, not an implementation
of a separately published trust-region algorithm. Exact simulator values are
recorded only for offline references and diagnostics, never optimizer choices
or hyperparameter selection. The follow-up was designed after the first study.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import networkx as nx
import numpy as np
import scipy
from scipy.optimize import minimize

from .experiment import graph_from, make_graph
from .provenance import CODE_ROOT, NUMERICAL_SOURCES, verify_frozen_sources
from .qaoa import MaxCutQAOA, exact_maxcut
from .records import environment_details
from .search import (BudgetExhausted, ShotOracle, descriptor, design_points,
                     fit_model, graph_prior, optimize, wrap)

FOLLOWUP_CONFIG = CODE_ROOT / "configs" / "followup.json"
FOLLOWUP_DATA = CODE_ROOT / "results" / "followup.json.gz"
REFINEMENT_DATA = CODE_ROOT / "results" / "refinement.json.gz"
FROZEN_SOURCES = (*NUMERICAL_SOURCES, "src/gcqaoa/followup.py", "scripts/run_followup.py",
                  "configs/followup.json")


def frozen_sources():
    """Include the active configuration when running a local protocol variant."""
    config_path = FOLLOWUP_CONFIG.resolve().relative_to(CODE_ROOT).as_posix()
    return (*FROZEN_SOURCES[:-1], config_path)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def content_hash(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def pool_moments(batches):
    """Recover pooled mean and unbiased variance from independent batch records.

    A record stores sample variance divided by its batch size. The within-batch
    sum of squares is therefore n*(n-1)*variance_of_mean. The between-batch
    correction is essential; averaging variance-of-mean fields is incorrect.
    """
    count = 0
    mean = 0.0
    m2 = 0.0
    for batch in batches:
        n = batch["shots"]
        value = batch["mean"]
        variance = batch["variance_of_mean"]
        if (not isinstance(n, int) or isinstance(n, bool) or n < 2 or
                not np.isfinite([value, variance]).all() or variance < 0):
            raise ValueError("Invalid batch moments")
        delta = value - mean
        m2 += n * (n - 1) * variance + delta * delta * count * n / (count + n)
        mean += delta * n / (count + n)
        count += n
    if count < 2:
        raise ValueError("At least two pooled observations are required")
    return count, float(mean), float(max(0.0, m2 / (count - 1)))


def bernstein_radius(count, sample_variance, alpha, horizon, looks):
    """Two-sided endpoint radius with total run-wise failure at most alpha.

    Each one-sided event receives delta=alpha/(4*horizon*looks). Theorem 4
    uses log(2/delta); reflection handles the other tail for outcomes [-1,0].
    Conditioning on past observations permits adaptively selected endpoints.
    The same fixed endpoint is freshly sampled at every cumulative look.
    """
    if count < 2 or sample_variance < 0 or not 0 < alpha < 1 or min(horizon, looks) < 1:
        raise ValueError("Invalid confidence-bound arguments")
    logarithm = np.log(8 * horizon * looks / alpha)
    return float(np.sqrt(2 * sample_variance * logarithm / count)
                 + 7 * logarithm / (3 * (count - 1)))


def bernstein_optimize(circuit, initial, shape, settings, seed, budget, observer=None):
    rng = np.random.default_rng(seed)
    oracle = ShotOracle(circuit, int(budget), rng, observer)
    x = np.asarray(initial, dtype=float).copy()
    dimension = len(x)
    radius = settings["radius"]
    horizon = settings["max_trials"]
    looks = settings["acceptance_looks"]
    incumbents = [{"shots": 0, "theta": wrap(x).tolist()}]
    decisions = []
    stop = "horizon"
    start = time.perf_counter()
    for iteration in range(horizon):
        oracle.iteration = iteration
        points, design = design_points(dimension, rng)
        locations = [wrap(x + radius * (shape @ point)) for point in points]
        try:
            means = oracle.batch(locations, settings["model_shots"], "model")
        except BudgetExhausted:
            stop = "budget"
            break
        coefficient, condition = fit_model(design, means, [settings["model_shots"]] * (dimension + 1))
        gradient = coefficient[1:]
        predicted = float(np.linalg.norm(gradient))
        if predicted < 1e-12:
            stop = "flat_model"
            break
        trial = wrap(x - radius * shape @ gradient / predicted)
        batches = [[], []]
        previous = 0
        lower = upper = None
        decision = "unresolved"
        intervals = []
        for cumulative in looks:
            try:
                oracle.batch([x, trial], cumulative - previous, "acceptance")
            except BudgetExhausted:
                stop = "budget"
                break
            for side in (0, 1):
                batches[side].append(oracle.records[-2 + side])
            moments = [pool_moments(part) for part in batches]
            bounds = [bernstein_radius(n, variance, settings["alpha"], horizon, len(looks))
                      for n, mean, variance in moments]
            decrease = moments[0][1] - moments[1][1]
            lower, upper = decrease - sum(bounds), decrease + sum(bounds)
            previous = cumulative
            intervals.append({"shots_per_point": cumulative,
                              "means": [part[1] for part in moments],
                              "sample_variances": [part[2] for part in moments],
                              "radii": bounds, "lower": lower, "upper": upper})
            if lower >= settings["eta"] * predicted:
                decision = "accepted"
                break
            if upper < settings["eta"] * predicted:
                decision = "rejected"
                break
        decisions.append({"iteration": iteration, "radius": radius, "predicted": predicted,
                          "condition": condition, "acceptance_shots_per_point": previous,
                          "lower_decrease": lower, "upper_decrease": upper,
                          "decision": decision, "center": wrap(x).tolist(),
                          "trial": trial.tolist(), "shots": oracle.shots, "intervals": intervals})
        if observer:
            observer("decision", decisions[-1])
        if decision == "accepted":
            x = trial
            incumbents.append({"shots": oracle.shots, "theta": x.tolist()})
            radius = min(settings["max_radius"], 1.5 * radius)
        else:
            radius = max(settings["min_radius"], 0.5 * radius)
        if oracle.budget - oracle.shots < (dimension + 1) * settings["model_shots"] + 2 * looks[0]:
            stop = "budget"
            break
    return {"theta": wrap(x).tolist(), "shots": oracle.shots, "calls": oracle.calls,
            "jobs": oracle.jobs, "seconds": time.perf_counter() - start, "stop": stop,
            "decisions": decisions, "incumbents": incumbents, "evaluations": oracle.records}


def fixed_schedule(depth):
    """Graph-independent schedule, declared before bank or test optimization."""
    return np.r_[np.linspace(0.35, 0.8, depth), np.linspace(0.35, 0.15, depth)]


def run_optimizer(circuit, initial, shape, settings, method, seed, budget, observer=None):
    if method in ("initialization_only", "fixed_schedule"):
        theta = wrap(initial).tolist()
        return {"theta": theta, "shots": 0, "calls": 0, "jobs": 0, "seconds": 0.0,
                "stop": "initialization_only", "decisions": [], "evaluations": [],
                "incumbents": [{"shots": 0, "theta": theta}]}
    if method.startswith("bernstein_"):
        return bernstein_optimize(circuit, initial, shape, settings, seed, budget, observer)
    return optimize(circuit, initial, shape, settings, method, seed, budget, observer)


def instances(config):
    rows, seen = [], []
    for panel, panel_seed in enumerate(config["panel_seeds"]):
        rng = np.random.default_rng(panel_seed)
        for split in ("bank", "validation", "test"):
            for family in config["families"]:
                for n in config[split + "_sizes"]:
                    for _ in range(1000):
                        seed = int(rng.integers(1, 2 ** 30))
                        graph = make_graph(family, n, seed)
                        if not any(len(old) == n and nx.is_isomorphic(old, graph) for old in seen):
                            break
                    else:
                        raise RuntimeError("Could not construct nonisomorphic split")
                    seen.append(graph)
                    rows.append({"id": f"panel{panel}_{split}_{family}_{n}", "panel": panel,
                                 "split": split, "family": family, "n": n, "seed": seed,
                                 "edges": sorted([sorted(edge) for edge in graph.edges()])})
    return rows


def build_reference(circuit, depth, config, seed):
    """Best-found comparator with all exact objective calls and starts retained."""
    rng = np.random.default_rng(seed)
    evaluations, starts = [], []
    start_time = time.perf_counter()
    for restart in range(config["reference_starts"]):
        initial = fixed_schedule(depth) if restart == 0 else np.r_[
            rng.uniform(-np.pi, np.pi, depth), rng.uniform(-np.pi / 4, np.pi / 4, depth)]
        def objective(theta):
            value = circuit.objective(theta)
            evaluations.append({"restart": restart, "theta": np.asarray(theta).tolist(), "objective": value})
            return value
        result = minimize(objective, initial, method="BFGS",
                          options={"maxiter": config["reference_maxiter"], "gtol": 1e-7})
        starts.append({"initial": initial.tolist(), "theta": wrap(result.x).tolist(),
                       "objective": float(result.fun), "calls": int(result.nfev),
                       "success": bool(result.success), "message": str(result.message)})
    best = min(starts, key=lambda row: row["objective"])
    return {"theta": best["theta"], "objective": best["objective"], "starts": starts,
            "exact_calls": len(evaluations), "evaluations": evaluations,
            "seed": seed, "seconds": time.perf_counter() - start_time}


def run_specifications(config, rows, phase, selected=None):
    specifications = []
    for graph_index, row in enumerate(rows):
        if row["split"] != phase:
            continue
        for depth in config["depths"]:
            candidates = range(len(config["tuning_candidates"])) if phase == "validation" else [None]
            repetitions = 1 if phase == "validation" else config["repetitions"]
            methods = config["tuning_methods"] if phase == "validation" else config["methods"]
            for candidate in candidates:
                setting = (config["tuning_candidates"][candidate] if candidate is not None
                           else selected[str(depth)])
                for replicate in range(repetitions):
                    # Matched initial RNG state across methods/candidates; fresh
                    # streams between graphs, depths, repetitions, and phases.
                    seed = config["panel_seeds"][row["panel"]] + 100000 * graph_index + 1000 * depth + replicate
                    for method in methods:
                        specifications.append({"phase": phase, "graph": row["id"],
                            "panel": row["panel"], "depth": depth, "replicate": replicate,
                            "candidate": candidate, "method": method, "budget": config["budget"],
                            "seed": seed, "settings": {**config["optimizer"], **setting}})
    return specifications


def select_settings(config, runs):
    """Select by independent finite-shot validation means, never test data."""
    selected, scores = {}, {}
    for depth in config["depths"]:
        scores[str(depth)] = []
        for index, candidate in enumerate(config["tuning_candidates"]):
            group = [row for row in runs if row["phase"] == "validation" and
                     row["depth"] == depth and row["candidate"] == index]
            score = float(np.mean([row["validation"]["mean"] for row in group]))
            scores[str(depth)].append(score)
        winner = int(np.argmin(scores[str(depth)]))
        selected[str(depth)] = config["tuning_candidates"][winner].copy()
    return selected, scores


def frozen_plan(data):
    return {"config_sha256": data["provenance"]["config_sha256"],
            "instances_sha256": content_hash(data["instances"]),
            "selected_settings": data["tuning"]["selected_settings"],
            "validation_runs_sha256": content_hash([row for row in data["runs"] if row["phase"] == "validation"])}


def evaluate_run(data, specification, observer=None):
    config = data["config"]
    row = next(row for row in data["instances"] if row["id"] == specification["graph"])
    circuit = MaxCutQAOA(graph_from(row))
    key = f"{row['id']}:p{specification['depth']}"
    prior = data["priors"][key]
    initial = (fixed_schedule(specification["depth"]) if specification["method"] == "fixed_schedule"
               else np.asarray(prior["theta"]))
    shape = (np.asarray(prior["shape"]) if specification["method"] == "bernstein_shape"
             else np.eye(2 * specification["depth"]))
    result = run_optimizer(circuit, initial, shape, specification["settings"], specification["method"],
                           specification["seed"], specification["budget"], observer)
    result.update(specification)
    result["initial"] = initial.tolist()
    result["shape"] = shape.tolist()
    result["initial_objective"] = circuit.objective(initial)
    result["objective"] = circuit.objective(result["theta"])
    result["reference_objective"] = data["references"][key]["objective"]
    result["signed_reference_gap"] = result["objective"] - result["reference_objective"]
    result["expected_ratio"] = -result["objective"] * circuit.m / row["maxcut"]
    for incumbent in result["incumbents"]:
        incumbent["objective"] = circuit.objective(incumbent["theta"])
    result["hitting_shots"] = {}
    for gap in config["target_gaps"]:
        hits = [inc["shots"] for inc in result["incumbents"]
                if inc["objective"] <= result["reference_objective"] + gap]
        result["hitting_shots"][str(gap)] = min(hits) if hits else None
    validation_seed = specification["seed"] + config["budget"] + 9000000
    mean, variance = circuit.sample(result["theta"], config["validation_shots"],
                                    np.random.default_rng(validation_seed))
    result["validation"] = {"shots": config["validation_shots"], "calls": 1, "jobs": 1,
                            "seed": validation_seed, "mean": mean, "variance_of_mean": variance}
    return result


def write_bundle(path, data):
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=1) as stream:
            stream.write(encoded(data))
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, path)


def run_study(output=FOLLOWUP_DATA):
    output = Path(output).resolve()
    output.relative_to(CODE_ROOT / "results")
    pending = output.with_name("." + output.name + ".pending")
    event_path = output.with_name("." + output.name + ".events")
    if output.exists() or pending.exists() or event_path.exists():
        raise ValueError("Follow-up output or interrupted evidence exists; choose a new output")
    config = json.loads(FOLLOWUP_CONFIG.read_text())
    environment = {"python": platform.python_version(), "numpy": np.__version__,
                   "scipy": scipy.__version__, "networkx": nx.__version__,
                   "machine": platform.machine(), "system": platform.system(),
                   "numerical_build": environment_details(),
                   "simulation_model": "exact statevector; independent multinomial observations; no device noise"}
    data = {"schema_version": 1, "status": "running", "started_utc": utc_now(),
            "config": config, "environment": environment,
            "provenance": {"source_sha256": {name: hashlib.sha256((CODE_ROOT / name).read_bytes()).hexdigest()
                            for name in frozen_sources()},
                           "config_sha256": content_hash(config), "environment_sha256": content_hash(environment)},
            "instances": instances(config), "references": {}, "priors": {}, "runs": [], "tuning": {}}
    data["execution_id"] = "followup-" + content_hash({"started": data["started_utc"], "provenance": data["provenance"]})[:20]
    start = time.perf_counter()
    write_bundle(pending, data)
    with event_path.open("x") as events:
        def observer(kind, event):
            events.write(encoded({"kind": kind, "event": event}).decode() + "\n")
            events.flush()
            os.fsync(events.fileno())
        try:
            for index, row in enumerate(data["instances"]):
                circuit = MaxCutQAOA(graph_from(row))
                row["maxcut"] = exact_maxcut(graph_from(row))[0]
                row["descriptor"] = descriptor(graph_from(row)).tolist()
                for depth in config["depths"]:
                    key = f"{row['id']}:p{depth}"
                    reference_seed = config["panel_seeds"][row["panel"]] + 10000 * index + depth
                    data["references"][key] = build_reference(circuit, depth, config, reference_seed)
                write_bundle(pending, data)
                print(f"follow-up references {index + 1}/{len(data['instances'])}: {row['id']}", flush=True)
            for row in data["instances"]:
                if row["split"] == "bank":
                    continue
                for depth in config["depths"]:
                    bank = [dict(other, theta=data["references"][f"{other['id']}:p{depth}"]["theta"])
                            for other in data["instances"] if other["split"] == "bank" and other["panel"] == row["panel"]]
                    theta, shape, information = graph_prior(graph_from(row), bank, "descriptor")
                    data["priors"][f"{row['id']}:p{depth}"] = {**information, "theta": theta.tolist(), "shape": shape.tolist()}
            for phase in ("validation", "test"):
                specifications = run_specifications(config, data["instances"], phase,
                                                     data["tuning"].get("selected_settings"))
                if phase == "test":
                    data["test_started_utc"] = utc_now()
                for index, specification in enumerate(specifications):
                    observer("run_start", specification)
                    data["runs"].append(evaluate_run(data, specification, observer))
                    if (index + 1) % 8 == 0 or index + 1 == len(specifications):
                        write_bundle(pending, data)
                        events.seek(0)
                        events.truncate()
                        events.flush()
                        print(f"follow-up {phase} {index + 1}/{len(specifications)}", flush=True)
                if phase == "validation":
                    selected, scores = select_settings(config, data["runs"])
                    data["tuning"] = {"selection_rule": "minimum mean independent validation-shot objective across both panels and both Bernstein shapes; candidate-order tie break",
                                      "selected_settings": selected, "scores": scores,
                                      "validation_completed_utc": utc_now(),
                                      "optimization_shots": sum(row["shots"] for row in data["runs"]),
                                      "validation_shots": sum(row["validation"]["shots"] for row in data["runs"]),
                                      "failures": []}
                    data["tuning"]["frozen_sha256"] = content_hash(frozen_plan(data))
                    write_bundle(pending, data)
                    print("follow-up validation settings frozen: " + json.dumps(selected), flush=True)
            data["status"] = "complete"
            data["finished_utc"] = utc_now()
            data["seconds"] = time.perf_counter() - start
            data["payload_sha256"] = content_hash({key: value for key, value in data.items() if key != "payload_sha256"})
            verify_followup(data)
            write_bundle(pending, data)
            os.link(pending, output)
            pending.unlink()
        except BaseException:
            data["status"] = "interrupted"
            write_bundle(pending, data)
            raise
    event_path.unlink()
    return data


def require_close(actual, expected, name, counters):
    counters["checks"] += 1
    if not np.allclose(actual, expected, rtol=2e-11, atol=2e-12, equal_nan=False):
        raise ValueError("Follow-up mismatch: " + name)


def compare_results(actual, expected, counters, path="run"):
    """Compare all deterministic run fields, excluding measured wall-clock time."""
    if isinstance(actual, dict):
        if set(actual) != set(expected):
            raise ValueError("Follow-up keys differ at " + path)
        for key in actual:
            if key != "seconds":
                compare_results(actual[key], expected[key], counters, path + "." + key)
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            raise ValueError("Follow-up lengths differ at " + path)
        for index, (first, second) in enumerate(zip(actual, expected)):
            compare_results(first, second, counters, path + "." + str(index))
    elif isinstance(actual, (int, float)) and not isinstance(actual, bool):
        require_close(actual, expected, path, counters)
    else:
        counters["checks"] += 1
        if actual != expected:
            raise ValueError("Follow-up value differs at " + path)


def verify_followup(data, replay=False, verify_sources=True):
    """Check complete protocol identity and optionally regenerate every run.

    Full replay repeats each stochastic optimizer from its recorded RNG seed,
    comparing every observation, decision, incumbent, and diagnostic. Reference
    objective calls are recomputed, while optimizer convergence trajectories are
    preserved as evidence rather than assumed portable across BLAS versions.
    """
    counters = {"checks": 0, "runs": 0, "reference_calls": 0, "replayed_runs": 0}
    if data.get("status") != "complete" or data.get("schema_version") != 1:
        raise ValueError("A complete recognized follow-up bundle is required")
    if data.get("payload_sha256") != content_hash({key: value for key, value in data.items() if key != "payload_sha256"}):
        raise ValueError("Follow-up payload hash mismatch")
    config = data["config"]
    provenance = data["provenance"]
    if provenance["config_sha256"] != content_hash(config) or provenance["environment_sha256"] != content_hash(data["environment"]):
        raise ValueError("Follow-up configuration/environment hash mismatch")
    required_sources = frozen_sources()
    if set(provenance["source_sha256"]) != set(required_sources):
        raise ValueError("Incomplete follow-up frozen source set")
    if verify_sources:
        verified = verify_frozen_sources(provenance["source_sha256"], required_paths=required_sources)
        if json.loads(verified["sources"][required_sources[-1]]) != config:
            raise ValueError("Follow-up configuration differs from frozen file")
    generated = instances(config)
    if len(generated) != len(data["instances"]):
        raise ValueError("Follow-up graph count mismatch")
    for expected, row in zip(generated, data["instances"]):
        if any(row[key] != value for key, value in expected.items()):
            raise ValueError("Follow-up graph generation mismatch")
        graph = graph_from(row)
        require_close(row["maxcut"], exact_maxcut(graph)[0], "MaxCut", counters)
        require_close(row["descriptor"], descriptor(graph), "descriptor", counters)
        circuit = MaxCutQAOA(graph)
        for depth in config["depths"]:
            reference = data["references"][f"{row['id']}:p{depth}"]
            if reference["exact_calls"] != len(reference["evaluations"]) or reference["exact_calls"] != sum(part["calls"] for part in reference["starts"]):
                raise ValueError("Follow-up reference accounting mismatch")
            if len(reference["starts"]) != config["reference_starts"]:
                raise ValueError("Follow-up reference restart count mismatch")
            best = min(reference["starts"], key=lambda part: part["objective"])
            require_close(reference["theta"], best["theta"], "reference minimizer", counters)
            require_close(reference["objective"], best["objective"], "reference minimum", counters)
            for restart in reference["starts"]:
                require_close(circuit.objective(restart["theta"]), restart["objective"], "reference restart", counters)
            if replay:
                for call in reference["evaluations"]:
                    require_close(circuit.objective(call["theta"]), call["objective"], "reference call", counters)
                    counters["reference_calls"] += 1
            if row["split"] != "bank":
                bank = [dict(other, theta=data["references"][f"{other['id']}:p{depth}"]["theta"])
                        for other in data["instances"] if other["split"] == "bank" and other["panel"] == row["panel"]]
                theta, shape, information = graph_prior(graph, bank, "descriptor")
                prior = data["priors"][f"{row['id']}:p{depth}"]
                require_close(theta, prior["theta"], "prior anchor", counters)
                require_close(shape, prior["shape"], "prior shape", counters)
                for name in ("scores", "weights", "anchor"):
                    require_close(information[name], prior[name], "prior " + name, counters)
    selected, scores = select_settings(config, data["runs"])
    compare_results(selected, data["tuning"]["selected_settings"], counters, "selected_settings")
    compare_results(scores, data["tuning"]["scores"], counters, "tuning_scores")
    if data["tuning"]["frozen_sha256"] != content_hash(frozen_plan(data)):
        raise ValueError("Frozen validation/test plan mismatch")
    if not data["started_utc"] <= data["tuning"]["validation_completed_utc"] <= data["test_started_utc"] <= data["finished_utc"]:
        raise ValueError("Follow-up chronology mismatch")
    specifications = (run_specifications(config, data["instances"], "validation") +
                      run_specifications(config, data["instances"], "test", selected))
    if len(specifications) != len(data["runs"]):
        raise ValueError("Incomplete follow-up run schedule")
    for specification, result in zip(specifications, data["runs"]):
        compare_results(specification, {name: result[name] for name in specification}, counters, "run specification")
        if result["calls"] != len(result["evaluations"]) or result["shots"] != sum(call["shots"] for call in result["evaluations"]) or result["shots"] > result["budget"]:
            raise ValueError("Follow-up shot/call accounting mismatch")
        if result["jobs"] != max([call["job"] for call in result["evaluations"]], default=0):
            raise ValueError("Follow-up job accounting mismatch")
        if result["method"] in ("initialization_only", "fixed_schedule") and (result["shots"] or result["decisions"]):
            raise ValueError("Initialization baseline spent optimization shots")
        if replay:
            fresh = evaluate_run(data, specification)
            compare_results(fresh, result, counters)
            counters["replayed_runs"] += 1
        counters["runs"] += 1
    validation_runs = [row for row in data["runs"] if row["phase"] == "validation"]
    if data["tuning"]["optimization_shots"] != sum(row["shots"] for row in validation_runs) or data["tuning"]["validation_shots"] != sum(row["validation"]["shots"] for row in validation_runs):
        raise ValueError("Follow-up tuning cost mismatch")
    return counters


def load_followup(path=FOLLOWUP_DATA, verify=True):
    with gzip.open(path, "rt") as stream:
        data = json.load(stream)
    if verify:
        verify_followup(data)
    return data


def refinement_config(historical):
    """Declare the revision protocol without selecting settings on test data.

    This is a post hoc follow-up to the historical experiment, not a claim of
    prospective preregistration. The two panels confound bank and target sets;
    they cannot identify the effect of changing a bank on a fixed target.
    """
    old = historical["config"]
    return {"protocol_version": 1, "budget": old["budget"],
            "validation_shots": old["validation_shots"],
            "depths": old["depths"], "repetitions": old["repetitions"],
            "methods": [f"revised_{rule}_{shape}" for rule in ("hoeffding", "bernstein")
                        for shape in ("iso", "shape")],
            "optimizer": old["optimizer"],
            "selected_settings": historical["tuning"]["selected_settings"],
            "target_gaps": old["target_gaps"], "seed": 2026092301,
            "diagnostic_panel": 0, "radii": [0.025, 0.05, 0.1, 0.2, 0.4],
            "model_shots": [256, 1024, 4096, 16384], "diagnostic_repetitions": 4,
            "gradient_difference_step": 1e-5,
            "power_model_shots": 1024, "power_model_replicate": 0,
            "power_repetitions": 128, "power_caps": [16384, 65536],
            "power_looks": [512, 2048, 8192, 16384, 65536],
            "power_alpha": old["optimizer"]["alpha"],
            "power_horizon": old["optimizer"]["max_trials"],
            "decision_policy": "stop on unresolved; contract only on certified rejection; radius-squared model allocation",
            "settings_rule": "reuse previously selected validation settings unchanged; no revision test tuning",
            "diagnostic_selection": "all panel-zero test graphs, both shapes/depths; power uses every radius at 1024 model shots and replicate zero"}


def evaluate_refinement(data, specification):
    """Execute a revised policy; exact diagnostics are computed after return."""
    from .search import refine
    row = next(row for row in data["instances"] if row["id"] == specification["graph"])
    circuit = MaxCutQAOA(graph_from(row))
    prior = data["priors"][f"{row['id']}:p{specification['depth']}"]
    initial = np.asarray(prior["theta"])
    shape = (np.asarray(prior["shape"]) if specification["method"].endswith("_shape")
             else np.eye(len(initial)))
    rule = specification["method"].split("_")[1]
    result = refine(circuit, initial, shape, specification["settings"],
                    specification["seed"], specification["budget"], bound=rule)
    result.update(specification)
    result.update(initial=initial.tolist(), shape=shape.tolist(),
                  initial_objective=circuit.objective(initial),
                  objective=circuit.objective(result["theta"]))
    key = f"{row['id']}:p{specification['depth']}"
    result["reference_objective"] = data["references"][key]["objective"]
    result["signed_reference_gap"] = result["objective"] - result["reference_objective"]
    result["expected_ratio"] = -result["objective"] * circuit.m / row["maxcut"]
    for incumbent in result["incumbents"]:
        incumbent["objective"] = circuit.objective(incumbent["theta"])
    for decision in result["decisions"]:
        decision["true_decrease"] = circuit.objective(decision["center"]) - circuit.objective(decision["trial"])
        decision["acceptance_margin"] = decision["true_decrease"] - specification["settings"]["eta"] * decision["predicted"]
    result["hitting_shots"] = {}
    for gap in data["config"]["target_gaps"]:
        hits = [inc["shots"] for inc in result["incumbents"]
                if inc["objective"] <= result["reference_objective"] + gap]
        result["hitting_shots"][str(gap)] = min(hits) if hits else None
    validation_seed = specification["seed"] + 900000000
    validation_shots = data["config"]["validation_shots"]
    mean, variance = circuit.sample(result["theta"], validation_shots,
                                    np.random.default_rng(validation_seed))
    result["validation"] = {"shots": validation_shots, "calls": 1, "jobs": 1,
                            "seed": validation_seed, "mean": mean, "variance_of_mean": variance}
    return result


def refinement_specifications(data):
    config = data["config"]
    specifications = []
    for index, row in enumerate(data["instances"]):
        if row["split"] != "test":
            continue
        for depth in config["depths"]:
            settings = {**config["optimizer"], **config["selected_settings"][str(depth)]}
            for replicate in range(config["repetitions"]):
                seed = config["seed"] + 100000 * index + 1000 * depth + replicate
                for method in config["methods"]:
                    specifications.append({"phase": "revision", "graph": row["id"],
                        "panel": row["panel"], "depth": depth, "replicate": replicate,
                        "method": method, "budget": config["budget"], "seed": seed,
                        "settings": settings})
    return specifications


def sample_distribution(distribution, shots, rng):
    """Sample cached exact distributions for offline diagnostics only."""
    outcomes, probabilities = distribution
    counts = rng.multinomial(shots, probabilities)
    mean = float(counts @ outcomes / shots)
    variance = float(counts @ ((outcomes - mean) ** 2) / (shots * (shots - 1)))
    return {"shots": shots, "mean": mean, "variance_of_mean": variance}


def affine_diagnostics(data):
    """Disentangle stochastic fitting error, affine bias, and proposal quality.

    A single design is shared over radii and sample counts for each target,
    depth and shape. Finite differences are offline reference derivatives;
    their step is recorded and never supplied to any optimizer.
    """
    config, records = data["config"], []
    rows = [row for row in data["instances"] if row["split"] == "test" and
            row["panel"] == config["diagnostic_panel"]]
    for graph_index, row in enumerate(rows):
        circuit = MaxCutQAOA(graph_from(row))
        for depth in config["depths"]:
            prior = data["priors"][f"{row['id']}:p{depth}"]
            theta = np.asarray(prior["theta"])
            objective = circuit.objective(theta)
            step = config["gradient_difference_step"]
            axes = np.eye(len(theta))
            true_gradient = np.array([(circuit.objective(theta + step * axis) -
                                      circuit.objective(theta - step * axis)) / (2 * step)
                                     for axis in axes])
            for shape_index, shape_kind in enumerate(("iso", "shape")):
                shape = axes if shape_kind == "iso" else np.asarray(prior["shape"])
                seed = config["seed"] + 50000000 + graph_index * 100000 + depth * 1000 + shape_index * 100
                points, design = design_points(len(theta), np.random.default_rng(seed))
                for radius_index, radius in enumerate(config["radii"]):
                    locations = [wrap(theta + radius * shape @ point) for point in points]
                    distributions = [circuit.distribution(location) for location in locations]
                    exact_means = np.array([outcomes @ probabilities for outcomes, probabilities in distributions])
                    exact_coefficient, condition = fit_model(design, exact_means, [1] * len(locations))
                    exact_gradient = np.linalg.solve(shape.T, exact_coefficient[1:]) / radius
                    exact_predicted = float(np.linalg.norm(exact_coefficient[1:]))
                    exact_trial = wrap(theta - radius * shape @ exact_coefficient[1:] / exact_predicted)
                    exact_decrease = objective - circuit.objective(exact_trial)
                    eta = config["optimizer"]["eta"]
                    for model_shots in config["model_shots"]:
                        for replicate in range(config["diagnostic_repetitions"]):
                            model_seed = [seed, radius_index, model_shots, replicate]
                            rng = np.random.default_rng(np.random.SeedSequence(model_seed))
                            batches = [sample_distribution(distribution, model_shots, rng)
                                       for distribution in distributions]
                            coefficient, _ = fit_model(design, [batch["mean"] for batch in batches],
                                                       [model_shots] * len(locations))
                            gradient = np.linalg.solve(shape.T, coefficient[1:]) / radius
                            predicted = float(np.linalg.norm(coefficient[1:]))
                            trial = wrap(theta - radius * shape @ coefficient[1:] / predicted)
                            decrease = objective - circuit.objective(trial)
                            records.append({"id": len(records), "graph": row["id"], "depth": depth,
                                "shape_kind": shape_kind, "shape": shape.tolist(), "radius": radius,
                                "model_shots": model_shots, "replicate": replicate, "model_seed": model_seed,
                                "design_seed": seed, "points": points.tolist(), "model_batches": batches,
                                "theta": theta.tolist(), "trial": trial.tolist(), "condition": condition,
                                "true_gradient": true_gradient.tolist(), "exact_fitted_gradient": exact_gradient.tolist(),
                                "sampled_gradient": gradient.tolist(), "gradient_difference_step": step,
                                "gradient_sampling_error": float(np.linalg.norm(gradient - exact_gradient)),
                                "gradient_total_error": float(np.linalg.norm(gradient - true_gradient)),
                                "exact_gradient_error": float(np.linalg.norm(exact_gradient - true_gradient)),
                                "predicted": predicted, "true_decrease": decrease,
                                "acceptance_margin": decrease - eta * predicted,
                                "exact_predicted": exact_predicted, "exact_trial": exact_trial.tolist(),
                                "exact_true_decrease": exact_decrease,
                                "exact_acceptance_margin": exact_decrease - eta * exact_predicted})
        print(f"refinement affine diagnostics {graph_index + 1}/{len(rows)}", flush=True)
    return records


def power_decision(observations, predicted, config, cap, rule):
    """Apply a finite-look rule to independently replayed frozen endpoints."""
    batches = [[], []]
    decision, total = "unresolved", 0
    for observation in observations:
        if observation["shots_per_point"] > cap:
            break
        total = observation["shots_per_point"]
        for side in (0, 1):
            batches[side].append(observation["batches"][side])
        moments = [pool_moments(part) for part in batches]
        if rule == "hoeffding":
            radii = [np.sqrt(np.log(4 * config["power_horizon"] * len(config["power_looks"]) /
                                   config["power_alpha"]) / (2 * total))] * 2
        elif rule == "bernstein":
            radii = [bernstein_radius(n, variance, config["power_alpha"],
                                     config["power_horizon"], len(config["power_looks"]))
                     for n, mean, variance in moments]
        else:
            raise ValueError("Unknown replay confidence rule")
        decrease = moments[0][1] - moments[1][1]
        threshold = config["optimizer"]["eta"] * predicted
        if decrease - sum(radii) >= threshold:
            decision = "accepted"
            break
        if decrease + sum(radii) < threshold:
            decision = "rejected"
            break
    return {"decision": decision, "shots": 2 * total}


def hoeffding_power_upper_bound(margin, config, cap):
    """Union upper bound on acceptance, not a necessary measurement bound.

    For two independent bounded endpoint means, P(error >= t) <= exp(-n t²).
    Acceptance at look n requires error >= 2*r_n-h. Early rejection can only
    lower acceptance probability, so summing across all declared looks is safe.
    """
    logarithm = np.log(4 * config["power_horizon"] * len(config["power_looks"]) /
                       config["power_alpha"])
    bounds = []
    for count in config["power_looks"]:
        if count <= cap:
            threshold = 2 * np.sqrt(logarithm / (2 * count)) - margin
            bounds.append(1.0 if threshold <= 0 else float(np.exp(-count * threshold ** 2)))
    return min(1.0, sum(bounds))


def endpoint_power(data):
    config = data["config"]
    chosen = [row for row in data["model_diagnostics"]
              if row["model_shots"] == config["power_model_shots"] and
              row["replicate"] == config["power_model_replicate"]]
    records, samples = [], []
    for proposal_index, proposal in enumerate(chosen):
        row = next(row for row in data["instances"] if row["id"] == proposal["graph"])
        circuit = MaxCutQAOA(graph_from(row))
        distributions = [circuit.distribution(proposal[name]) for name in ("theta", "trial")]
        outcomes = {(cap, rule): [] for cap in config["power_caps"] for rule in ("hoeffding", "bernstein")}
        for replicate in range(config["power_repetitions"]):
            # Separate streams for the two endpoints and every fixed proposal.
            seed = [config["seed"], 70000000, proposal["id"], replicate]
            streams = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(2)]
            observations, previous = [], 0
            for count in config["power_looks"]:
                observations.append({"shots_per_point": count,
                    "batches": [sample_distribution(distribution, count - previous, rng)
                                for distribution, rng in zip(distributions, streams)]})
                previous = count
            samples.append({"proposal_id": proposal["id"], "replicate": replicate,
                            "seed": seed, "observations": observations})
            for (cap, rule), decisions in outcomes.items():
                decisions.append(power_decision(observations, proposal["predicted"], config, cap, rule))
        for (cap, rule), decisions in outcomes.items():
            records.append({"proposal_id": proposal["id"], "graph": proposal["graph"],
                "depth": proposal["depth"], "shape_kind": proposal["shape_kind"],
                "radius": proposal["radius"], "predicted": proposal["predicted"],
                "true_decrease": proposal["true_decrease"], "acceptance_margin": proposal["acceptance_margin"],
                "cap_per_endpoint": cap, "rule": rule, "repeats": len(decisions),
                "acceptance_probability": float(np.mean([part["decision"] == "accepted" for part in decisions])),
                "rejection_probability": float(np.mean([part["decision"] == "rejected" for part in decisions])),
                "unresolved_fraction": float(np.mean([part["decision"] == "unresolved" for part in decisions])),
                "mean_acceptance_shots": float(np.mean([part["shots"] for part in decisions])),
                "outcomes": decisions,
                "hoeffding_acceptance_upper_bound": hoeffding_power_upper_bound(proposal["acceptance_margin"], config, cap)})
        if (proposal_index + 1) % 16 == 0:
            print(f"refinement endpoint power {proposal_index + 1}/{len(chosen)}", flush=True)
    return records, samples


def run_refinement(output=REFINEMENT_DATA):
    """Run a separately dated revision without changing historical evidence."""
    output = Path(output).resolve()
    output.relative_to(CODE_ROOT / "results")
    pending = output.with_name("." + output.name + ".pending")
    if output.exists() or pending.exists():
        raise ValueError("Refinement output or interrupted evidence exists; choose a new output")
    historical = load_followup()
    config = refinement_config(historical)
    sources = {name: (CODE_ROOT / name).read_text() for name in frozen_sources()}
    environment = {"python": platform.python_version(), "numpy": np.__version__,
                   "scipy": scipy.__version__, "networkx": nx.__version__,
                   "numerical_build": environment_details(),
                   "simulation_model": "exact statevector; independent multinomial observations; no device noise"}
    data = {"schema_version": 1, "study": "refinement", "status": "running",
            "started_utc": utc_now(), "config": config, "environment": environment,
            "provenance": {"source_text": sources,
                "source_sha256": {name: hashlib.sha256(value.encode()).hexdigest() for name, value in sources.items()},
                "config_sha256": content_hash(config), "environment_sha256": content_hash(environment),
                "historical_path": "results/followup.json.gz",
                "historical_sha256": hashlib.sha256(FOLLOWUP_DATA.read_bytes()).hexdigest(),
                "historical_payload_sha256": historical["payload_sha256"]},
            "instances": copy.deepcopy(historical["instances"]),
            "priors": copy.deepcopy(historical["priors"]),
            "references": {key: {name: value[name] for name in ("theta", "objective", "exact_calls")}
                           for key, value in historical["references"].items()},
            "baseline_runs": [dict(copy.deepcopy(row), historical_run_index=index)
                              for index, row in enumerate(historical["runs"]) if row["phase"] == "test"],
            "runs": [], "model_diagnostics": [], "power": [], "power_samples": []}
    data["execution_id"] = "refinement-" + content_hash({"started": data["started_utc"], "config": config,
                                                         "source_sha256": data["provenance"]["source_sha256"]})[:20]
    start = time.perf_counter()
    data["frozen_plan_sha256"] = content_hash({"config": config, "priors": data["priors"],
                                              "specifications": refinement_specifications(data)})
    write_bundle(pending, data)
    try:
        specifications = refinement_specifications(data)
        for index, specification in enumerate(specifications):
            data["runs"].append(evaluate_refinement(data, specification))
            if (index + 1) % 16 == 0:
                write_bundle(pending, data)
                print(f"refinement factorial {index + 1}/{len(specifications)}", flush=True)
        data["model_diagnostics"] = affine_diagnostics(data)
        data["proposals_frozen_sha256"] = content_hash(data["model_diagnostics"])
        data["power_started_utc"] = utc_now()
        write_bundle(pending, data)
        data["power"], data["power_samples"] = endpoint_power(data)
        data["status"] = "complete"
        data["finished_utc"] = utc_now()
        data["seconds"] = time.perf_counter() - start
        data["payload_sha256"] = content_hash({key: value for key, value in data.items() if key != "payload_sha256"})
        verify_refinement(data)
        write_bundle(pending, data)
        os.link(pending, output)
        pending.unlink()
    except BaseException:
        data["status"] = "interrupted"
        write_bundle(pending, data)
        raise
    return data


def verify_refinement(data, replay=False):
    """Validate immutable identities, controls, costs and optionally RNG replay."""
    if data.get("study") != "refinement" or data.get("schema_version") != 1 or data.get("status") != "complete":
        raise ValueError("A complete recognized refinement bundle is required")
    if data.get("payload_sha256") != content_hash({key: value for key, value in data.items() if key != "payload_sha256"}):
        raise ValueError("Refinement payload hash mismatch")
    provenance = data["provenance"]
    if provenance["config_sha256"] != content_hash(data["config"]) or provenance["environment_sha256"] != content_hash(data["environment"]):
        raise ValueError("Refinement configuration/environment mismatch")
    if set(provenance["source_sha256"]) != set(frozen_sources()) or set(provenance["source_text"]) != set(frozen_sources()):
        raise ValueError("Refinement frozen source set mismatch")
    if any(hashlib.sha256(provenance["source_text"][name].encode()).hexdigest() != expected
           for name, expected in provenance["source_sha256"].items()):
        raise ValueError("Refinement embedded source hash mismatch")
    if hashlib.sha256(FOLLOWUP_DATA.read_bytes()).hexdigest() != provenance["historical_sha256"]:
        raise ValueError("Refinement historical bundle identity mismatch")
    historical = load_followup(verify=False)
    if historical["payload_sha256"] != provenance["historical_payload_sha256"]:
        raise ValueError("Refinement historical payload identity mismatch")
    counters = {"checks": 0, "runs": 0, "replayed_runs": 0,
                "model_diagnostics": len(data["model_diagnostics"]), "power_conditions": len(data["power"])}
    compare_results(refinement_config(historical), data["config"], counters, "revision config")
    compare_results(historical["instances"], data["instances"], counters, "historical targets")
    compare_results(historical["priors"], data["priors"], counters, "historical anchors")
    for baseline in data["baseline_runs"]:
        compare_results(historical["runs"][baseline["historical_run_index"]],
                        {key: value for key, value in baseline.items() if key != "historical_run_index"},
                        counters, "historical comparator")
    specifications = refinement_specifications(data)
    if len(specifications) != len(data["runs"]):
        raise ValueError("Incomplete refinement factorial")
    if data["frozen_plan_sha256"] != content_hash({"config": data["config"], "priors": data["priors"],
                                                   "specifications": specifications}):
        raise ValueError("Refinement frozen plan mismatch")
    for specification, result in zip(specifications, data["runs"]):
        compare_results(specification, {key: result[key] for key in specification}, counters, "refinement specification")
        if (result["shots"] != sum(call["shots"] for call in result["evaluations"]) or
                result["calls"] != len(result["evaluations"]) or result["shots"] > result["budget"]):
            raise ValueError("Refinement resource accounting mismatch")
        if replay:
            compare_results(evaluate_refinement(data, specification), result, counters, "refinement replay")
            counters["replayed_runs"] += 1
        counters["runs"] += 1
    if data["proposals_frozen_sha256"] != content_hash(data["model_diagnostics"]):
        raise ValueError("Refinement frozen proposals mismatch")
    if not data["started_utc"] <= data["power_started_utc"] <= data["finished_utc"]:
        raise ValueError("Refinement chronology mismatch")
    if replay:
        compare_results(affine_diagnostics(data), data["model_diagnostics"], counters, "affine diagnostic replay")
        power, samples = endpoint_power(data)
        compare_results(power, data["power"], counters, "endpoint power replay")
        compare_results(samples, data["power_samples"], counters, "endpoint sample replay")
    return counters


def load_refinement(path=REFINEMENT_DATA, verify=True):
    with gzip.open(path, "rt") as stream:
        data = json.load(stream)
    if verify:
        verify_refinement(data)
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--revision", action="store_true", help="run the matched radius-aware revision and offline proposal/power diagnostics")
    parser.add_argument("--verify", action="store_true", help="verify the saved artifact instead of running a study")
    parser.add_argument("--full", action="store_true", help="regenerate all objective observations and optimizer decisions")
    args = parser.parse_args(argv)
    args.output = args.output or (REFINEMENT_DATA if args.revision else FOLLOWUP_DATA)
    if args.full and not args.verify:
        parser.error("--full requires --verify")
    if args.verify:
        loader, verifier = (load_refinement, verify_refinement) if args.revision else (load_followup, verify_followup)
        print(json.dumps(verifier(loader(args.output, verify=False), replay=args.full), sort_keys=True))
    else:
        data = (run_refinement if args.revision else run_study)(args.output)
        print(json.dumps({"output": str(args.output.resolve().relative_to(CODE_ROOT.parent)),
                          "runs": len(data["runs"]), "seconds": data["seconds"]}, sort_keys=True))
