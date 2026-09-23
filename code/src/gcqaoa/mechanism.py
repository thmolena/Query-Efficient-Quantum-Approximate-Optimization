"""Controlled attribution and simulator-informed prediction of affine refinement.

Run ``python -m gcqaoa.mechanism`` from an installed checkout, or verify with
``--verify --full``. This separately dated study never rewrites prior bundles.
Exact target expectations and variances inform the offline predictor; they are
not an implementable finite-shot pilot and never enter optimizer decisions.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import networkx as nx
import numpy as np
import scipy

from .experiment import graph_from, make_graph
from .followup import (content_hash, utc_now, write_bundle,
                       load_followup, load_refinement, power_decision, sample_distribution)
from .provenance import CODE_ROOT, NUMERICAL_SOURCES
from .qaoa import MaxCutQAOA, exact_maxcut
from .records import environment_details
from .search import descriptor, design_points, fit_model, graph_prior, refine, wrap

MECHANISM_DATA = CODE_ROOT / "results" / "mechanism.json.gz"
SOURCES = (*NUMERICAL_SOURCES, "src/gcqaoa/followup.py", "src/gcqaoa/mechanism.py")
HISTORICAL = ("study", "followup", "transport", "refinement")


def compare_results(actual, expected, counters, path="record"):
    """Compare deterministic replay fields, vectorizing stored sample vectors."""
    if isinstance(actual, dict):
        if set(actual) != set(expected):
            raise ValueError("Mechanism keys differ at " + path)
        for key in actual:
            if key != "seconds":
                compare_results(actual[key], expected[key], counters, path + "." + key)
    elif isinstance(actual, list):
        if len(actual) != len(expected):
            raise ValueError("Mechanism lengths differ at " + path)
        if actual and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in actual):
            counters["checks"] += len(actual)
            if not np.allclose(actual, expected, rtol=2e-11, atol=2e-12):
                raise ValueError("Mechanism values differ at " + path)
        else:
            for index, (first, second) in enumerate(zip(actual, expected)):
                compare_results(first, second, counters, path + "." + str(index))
    elif isinstance(actual, (int, float)) and not isinstance(actual, bool):
        counters["checks"] += 1
        if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=2e-11, abs_tol=2e-12):
            raise ValueError("Mechanism value differs at " + path)
    else:
        counters["checks"] += 1
        if actual != expected:
            raise ValueError("Mechanism value differs at " + path)


def configuration(historical):
    return {"version": 1, "seed": 2026092317, "fresh_seed": 2026092329,
            "depths": [2, 3], "families": historical["config"]["families"],
            "fresh_sizes": [12, 14], "bank_panel": 0,
            "radii": [0.025, 0.05, 0.1, 0.2, 0.4],
            "model_shots": [16, 64, 256, 1024, 4096, 16384],
            "base_model_shots": 256, "base_radius": 0.1,
            "fit_repetitions": 128, "proposal_repetitions": 32,
            "gaussian_repetitions": 256, "gaussian_population": "fresh",
            "gradient_steps": [1e-5, 5e-6], "gradient_check_tolerance": 1e-7,
            "optimizer": historical["config"]["optimizer"],
            "selected_settings": historical["tuning"]["selected_settings"],
            "attribution_repetitions": 2, "budget": 65536, "validation_shots": 8192,
            "attribution_cells": [[allocation, stopping] for allocation in ("fixed", "radius")
                                  for stopping in ("stop", "continue")],
            "continuation_semantics": "hold radius after unresolved; refit with fresh data; only certified rejection contracts",
            "online_looks": [512, 2048, 8192, 16384],
            "online_pair_budgets": [1024, 4096, 16384, 32768],
            "cohort_rule": "fixed true-margin sign classes; every proposal retained at every budget; weights constant within each reported curve",
            "predictor_scope": "offline simulator-informed conditional bias/variance and Gaussian model; not a measurement implementable pilot",
            "fresh_protocol": "fixed panel-zero bank; nonisomorphic to all previous bank/test/validation graphs; no tuning on fresh targets"}


def historical_bundles():
    data = {}
    for name in HISTORICAL:
        with gzip.open(CODE_ROOT / "results" / (name + ".json.gz"), "rt") as stream:
            data[name] = json.load(stream)
    return data


def prepare_targets(config, bundles):
    historical = bundles["followup"]
    old = [dict(copy.deepcopy(row), split="diagnostic") for row in historical["instances"]
           if row["split"] == "test" and row["panel"] == 0]
    seen = [graph_from(row) for bundle in bundles.values() for row in bundle.get("instances", [])]
    rng = np.random.default_rng(config["fresh_seed"])
    fresh = []
    for family in config["families"]:
        for n in config["fresh_sizes"]:
            for _ in range(1000):
                seed = int(rng.integers(1, 2 ** 30))
                graph = make_graph(family, n, seed)
                if not any(len(other) == n and nx.is_isomorphic(graph, other) for other in seen):
                    break
            else:
                raise RuntimeError("No disjoint fresh graph found")
            seen.append(graph)
            fresh.append({"id": f"fresh_{family}_{n}", "split": "fresh", "panel": 0,
                          "family": family, "n": n, "seed": seed,
                          "edges": sorted([sorted(edge) for edge in graph.edges()]),
                          "maxcut": exact_maxcut(graph)[0], "descriptor": descriptor(graph).tolist()})
    priors = {f"{row['id']}:p{depth}": copy.deepcopy(historical["priors"][f"{row['id']}:p{depth}"])
              for row in old for depth in config["depths"]}
    for row in fresh:
        for depth in config["depths"]:
            bank = [dict(donor, theta=historical["references"][f"{donor['id']}:p{depth}"]["theta"])
                    for donor in historical["instances"] if donor["split"] == "bank" and
                    donor["panel"] == config["bank_panel"]]
            theta, shape, information = graph_prior(graph_from(row), bank, "descriptor")
            priors[f"{row['id']}:p{depth}"] = {**information, "theta": theta.tolist(), "shape": shape.tolist()}
    return old + fresh, priors


def ledger_attribution(revision):
    rows = []
    for run in revision["runs"]:
        base = run["settings"]["radius"]
        models = run["models"]
        radii = [model["radius"] for model in models] + [run["final_radius"]]
        rows.append({"graph": run["graph"], "depth": run["depth"], "replicate": run["replicate"],
                     "method": run["method"], "shots": run["shots"], "stop_reason": run["stop_reason"],
                     "radius_changes": sum(not np.isclose(first, second) for first, second in zip(radii, radii[1:])),
                     "noninitial_radius_models": sum(not np.isclose(model["radius"], base) for model in models),
                     "above_base_shot_models": sum(model["model_shots_per_point"] > run["settings"]["model_shots"]
                                                   for model in models),
                     "below_base_shot_models": sum(model["model_shots_per_point"] < run["settings"]["model_shots"]
                                                   for model in models),
                     "model_allocations": [{key: model[key] for key in ("radius", "model_shots_per_point", "model_shots")}
                                           for model in models]})
    return rows


def historical_comparison(revision):
    original = {(row["graph"], row["depth"], row["replicate"]): row for row in revision["baseline_runs"]
                if row["method"] == "bernstein_shape"}
    paired = []
    for row in revision["runs"]:
        if row["method"] != "revised_bernstein_shape":
            continue
        old = original[(row["graph"], row["depth"], row["replicate"])]
        if not np.array_equal(row["initial"], old["initial"]):
            raise ValueError("Historical shaped comparison does not share an anchor")
        old_gain = 100 * (old["initial_objective"] - old["objective"])
        new_gain = 100 * (row["initial_objective"] - row["objective"])
        paired.append({"graph": row["graph"], "depth": row["depth"], "replicate": row["replicate"],
                       "original_gain_pp": old_gain, "revised_gain_pp": new_gain,
                       "gain_difference_pp": new_gain - old_gain,
                       "original_shots": old["shots"], "revised_shots": row["shots"],
                       "cost_difference_shots": row["shots"] - old["shots"]})
    return paired


def online_power(config, revision, extended=False):
    """Recompute the online J=4 rule, retaining fixed proposals and streams."""
    records = []
    by_proposal = {}
    for row in revision["power_samples"]:
        by_proposal.setdefault(row["proposal_id"], []).append(row)
    proposals = {row["id"]: row for row in revision["model_diagnostics"]}
    looks = revision["config"]["power_looks"] if extended else config["online_looks"]
    budgets = [2 * look for look in looks] if extended else config["online_pair_budgets"]
    power_config = {**revision["config"], "power_looks": looks}
    for identifier, samples in by_proposal.items():
        proposal = proposals[identifier]
        cohort = ("worsening" if proposal["true_decrease"] <= 0 else "improving_insufficient"
                  if proposal["acceptance_margin"] <= 0 else "positive_margin")
        model_cost = (2 * proposal["depth"] + 1) * proposal["model_shots"]
        for pair_budget in budgets:
            available = pair_budget if extended else min(pair_budget, config["budget"] - model_cost)
            affordable = [look for look in looks if 2 * look <= available]
            cap = max(affordable, default=0)
            for rule in ("hoeffding", "bernstein"):
                outcomes = [power_decision(sample["observations"], proposal["predicted"],
                                            power_config, cap, rule) for sample in samples]
                records.append({"proposal_id": identifier, "graph": proposal["graph"],
                    "depth": proposal["depth"], "shape_kind": proposal["shape_kind"], "radius": proposal["radius"],
                    "acceptance_margin": proposal["acceptance_margin"], "true_decrease": proposal["true_decrease"],
                    "cohort": cohort, "rule": rule, "J": len(looks), "endpoint_pair_budget": pair_budget,
                    "cap_per_endpoint": cap, "model_shots": model_cost,
                    "affordable_looks": affordable, "repeats": len(outcomes), "outcomes": outcomes,
                    "acceptance_probability": float(np.mean([part["decision"] == "accepted" for part in outcomes])),
                    "unresolved_fraction": float(np.mean([part["decision"] == "unresolved" for part in outcomes])),
                    "mean_acceptance_shots": float(np.mean([part["shots"] for part in outcomes]))})
    limits = []
    for run in ([] if extended else revision["runs"]):
        for decision in run["decisions"]:
            spent_before = decision["shots"] - decision["acceptance_shots"]
            remaining = run["budget"] - spent_before
            limits.append({"graph": run["graph"], "depth": run["depth"], "replicate": run["replicate"],
                           "method": run["method"], "iteration": decision["iteration"],
                           "shots_before_acceptance": spent_before, "remaining_shots": remaining,
                           "affordable_looks": [look for look in config["online_looks"] if 2 * look <= remaining],
                           "observed_looks": [part["shots_per_point"] for part in decision["intervals"]],
                           "decision": decision["decision"]})
    return records, limits


def allocation_specs(config, historical):
    specs = []
    for index, row in enumerate(historical["instances"]):
        if row["split"] != "test":
            continue
        for depth in config["depths"]:
            for replicate in range(config["attribution_repetitions"]):
                for allocation, stopping in config["attribution_cells"]:
                    specs.append({"graph": row["id"], "depth": depth, "replicate": replicate,
                                  "method": f"{allocation}_{stopping}", "model_sampling": allocation,
                                  "unresolved_policy": stopping,
                                  "seed": config["seed"] + 100000 * index + 1000 * depth + replicate})
    return specs


def allocation_run(config, historical, spec):
    row = next(row for row in historical["instances"] if row["id"] == spec["graph"])
    circuit = MaxCutQAOA(graph_from(row))
    key = f"{row['id']}:p{spec['depth']}"
    prior = historical["priors"][key]
    initial, shape = np.array(prior["theta"]), np.array(prior["shape"])
    settings = {**config["optimizer"], **config["selected_settings"][str(spec["depth"])]}
    result = refine(circuit, initial, shape, settings, spec["seed"], config["budget"], bound="bernstein",
                    model_sampling=spec["model_sampling"], unresolved_policy=spec["unresolved_policy"])
    result.update(spec)
    result.update(panel=row["panel"], initial=initial.tolist(), shape=shape.tolist(), settings=settings, budget=config["budget"],
                  initial_objective=circuit.objective(initial), objective=circuit.objective(result["theta"]),
                  reference_objective=historical["references"][key]["objective"])
    result["signed_reference_gap"] = result["objective"] - result["reference_objective"]
    for incumbent in result["incumbents"]:
        incumbent["objective"] = circuit.objective(incumbent["theta"])
    for decision in result["decisions"]:
        decision["true_decrease"] = circuit.objective(decision["center"]) - circuit.objective(decision["trial"])
        decision["acceptance_margin"] = decision["true_decrease"] - settings["eta"] * decision["predicted"]
    seed = spec["seed"] + 800000000
    mean, variance = circuit.sample(result["theta"], config["validation_shots"], np.random.default_rng(seed))
    result["validation"] = {"seed": seed, "shots": config["validation_shots"], "calls": 1, "jobs": 1,
                            "mean": mean, "variance_of_mean": variance}
    return result


def gradient_map(design, shape, radius):
    """Map independent objective means to a physical-coordinate gradient."""
    inverse = np.linalg.solve(np.asarray(design), np.eye(len(design)))
    return np.linalg.solve(np.asarray(shape).T, inverse[1:, :]) / radius


def variance_prediction(mapping, means, per_shot_variances, true_gradient, shots):
    fitted = mapping @ np.asarray(means)
    bias = fitted - np.asarray(true_gradient)
    covariance = (mapping * (np.asarray(per_shot_variances) / shots)) @ mapping.T
    return {"exact_fitted_gradient": fitted.tolist(), "exact_bias_sq": float(bias @ bias),
            "sampling_mse": float(np.trace(covariance)),
            "total_mse": float(bias @ bias + np.trace(covariance)),
            "gradient_covariance": covariance.tolist()}


def proposal_value(circuit, theta, shape, radius, coefficient, eta, center_objective):
    predicted = float(np.linalg.norm(coefficient[1:]))
    if predicted < 1e-12:
        trial, decrease = theta.copy(), 0.0
    else:
        trial = wrap(theta - radius * shape @ coefficient[1:] / predicted)
        decrease = center_objective - circuit.objective(trial)
    return {"theta": trial.tolist(), "predicted": predicted, "true_decrease": decrease,
            "acceptance_margin": decrease - eta * predicted}


def predict_mechanism(data):
    config, predictions, gradients = data["config"], [], []
    for graph_index, row in enumerate(data["instances"]):
        circuit = MaxCutQAOA(graph_from(row))
        for depth in config["depths"]:
            prior = data["priors"][f"{row['id']}:p{depth}"]
            theta = np.asarray(prior["theta"])
            objective = circuit.objective(theta)
            axes = np.eye(len(theta))
            estimates = [np.array([(circuit.objective(theta + step * axis) -
                                   circuit.objective(theta - step * axis)) / (2 * step) for axis in axes])
                         for step in config["gradient_steps"]]
            error = float(np.linalg.norm(estimates[0] - estimates[1]))
            if error > config["gradient_check_tolerance"]:
                raise ValueError("Reference gradient fails independent step check")
            gradients.append({"graph": row["id"], "depth": depth, "steps": config["gradient_steps"],
                              "gradients": [value.tolist() for value in estimates], "difference_norm": error,
                              "exact_calls": 1 + 2 * len(theta) * len(config["gradient_steps"])})
            for shape_index, shape_kind in enumerate(("iso", "shape")):
                shape = axes if shape_kind == "iso" else np.asarray(prior["shape"])
                design_seed = [config["seed"], 110000000, graph_index, depth, shape_index]
                points, design = design_points(len(theta), np.random.default_rng(np.random.SeedSequence(design_seed)))
                inverse = np.linalg.solve(design, np.eye(len(design)))
                for radius_index, radius in enumerate(config["radii"]):
                    locations = [wrap(theta + radius * shape @ point) for point in points]
                    distributions = [circuit.distribution(location) for location in locations]
                    means = np.array([outcomes @ probabilities for outcomes, probabilities in distributions])
                    variances = np.array([((outcomes - mean) ** 2) @ probabilities
                                          for (outcomes, probabilities), mean in zip(distributions, means)])
                    mapping = gradient_map(design, shape, radius)
                    predicted = variance_prediction(mapping, means, variances, estimates[0], 1)
                    coefficient = inverse @ means
                    schedule = max(2, int(np.ceil(config["base_model_shots"] * (config["base_radius"] / radius) ** 2)))
                    exact_proposal = proposal_value(circuit, theta, shape, radius, coefficient,
                                                    config["optimizer"]["eta"], objective)
                    gaussian_seed = [config["seed"], 120000000, graph_index, depth, shape_index, radius_index]
                    gaussian = []
                    if row["split"] == config["gaussian_population"]:
                        rng = np.random.default_rng(np.random.SeedSequence(gaussian_seed))
                        for _ in range(config["gaussian_repetitions"]):
                            gaussian_means = means + rng.normal(size=len(means)) * np.sqrt(variances / schedule)
                            gaussian.append(proposal_value(circuit, theta, shape, radius, inverse @ gaussian_means,
                                                            config["optimizer"]["eta"], objective))
                    predictions.append({"id": len(predictions), "graph": row["id"], "split": row["split"],
                        "depth": depth, "shape_kind": shape_kind, "shape": shape.tolist(), "radius": radius,
                        "scheduled_shots": schedule, "theta": theta.tolist(), "center_objective": objective,
                        "design_seed": design_seed, "points": points.tolist(), "design": design.tolist(),
                        "locations": [value.tolist() for value in locations], "L": mapping.tolist(),
                        "L_operator_norm": float(np.linalg.norm(mapping, 2)),
                        "L_frobenius_norm": float(np.linalg.norm(mapping, "fro")),
                        "exact_means": means.tolist(), "per_shot_variances": variances.tolist(),
                        "distributions": [{"outcomes": values.tolist(), "probabilities": probabilities.tolist()}
                                          for values, probabilities in distributions],
                        "true_gradient": estimates[0].tolist(),
                        "exact_fitted_gradient": predicted["exact_fitted_gradient"],
                        "exact_bias_sq": predicted["exact_bias_sq"],
                        "variance_trace_per_shot": predicted["sampling_mse"],
                        "exact_acceptance_margin": exact_proposal["acceptance_margin"],
                        "exact_true_decrease": exact_proposal["true_decrease"],
                        "exact_proposal": exact_proposal, "gaussian_seed": gaussian_seed,
                        "gaussian_proposals": gaussian,
                        "exact_calls": len(locations) + int(exact_proposal["predicted"] >= 1e-12) +
                                       sum(part["predicted"] >= 1e-12 for part in gaussian),
                        "gaussian_positive_margin_probability": (float(np.mean([part["acceptance_margin"] > 0
                                                                                 for part in gaussian])) if gaussian else None)})
        print(f"mechanism predictions {graph_index + 1}/{len(data['instances'])}", flush=True)
    return predictions, gradients


def observe_mechanism(data):
    config, diagnostics, proposals = data["config"], [], []
    circuits = {row["id"]: MaxCutQAOA(graph_from(row)) for row in data["instances"]}
    for prediction in data["predictions"]:
        design, mapping = np.asarray(prediction["design"]), np.asarray(prediction["L"])
        inverse = np.linalg.solve(design, np.eye(len(design)))
        shape = np.asarray(prediction["shape"])
        exact = np.asarray(prediction["exact_fitted_gradient"])
        truth = np.asarray(prediction["true_gradient"])
        distributions = [(np.asarray(row["outcomes"]), np.asarray(row["probabilities"]))
                         for row in prediction["distributions"]]
        for shots in config["model_shots"]:
            samples, observed_proposals = [], []
            sampling_errors, total_errors = [], []
            seed = [config["seed"], 130000000, prediction["id"], shots]
            rng = np.random.default_rng(np.random.SeedSequence(seed))
            for replicate in range(config["fit_repetitions"]):
                batches = [sample_distribution(distribution, shots, rng) for distribution in distributions]
                means = np.array([batch["mean"] for batch in batches])
                gradient = mapping @ means
                sampling_error = float(np.sum((gradient - exact) ** 2))
                total_error = float(np.sum((gradient - truth) ** 2))
                sampling_errors.append(sampling_error)
                total_errors.append(total_error)
                samples.append({"means": means.tolist(),
                                "variance_of_means": [batch["variance_of_mean"] for batch in batches],
                                "sampling_squared_error": sampling_error, "total_squared_error": total_error})
                if shots == prediction["scheduled_shots"] and replicate < config["proposal_repetitions"]:
                    observed_proposals.append(proposal_value(circuits[prediction["graph"]],
                        np.asarray(prediction["theta"]), shape, prediction["radius"], inverse @ means,
                        config["optimizer"]["eta"], prediction["center_objective"]))
            prediction_sampling = prediction["variance_trace_per_shot"] / shots
            diagnostics.append({"prediction_id": prediction["id"],
                **{key: prediction[key] for key in ("graph", "split", "depth", "shape_kind", "radius")},
                "model_shots": shots, "is_schedule": shots == prediction["scheduled_shots"],
                "repetitions": len(samples), "seed": seed, "samples": samples,
                "predicted_sampling_mse": prediction_sampling,
                "predicted_total_mse": prediction_sampling + prediction["exact_bias_sq"],
                "bias_squared": prediction["exact_bias_sq"], "propagated_variance": prediction_sampling,
                "L_operator_norm": prediction["L_operator_norm"],
                "L_frobenius_norm": prediction["L_frobenius_norm"],
                "observed_sampling_mse": float(np.mean(sampling_errors)),
                "observed_total_mse": float(np.mean(total_errors)),
                "rms_sampling_error": float(np.sqrt(np.mean(sampling_errors))),
                "rms_total_error": float(np.sqrt(np.mean(total_errors))),
                "exact_gradient_error": float(np.sqrt(prediction["exact_bias_sq"]))})
            if observed_proposals:
                proposals.append({"prediction_id": prediction["id"],
                    **{key: prediction[key] for key in ("graph", "split", "depth", "shape_kind", "radius")},
                    "model_shots": shots, "repetitions": len(observed_proposals), "proposals": observed_proposals,
                    "observed_positive_margin_fraction": float(np.mean([part["acceptance_margin"] > 0 for part in observed_proposals])),
                    "mean_acceptance_margin": float(np.mean([part["acceptance_margin"] for part in observed_proposals])),
                    "mean_true_decrease": float(np.mean([part["true_decrease"] for part in observed_proposals]))})
        if (prediction["id"] + 1) % 20 == 0:
            print(f"mechanism observations {prediction['id'] + 1}/{len(data['predictions'])}", flush=True)
    return diagnostics, proposals


def run_mechanism(output=MECHANISM_DATA):
    output = Path(output).resolve()
    output.relative_to(CODE_ROOT / "results")
    pending = output.with_name("." + output.name + ".pending")
    if output.exists() or pending.exists():
        raise ValueError("Mechanism output or interrupted evidence exists; choose a new output")
    bundles = historical_bundles()
    config = configuration(bundles["followup"])
    targets, priors = prepare_targets(config, bundles)
    sources = {name: (CODE_ROOT / name).read_text() for name in SOURCES}
    data = {"schema_version": 1, "study": "mechanism", "status": "running", "started_utc": utc_now(),
            "config": config, "instances": targets, "priors": priors,
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "networkx": nx.__version__, "numerical_build": environment_details()},
            "provenance": {"source_text": sources,
                "source_sha256": {name: hashlib.sha256(value.encode()).hexdigest() for name, value in sources.items()},
                "historical_sha256": {name: hashlib.sha256((CODE_ROOT / "results" / (name + ".json.gz")).read_bytes()).hexdigest()
                                      for name in HISTORICAL}},
            "ledger_attribution": ledger_attribution(bundles["refinement"]),
            "historical_comparison": historical_comparison(bundles["refinement"]),
            "attribution_runs": [], "predictions": [], "gradient_checks": [],
            "model_diagnostics": [], "schedule_proposals": []}
    data["online_power"], data["online_stage_limits"] = online_power(config, bundles["refinement"])
    data["extended_power"], _ = online_power(config, bundles["refinement"], extended=True)
    data["extended_power_source"] = {"bundle": "refinement", "field": "power", "J": 5,
                                    "note": "separate offline diagnostic, not an online policy"}
    data["plan_sha256"] = content_hash({"config": config, "instances": targets, "priors": priors,
                                        "attribution_specs": allocation_specs(config, bundles["followup"])})
    data["plan_frozen_utc"] = utc_now()
    data["execution_id"] = "mechanism-" + content_hash({"plan": data["plan_sha256"], "started": data["started_utc"]})[:20]
    start = time.perf_counter()
    write_bundle(pending, data)
    try:
        data["predictions"], data["gradient_checks"] = predict_mechanism(data)
        data["prediction_sha256"] = content_hash(data["predictions"])
        data["predictions_frozen_utc"] = utc_now()
        write_bundle(pending, data)
        data["observations_started_utc"] = utc_now()
        data["model_diagnostics"], data["schedule_proposals"] = observe_mechanism(data)
        write_bundle(pending, data)
        specs = allocation_specs(config, bundles["followup"])
        for index, spec in enumerate(specs):
            data["attribution_runs"].append(allocation_run(config, bundles["followup"], spec))
            if (index + 1) % 32 == 0:
                print(f"mechanism attribution {index + 1}/{len(specs)}", flush=True)
        data["status"] = "complete"
        data["finished_utc"] = utc_now()
        data["seconds"] = time.perf_counter() - start
        data["accounting"] = {"attribution_optimization_shots": sum(row["shots"] for row in data["attribution_runs"]),
            "attribution_validation_shots": sum(row["validation"]["shots"] for row in data["attribution_runs"]),
            "mechanism_observation_shots": sum(row["repetitions"] * (2 * row["depth"] + 1) * row["model_shots"]
                                               for row in data["model_diagnostics"]),
            "online_power_new_shots": 0,
            "gaussian_samples": sum(len(row["gaussian_proposals"]) for row in data["predictions"]),
            "scheduled_sampled_proposals": sum(row["repetitions"] for row in data["schedule_proposals"]),
            "prediction_exact_calls": sum(row["exact_calls"] for row in data["predictions"] + data["gradient_checks"]),
            "scheduled_proposal_diagnostic_exact_calls": sum(sum(part["predicted"] >= 1e-12 for part in row["proposals"])
                                                            for row in data["schedule_proposals"])}
        data["payload_sha256"] = content_hash({key: value for key, value in data.items() if key != "payload_sha256"})
        verify_mechanism(data)
        write_bundle(pending, data)
        os.link(pending, output)
        pending.unlink()
    except BaseException:
        data["status"] = "interrupted"
        write_bundle(pending, data)
        raise
    return data


def verify_mechanism(data, replay=False):
    if data.get("study") != "mechanism" or data.get("schema_version") != 1 or data.get("status") != "complete":
        raise ValueError("A complete mechanism study is required")
    if data["payload_sha256"] != content_hash({key: value for key, value in data.items() if key != "payload_sha256"}):
        raise ValueError("Mechanism payload hash mismatch")
    provenance = data["provenance"]
    if set(provenance["source_text"]) != set(SOURCES) or set(provenance["source_sha256"]) != set(SOURCES):
        raise ValueError("Mechanism source set mismatch")
    for name, digest in provenance["source_sha256"].items():
        if hashlib.sha256(provenance["source_text"][name].encode()).hexdigest() != digest:
            raise ValueError("Mechanism embedded source hash mismatch")
        if replay and hashlib.sha256((CODE_ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Full mechanism replay requires the recorded numerical sources: " + name)
    for name in HISTORICAL:
        if hashlib.sha256((CODE_ROOT / "results" / (name + ".json.gz")).read_bytes()).hexdigest() != provenance["historical_sha256"][name]:
            raise ValueError("Mechanism historical evidence identity mismatch")
    bundles = historical_bundles()
    config = data["config"]
    counters = {"checks": 0, "attribution_runs": len(data["attribution_runs"]),
                "predictions": len(data["predictions"]), "diagnostics": len(data["model_diagnostics"]),
                "replayed_runs": 0}
    compare_results(configuration(bundles["followup"]), config, counters, "mechanism protocol")
    specs = allocation_specs(config, bundles["followup"])
    if len(specs) != len(data["attribution_runs"]):
        raise ValueError("Incomplete controlled attribution")
    if data["plan_sha256"] != content_hash({"config": config, "instances": data["instances"], "priors": data["priors"],
                                           "attribution_specs": specs}):
        raise ValueError("Mechanism frozen plan mismatch")
    if data["prediction_sha256"] != content_hash(data["predictions"]):
        raise ValueError("Mechanism frozen prediction mismatch")
    if not data["started_utc"] <= data["plan_frozen_utc"] <= data["predictions_frozen_utc"] <= data["observations_started_utc"] <= data["finished_utc"]:
        raise ValueError("Mechanism chronology mismatch")
    compare_results(ledger_attribution(bundles["refinement"]), data["ledger_attribution"], counters, "saved ledger attribution")
    compare_results(historical_comparison(bundles["refinement"]), data["historical_comparison"], counters, "historical comparison")
    online, limits = online_power(config, bundles["refinement"])
    compare_results(online, data["online_power"], counters, "online power recomputation")
    compare_results(limits, data["online_stage_limits"], counters, "online stage limits")
    extended, _ = online_power(config, bundles["refinement"], extended=True)
    compare_results(extended, data["extended_power"], counters, "extended offline power recomputation")
    if len(data["predictions"]) != len(data["instances"]) * len(config["depths"]) * 2 * len(config["radii"]):
        raise ValueError("Incomplete mechanism prediction grid")
    if len(data["model_diagnostics"]) != len(data["predictions"]) * len(config["model_shots"]):
        raise ValueError("Incomplete mechanism observation grid")
    if len(data["schedule_proposals"]) != len(data["predictions"]):
        raise ValueError("Incomplete operating-schedule proposal grid")
    for prediction in data["predictions"]:
        matrix = gradient_map(prediction["design"], prediction["shape"], prediction["radius"])
        compare_results(matrix.tolist(), prediction["L"], counters, "gradient fitting map")
        expected = variance_prediction(matrix, prediction["exact_means"], prediction["per_shot_variances"],
                                       prediction["true_gradient"], 1)
        compare_results(expected["exact_bias_sq"], prediction["exact_bias_sq"], counters, "squared bias")
        compare_results(expected["sampling_mse"], prediction["variance_trace_per_shot"], counters, "propagated variance")
    for spec, result in zip(specs, data["attribution_runs"]):
        compare_results(spec, {key: result[key] for key in spec}, counters, "attribution specification")
        if result["shots"] != sum(row["shots"] for row in result["evaluations"]) or result["shots"] > config["budget"]:
            raise ValueError("Attribution measurement accounting mismatch")
        if replay:
            compare_results(allocation_run(config, bundles["followup"], spec), result, counters, "controlled optimizer replay")
            counters["replayed_runs"] += 1
    if replay:
        targets, priors = prepare_targets(config, bundles)
        compare_results(targets, data["instances"], counters, "fresh graph regeneration")
        compare_results(priors, data["priors"], counters, "fixed-bank prior regeneration")
        predictions, gradients = predict_mechanism(data)
        compare_results(predictions, data["predictions"], counters, "prediction replay")
        compare_results(gradients, data["gradient_checks"], counters, "reference derivative replay")
        observations, proposals = observe_mechanism(data)
        compare_results(observations, data["model_diagnostics"], counters, "independent-fit observation replay")
        compare_results(proposals, data["schedule_proposals"], counters, "sampled proposal replay")
    return counters


def load_mechanism(path=MECHANISM_DATA, verify=True):
    with gzip.open(path, "rt") as stream:
        data = json.load(stream)
    if verify:
        verify_mechanism(data)
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MECHANISM_DATA)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    if args.full and not args.verify:
        parser.error("--full requires --verify")
    if args.verify:
        print(json.dumps(verify_mechanism(load_mechanism(args.output, verify=False), replay=args.full), sort_keys=True))
    else:
        data = run_mechanism(args.output)
        print(json.dumps({"output": str(args.output), "seconds": data["seconds"], "runs": len(data["attribution_runs"])}, sort_keys=True))


if __name__ == "__main__":
    main()
