"""Post hoc calibration and matched proposal-to-decision QAOA diagnostics.

All prediction inputs and saved proposal populations are inherited unchanged
from mechanism.json.gz. The new endpoint experiment freezes its protocol before
new measurements. It is an exact-target-informed conditional diagnostic, not
a new optimizer benchmark or a measurement-available controller.
"""
from __future__ import annotations

import argparse
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
from scipy.stats import hypergeom

from .experiment import graph_from
from .followup import content_hash, utc_now, write_bundle, power_decision, sample_distribution
from .mechanism import compare_results
from .provenance import CODE_ROOT, NUMERICAL_SOURCES
from .qaoa import MaxCutQAOA

DECISION_DATA = CODE_ROOT / "results" / "decision.json.gz"
MECHANISM_DATA = CODE_ROOT / "results" / "mechanism.json.gz"
SOURCES = (*NUMERICAL_SOURCES, "src/gcqaoa/followup.py", "src/gcqaoa/mechanism.py", "src/gcqaoa/decision.py")


def configuration():
    return {"version": 1, "seed": 2026092331, "budget": 65536,
            "endpoint_looks": [512, 2048, 8192, 16384],
            "endpoint_pair_budgets": [1024, 4096, 16384, 32768],
            "alpha": 0.05, "horizon": 16, "eta": 0.1,
            "populations": {"gaussian": 1, "sampled": 8},
            "rules": ["hoeffding", "bernstein"],
            "null_repetitions": 10000,
            "population_rule": "all fresh shaped actual-schedule conditions; all 256 saved Gaussian and 32 saved multinomial proposals per condition; no sign or success filtering",
            "pairing": "independent new endpoint streams per proposal and replay, shared across both confidence rules and budget prefixes",
            "prediction_scope": "post hoc comparison of previously frozen exact-target-informed Gaussian proposals with saved finite-shot proposals; not a fresh optimizer benchmark",
            "baseline_scope": "post hoc radius-only and radius-plus-depth condition means fitted on original eight diagnostic targets and evaluated on eight fresh targets",
            "uncertainty_scope": "graphs are population comparison units; endpoint/model repetitions estimate conditional simulation uncertainty; conditional null is not an approximation-error confidence interval"}


def load_source():
    with gzip.open(MECHANISM_DATA, "rt") as stream:
        return json.load(stream)


def stopping_savings(source):
    records = []
    for depth in (2, 3):
        for allocation in ("fixed", "radius"):
            groups = {policy: {(row["graph"], row["replicate"]): row for row in source["attribution_runs"]
                              if row["depth"] == depth and row["method"] == allocation + "_" + policy}
                      for policy in ("stop", "continue")}
            paired = []
            for key, stopped in groups["stop"].items():
                continued = groups["continue"][key]
                paired.append({"graph": key[0], "replicate": key[1], "panel": stopped["panel"],
                               "stop_shots": stopped["shots"], "continue_shots": continued["shots"],
                               "shot_difference": stopped["shots"] - continued["shots"],
                               "identical_output": stopped["theta"] == continued["theta"]})
            stop = float(np.mean([row["stop_shots"] for row in paired]))
            continued = float(np.mean([row["continue_shots"] for row in paired]))
            records.append({"depth": depth, "allocation": allocation, "mean_stop_shots": stop,
                            "mean_continue_shots": continued, "fraction_saved": 1 - stop / continued,
                            "paired": paired})
    return records


def conditional_null(k_observed, n_observed, k_gaussian, n_gaussian):
    """Exact null law for a frequency difference at a fixed condition.

    Conditional on total successes, equal Bernoulli probabilities imply a
    hypergeometric allocation between the independent two sample sizes.
    This is a reference under equality, not an interval for systematic bias.
    """
    total, successes = n_observed + n_gaussian, k_observed + k_gaussian
    lower, upper = max(0, successes - n_gaussian), min(n_observed, successes)
    support = np.arange(lower, upper + 1)
    probabilities = hypergeom.pmf(support, total, successes, n_observed)
    difference = support / n_observed - (successes - support) / n_gaussian
    return {"expected_abs_error": float(probabilities @ np.abs(difference)),
            "variance": float(probabilities @ difference ** 2),
            "total_successes": int(successes), "total_draws": int(total)}


def calibration_analysis(source, config):
    predictions = {row["id"]: row for row in source["predictions"]}
    observed = source["schedule_proposals"]
    records = []
    for row in observed:
        if row["split"] != "fresh":
            continue
        prediction = predictions[row["prediction_id"]]
        k_observed = sum(part["acceptance_margin"] > 0 for part in row["proposals"])
        k_gaussian = sum(part["acceptance_margin"] > 0 for part in prediction["gaussian_proposals"])
        n_observed, n_gaussian = len(row["proposals"]), len(prediction["gaussian_proposals"])
        rate = k_observed / n_observed
        null = conditional_null(k_observed, n_observed, k_gaussian, n_gaussian)
        for method in ("gaussian", "radius_depth", "radius_only"):
            training = [] if method == "gaussian" else [part for part in observed
                if part["split"] == "diagnostic" and part["shape_kind"] == row["shape_kind"]
                and part["radius"] == row["radius"]
                and (method == "radius_only" or part["depth"] == row["depth"])]
            probability = (k_gaussian / n_gaussian if method == "gaussian" else
                           float(np.mean([part["observed_positive_margin_fraction"] for part in training])))
            records.append({key: row[key] for key in ("prediction_id", "graph", "depth", "radius", "shape_kind")} | {
                "predictor": method, "predicted_probability": probability, "observed_probability": rate,
                "k_observed": k_observed, "n_observed": n_observed,
                "k_gaussian": k_gaussian, "n_gaussian": n_gaussian,
                "residual": rate - probability, "abs_error": abs(rate - probability),
                "brier_score": rate * (1 - probability) ** 2 + (1 - rate) * probability ** 2,
                "training_prediction_ids": [part["prediction_id"] for part in training],
                "null_expected_abs_error": null["expected_abs_error"], "null_variance": null["variance"]})
    null_records = []
    for shape in ("iso", "shape"):
        rows = [row for row in records if row["shape_kind"] == shape and row["predictor"] == "gaussian"]
        successes = np.array([row["k_observed"] + row["k_gaussian"] for row in rows])
        n_observed = np.array([row["n_observed"] for row in rows])
        n_gaussian = np.array([row["n_gaussian"] for row in rows])
        seed = [config["seed"], 100000000, int(shape == "shape")]
        rng = np.random.default_rng(np.random.SeedSequence(seed))
        draws = rng.hypergeometric(successes, n_observed + n_gaussian - successes, n_observed,
                                  size=(config["null_repetitions"], len(rows)))
        differences = draws / n_observed - (successes - draws) / n_gaussian
        means = np.mean(np.abs(differences), axis=1)
        observed_mae = float(np.mean([row["abs_error"] for row in rows]))
        null_records.append({"shape_kind": shape, "conditions": len(rows), "seed": seed,
            "observed_mae": observed_mae,
            "expected_null_mae": float(np.mean([row["null_expected_abs_error"] for row in rows])),
            "null_mae_quantiles": np.quantile(means, [0.025, 0.5, 0.975]).tolist(),
            "conditional_tail_frequency": float((1 + np.count_nonzero(means >= observed_mae)) / (len(means) + 1)),
            "null_mean_absolute_errors": means.tolist()})
    return records, null_records


def selected_conditions(source):
    predictions = {row["id"]: row for row in source["predictions"]}
    return [(predictions[row["prediction_id"]], row) for row in source["schedule_proposals"]
            if row["split"] == "fresh" and row["shape_kind"] == "shape"]


def available_looks(model_cost, pair_budget, config):
    if model_cost < 0 or model_cost > config["budget"]:
        raise ValueError("Model cost is outside the total attempt budget")
    allowance = min(pair_budget, config["budget"] - model_cost)
    return [look for look in config["endpoint_looks"] if 2 * look <= allowance]


def summarize_condition(prediction, population, proposals, samples, config):
    records = []
    model_cost = (2 * prediction["depth"] + 1) * prediction["scheduled_shots"]
    replay_config = {"power_looks": config["endpoint_looks"], "power_alpha": config["alpha"],
                     "power_horizon": config["horizon"], "optimizer": {"eta": config["eta"]}}
    for pair_budget in config["endpoint_pair_budgets"]:
        looks = available_looks(model_cost, pair_budget, config)
        cap = max(looks, default=0)
        for rule in config["rules"]:
            outcomes = []
            for proposal_index, (proposal, record) in enumerate(zip(proposals, samples)):
                decisions = [power_decision(replay["observations"], proposal["predicted"], replay_config, cap, rule)
                             for replay in record["replays"]]
                outcomes.append({"proposal_index": proposal_index,
                    "true_decrease": proposal["true_decrease"], "acceptance_margin": proposal["acceptance_margin"],
                    "decisions": [part["decision"] for part in decisions],
                    "endpoint_shots": [part["shots"] for part in decisions]})
            decisions = [decision for row in outcomes for decision in row["decisions"]]
            expenditures = [shots for row in outcomes for shots in row["endpoint_shots"]]
            gains = [row["true_decrease"] if decision == "accepted" else 0.0
                     for row in outcomes for decision in row["decisions"]]
            records.append({key: prediction[key] for key in ("graph", "depth", "radius", "shape_kind")} | {
                "prediction_id": prediction["id"], "population": population, "rule": rule,
                "model_shots": prediction["scheduled_shots"], "model_cost": model_cost,
                "endpoint_pair_budget": pair_budget, "available_pair_budget": min(pair_budget, config["budget"] - model_cost),
                "affordable_looks": looks, "n_proposals": len(proposals),
                "replays_per_proposal": config["populations"][population], "endpoint_replays": len(decisions),
                "acceptance_probability": float(np.mean([decision == "accepted" for decision in decisions])),
                "rejection_probability": float(np.mean([decision == "rejected" for decision in decisions])),
                "unresolved_fraction": float(np.mean([decision == "unresolved" for decision in decisions])),
                "mean_accepted_true_gain_pp": 100 * float(np.mean(gains)),
                "mean_endpoint_shots": float(np.mean(expenditures)),
                "mean_total_shots": model_cost + float(np.mean(expenditures)),
                "false_acceptances": sum(decision == "accepted" and row["acceptance_margin"] < 0
                                         for row in outcomes for decision in row["decisions"]),
                "proposal_outcomes": outcomes})
    return records


def joint_endpoint_experiment(source, config):
    rows = {row["id"]: row for row in source["instances"]}
    circuits, centers = {}, {}
    endpoint_samples, joint_records = [], []
    conditions = selected_conditions(source)
    for index, (prediction, observed) in enumerate(conditions):
        graph = prediction["graph"]
        if graph not in circuits:
            circuits[graph] = MaxCutQAOA(graph_from(rows[graph]))
        circuit = circuits[graph]
        center_key = (graph, prediction["depth"])
        if center_key not in centers:
            centers[center_key] = circuit.distribution(prediction["theta"])
        for population in config["populations"]:
            proposals = prediction["gaussian_proposals"] if population == "gaussian" else observed["proposals"]
            samples = []
            for proposal_index, proposal in enumerate(proposals):
                distributions = [centers[center_key], circuit.distribution(proposal["theta"])]
                seed = [config["seed"], 200000000, prediction["id"], int(population == "sampled"), proposal_index]
                replays = []
                for replicate, child in enumerate(np.random.SeedSequence(seed).spawn(config["populations"][population])):
                    streams = [np.random.default_rng(part) for part in child.spawn(2)]
                    observations, previous = [], 0
                    for look in config["endpoint_looks"]:
                        observations.append({"shots_per_point": look,
                            "batches": [sample_distribution(distribution, look - previous, rng)
                                        for distribution, rng in zip(distributions, streams)]})
                        previous = look
                    replays.append({"replicate": replicate, "observations": observations})
                record = {"prediction_id": prediction["id"], "population": population,
                          "proposal_index": proposal_index, "seed": seed, "replays": replays}
                samples.append(record)
                endpoint_samples.append(record)
            joint_records.extend(summarize_condition(prediction, population, proposals, samples, config))
        if (index + 1) % 8 == 0:
            print(f"matched decision conditions {index + 1}/{len(conditions)}", flush=True)
    return joint_records, endpoint_samples


def run_decision(output=DECISION_DATA):
    output = Path(output).resolve()
    output.relative_to(CODE_ROOT / "results")
    pending = output.with_name("." + output.name + ".pending")
    if output.exists() or pending.exists():
        raise ValueError("Decision output or interrupted evidence exists; choose a new output")
    source, config = load_source(), configuration()
    source_text = {name: (CODE_ROOT / name).read_text() for name in SOURCES}
    data = {"study": "decision", "schema_version": 1, "status": "running", "started_utc": utc_now(),
            "config": config,
            "environment": {"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "networkx": nx.__version__},
            "provenance": {"mechanism_sha256": hashlib.sha256(MECHANISM_DATA.read_bytes()).hexdigest(),
                "mechanism_payload_sha256": source["payload_sha256"], "source_text": source_text,
                "source_sha256": {name: hashlib.sha256(text.encode()).hexdigest() for name, text in source_text.items()}},
            "instances": [row for row in source["instances"] if row["split"] == "fresh"],
            "stopping_savings": stopping_savings(source), "joint_records": [], "endpoint_samples": []}
    data["calibration_records"], data["calibration_null"] = calibration_analysis(source, config)
    data["population_sha256"] = content_hash(selected_conditions(source))
    data["plan_sha256"] = content_hash({"config": config, "population_sha256": data["population_sha256"],
                                       "source_sha256": data["provenance"]["source_sha256"]})
    data["plan_frozen_utc"] = utc_now()
    data["execution_id"] = "decision-" + content_hash({"plan": data["plan_sha256"], "started": data["started_utc"]})[:20]
    write_bundle(pending, data)
    start = time.perf_counter()
    try:
        data["endpoint_sampling_started_utc"] = utc_now()
        data["joint_records"], data["endpoint_samples"] = joint_endpoint_experiment(source, config)
        generated = sum(sum(batch["shots"] for batch in observation["batches"])
                        for row in data["endpoint_samples"] for replay in row["replays"]
                        for observation in replay["observations"])
        data["accounting"] = {"new_endpoint_simulated_shots": generated,
            "new_model_shots": 0, "endpoint_replays": sum(len(row["replays"]) for row in data["endpoint_samples"]),
            "exact_distribution_queries": len(data["endpoint_samples"]) + len({(row["graph"], row["depth"]) for row in data["joint_records"]}),
            "note": "model costs in per-attempt totals are charged from the saved proposal protocol; new sampling generates endpoint streams only; all prefixes/rules reuse each stream"}
        data["status"] = "complete"
        data["finished_utc"] = utc_now()
        data["seconds"] = time.perf_counter() - start
        data["payload_sha256"] = content_hash({key: value for key, value in data.items() if key != "payload_sha256"})
        verify_decision(data)
        write_bundle(pending, data)
        os.link(pending, output)
        pending.unlink()
    except BaseException:
        data["status"] = "interrupted"
        write_bundle(pending, data)
        raise
    return data


def verify_decision(data, replay=False):
    if data.get("study") != "decision" or data.get("schema_version") != 1 or data.get("status") != "complete":
        raise ValueError("A complete matched-decision bundle is required")
    if data["payload_sha256"] != content_hash({key: value for key, value in data.items() if key != "payload_sha256"}):
        raise ValueError("Decision payload hash mismatch")
    provenance = data["provenance"]
    if set(provenance["source_text"]) != set(SOURCES) or set(provenance["source_sha256"]) != set(SOURCES):
        raise ValueError("Decision source set mismatch")
    for name, expected in provenance["source_sha256"].items():
        if hashlib.sha256(provenance["source_text"][name].encode()).hexdigest() != expected:
            raise ValueError("Decision embedded source hash mismatch")
        if replay and hashlib.sha256((CODE_ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Full decision replay requires the recorded numerical sources: " + name)
    if hashlib.sha256(MECHANISM_DATA.read_bytes()).hexdigest() != provenance["mechanism_sha256"]:
        raise ValueError("Decision source experiment identity mismatch")
    source, config = load_source(), data["config"]
    if source["payload_sha256"] != provenance["mechanism_payload_sha256"]:
        raise ValueError("Decision source payload identity mismatch")
    counters = {"checks": 0, "conditions": len(selected_conditions(source)),
                "joint_records": len(data["joint_records"]), "replayed": False}
    compare_results(configuration(), config, counters, "decision protocol")
    compare_results([row for row in source["instances"] if row["split"] == "fresh"], data["instances"], counters, "fresh target identity")
    compare_results(stopping_savings(source), data["stopping_savings"], counters, "stopping savings")
    calibration, null = calibration_analysis(source, config)
    compare_results(calibration, data["calibration_records"], counters, "prediction benchmarks")
    compare_results(null, data["calibration_null"], counters, "conditional Monte Carlo calibration")
    if data["population_sha256"] != content_hash(selected_conditions(source)):
        raise ValueError("Decision frozen proposal population mismatch")
    if data["plan_sha256"] != content_hash({"config": config, "population_sha256": data["population_sha256"],
                                           "source_sha256": provenance["source_sha256"]}):
        raise ValueError("Decision frozen plan mismatch")
    if not data["started_utc"] <= data["plan_frozen_utc"] <= data["endpoint_sampling_started_utc"] <= data["finished_utc"]:
        raise ValueError("Decision chronology mismatch")
    lookup = {}
    sample_keys = set()
    generated_shots = endpoint_replays = 0
    for sample in data["endpoint_samples"]:
        key = (sample["prediction_id"], sample["population"], sample["proposal_index"])
        if key in sample_keys:
            raise ValueError("Duplicate decision endpoint sample")
        sample_keys.add(key)
        for replicate, replay_record in enumerate(sample["replays"]):
            if replay_record["replicate"] != replicate or [part["shots_per_point"] for part in replay_record["observations"]] != config["endpoint_looks"]:
                raise ValueError("Decision endpoint replay schedule mismatch")
            previous = 0
            for observation in replay_record["observations"]:
                count = observation["shots_per_point"] - previous
                if len(observation["batches"]) != 2:
                    raise ValueError("Decision endpoint pair missing")
                for batch in observation["batches"]:
                    if (batch["shots"] != count or not -1 <= batch["mean"] <= 0 or
                            not 0 <= batch["variance_of_mean"] <= 1 / (4 * (count - 1)) + 1e-12):
                        raise ValueError("Decision endpoint batch accounting/moments mismatch")
                    generated_shots += batch["shots"]
                previous = observation["shots_per_point"]
            endpoint_replays += 1
        lookup.setdefault((sample["prediction_id"], sample["population"]), []).append(sample)
    reconstructed = []
    expected_keys = set()
    for prediction, observed in selected_conditions(source):
        for population in config["populations"]:
            proposals = prediction["gaussian_proposals"] if population == "gaussian" else observed["proposals"]
            samples = lookup[(prediction["id"], population)]
            expected_keys.update((prediction["id"], population, index) for index in range(len(proposals)))
            if len(samples) != len(proposals) or any(row["proposal_index"] != index or len(row["replays"]) != config["populations"][population]
                                                    for index, row in enumerate(samples)):
                raise ValueError("Decision endpoint sample coverage mismatch")
            reconstructed.extend(summarize_condition(prediction, population, proposals, samples, config))
    if sample_keys != expected_keys:
        raise ValueError("Decision endpoint population coverage mismatch")
    compare_results(reconstructed, data["joint_records"], counters, "decision/gain/cost reconstruction")
    if any(row["mean_total_shots"] > config["budget"] or row["mean_endpoint_shots"] > row["available_pair_budget"]
           for row in data["joint_records"]):
        raise ValueError("Decision budget exceeded")
    for row in data["joint_records"]:
        for proposal in row["proposal_outcomes"]:
            if any(shots > row["available_pair_budget"] or row["model_cost"] + shots > config["budget"]
                   for shots in proposal["endpoint_shots"]):
                raise ValueError("Individual proposal exceeds its decision/total budget")
    expected_accounting = {"new_endpoint_simulated_shots": generated_shots, "new_model_shots": 0,
        "endpoint_replays": endpoint_replays,
        "exact_distribution_queries": len(data["endpoint_samples"]) + len({(row["graph"], row["depth"]) for row in data["joint_records"]})}
    compare_results(expected_accounting, {key: data["accounting"][key] for key in expected_accounting}, counters, "decision resource accounting")
    if replay:
        records, samples = joint_endpoint_experiment(source, config)
        compare_results(records, data["joint_records"], counters, "joint decision replay")
        compare_results(samples, data["endpoint_samples"], counters, "fresh endpoint sample replay")
        counters["replayed"] = True
    return counters


def load_decision(path=DECISION_DATA, verify=True):
    with gzip.open(path, "rt") as stream:
        data = json.load(stream)
    if verify:
        verify_decision(data)
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DECISION_DATA)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    if args.full and not args.verify:
        parser.error("--full requires --verify")
    if args.verify:
        print(json.dumps(verify_decision(load_decision(args.output, verify=False), replay=args.full), sort_keys=True))
    else:
        data = run_decision(args.output)
        print(json.dumps({"output": str(args.output), "seconds": data["seconds"], "joint_records": len(data["joint_records"])}, sort_keys=True))


if __name__ == "__main__":
    main()
