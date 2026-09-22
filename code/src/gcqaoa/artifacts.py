"""Derive public graph manifests, graph-unit statistics, and selected run exports.

The preserved compressed study is the complete released run bundle. Summary intervals
resample graphs after averaging repeated shot seeds; pairwise intervals use
the same graph on both sides. Exported event streams distinguish optimization,
independent validation, and offline exact reference costs explicitly.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np

from .provenance import CANONICAL_DATA, CODE_ROOT, REPO_ROOT, digest, load_study


def ci(values):
    """Seeded percentile bootstrap of graph means, preserving the paper protocol."""
    values = np.array(values, dtype=float)
    rng = np.random.default_rng(5713)
    samples = values[rng.integers(0, len(values), (10000, len(values)))].mean(1)
    return float(values.mean()), *map(float, np.quantile(samples, [.025, .975]))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def lineage(data, path):
    return {"execution_run_id": data["execution_run_id"],
            "study_path": Path(path).resolve().relative_to(REPO_ROOT).as_posix(),
            "study_sha256": digest(path), "config_path": data["config_path"],
            "config_sha256": data["source_sha256"][data["config_path"]],
            "source_commit": data["source_commit"], "source_dirty": data["source_dirty"]}


def generate_graphs(data, path=CANONICAL_DATA):
    """Export actual graphs/splits/labels, verifying graph-generation replay."""
    from .experiment import instances
    generated = instances(data["config"])
    for expected, actual in zip(generated, data["instances"]):
        if any(expected[key] != actual[key] for key in expected):
            raise ValueError(f"Graph generation does not reproduce {actual['id']}")
    if len(generated) != len(data["instances"]):
        raise ValueError("Graph-generation count mismatch")
    target = CODE_ROOT / "data" if Path(path).resolve() == CANONICAL_DATA else Path(path).resolve().parent / "data"
    provenance = lineage(data, path)
    graph_rows = [{key: row[key] for key in ("id", "split", "family", "n", "seed", "edges", "maxcut")}
                  for row in data["instances"]]
    target.mkdir(parents=True, exist_ok=True)
    with (target / "graphs.jsonl").open("w") as stream:
        for row in graph_rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    split_rows = {split: [row["id"] for row in data["instances"] if row["split"] == split]
                  for split in ("bank", "test")}
    write_json(target / "splits.json", {"provenance": provenance, "splits": split_rows,
                                        "disjointness": "pairwise exact graph nonisomorphism across both splits"})
    references = {key: reference for key, reference in data["references"].items()
                  if key.split(":p")[0] in split_rows["bank"]}
    write_json(target / "reference_bank.json", {"provenance": provenance, "references": references,
               "reference_status": "best found among saved exact-objective multistarts; not certified QAOA optima",
               "resource_units": "exact simulator objective calls; no assigned hardware-shot equivalent"})
    return [target / name for name in ("graphs.jsonl", "splits.json", "reference_bank.json")]


def metrics(run):
    validation = run["validation"]
    success = run["hitting_shots"] is not None
    return {"objective": run["objective"], "expected_ratio": run["expected_ratio"],
            "signed_reference_gap": run["signed_reference_gap"], "success_fraction": float(success),
            "capped_target_shots": run["hitting_shots"] if success else run["budget"],
            "optimization_shots": run["shots"], "optimization_calls": run["calls"],
            "optimization_jobs": run["jobs"], "validation_shots": validation["shots"],
            "validation_calls": validation["calls"], "validation_jobs": validation["jobs"],
            "total_shots": run["shots"] + validation["shots"],
            "total_calls": run["calls"] + validation["calls"],
            "total_jobs": run["jobs"] + validation["jobs"], "optimization_seconds": run["seconds"]}


def analyze(data, path=CANONICAL_DATA):
    """Write aggregate and paired CSVs with graphs as the resampling unit."""
    groups = {}
    for run in data["runs"]:
        key = (run["depth"], run["budget"], run["method"])
        groups.setdefault(key, {}).setdefault(run["graph"], []).append(metrics(run))
    graph_units = {}
    summaries = []
    for (p, budget, method), graphs in sorted(groups.items()):
        means = {graph: {field: float(np.mean([row[field] for row in graphs[graph]]))
                         for field in graphs[graph][0]} for graph in sorted(graphs)}
        graph_units[(p, budget, method)] = means
        row = {"execution_run_id": data["execution_run_id"], "depth": p, "budget": budget,
               "method": method, "n_graphs": len(graphs), "n_runs": sum(map(len, graphs.values())),
               "resampling_unit": "graph mean over shot repetitions"}
        for field in next(iter(means.values())):
            average, lower, upper = ci([values[field] for values in means.values()])
            row.update({field + "_mean": average, field + "_ci_low": lower, field + "_ci_high": upper})
        summaries.append(row)
    paired = []
    fields = ("objective", "expected_ratio", "signed_reference_gap", "success_fraction",
              "capped_target_shots", "optimization_shots", "optimization_calls", "optimization_jobs")
    for p, budget in itertools.product(data["config"]["depths"], data["config"]["budgets"]):
        for first, second in itertools.combinations(data["config"]["methods"], 2):
            a, b = graph_units[(p, budget, first)], graph_units[(p, budget, second)]
            if set(a) != set(b):
                raise ValueError("Paired methods do not contain the same graphs")
            for field in fields:
                average, lower, upper = ci([a[graph][field] - b[graph][field] for graph in sorted(a)])
                paired.append({"execution_run_id": data["execution_run_id"], "depth": p, "budget": budget,
                               "method_a": first, "method_b": second, "metric": field, "n_graphs": len(a),
                               "difference": "a_minus_b", "mean_difference": average,
                               "ci_low": lower, "ci_high": upper,
                               "smaller_is_better": field not in ("expected_ratio", "success_fraction"),
                               "interval": "unadjusted 95 percent paired graph percentile bootstrap"})
    target = Path(path).resolve().parent / "aggregate"
    write_csv(target / "summary.csv", summaries)
    write_csv(target / "paired_comparisons.csv", paired)
    return [target / "summary.csv", target / "paired_comparisons.csv"]


def export_run(data, data_path, selected_id, output=None):
    """Materialize one run's four interoperable files under ignored exports/."""
    matches = [run for run in data["runs"] if run["run_id"] == selected_id]
    if len(matches) != 1:
        raise ValueError("--run-id must identify exactly one saved run; use --list to inspect IDs")
    run = matches[0]
    target = Path(output).resolve() if output else CODE_ROOT / "results" / "exports" / selected_id.replace(":", "_")
    target.relative_to(CODE_ROOT / "results" / "exports")
    target.mkdir(parents=True, exist_ok=False)
    provenance = lineage(data, data_path)
    selection = {key: run[key] for key in ("run_id", "graph", "depth", "replicate", "budget", "method", "seed")}
    write_json(target / "config.json", {"study": data["config"], "selection": selection})
    graph = next(row for row in data["instances"] if row["id"] == run["graph"])
    write_json(target / "metadata.json", {"provenance": provenance, "selection": selection,
               "environment": data["environment"], "source_sha256": data["source_sha256"], "graph": graph,
               "reference": data["references"][f"{run['graph']}:p{run['depth']}"],
               "units": {"optimization": "finite shots, objective calls, simulated batches",
                         "validation": "separate independent final measurement",
                         "offline_reference": "exact simulator calls, not quantum shots"}})
    events = []
    for record in run["evaluations"]:
        events.append((record["cumulative_shots"], 0, {"event_type": "evaluation", "resource_class": "optimization", **record}))
    for record in run["decisions"]:
        events.append((record["shots"], 1, {"event_type": "decision", "resource_class": "classical", **record}))
    for record in run["incumbents"]:
        events.append((record["shots"], 2, {"event_type": "incumbent", "resource_class": "diagnostic_exact", **record}))
    ordered = [item[2] for item in sorted(events, key=lambda item: item[:2])]
    ordered.append({"event_type": "validation", "resource_class": "validation", **run["validation"]})
    with (target / "events.jsonl").open("w") as stream:
        for index, event in enumerate(ordered):
            stream.write(json.dumps({"event_index": index, "run_id": selected_id, **event}, allow_nan=False) + "\n")
    summary = {key: value for key, value in run.items() if key not in ("evaluations", "decisions", "incumbents")}
    summary["resource_totals"] = metrics(run)
    summary["offline_bank_exact_calls"] = sum(reference["exact_calls"] for key, reference in data["references"].items()
                                               if key.startswith("bank_"))
    summary["target_reference_exact_calls"] = data["references"][f"{run['graph']}:p{run['depth']}"]["exact_calls"]
    write_json(target / "summary.json", summary)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("graphs", "analyze", "export"))
    parser.add_argument("--data", type=Path, default=CANONICAL_DATA)
    parser.add_argument("--run-id")
    parser.add_argument("--list", action="store_true", help="list exportable run IDs")
    parser.add_argument("--output", type=Path, help="selected-run directory beneath code/results/exports")
    arguments = parser.parse_args(argv)
    data = load_study(arguments.data)
    if arguments.action == "graphs":
        for path in generate_graphs(data, arguments.data):
            print(path)
    elif arguments.action == "analyze":
        for path in analyze(data, arguments.data):
            print(path)
    elif arguments.list:
        for run in data["runs"]:
            print(run["run_id"])
    elif arguments.run_id:
        print(export_run(data, arguments.data, arguments.run_id, arguments.output))
    else:
        parser.error("export requires --run-id or --list")


if __name__ == "__main__":
    main()
