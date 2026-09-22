"""Durable, append-only observations for new experiment executions.

Each completed sample is flushed before the optimizer continues. An orderly
interrupt preserves its preceding observations and labels incomplete work;
an uncatchable termination can leave a running status, but does not turn the
last durable observation into a successful result. Existing executions are
never reused. The released compressed study remains an independent artifact.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile


def environment_details():
    """Capture numerical build identities without compiler flags or local paths."""
    import numpy
    import scipy

    result = {}
    for name, module in (("numpy", numpy), ("scipy", scipy)):
        configuration = module.show_config(mode="dicts")
        compilers = {language: {key: value for key, value in compiler.items()
                                if key in ("name", "version", "linker")}
                     for language, compiler in configuration.get("Compilers", {}).items()}
        machine = configuration.get("Machine Information", {})
        architecture = {kind: {key: value for key, value in machine.get(kind, {}).items()
                              if key in ("cpu", "family", "endian", "system")}
                        for kind in ("host", "build")}
        dependencies = {}
        for kind in ("blas", "lapack"):
            source = configuration.get("Build Dependencies", {}).get(kind, {})
            details = {key: value for key, value in source.items()
                       if key in ("name", "version", "found", "detection method", "has ilp64")}
            # A generic BLAS link can hide the backend; record resolved library
            # basenames only, never the containing installation directory.
            library_directory = source.get("lib directory")
            if library_directory:
                libraries = Path(library_directory).glob(f"lib{kind}.*")
                details["resolved_library_names"] = sorted({path.resolve().name for path in libraries})
            dependencies[kind] = details
        result[name] = {"version": module.__version__, "compilers": compilers,
                        "architecture": architecture, "linear_algebra": dependencies}
    return result


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _atomic_json(path, value):
    """Replace one snapshot after syncing its bytes, within the same directory."""
    encoded = _json_bytes(value)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".record-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class ExecutionRecorder:
    """Reserve a new execution directory and persist observations as they occur.

    ``target`` is the final compressed study path. Its parent must not exist.
    The runner retains responsibility for creating that final study only after
    successful completion; this recorder retains partial evidence separately.
    """

    def __init__(self, target):
        self.target = Path(target)
        self.directory = self.target.parent
        self.directory.mkdir(parents=True, exist_ok=False)
        self.active = None
        self._events = None
        self._counts = {"completed": 0, "failed": 0, "interrupted": 0}
        self._status = "running"
        self._error = None
        self._execution_id = None

    def start(self, data):
        if self._execution_id is not None:
            raise RuntimeError("Execution recording has already started")
        self._execution_id = data["execution_run_id"]
        _atomic_json(self.directory / "config.json", data["config"])
        self.checkpoint(data)

    def checkpoint(self, data):
        if data["execution_run_id"] != self._execution_id:
            raise ValueError("Execution identity changed during recording")
        metadata = {key: value for key, value in data.items() if key not in ("runs", "config")}
        metadata["record_schema"] = 1
        metadata["persistence"] = "Completed events are flushed and synced before optimization continues"
        _atomic_json(self.directory / "metadata.json", metadata)
        self._write_execution_summary()

    def _write_execution_summary(self):
        summary = {"execution_run_id": self._execution_id, "status": self._status,
                   "run_counts": self._counts.copy(),
                   "active_run_id": self.active["selection"]["run_id"] if self.active else None,
                   "study_file": self.target.name if self._status == "completed" and self.target.is_file() else None}
        if self._error is not None:
            summary["exception_class"] = self._error
        _atomic_json(self.directory / "summary.json", summary)

    def start_run(self, selection, data, initial, shape):
        if self.active is not None or self._status != "running":
            raise RuntimeError("A run is already active or the execution has ended")
        if data["execution_run_id"] != self._execution_id:
            raise ValueError("Run belongs to a different execution")
        identifier = selection["run_id"]
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]+", identifier):
            raise ValueError("Run identifier cannot be represented as a safe directory name")
        directory = self.directory / "runs" / identifier.replace(":", "_")
        directory.mkdir(parents=True, exist_ok=False)
        graph = next(row for row in data["instances"] if row["id"] == selection["graph"])
        reference = data["references"][f"{selection['graph']}:p{selection['depth']}"]
        array_value = lambda value: value.tolist() if hasattr(value, "tolist") else value
        _atomic_json(directory / "config.json", {"study": data["config"], "selection": selection,
                     "initial": array_value(initial), "shape": array_value(shape)})
        provenance_keys = ("execution_run_id", "config_path", "source_commit", "source_dirty",
                           "working_tree_dirty", "started_utc")
        _atomic_json(directory / "metadata.json", {
            "record_schema": 1, "selection": selection,
            "provenance": {key: data[key] for key in provenance_keys if key in data},
            "source_sha256": data["source_sha256"], "environment": data["environment"],
            "graph": graph, "reference": reference,
            "iteration_convention": "Zero-based optimizer iteration; COBYLA uses objective-call index",
            "offline_bank_exact_calls": sum(row["exact_calls"] for key, row in data["references"].items()
                                              if key.startswith("bank_")),
            "units": {"optimization": "finite shots, objective calls, simulated batches",
                      "validation": "separate independent final measurement",
                      "offline_reference": "exact simulator calls, not quantum shots"}})
        self.active = {"directory": directory, "selection": dict(selection), "event_count": 0,
                       "shots": 0, "calls": 0, "jobs": 0, "decisions": 0, "accepted": 0,
                       "last_incumbent": None, "validation": None}
        self._events = (directory / "events.jsonl").open("x")
        self._write_run_summary("running")
        self._write_execution_summary()

    def event(self, event_type, record):
        if self.active is None or self._events is None:
            raise RuntimeError("An observation requires an active run")
        resource = {"evaluation": "optimization", "incumbent": "classical",
                    "decision": "classical", "validation": "validation"}
        if event_type not in resource:
            raise ValueError("Unknown event type")
        if event_type == "validation" and self.active["validation"] is not None:
            raise ValueError("A final validation has already been recorded")
        event = {**record, **self.active["selection"], "event_index": self.active["event_count"],
                 "event_type": event_type, "resource_class": resource[event_type]}
        encoded = json.dumps(event, sort_keys=True, allow_nan=False) + "\n"
        self._events.write(encoded)
        self._events.flush()
        os.fsync(self._events.fileno())
        self._apply_event(event_type, record)

    def _apply_event(self, event_type, record):
        if event_type == "evaluation":
            updates = {"shots": self.active["shots"] + record["shots"],
                       "calls": self.active["calls"] + 1,
                       "jobs": max(self.active["jobs"], record["job"])}
        elif event_type == "incumbent":
            updates = {"last_incumbent": dict(record)}
        elif event_type == "decision":
            updates = {"decisions": self.active["decisions"] + 1,
                       "accepted": self.active["accepted"] + int(record["decision"] == "accepted")}
        else:
            updates = {"validation": dict(record)}
        self.active.update(updates)
        self.active["event_count"] += 1

    def _recover_active_counts(self):
        """Read the durable prefix if interruption preceded in-memory updates."""
        if self._events is not None:
            self._events.flush()
        active = self.active
        for key in ("event_count", "shots", "calls", "jobs", "decisions", "accepted"):
            active[key] = 0
        active["last_incumbent"] = active["validation"] = None
        reserved = set(active["selection"]) | {"event_type", "event_index", "resource_class"}
        lines = (active["directory"] / "events.jsonl").read_bytes().splitlines(keepends=True)
        for index, line in enumerate(lines):
            try:
                event = json.loads(line)
                if event["run_id"] != active["selection"]["run_id"] or event["event_index"] != index:
                    raise ValueError("Event identity or order is inconsistent")
                event_type = event["event_type"]
                if event_type not in ("evaluation", "incumbent", "decision", "validation"):
                    raise ValueError("Unknown event type")
                record = {key: value for key, value in event.items() if key not in reserved}
                self._apply_event(event_type, record)
            except (ValueError, KeyError, TypeError, UnicodeDecodeError):
                active["incomplete_event_line"] = index + 1
                break
            if not line.endswith(b"\n"):
                active["incomplete_event_line"] = index + 1

    def _write_run_summary(self, status, result=None, error=None):
        active = self.active
        excluded = ("evaluations", "incumbents", "decisions")
        summary = {key: value for key, value in (result or {}).items() if key not in excluded}
        summary.update(active["selection"])
        summary.update(status=status, event_count=active["event_count"],
                       completed_decisions=active["decisions"], accepted_decisions=active["accepted"],
                       last_incumbent=active["last_incumbent"])
        validation = active["validation"] or {}
        summary["resource_totals"] = {
            "optimization_shots": active["shots"], "optimization_calls": active["calls"],
            "optimization_jobs": active["jobs"],
            "validation_shots": validation.get("shots", 0),
            "validation_calls": validation.get("calls", 0),
            "validation_jobs": validation.get("jobs", 0),
            "total_shots": active["shots"] + validation.get("shots", 0),
            "total_calls": active["calls"] + validation.get("calls", 0),
            "total_jobs": active["jobs"] + validation.get("jobs", 0)}
        if validation:
            summary["validation"] = validation
        if error is not None:
            summary["exception_class"] = type(error).__name__
        if "incomplete_event_line" in active:
            summary["incomplete_event_line"] = active["incomplete_event_line"]
            summary["resource_accounting"] = "Only parseable event records are included in resource totals"
        _atomic_json(active["directory"] / "summary.json", summary)

    def finish_run(self, result):
        if self.active is None:
            raise RuntimeError("No run is active")
        if result["run_id"] != self.active["selection"]["run_id"]:
            raise ValueError("Result belongs to a different run")
        if "validation" in result and self.active["validation"] is None:
            self.event("validation", result["validation"])
        for key in ("shots", "calls", "jobs"):
            if result[key] != self.active[key]:
                raise ValueError("Completed result disagrees with durable observation counts")
        self._write_run_summary("completed", result=result)
        self._events.close()
        self._events = None
        self.active = None
        self._counts["completed"] += 1
        self._write_execution_summary()

    def finish(self, data):
        if self.active is not None:
            raise RuntimeError("Cannot finish an execution with an active run")
        self._status = "completed"
        self.checkpoint(data)

    def abort(self, data, error):
        status = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        self._status = status
        self._error = type(error).__name__
        if self.active is not None:
            try:
                summary_path = self.active["directory"] / "summary.json"
                saved = json.loads(summary_path.read_text()) if summary_path.exists() else {}
                # finish_run may have committed its complete summary just
                # before an interrupt. Preserve that already-finished run.
                if saved.get("status") != "completed":
                    self._recover_active_counts()
                    self._write_run_summary(status, error=error)
            finally:
                if self._events is not None:
                    self._events.close()
                self._events = None
            self.active = None
        self._counts = {key: 0 for key in ("completed", "failed", "interrupted")}
        for summary_path in self.directory.glob("runs/*/summary.json"):
            saved_status = json.loads(summary_path.read_text()).get("status")
            if saved_status in self._counts:
                self._counts[saved_status] += 1
        self.checkpoint(data)
