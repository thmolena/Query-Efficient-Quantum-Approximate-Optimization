"""Verify durable interruption evidence and numerically inert observation hooks."""
from copy import deepcopy
import gzip
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import networkx as nx
import numpy as np

from gcqaoa.provenance import CODE_ROOT
from gcqaoa import experiment
from gcqaoa.qaoa import MaxCutQAOA
from gcqaoa.records import ExecutionRecorder, environment_details
from gcqaoa.search import optimize


def study_fixture():
    return {"execution_run_id": "study-record-test", "config": {"seed": 93},
            "config_path": "configs/smoke.json", "source_commit": "a" * 40,
            "source_dirty": False, "working_tree_dirty": False,
            "source_sha256": {"configs/smoke.json": "b" * 64},
            "environment": {"python": "test"}, "started_utc": "2026-01-01T00:00:00Z",
            "instances": [{"id": "test_graph", "edges": [[0, 1]], "n": 2}],
            "references": {"test_graph:p1": {"objective": -1, "exact_calls": 3},
                           "bank_graph:p1": {"exact_calls": 4}}, "priors": {}, "runs": []}


def selection_fixture(index=0):
    return {"run_id": f"study-record-test:test_graph:p1:r{index}:b100:a",
            "graph": "test_graph", "depth": 1, "replicate": index,
            "budget": 100, "method": "a", "seed": 93}


def read_json(path):
    return json.loads(path.read_text())


def settings_fixture():
    return {"radius": .18, "max_radius": .3, "min_radius": .01,
            "model_shots": 32, "baseline_shots": 128, "max_trials": 3,
            "acceptance_looks": [64, 256], "alpha": .05, "eta": .1}


def without_timing(value):
    if isinstance(value, dict):
        return {key: without_timing(item) for key, item in value.items() if key != "seconds"}
    if isinstance(value, list):
        return [without_timing(item) for item in value]
    return value


class DurableRecordTests(unittest.TestCase):
    def setUp(self):
        self.scratch = CODE_ROOT / ".validation"
        self.scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="records-", dir=self.scratch)
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()
        try:
            self.scratch.rmdir()
        except OSError:
            pass

    def start_recorder(self, name="execution", index=0):
        data = study_fixture()
        recorder = ExecutionRecorder(self.directory / name / "study.json.gz")
        recorder.start(data)
        recorder.start_run(selection_fixture(index), data, np.array([.3, .15]), np.eye(2))
        return recorder, data

    def test_existing_execution_is_never_overwritten(self):
        recorder, data = self.start_recorder()
        original = (recorder.directory / "config.json").read_bytes()
        with self.assertRaises(FileExistsError):
            ExecutionRecorder(recorder.target)
        recorder.abort(data, KeyboardInterrupt())
        self.assertEqual((recorder.directory / "config.json").read_bytes(), original)

    def test_failure_and_interruption_preserve_completed_sample_midbatch(self):
        for index, error_type in enumerate((ArithmeticError, KeyboardInterrupt, SystemExit)):
            with self.subTest(error=error_type.__name__):
                recorder, data = self.start_recorder(name=f"execution-{index}")
                run_directory = recorder.active["directory"]
                class FailingSampler:
                    calls = 0

                    def sample(self, theta, shots, rng):
                        self.calls += 1
                        if self.calls == 2:
                            raise error_type("private diagnostic must not enter public records")
                        return -.5, .01

                try:
                    optimize(FailingSampler(), [.3, .15], np.eye(2), settings_fixture(),
                             "trust", 93, 1000, observer=recorder.event)
                except BaseException as error:
                    recorder.abort(data, error)
                else:
                    self.fail("Injected sampler failure was not observed")
                events = [json.loads(line) for line in (run_directory / "events.jsonl").read_text().splitlines()]
                self.assertEqual([event["event_type"] for event in events], ["incumbent", "evaluation"])
                self.assertEqual([event["event_index"] for event in events], [0, 1])
                expected_status = "failed" if error_type is ArithmeticError else "interrupted"
                summary = read_json(run_directory / "summary.json")
                self.assertEqual(summary["status"], expected_status)
                self.assertEqual(summary["exception_class"], error_type.__name__)
                self.assertEqual(summary["resource_totals"]["optimization_shots"], 32)
                self.assertEqual(summary["resource_totals"]["optimization_calls"], 1)
                self.assertEqual(summary["resource_totals"]["optimization_jobs"], 1)
                self.assertEqual(summary["resource_totals"]["validation_shots"], 0)
                self.assertEqual(summary["last_incumbent"]["shots"], 0)
                self.assertEqual(read_json(recorder.directory / "summary.json")["run_counts"][expected_status], 1)
                self.assertFalse(recorder.target.exists())
                self.assertEqual({path.name for path in run_directory.iterdir()},
                                 {"config.json", "metadata.json", "events.jsonl", "summary.json"})
                self.assertNotIn("private diagnostic", (run_directory / "summary.json").read_text())

    def test_completed_nonsuccessful_run_survives_later_interruption(self):
        recorder, data = self.start_recorder()
        first_directory = recorder.active["directory"]
        evaluation = {"call": 1, "job": 1, "shots": 4, "cumulative_shots": 4,
                      "theta": [.3, .15], "stage": "model", "mean": -.5,
                      "variance_of_mean": .01, "seconds": .01}
        recorder.event("evaluation", evaluation)
        result = {**selection_fixture(), "shots": 4, "calls": 1, "jobs": 1, "stop": "budget",
                  "hitting_shots": None, "evaluations": [evaluation], "decisions": [], "incumbents": [],
                  "validation": {"shots": 16, "calls": 1, "jobs": 1, "mean": -.5}}
        recorder.finish_run(result)
        data["runs"].append(result)
        first_summary = (first_directory / "summary.json").read_bytes()
        recorder.start_run(selection_fixture(1), data, [.3, .15], np.eye(2))
        recorder.abort(data, KeyboardInterrupt())
        self.assertEqual((first_directory / "summary.json").read_bytes(), first_summary)
        summary = json.loads(first_summary)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["stop"], "budget")
        self.assertIsNone(summary["hitting_shots"])
        self.assertEqual(summary["resource_totals"]["total_shots"], 20)
        self.assertEqual(read_json(recorder.directory / "summary.json")["run_counts"],
                         {"completed": 1, "failed": 0, "interrupted": 1})

    def test_completed_event_is_synced_before_optimizer_continues(self):
        recorder, data = self.start_recorder()
        with patch("gcqaoa.records.os.fsync") as sync:
            recorder.event("incumbent", {"shots": 0, "theta": [.3, .15]})
            sync.assert_called_once()
            events = (recorder.active["directory"] / "events.jsonl").read_text()
            self.assertEqual(json.loads(events)["event_type"], "incumbent")
        recorder.abort(data, KeyboardInterrupt())

    def test_interrupt_after_sync_recovers_resource_counts_from_event_stream(self):
        recorder, data = self.start_recorder()
        run_directory = recorder.active["directory"]
        original_sync = os.fsync

        def sync_then_interrupt(descriptor):
            original_sync(descriptor)
            raise KeyboardInterrupt()

        with patch("gcqaoa.records.os.fsync", side_effect=sync_then_interrupt), self.assertRaises(KeyboardInterrupt):
            recorder.event("evaluation", {"call": 1, "job": 1, "shots": 32,
                                           "cumulative_shots": 32, "mean": -.5})
        self.assertEqual(recorder.active["shots"], 0)
        recorder.abort(data, KeyboardInterrupt())
        summary = read_json(run_directory / "summary.json")
        self.assertEqual(summary["event_count"], 1)
        self.assertEqual(summary["resource_totals"]["optimization_shots"], 32)
        self.assertEqual(summary["resource_totals"]["optimization_calls"], 1)

    def test_partial_trailing_event_is_preserved_and_flagged(self):
        recorder, data = self.start_recorder()
        run_directory = recorder.active["directory"]
        recorder.event("evaluation", {"call": 1, "job": 1, "shots": 32,
                                       "cumulative_shots": 32, "mean": -.5})
        recorder._events.write('{"event_type": "evaluation", "shots":')
        recorder.abort(data, KeyboardInterrupt())
        summary = read_json(run_directory / "summary.json")
        self.assertEqual(summary["incomplete_event_line"], 2)
        self.assertEqual(summary["resource_totals"]["optimization_shots"], 32)
        self.assertTrue((run_directory / "events.jsonl").read_text().endswith('"shots":'))

    def test_interrupt_after_run_summary_commit_preserves_completed_status(self):
        recorder, data = self.start_recorder()
        directory = recorder.active["directory"]
        result = {**selection_fixture(), "shots": 0, "calls": 0, "jobs": 0, "stop": "budget"}
        write_summary = recorder._write_run_summary

        def interrupt_after_write(*args, **kwargs):
            write_summary(*args, **kwargs)
            raise KeyboardInterrupt()

        with patch.object(recorder, "_write_run_summary", side_effect=interrupt_after_write):
            with self.assertRaises(KeyboardInterrupt):
                recorder.finish_run(result)
        completed = (directory / "summary.json").read_bytes()
        recorder.abort(data, KeyboardInterrupt())
        self.assertEqual((directory / "summary.json").read_bytes(), completed)
        self.assertEqual(read_json(recorder.directory / "summary.json")["run_counts"],
                         {"completed": 1, "failed": 0, "interrupted": 0})

    def test_checkpoint_does_not_duplicate_all_completed_runs(self):
        recorder, data = self.start_recorder()
        data["references"]["another:p1"] = {"exact_calls": 11}
        data["runs"] = [{"not": "duplicated"}]
        recorder.checkpoint(data)
        metadata = read_json(recorder.directory / "metadata.json")
        self.assertNotIn("runs", metadata)
        self.assertEqual(metadata["references"]["another:p1"]["exact_calls"], 11)
        recorder.abort(data, KeyboardInterrupt())
        self.assertFalse(any(path.name.startswith(".record-") for path in recorder.directory.rglob("*")))

    def test_experiment_interrupt_handler_preserves_optimizer_observation(self):
        data = study_fixture()
        config = {"seed": 93, "depths": [1], "repetitions": 1, "budgets": [1000],
                  "methods": ["descriptor_iso"], "optimizer": settings_fixture(),
                  "validation_shots": 16, "target_gap": .02}
        graphs = [{"id": "bank_graph", "split": "bank", "n": 4,
                   "edges": [[0, 1], [1, 2], [2, 3]]},
                  {"id": "test_graph", "split": "test", "n": 4,
                   "edges": [[0, 1], [1, 2], [2, 3], [0, 3]]}]
        metadata = {key: data[key] for key in ("execution_run_id", "config_path", "source_commit",
                    "source_dirty", "working_tree_dirty", "source_sha256", "started_utc")}
        recorder = ExecutionRecorder(self.directory / "integrated" / "study.json.gz")
        original_sample = MaxCutQAOA.sample
        calls = []

        def interrupted_sample(circuit, theta, shots, rng):
            calls.append(shots)
            if len(calls) == 2:
                raise KeyboardInterrupt()
            return original_sample(circuit, theta, shots, rng)

        def inexpensive_reference(circuit, depth, config, seed):
            return {"theta": [.3, .15], "objective": circuit.objective([.3, .15]),
                    "exact_calls": 1, "starts": [], "seconds": 0.}

        with patch.object(experiment, "instances", return_value=graphs), \
                patch.object(experiment, "reference", side_effect=inexpensive_reference), \
                patch.object(MaxCutQAOA, "sample", new=interrupted_sample), \
                self.assertRaises(KeyboardInterrupt):
            experiment.run(config, metadata=metadata, recorder=recorder)
        execution_summary = read_json(recorder.directory / "summary.json")
        self.assertEqual(execution_summary["status"], "interrupted")
        self.assertEqual(execution_summary["run_counts"]["interrupted"], 1)
        run_directory, = (recorder.directory / "runs").iterdir()
        run_summary = read_json(run_directory / "summary.json")
        self.assertEqual(run_summary["status"], "interrupted")
        self.assertEqual(run_summary["resource_totals"]["optimization_shots"], 32)
        events = [json.loads(line) for line in (run_directory / "events.jsonl").read_text().splitlines()]
        self.assertEqual([event["event_type"] for event in events], ["incumbent", "evaluation"])
        self.assertEqual(events[1]["iteration"], 0)
        self.assertEqual(events[1]["method"], "descriptor_iso")
        self.assertEqual(events[1]["graph"], "test_graph")
        self.assertFalse(recorder.target.exists())

    def test_main_marks_complete_only_after_compressed_bundle_exists(self):
        config = self.directory / "config.json"
        config.write_text(json.dumps({"seed": 93}))
        target = self.directory / "completed" / "study.json.gz"
        data = study_fixture()
        data["seconds"] = 0.
        original_link = os.link
        before_publication = []

        def fake_run(config, config_path, *, metadata, recorder):
            recorder.start(data)
            return data

        def inspect_publication(source, destination):
            before_publication.append(read_json(target.parent / "summary.json"))
            self.assertFalse(target.exists())
            original_link(source, destination)

        with patch.object(experiment, "execution_metadata", return_value=data), \
                patch.object(experiment, "safe_output", return_value=target), \
                patch.object(experiment, "run", side_effect=fake_run), \
                patch.object(experiment.os, "link", side_effect=inspect_publication):
            experiment.main(["--config", str(config)])
        self.assertEqual(before_publication[0]["status"], "running")
        self.assertIsNone(before_publication[0]["study_file"])
        summary = read_json(target.parent / "summary.json")
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["study_file"], target.name)
        with gzip.open(target, "rt") as stream:
            self.assertEqual(json.load(stream)["execution_run_id"], data["execution_run_id"])
        self.assertFalse((target.parent / ".study.json.gz.pending").exists())


class ObserverAndEnvironmentTests(unittest.TestCase):
    def test_observer_does_not_change_random_stream_or_optimizer_result(self):
        circuit = MaxCutQAOA(nx.path_graph(3))
        for method in ("trust", "cobyla", "spsa", "fixed_shots", "fixed_radius", "no_repair"):
            with self.subTest(method=method):
                events = []
                plain = optimize(circuit, [.3, .15], np.eye(2), settings_fixture(), method, 93, 1700)
                recorded = optimize(circuit, [.3, .15], np.eye(2), settings_fixture(), method, 93, 1700,
                                    observer=lambda kind, record: events.append((kind, deepcopy(record))))
                self.assertEqual(without_timing(plain), without_timing(recorded))
                for event_type, field in (("evaluation", "evaluations"), ("incumbent", "incumbents"),
                                          ("decision", "decisions")):
                    selected = [record for kind, record in events if kind == event_type]
                    self.assertEqual(selected, recorded[field])

    def test_environment_build_summary_omits_private_path_fields(self):
        metadata = {"Compilers": {"c": {"name": "clang", "version": "19", "linker": "ld64",
                                        "args": "-I/private/author/include"}},
                    "Build Dependencies": {"blas": {"name": "blas", "version": "3.9",
                                                       "include directory": "/private/author/include"}},
                    "Machine Information": {"host": {"cpu": "arm64", "system": "darwin"}},
                    "Python Information": {"path": "/private/author/python"}}
        with patch("numpy.show_config", return_value=metadata), patch("scipy.show_config", return_value=metadata):
            result = environment_details()
        encoded = json.dumps(result)
        self.assertNotIn("/private", encoded)
        self.assertEqual(result["numpy"]["compilers"]["c"]["name"], "clang")
        self.assertEqual(result["scipy"]["linear_algebra"]["blas"]["version"], "3.9")


if __name__ == "__main__":
    unittest.main()
