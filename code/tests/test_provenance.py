"""Verify portable source identity without rewriting historical experiment data."""
from contextlib import contextmanager
import copy
import gzip
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from gcqaoa import provenance


class SourceIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_code = provenance.CODE_ROOT
        cls.bundles = {}
        cls.bundle_digests = {}
        for name in ("study", "followup", "transport"):
            path = provenance.CODE_ROOT / f"results/{name}.json.gz"
            with gzip.open(path, "rt") as stream:
                recorded = json.load(stream)
            cls.bundles[name] = {"source_sha256": recorded["provenance"]["source_sha256"]} if name == "followup" else {
                key: recorded[key] for key in ("source_sha256", "source_commit", "source_dirty", "config_path", "config")
                if key in recorded}
            cls.bundle_digests[name] = provenance.digest(path)
        cls.historical = cls.bundles["study"]

    @contextmanager
    def portable_tree(self, snapshots=True):
        """Materialize only public source inputs in an actual directory without .git."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            code = root / "code"
            paths = set().union(*(set(data["source_sha256"]) for data in self.bundles.values()))
            for name in paths:
                path = code / name
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.original_code / name, path)
            archived = code / "data/source_snapshots"
            if snapshots:
                shutil.copytree(provenance.SOURCE_SNAPSHOTS, archived)
            with patch.multiple(provenance, CODE_ROOT=code, REPO_ROOT=root, SOURCE_SNAPSHOTS=archived), \
                    patch("gcqaoa.provenance.subprocess.check_output", side_effect=AssertionError("Git unnecessary")):
                self.assertFalse((root / ".git").exists())
                yield code

    def test_matching_checkout_needs_no_git_or_snapshot(self):
        data = copy.deepcopy(self.historical)
        with self.portable_tree(snapshots=False) as code:
            data["source_sha256"] = {name: provenance.digest(code / name)
                                     for name in (*provenance.NUMERICAL_SOURCES, data["config_path"])}
            data["source_commit"], data["source_dirty"] = None, None
            result = provenance.verify_source_identity(data)
        self.assertEqual(result["mode"], "working-tree")
        self.assertEqual(result["verified_sha256"], data["source_sha256"])

    def test_all_three_bundles_verify_without_git(self):
        with self.portable_tree():
            for name, data in self.bundles.items():
                before = copy.deepcopy(data)
                with self.subTest(study=name):
                    result = provenance.verify_frozen_sources(data["source_sha256"], required_paths=data["source_sha256"])
                    self.assertEqual(result["mode"], "source-snapshot")
                    self.assertEqual(result["verified_sha256"], data["source_sha256"])
                    self.assertEqual(data, before)
            result = provenance.verify_source_identity(self.historical)
            self.assertEqual(result["source_commit"], self.historical["source_commit"])

    def test_supplied_manifest_binds_archives_to_unmodified_bundles(self):
        with patch("gcqaoa.provenance.subprocess.check_output", side_effect=AssertionError("Git unnecessary")):
            entries = provenance.verify_source_snapshots()
        self.assertEqual([row["study"] for row in entries], ["study", "followup", "transport"])
        for row in entries:
            self.assertEqual(row["bundle_sha256"], self.bundle_digests[row["study"]])

    def test_changed_current_source_uses_archive_without_rewriting_it(self):
        with self.portable_tree() as code:
            changed = code / "src/gcqaoa/qaoa.py"
            payload = changed.read_bytes() + b"\n# Local development change.\n"
            changed.write_bytes(payload)
            config = code / self.historical["config_path"]
            revised_config = json.loads(config.read_text())
            revised_config["seed"] += 1
            config.write_text(json.dumps(revised_config))
            result = provenance.verify_source_identity(self.historical)
            self.assertEqual(result["mode"], "source-snapshot")
            self.assertEqual(result["config"], self.historical["config"])
            self.assertEqual(changed.read_bytes(), payload)
            self.assertEqual(json.loads(config.read_text()), revised_config)

    def test_new_execution_with_changed_source_verifies_against_current_files(self):
        with self.portable_tree(snapshots=False) as code:
            changed = code / "src/gcqaoa/qaoa.py"
            changed.write_bytes(changed.read_bytes() + b"\n# New execution implementation.\n")
            data = copy.deepcopy(self.historical)
            data["source_sha256"] = {name: provenance.digest(code / name) for name in data["source_sha256"]}
            data["source_commit"], data["source_dirty"] = None, None
            self.assertEqual(provenance.verify_source_identity(data)["mode"], "working-tree")

    def test_source_paths_and_complete_hash_set_are_required(self):
        for changed in ("missing", "extra", "traversal", "absolute", "bad_hash"):
            data = copy.deepcopy(self.historical)
            with self.subTest(changed=changed):
                if changed == "missing":
                    data["source_sha256"].pop("src/gcqaoa/qaoa.py")
                elif changed == "extra":
                    data["source_sha256"]["src/gcqaoa/report.py"] = "0" * 64
                elif changed in ("traversal", "absolute"):
                    data["config_path"] = "../configs/paper.json" if changed == "traversal" else "/configs/paper.json"
                else:
                    data["source_sha256"]["src/gcqaoa/qaoa.py"] = "not-a-hash"
                with self.assertRaises(ValueError):
                    provenance.verify_source_identity(data)

    def test_generic_verifier_rejects_unsafe_paths_even_in_claimed_required_set(self):
        for name in ("../outside.py", "/outside.py", "src/../outside.py", "src//outside.py"):
            with self.subTest(path=name), self.assertRaisesRegex(ValueError, "normalized relative"):
                provenance.verify_frozen_sources({name: "0" * 64}, required_paths=(name,))

    def test_missing_snapshot_has_actionable_error(self):
        with self.portable_tree(snapshots=False):
            with self.assertRaisesRegex(ValueError, "Restore the complete code/data/source_snapshots"):
                provenance.verify_source_identity(self.historical)

    def test_snapshot_hash_mismatch_fails(self):
        with self.portable_tree():
            snapshot = provenance.SOURCE_SNAPSHOTS / provenance.source_snapshot_id(self.historical["source_sha256"])
            changed = snapshot / "src/gcqaoa/qaoa.py"
            changed.write_bytes(changed.read_bytes() + b"\n# Corrupted snapshot.\n")
            with self.assertRaisesRegex(ValueError, "snapshot SHA-256 mismatch"):
                provenance.verify_source_identity(self.historical)

    def test_snapshot_rejects_extra_and_missing_files(self):
        for changed in ("extra", "missing"):
            with self.subTest(changed=changed), self.portable_tree():
                snapshot = provenance.SOURCE_SNAPSHOTS / provenance.source_snapshot_id(self.historical["source_sha256"])
                if changed == "extra":
                    (snapshot / "extra.txt").write_text("unexpected")
                else:
                    (snapshot / "src/gcqaoa/qaoa.py").unlink()
                with self.assertRaisesRegex(ValueError, "exactly its recorded"):
                    provenance.verify_source_identity(self.historical)

    def test_snapshot_symlink_is_not_a_source_file(self):
        with self.portable_tree() as code:
            snapshot = provenance.SOURCE_SNAPSHOTS / provenance.source_snapshot_id(self.historical["source_sha256"])
            changed = snapshot / "src/gcqaoa/qaoa.py"
            changed.unlink()
            changed.symlink_to(code / "src/gcqaoa/qaoa.py")
            with self.assertRaisesRegex(ValueError, "regular local source"):
                provenance.verify_source_identity(self.historical)

    def test_archive_does_not_excuse_changed_recorded_hashes_or_configuration(self):
        for changed in ("hash", "configuration"):
            data = copy.deepcopy(self.historical)
            if changed == "hash":
                data["source_sha256"]["src/gcqaoa/qaoa.py"] = "0" * 64
            else:
                data["config"]["seed"] += 1
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                provenance.verify_source_identity(data)

    def test_no_git_directory_records_unknown_git_identity(self):
        with self.portable_tree(snapshots=False):
            self.assertEqual(provenance.git_identity(), {
                "source_commit": None, "source_dirty": None, "working_tree_dirty": None})


if __name__ == "__main__":
    unittest.main()
