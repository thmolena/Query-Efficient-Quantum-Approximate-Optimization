"""Repository paths, source identities, and read-only canonical-bundle loading.

The editable source checkout is the reproducibility unit. Source hashes use
paths relative to ``code/``; generated result and manuscript files never enter
the numerical source identity. Run IDs identify one execution and its frozen
configuration rather than claiming that measured wall times are deterministic.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess


CODE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = CODE_ROOT.parent
PAPER_CONFIG = CODE_ROOT / "configs" / "paper.json"
CANONICAL_DATA = CODE_ROOT / "results" / "study.json.gz"
NUMERICAL_SOURCES = tuple("src/gcqaoa/" + name for name in
                          ("__init__.py", "provenance.py", "qaoa.py", "search.py", "experiment.py", "records.py"))
LEGACY_NUMERICAL_SOURCES = tuple("src/gcqaoa/" + name for name in
                                 ("__init__.py", "provenance.py", "qaoa.py", "search.py", "experiment.py"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_identity(paths=()):
    """Record the checked-out commit and tracked/untracked source dirty state."""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                         stderr=subprocess.DEVNULL, text=True).strip()
        command = ["git", "status", "--porcelain", "--untracked-files=all"]
        entire = subprocess.check_output(command, cwd=REPO_ROOT, text=True)
        selected = subprocess.check_output(command + ["--"] + ["code/" + p for p in paths],
                                           cwd=REPO_ROOT, text=True) if paths else entire
        return {"source_commit": commit, "source_dirty": bool(selected.strip()),
                "working_tree_dirty": bool(entire.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"source_commit": None, "source_dirty": None, "working_tree_dirty": None}


def execution_metadata(config_path):
    config_path = Path(config_path).resolve()
    config_relative = config_path.relative_to(CODE_ROOT).as_posix()
    hashes = {name: digest(CODE_ROOT / name) for name in (*NUMERICAL_SOURCES, config_relative)}
    metadata = {"config_path": config_relative, "source_sha256": hashes,
                "started_utc": datetime.now(timezone.utc).isoformat(), **git_identity(tuple(hashes))}
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    metadata["execution_run_id"] = "study-" + hashlib.sha256(encoded).hexdigest()[:20]
    return metadata


def run_id(execution_run_id, run):
    return (f"{execution_run_id}:{run['graph']}:p{run['depth']}:r{run['replicate']}:"
            f"b{run['budget']}:{run['method']}")


def verify_source_identity(data):
    """Verify complete frozen sources on disk or in their clean recorded commit.

    Historical verification proves the identity of the recorded implementation;
    it does not assert that newer code has identical numerical behavior. A clone
    must retain the recorded commit in the ancestry of ``main`` when current
    sources differ. No hashes or execution metadata are rewritten.
    """
    config_path = data.get("config_path")
    allowed_configs = {path.relative_to(CODE_ROOT).as_posix()
                       for path in (CODE_ROOT / "configs").glob("*.json")}
    if not isinstance(config_path, str) or config_path not in allowed_configs:
        raise ValueError("Recorded configuration must name an existing code/configs/*.json file")
    hashes = data.get("source_sha256")
    required_sets = [set((*sources, config_path))
                     for sources in (LEGACY_NUMERICAL_SOURCES, NUMERICAL_SOURCES)]
    if not isinstance(hashes, dict) or set(hashes) not in required_sets:
        raise ValueError("Saved study has an incomplete or unrecognized frozen source hash set")
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
           for value in hashes.values()):
        raise ValueError("Frozen source hashes must be lowercase SHA-256 hexadecimal strings")
    current = {}
    for name in hashes:
        path = CODE_ROOT / name
        if not path.is_file() or path.resolve() != path:
            raise ValueError(f"Frozen source is missing or is not a regular local source path: {name}")
        current[name] = path.read_bytes()
    matches_disk = all(hashlib.sha256(current[name]).hexdigest() == expected
                       for name, expected in hashes.items())
    commit = data.get("source_commit")
    verified = current
    mode = "working-tree"
    if not matches_disk:
        if data.get("source_dirty") is not False:
            raise ValueError("Changed sources require a recorded clean source commit (source_dirty must be false)")
        if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise ValueError("Changed sources require a valid recorded 40-hex Git source commit")

        def git_bytes(*arguments):
            return subprocess.check_output(["git", "--no-replace-objects", *arguments], cwd=REPO_ROOT,
                                           stderr=subprocess.PIPE)

        try:
            resolved = git_bytes("rev-parse", "--verify", commit + "^{commit}").decode().strip()
            if resolved != commit:
                raise ValueError("Recorded source identity is not an exact Git commit")
            git_bytes("merge-base", "--is-ancestor", commit, "refs/heads/main")
            entries = git_bytes("ls-tree", "-z", commit, "--", *("code/" + name for name in hashes))
            blobs = {}
            for entry in entries.split(b"\0"):
                if not entry:
                    continue
                metadata, filename = entry.split(b"\t", 1)
                mode_bits, kind, object_id = metadata.split()
                name = filename.decode()
                if mode_bits not in (b"100644", b"100755") or kind != b"blob":
                    raise ValueError(f"Historical source is not a regular Git blob: {name}")
                blobs[name] = object_id.decode()
            if set(blobs) != {"code/" + name for name in hashes}:
                raise ValueError("Recorded commit does not contain the complete frozen source set")
            verified = {name: git_bytes("cat-file", "blob", blobs["code/" + name]) for name in hashes}
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError("Current sources differ from this study; its recorded clean source commit "
                             "must be available as an ancestor of local main. Use a full Git clone "
                             "with the preserved history; do not replace the released data.") from error
        for name, expected in hashes.items():
            if hashlib.sha256(verified[name]).hexdigest() != expected:
                raise ValueError(f"Recorded Git commit does not match frozen source SHA256: {name}")
        mode = "recorded-commit"
    config = json.loads(verified[config_path])
    if data.get("config") != config:
        raise ValueError("Saved configuration differs from its verified configuration source")
    return {"mode": mode, "source_commit": commit, "verified_sha256": dict(hashes), "config": config}


def load_study(path=CANONICAL_DATA, *, verify_sources=True):
    """Read a bundle and verify its current or preserved historical sources."""
    with gzip.open(path, "rt") as stream:
        data = json.load(stream)
    if verify_sources:
        verify_source_identity(data)
    return data


def safe_output(config_path, output=None, *, execution_id=None):
    """Require a fresh execution directory and never replace saved evidence."""
    config_path = Path(config_path).resolve()
    config_path.relative_to(CODE_ROOT / "configs")
    if output is None:
        execution_id = execution_id or execution_metadata(config_path)["execution_run_id"]
        if not re.fullmatch(r"study-[0-9a-f]{20}", execution_id):
            raise ValueError("Invalid execution ID")
        output = CODE_ROOT / "results" / "executions" / execution_id / "study.json.gz"
    output = Path(output).resolve()
    output.relative_to(CODE_ROOT / "results")
    if output == CANONICAL_DATA:
        raise ValueError("The released paper bundle is immutable; use a new execution directory")
    if output.name != "study.json.gz":
        raise ValueError("Output must be named study.json.gz inside a new execution directory")
    if output.exists() or output.parent.exists():
        raise ValueError("Execution directory already exists; choose a new destination")
    return output
