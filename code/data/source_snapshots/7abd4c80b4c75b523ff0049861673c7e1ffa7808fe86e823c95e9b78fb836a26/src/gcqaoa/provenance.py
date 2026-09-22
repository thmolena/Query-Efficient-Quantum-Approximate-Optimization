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
import subprocess


CODE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = CODE_ROOT.parent
PAPER_CONFIG = CODE_ROOT / "configs" / "paper.json"
CANONICAL_DATA = CODE_ROOT / "results" / "study.json.gz"
NUMERICAL_SOURCES = tuple("src/gcqaoa/" + name for name in
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


def load_study(path=CANONICAL_DATA, *, verify_sources=True):
    """Read a bundle; by default require its frozen sources to match disk."""
    with gzip.open(path, "rt") as stream:
        data = json.load(stream)
    if verify_sources:
        for name, expected in data["source_sha256"].items():
            if digest(CODE_ROOT / name) != expected:
                raise ValueError(f"Saved study does not match {name}; rerun the frozen experiment")
        config = json.loads((CODE_ROOT / data["config_path"]).read_text())
        if data["config"] != config:
            raise ValueError("Saved and public configurations differ")
    return data


def safe_output(config_path, output=None):
    """Keep all run outputs under results and protect the canonical paper run."""
    config_path = Path(config_path).resolve()
    config_path.relative_to(CODE_ROOT / "configs")
    if output is None:
        output = CANONICAL_DATA if config_path == PAPER_CONFIG else CODE_ROOT / "results" / config_path.stem / "study.json.gz"
    output = Path(output).resolve()
    output.relative_to(CODE_ROOT / "results")
    if output == CANONICAL_DATA and config_path != PAPER_CONFIG:
        raise ValueError("Only configs/paper.json may write the canonical paper bundle")
    return output
