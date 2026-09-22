"""Repository paths, source identities, and read-only canonical-bundle loading.

Source hashes use paths relative to ``code/``. Supplied experiments retain
their exact recorded source bytes in portable snapshots; Git history is not
needed to verify them. Run IDs identify one execution and its frozen
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
SOURCE_SNAPSHOTS = CODE_ROOT / "data" / "source_snapshots"
NUMERICAL_SOURCES = tuple("src/gcqaoa/" + name for name in
                          ("__init__.py", "provenance.py", "qaoa.py", "search.py", "experiment.py", "records.py"))
LEGACY_NUMERICAL_SOURCES = tuple("src/gcqaoa/" + name for name in
                                 ("__init__.py", "provenance.py", "qaoa.py", "search.py", "experiment.py"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_identity(paths=()):
    """Record the checked-out commit and tracked/untracked source dirty state."""
    unknown = {"source_commit": None, "source_dirty": None, "working_tree_dirty": None}
    if not (REPO_ROOT / ".git").exists():
        return unknown
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
        return unknown


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


def source_snapshot_id(hashes):
    """Identify a complete frozen hash mapping independently of Git metadata."""
    encoded = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_hashes(hashes, required_paths):
    if not isinstance(hashes, dict) or not hashes or set(hashes) != set(required_paths):
        raise ValueError("Saved study has an incomplete or unrecognized frozen source hash set")
    if any(not isinstance(name, str) or Path(name).is_absolute() or
           Path(name).as_posix() != name or ".." in Path(name).parts for name in hashes):
        raise ValueError("Frozen source paths must be normalized relative local paths")
    if any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
           for value in hashes.values()):
        raise ValueError("Frozen source hashes must be lowercase SHA-256 hexadecimal strings")


def _read_sources(root, hashes):
    sources = {}
    for name in hashes:
        path = root / name
        if not path.is_file() or path.resolve() != path:
            raise ValueError(f"Frozen source is missing or is not a regular local source path: {name}")
        sources[name] = path.read_bytes()
    return sources


def _snapshot_sources(hashes):
    snapshot = SOURCE_SNAPSHOTS / source_snapshot_id(hashes)
    if not snapshot.is_dir() or snapshot.resolve() != snapshot:
        raise ValueError("Current sources differ from this study and its source snapshot is missing. "
                         "Restore the complete code/data/source_snapshots directory from the same "
                         "public package; do not rewrite the recorded experiment hashes.")
    files = {path.relative_to(snapshot).as_posix() for path in snapshot.rglob("*")
             if path.is_file() or path.is_symlink()}
    if files != set(hashes):
        raise ValueError("Source snapshot must contain exactly its recorded frozen source set")
    sources = _read_sources(snapshot, hashes)
    for name, expected in hashes.items():
        if hashlib.sha256(sources[name]).hexdigest() != expected:
            raise ValueError(f"Source snapshot SHA-256 mismatch: {name}")
    return sources


def verify_frozen_sources(hashes, *, required_paths):
    """Verify source bytes on disk or in a portable, content-addressed snapshot.

    Matching current sources support new executions without a supplied snapshot.
    Historical sources are read as data and never imported or executed. A
    successful snapshot check proves byte identity, not behavioral equivalence
    between the archived implementation and the current one.
    """
    _validate_hashes(hashes, required_paths)
    current = _read_sources(CODE_ROOT, hashes)
    if all(hashlib.sha256(current[name]).hexdigest() == expected for name, expected in hashes.items()):
        sources, mode = current, "working-tree"
    else:
        sources, mode = _snapshot_sources(hashes), "source-snapshot"
    return {"mode": mode, "verified_sha256": dict(hashes), "sources": sources}


def verify_source_snapshots():
    """Check all supplied archives and their exact links to unchanged bundles."""
    manifest = json.loads((SOURCE_SNAPSHOTS / "manifest.json").read_text())
    entries = manifest.get("snapshots", [])
    if manifest.get("schema_version") != 1 or [row.get("study") for row in entries] != ["study", "followup", "transport"]:
        raise ValueError("Unrecognized supplied source-snapshot manifest")
    expected_files = {"manifest.json"}
    for row in entries:
        name = row["study"]
        if row.get("bundle_path") != f"results/{name}.json.gz":
            raise ValueError("Source-snapshot manifest has an invalid bundle path")
        bundle = CODE_ROOT / row["bundle_path"]
        if digest(bundle) != row.get("bundle_sha256"):
            raise ValueError(f"Source-snapshot bundle SHA-256 mismatch: {name}")
        with gzip.open(bundle, "rt") as stream:
            data = json.load(stream)
        hashes = data["provenance"]["source_sha256"] if name == "followup" else data["source_sha256"]
        _validate_hashes(hashes, hashes)
        if row.get("source_sha256") != hashes or row.get("snapshot_id") != source_snapshot_id(hashes):
            raise ValueError(f"Source snapshot does not identify the recorded sources: {name}")
        if name == "study" and any(row.get(key) != data.get(key) for key in ("source_commit", "source_dirty")):
            raise ValueError("Source snapshot changes the originally recorded Git metadata")
        _snapshot_sources(hashes)
        expected_files.update(row["snapshot_id"] + "/" + path for path in hashes)
    actual_files = {path.relative_to(SOURCE_SNAPSHOTS).as_posix() for path in SOURCE_SNAPSHOTS.rglob("*")
                    if path.is_file() or path.is_symlink()}
    if actual_files != expected_files:
        raise ValueError("Unexpected files in the supplied source snapshots")
    return entries


def verify_source_identity(data):
    """Verify the study's complete source set without requiring Git history.

    Originally recorded Git metadata is retained as historical information.
    Verification uses the source hashes, including the configuration source,
    and does not rewrite hashes or execution identifiers.
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
    result = verify_frozen_sources(hashes, required_paths=hashes)
    config = json.loads(result["sources"][config_path])
    if data.get("config") != config:
        raise ValueError("Saved configuration differs from its verified configuration source")
    return {"mode": result["mode"], "source_commit": data.get("source_commit"),
            "verified_sha256": result["verified_sha256"], "config": config}


def load_study(path=CANONICAL_DATA, *, verify_sources=True):
    """Read a bundle and verify its current or archived historical sources."""
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
