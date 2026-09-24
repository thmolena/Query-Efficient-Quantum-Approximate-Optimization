#!/usr/bin/env python3
"""Check public contents, manuscript dependencies, provenance, and optional numerical replay."""
import argparse
import gzip
import hashlib
from html import unescape
from html.parser import HTMLParser
import io
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit
import zipfile

from pypdf import PdfReader
from pypdf.generic import NullObject

from build_paper import ROOT, SUBMISSION, active_tex, embedded_pdfs, inline_bibliography, manuscript_files
sys.path.insert(0, str(ROOT / "code/src"))
from gcqaoa.provenance import verify_source_identity, verify_source_snapshots
from gcqaoa.report import (WEB_BEGIN, WEB_END, EXPERIMENTS_BEGIN, EXPERIMENTS_END,
                          website_results, website_additional)

MANIFEST = ROOT / "code/results/aggregate/release-manifest.json"
EXCLUDED = {".git", ".validation", ".venv", ".replay-env", "__pycache__", ".pytest_cache", ".render-cache",
            ".pdf-preview", "exports", "build", "arxiv"}
TITLE = "Query-Efficient Quantum Approximate Optimization via Graph-Conditioned Trust Regions"
AUTHOR = "Molena Huynh"


def load_rules(filename, *, root=ROOT):
    """Load optional private literals without disclosing their contents or path."""
    if filename is None:
        return ()
    try:
        path = Path(filename).resolve(strict=True)
        if path.is_relative_to(root.resolve()):
            raise ValueError("Private rules must be outside the repository")
        rules = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("Private rules could not be loaded") from None
    if not isinstance(rules, list) or not all(isinstance(rule, str) and rule.strip() for rule in rules):
        raise ValueError("Private rules must be a JSON array of nonempty literal phrases")
    return tuple(rules)


class PublicAudit:
    """Inspect content with diagnostics that contain no matching text or paths.

    Object identifiers are SHA-256 digests of inspected bytes, or Git object
    identifiers during history inspection. The optional rules stay external;
    they are never copied into manifests, reports, or command output.
    """

    def __init__(self, rules=()):
        self.rules = tuple(" ".join(rule.casefold().split()) for rule in rules)
        self.errors = set()
        # Superseded drafts legitimately carry an earlier title and, early on, no
        # title at all. Requiring every historical PDF to bear the final title is
        # not a release property, so identity checks apply to current content only.
        # Everything that matters for publication -- venue and administrative-status
        # text, nonstandard information fields, embedded objects, actions, links and
        # page text -- is still audited on history.
        self.historical = False

    def reject(self, reason, object_id):
        self.errors.add(f"{reason}; object {object_id}")

    def text(self, content, object_id):
        if not self.rules:
            return
        normalized = " ".join(unescape(content).casefold().split())
        for index, rule in enumerate(self.rules, 1):
            if rule in normalized:
                self.reject(f"Private rule {index} matched", object_id)

    def json_strings(self, value, object_id):
        if isinstance(value, str):
            self.text(value, object_id)
        elif isinstance(value, dict):
            for key, item in value.items():
                self.text(key, object_id)
                self.json_strings(item, object_id)
        elif isinstance(value, list):
            for item in value:
                self.json_strings(item, object_id)

    def manuscript(self, content, object_id):
        # Comments do not activate a class or bibliography style.
        active = active_tex(content)
        classes = re.findall(r"\\documentclass\s*(?:\[[^]]*\])?\s*\{([^}]+)\}", active)
        styles = re.findall(r"\\bibliographystyle\s*\{([^}]+)\}", active)
        try:
            inline = inline_bibliography(content)
            figures = embedded_pdfs(content)
        except ValueError:
            self.reject("Invalid self-contained manuscript assets or bibliography", object_id)
            return
        if classes != ["article"] or (styles not in ([], ["plainnat"]) if inline else styles != ["plainnat"]):
            self.reject("Manuscript must use the generic article class and inline or plainnat bibliography", object_id)
        for name, payload in figures.items():
            self.blob(name, payload)

    def pdf(self, payload, object_id, *, main=False):
        """Check PDF metadata, embedded objects, actions, links, and page text."""
        previous_logging = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            reader = PdfReader(io.BytesIO(payload), strict=True)
            if reader.is_encrypted:
                self.reject("Encrypted PDF cannot be audited", object_id)
                return
            page_ids = {(page.indirect_reference.idnum, page.indirect_reference.generation)
                        for page in reader.pages if page.indirect_reference is not None}

            def safe_destination(destination):
                if hasattr(destination, "get_object"):
                    destination = destination.get_object()
                if isinstance(destination, str):
                    named = reader.named_destinations.get(destination)
                    return named is not None and safe_destination(named.dest_array)
                if not isinstance(destination, (list, tuple)) or len(destination) < 2:
                    return False
                page, mode = destination[:2]
                if not hasattr(page, "idnum") or (page.idnum, page.generation) not in page_ids:
                    return False
                lengths = {"/Fit": 2, "/XYZ": 5, "/FitH": 3, "/FitV": 3, "/FitR": 6,
                           "/FitB": 2, "/FitBH": 3, "/FitBV": 3}
                return (lengths.get(mode) == len(destination) and
                        all(isinstance(value, (int, float, NullObject)) for value in destination[2:]))

            def safe_open_action(action):
                if hasattr(action, "get_object"):
                    action = action.get_object()
                if isinstance(action, dict):
                    return (set(action) == {"/S", "/D"} and action["/S"] == "/GoTo" and
                            safe_destination(action["/D"]))
                return safe_destination(action)

            metadata = reader.metadata or {}
            if main:
                allowed = {"/Title", "/Author", "/Subject", "/Keywords", "/Creator", "/Producer",
                           "/CreationDate", "/ModDate", "/Trapped"}
                if set(metadata) - allowed:
                    self.reject("Manuscript PDF has nonstandard information fields", object_id)
                if not self.historical:
                    if metadata.get("/Title") != TITLE or metadata.get("/Author") != AUTHOR:
                        self.reject("Manuscript PDF title or author metadata is inconsistent", object_id)
                    if not str(metadata.get("/Subject", "")).strip() or not str(metadata.get("/Keywords", "")).strip():
                        self.reject("Manuscript PDF needs scientific subject and keywords", object_id)
                factual = " ".join(str(metadata.get(key, "")) for key in
                                   ("/Title", "/Author", "/Subject", "/Keywords"))
                if re.search(r"\b(?:submitted|submission|under\s+review|accepted|rejected|resubmitted|"
                             r"decision|journal|venue|manuscript\s+(?:id|number))\b", factual, re.I):
                    self.reject("Manuscript PDF contains administrative status metadata", object_id)
            seen = set()

            def inspect(value):
                if hasattr(value, "idnum"):
                    identity = (value.idnum, value.generation)
                    if identity in seen:
                        return
                    seen.add(identity)
                    value = value.get_object()
                if isinstance(value, dict):
                    if "/EmbeddedFiles" in value or "/EF" in value or "/AF" in value or value.get("/Type") == "/EmbeddedFile":
                        self.reject("PDF contains embedded attachments", object_id)
                    if any(key in value for key in ("/JavaScript", "/JS", "/AA")):
                        self.reject("PDF contains automatic or scripted actions", object_id)
                    if "/OpenAction" in value and not safe_open_action(value["/OpenAction"]):
                        self.reject("PDF contains an unsafe document-opening action", object_id)
                    if value.get("/S") in {"/JavaScript", "/Launch", "/GoToR", "/GoToE", "/SubmitForm",
                                           "/ImportData", "/Rendition", "/Movie", "/Sound", "/Named"}:
                        self.reject("PDF contains an unsafe action", object_id)
                    for key, item in value.items():
                        self.text(str(key), object_id)
                        inspect(item)
                    if "/Metadata" in value:
                        packet = value["/Metadata"].get_object().get_data()
                        self.text(packet.decode("utf-8", "replace"), object_id)
                elif isinstance(value, (list, tuple)):
                    for item in value:
                        inspect(item)
                elif isinstance(value, str):
                    self.text(value, object_id)
                elif isinstance(value, bytes):
                    self.text(value.decode("utf-8", "replace"), object_id)

            inspect(reader.trailer)
            for page in reader.pages:
                self.text(page.extract_text(), object_id)
                annotations = page.get("/Annots", [])
                if hasattr(annotations, "get_object"):
                    annotations = annotations.get_object()
                for reference in annotations:
                    annotation = reference.get_object()
                    if annotation.get("/Subtype") != "/Link":
                        self.reject("PDF contains a non-link annotation", object_id)
                    action = annotation.get("/A")
                    if action is None:
                        continue
                    action = action.get_object()
                    if action.get("/S") not in {"/GoTo", "/URI"} or "/Next" in action:
                        self.reject("PDF link contains an unsafe action", object_id)
                    if action.get("/S") == "/URI":
                        uri = str(action.get("/URI", ""))
                        parsed = urlsplit(uri)
                        if (parsed.scheme not in {"https", "http", "mailto"} or
                                any(ord(character) < 32 for character in uri) or
                                (parsed.scheme in {"https", "http"} and
                                 (not parsed.netloc or parsed.username or parsed.password)) or
                                (parsed.scheme == "mailto" and not parsed.path)):
                            self.reject("PDF link has an unsafe URI", object_id)
        except Exception:
            # Parser exceptions can include text taken from the inspected PDF.
            self.reject("PDF could not be completely audited", object_id)
        finally:
            logging.disable(previous_logging)

    def blob(self, name, payload, *, object_id=None):
        object_id = object_id or hashlib.sha256(payload).hexdigest()
        self.text(name, object_id)
        lowered = name.lower()
        if lowered.endswith(".pdf"):
            self.pdf(payload, object_id, main=Path(name).name == "main.pdf")
        elif lowered.endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                    self.text(archive.comment.decode("utf-8", "replace"), object_id)
                    for member in archive.infolist():
                        self.text(member.filename, object_id)
                        self.text(member.comment.decode("utf-8", "replace"), object_id)
                        if not member.is_dir():
                            self.blob(member.filename, archive.read(member))
            except (OSError, ValueError, zipfile.BadZipFile):
                self.reject("Archive could not be completely audited", object_id)
        elif lowered.endswith(".gz"):
            if self.rules:
                try:
                    self.blob(name[:-3], gzip.decompress(payload))
                except (OSError, EOFError):
                    self.reject("Compressed evidence could not be completely audited", object_id)
        else:
            content = payload.decode("utf-8", "replace")
            self.text(content, object_id)
            if Path(name).name == "main.tex":
                self.manuscript(content, object_id)
            if self.rules and lowered.endswith((".json", ".jsonl")):
                try:
                    if lowered.endswith(".jsonl"):
                        for line in content.splitlines():
                            if line.strip():
                                self.json_strings(json.loads(line), object_id)
                    else:
                        self.json_strings(json.loads(content), object_id)
                except (ValueError, RecursionError):
                    self.reject("Structured evidence could not be completely audited", object_id)

    def history(self):
        """Audit only publication refs: local branches and tags, never private refs."""
        self.historical = True
        try:
            self._history()
        finally:
            self.historical = False

    def _history(self):
        def git(*arguments):
            return subprocess.check_output(["git", "--no-replace-objects", *arguments], cwd=ROOT,
                                           stderr=subprocess.PIPE)

        refs = git("for-each-ref", "--format=%(refname) %(objectname) %(objecttype)",
                   "refs/heads", "refs/tags").decode().splitlines()
        names = []
        for entry in refs:
            name, object_id, kind = entry.split()
            names.append(name)
            self.text(name, object_id)
            while kind == "tag":
                tag = git("cat-file", "tag", object_id).decode("utf-8", "replace")
                self.text(tag, object_id)
                header = dict(line.split(" ", 1) for line in tag.split("\n\n", 1)[0].splitlines())
                object_id, kind = header["object"], header["type"]
        if not names:
            self.reject("No publication history is available", "0" * 40)
            return
        seen = set()
        for commit in git("rev-list", *names).decode().splitlines():
            self.text(git("cat-file", "commit", commit).decode("utf-8", "replace"), commit)
            for entry in git("ls-tree", "-rz", "--full-tree", commit).split(b"\0"):
                if not entry:
                    continue
                metadata, raw_name = entry.split(b"\t", 1)
                _, kind, object_id = metadata.decode().split()
                name = raw_name.decode("utf-8", "replace")
                self.text(name, object_id)
                key = (object_id, Path(name).suffix.lower(), Path(name).name in {"main.tex", "main.pdf"})
                if kind == "blob" and key not in seen:
                    seen.add(key)
                    self.blob(name, git("cat-file", "blob", object_id), object_id=object_id)

    def finish(self):
        if self.errors:
            raise RuntimeError("\n".join(sorted(self.errors)))


def public_files():
    return sorted(path for path in ROOT.rglob("*") if path.is_file() and
                  not any(path.is_relative_to(ROOT / "code/results" / name) for name in ("smoke", "pilot", "scratch", "executions")) and
                  not any(part in EXCLUDED or part.endswith(".egg-info") or
                          part.startswith((".paper-", ".arxiv-")) for part in path.relative_to(ROOT).parts))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_bibliography(tex, bibliography, compiled=None):
    """Check citation coverage and duplicate identifiers, not scholarly support."""
    source = active_tex(tex)
    if re.search(r"\\nocite(?:\[[^]]*\])?\s*\{[^}]*\*", source):
        raise ValueError("Wildcard bibliography inclusion is not permitted")
    commands = re.findall(r"\\cite(?:[pt]|author|year(?:par)?)?\*?"
                          r"(?:\[[^]]*\]){0,2}\s*\{([^}]+)\}", source)
    cited = [key.strip() for command in commands for key in command.split(",")]
    starts = list(re.finditer(r"(?m)^\s*@([A-Za-z]+)\s*\{\s*([^,\s]+)\s*,", bibliography))
    entries, identifiers = {}, {}
    for index, match in enumerate(starts):
        kind, key = match.groups()
        if kind.casefold() in {"comment", "preamble", "string"}:
            continue
        if key in entries:
            raise ValueError(f"Duplicate bibliography key: {key}")
        entry = bibliography[match.end():starts[index + 1].start() if index + 1 < len(starts) else len(bibliography)]
        entries[key] = entry
        for field in ("author", "title", "year", "annote"):
            if not re.search(rf"(?im)^\s*{field}\s*=\s*\{{\s*[^}}\s]", entry):
                raise ValueError(f"Bibliography entry lacks {field}: {key}")
        for field in ("doi", "eprint"):
            value = re.search(rf'(?im)^\s*{field}\s*=\s*[{{"]([^}}"]+)', entry)
            if value is None:
                continue
            normalized = value.group(1).strip().casefold()
            normalized = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", normalized)
            normalized = re.sub(r"^arxiv:", "", normalized)
            if field == "eprint":
                normalized = re.sub(r"v\d+$", "", normalized)
            identity = field, normalized
            if identity in identifiers:
                raise ValueError(f"Duplicate {field} in {identifiers[identity]} and {key}")
            identifiers[identity] = key
    missing, unused = set(cited) - set(entries), set(entries) - set(cited)
    if missing or unused:
        raise ValueError(f"Citation coverage differs: missing={sorted(missing)}, unused={sorted(unused)}")
    if compiled is not None:
        rendered = re.findall(r"\\bibitem(?:\[[^]]*\])?\s*\{([^}]+)\}", active_tex(compiled))
        if len(rendered) != len(set(rendered)) or set(rendered) != set(cited):
            raise ValueError("Compiled bibliography differs from cited references")
    return {"distinct_references": len(entries), "citation_commands": len(commands),
            "reference_mentions": len(cited)}


def check_followup_replay(output, expected_runs):
    """Require a complete replay report, rather than a successful no-op exit."""
    try:
        result = json.loads(output)
    except (TypeError, ValueError):
        raise ValueError("Follow-up replay did not return a valid completion report") from None
    if (not isinstance(result, dict) or result.get("runs") != expected_runs or
            result.get("replayed_runs") != expected_runs or
            not isinstance(result.get("checks"), int) or result["checks"] <= 0):
        raise ValueError("Follow-up replay did not verify every saved run")
    return result


def check_transport_status(record, protocol):
    """Derive stopping claims independently of the saved convergence flags."""
    import math
    iterations=record['iterations']
    if type(iterations) is not int or type(record['feasible']) is not bool or type(record['converged']) is not bool:
        raise ValueError('Transport stopping fields have invalid types')
    if 'gap' in record:
        metric=record['gap']
        valid_range=0 <= iterations <= protocol['fw_limit']
        expected=metric <= protocol['fw_tolerance'] and record['feasible']
        if iterations != len(record['trace']):
            raise ValueError('Transport iteration count differs from its trace')
    else:
        metric=record['residual']
        valid_range=1 <= iterations <= protocol['outer_limit']
        expected=metric < protocol['outer_tolerance'] and record['feasible']
    if not valid_range or not math.isfinite(metric) or metric < 0:
        raise ValueError('Transport stopping diagnostic is outside its valid range')
    if record['converged'] != expected:
        raise ValueError('Transport convergence flag differs from its stopping diagnostic')


def compare_transport_replay(actual, expected):
    """Check every deterministic solver field, including all unselected starts."""
    import numpy as np
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual)!=set(expected):
            raise ValueError('Transport replay fields differ')
        for key in expected:
            compare_transport_replay(actual[key], expected[key])
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual)!=len(expected):
            raise ValueError('Transport replay array lengths differ')
        for a,b in zip(actual,expected):
            compare_transport_replay(a,b)
    elif type(expected) in (bool,int):
        if type(actual) is not type(expected) or actual!=expected:
            raise ValueError('Transport replay discrete diagnostics differ')
    elif isinstance(expected,float):
        if not np.isfinite(actual) or not np.isclose(actual,expected,rtol=2e-8,atol=2e-9):
            raise ValueError('Transport replay numerical diagnostics differ')
    elif actual!=expected:
        raise ValueError('Transport replay value differs')


def check_transport_diagnostics(data, *, replay=False):
    from gcqaoa.transport_audit import pair_solvers
    from gcqaoa.provenance import load_study
    import numpy as np
    for pair in data['pairs']:
        for solver in pair['solvers'].values():
            for record in solver.get('starts',[solver]):
                check_transport_status(record,data['protocol'])
    if replay:
        canonical=load_study()
        rows={r['id']:r for r in canonical['instances']}
        banks=[r['id'] for r in canonical['instances'] if r['split']=='bank']
        targets=[r['id'] for r in canonical['instances'] if r['split']=='test']
        for index,pair in enumerate(data['pairs'],1):
            seed=data['protocol']['seed']+100*targets.index(pair['test'])+3*banks.index(pair['bank'])
            if pair['seed']!=seed:
                raise ValueError('Transport seed does not match the frozen pair schedule')
            D,E=np.array(rows[pair['test']]['structure']),np.array(rows[pair['bank']]['structure'])
            fresh=pair_solvers(D,E,seed,data['protocol'])
            compare_transport_replay(pair['solvers'],fresh)
            if index%len(banks)==0:
                print(f'Transport replay: {index}/{len(data["pairs"])} pairs, including every start.',flush=True)


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets, self.ids = [], set()

    def handle_starttag(self, tag, attributes):
        values = dict(attributes)
        if "id" in values:
            self.ids.add(values["id"])
        for key in ("href", "src"):
            if key in values:
                self.targets.append(values[key])


def check(*, rules=(), history=False, allow_manifest_creation=False):
    files = public_files()
    audit = PublicAudit(rules)
    for path in files:
        audit.blob(path.relative_to(ROOT).as_posix(), path.read_bytes())
    if history:
        audit.history()
    # Stop before path-bearing technical diagnostics when private rules match.
    audit.finish()
    errors = []
    require = lambda condition, message: errors.append(message) if not condition else None
    visible = {path.name for path in ROOT.iterdir() if not path.name.startswith(".")}
    require(visible == {"README.md", "index.html", "code", "submission"}, "Exactly four visible root items")
    required = ["README.md", "index.html", "code/pyproject.toml", "code/LICENSE", "code/requirements-lock.txt",
                "code/data/graphs.jsonl", "code/data/splits.json", "code/data/source_snapshots/manifest.json",
                "code/data/reference_bank.json", "code/results/study.json.gz",
                "code/results/aggregate/summary.csv", "code/results/aggregate/paired_comparisons.csv",
                "code/results/aggregate/figure_manifest.json", "code/results/aggregate/environment.json",
                "submission/main.pdf", "submission/main.bbl", "submission/LICENSE",
                "code/docs/METHOD.md", "code/docs/EVIDENCE.md", "code/docs/PROVENANCE.md",
                "code/configs/followup.json", "code/results/followup.json.gz", "code/results/transport.json.gz",
                "code/results/refinement.json.gz", "code/results/mechanism.json.gz",
                "code/src/gcqaoa/mechanism.py", "code/tests/test_mechanism.py",
                "code/src/gcqaoa/decision.py", "code/tests/test_decision.py",
                "code/results/decision.json.gz",
                "submission/tables/refinement.tex"]
    for name in required:
        require((ROOT / name).is_file() and (ROOT / name).stat().st_size > 0, f"Missing or empty: {name}")
    snapshots = verify_source_snapshots()
    print(f"Source snapshots: {len(snapshots)} complete source sets verified without Git history.", flush=True)
    followup, transport = None, None
    if (ROOT / "code/results/followup.json.gz").is_file():
        from gcqaoa.followup import load_followup
        followup=load_followup()
    if (ROOT / "code/results/transport.json.gz").is_file():
        from gcqaoa.transport_audit import verify as verify_transport
        verify_transport()
        with gzip.open(ROOT / "code/results/transport.json.gz", "rt") as stream:
            transport=json.load(stream)
        check_transport_diagnostics(transport)
    if (ROOT / "code/results/refinement.json.gz").is_file():
        from gcqaoa.followup import load_refinement, verify_refinement
        revision = verify_refinement(load_refinement(verify=False))
        require(revision["runs"] == 256, "Incomplete revision factorial")
        print("Revision evidence: " + json.dumps(revision, sort_keys=True), flush=True)
    if (ROOT / "code/results/mechanism.json.gz").is_file():
        from gcqaoa.mechanism import load_mechanism, verify_mechanism
        mechanism = verify_mechanism(load_mechanism(verify=False))
        require(mechanism["attribution_runs"] == 256 and mechanism["predictions"] == 320
                and mechanism["diagnostics"] == 1920, "Incomplete mechanism experiment")
        print("Mechanism evidence: " + json.dumps(mechanism, sort_keys=True), flush=True)
    if (ROOT / "code/results/decision.json.gz").is_file():
        from gcqaoa.decision import load_decision, verify_decision
        decision = verify_decision(load_decision(verify=False))
        require(decision["conditions"] == 80 and decision["joint_records"] == 1280,
                "Incomplete joint proposal-and-endpoint diagnostic")
        print("Joint decision evidence: " + json.dumps(decision, sort_keys=True), flush=True)
    figure_manifest = ROOT / "code/results/aggregate/figure_manifest.json"
    if figure_manifest.is_file():
        figures = json.loads(figure_manifest.read_text())
        require(figures["provenance"]["study_sha256"] == digest(ROOT / "code/results/study.json.gz"),
                "Figures were generated from a different study bundle")
        for name, expected in figures["analysis_source_sha256"].items():
            require(digest(ROOT / "code" / name) == expected, f"Stale figure source: {name}")
        for record in figures["figures"] + figures.get("tables", []):
            require(digest(ROOT / record["path"]) == record["sha256"], f"Stale figure/table: {record['path']}")
        with gzip.open(ROOT / "code/results/study.json.gz", "rt") as stream:
            data = json.load(stream)
        verify_source_identity(data)
        html=(ROOT / "index.html").read_text()
        require(html.count(WEB_BEGIN)==1 and html.count(WEB_END)==1,
                "Website needs one generated canonical-results block")
        if html.count(WEB_BEGIN)==1 and html.count(WEB_END)==1:
            block=WEB_BEGIN+html.split(WEB_BEGIN,1)[1].split(WEB_END,1)[0]+WEB_END
            require(block==website_results(data), "Website figure or table differs from the preserved study")
            require(figures.get("website",{}).get("block_sha256")==hashlib.sha256(block.encode()).hexdigest(),
                    "Website provenance does not match the generated results")
        if followup is not None and transport is not None:
            require(html.count(EXPERIMENTS_BEGIN)==1 and html.count(EXPERIMENTS_END)==1,
                    "Website needs one generated additional-results block")
            if html.count(EXPERIMENTS_BEGIN)==1 and html.count(EXPERIMENTS_END)==1:
                block=EXPERIMENTS_BEGIN+html.split(EXPERIMENTS_BEGIN,1)[1].split(EXPERIMENTS_END,1)[0]+EXPERIMENTS_END
                require(block==website_additional(followup,transport,data), "Website follow-up or transport results are stale")
            require(figures.get("additional_studies")=={name:digest(ROOT / "code/results" / name)
                    for name in ("followup.json.gz", "transport.json.gz", "refinement.json.gz", "mechanism.json.gz", "decision.json.gz")},
                    "Manuscript result provenance omits or differs from additional studies")
        require(figures["numerical_source_sha256"] == data["source_sha256"],
                "Figures do not identify the recorded execution's numerical sources")
        graph_path = ROOT / "code/data/graphs.jsonl"
        if graph_path.is_file():
            graph_rows = [json.loads(line) for line in graph_path.read_text().splitlines()]
            expected_rows = [{key: row[key] for key in ("id", "split", "family", "n", "seed", "edges", "maxcut")}
                             for row in data["instances"]]
            require(graph_rows == expected_rows, "Exported graph inputs differ from the recorded study")
        for name in ("splits.json", "reference_bank.json"):
            path = ROOT / "code/data" / name
            if path.is_file():
                require(json.loads(path.read_text())["provenance"]["study_sha256"] == digest(ROOT / "code/results/study.json.gz"),
                        f"Stale data export: {name}")
    credential_patterns = [r"gh[pousr]_[A-Za-z0-9]{30,}", r"AKIA[0-9A-Z]{16}",
                           r"sk-[A-Za-z0-9_-]{40,}", r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"]
    for path in files:
        name = path.relative_to(ROOT).as_posix()
        require(path.name != ".DS_Store" and path.suffix not in {".pyc", ".aux", ".log", ".blg", ".bak"},
                f"Generated or private debris: {name}")
        require(path.name != ".env" and not path.name.startswith(".env."), f"Private environment file: {name}")
        if path.suffix.lower() in {".py", ".tex", ".rst", ".md", ".html", ".json", ".jsonl", ".csv", ".toml", ".txt"}:
            content = path.read_text()
            require(not re.search(r"/(?:Users|home)/[A-Za-z0-9._-]+/", content), f"Personal filesystem path: {name}")
            require(not any(re.search(pattern, content) for pattern in credential_patterns), f"Credential pattern: {name}")
    page = Links()
    page.feed((ROOT / "index.html").read_text())
    for target in page.targets:
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc:
            continue
        if parsed.path:
            path = (ROOT / unquote(parsed.path)).resolve()
            require(path.is_relative_to(ROOT) and
                    (path.is_file() or (allow_manifest_creation and path==MANIFEST)),
                    f"Broken website link: {target}")
        elif parsed.fragment:
            require(parsed.fragment in page.ids, f"Broken website anchor: {target}")
    manuscript = (SUBMISSION / "main.tex").read_text()
    require("\\today" not in manuscript, "Manuscript uses a moving date")
    inline = inline_bibliography(manuscript)
    require("\\input{main.bbl}" not in active_tex(manuscript) and
            (inline or "\\bibliography{refs}" in active_tex(manuscript)),
            "Use one inline bibliography or the editable refs.bib workflow")
    for name in ("README.md", "index.html", "submission/main.tex"):
        content = (ROOT / name).read_text()
        if name.endswith(".tex"):
            content = " ".join(content.replace("\\\\", " ").split())
        require(TITLE in content, f"Title inconsistent in {name}")
    tex_sources = "\n".join((SUBMISSION / name).read_text() for name in manuscript_files()
                            if name.suffix == ".tex")
    reference_counts = check_bibliography(tex_sources, (SUBMISSION / "refs.bib").read_text(),
                                         tex_sources if inline else (SUBMISSION / "main.bbl").read_text())
    print("Bibliography: " + json.dumps(reference_counts, sort_keys=True), flush=True)
    archive = SUBMISSION / "dist/arxiv-2604.24803-v2-source.zip"
    expected = {path.as_posix() for path in manuscript_files()}
    if archive.is_file():
        with zipfile.ZipFile(archive) as bundle:
            require(set(bundle.namelist()) == expected, "arXiv archive has missing or extraneous files")
            for name in bundle.namelist():
                require(name in expected and bundle.read(name) == (SUBMISSION / name).read_bytes(),
                        f"Stale arXiv archive dependency: {name}")
        checksum = archive.with_suffix(".zip.sha256")
        require(checksum.is_file() and checksum.read_text().split()[0] == digest(archive), "Archive checksum mismatch")
    elif archive.with_suffix(".zip.sha256").exists():
        require(False, "Source archive checksum exists without its archive")
    if errors:
        raise RuntimeError("\n".join(errors))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="also run tests, canonical record reconstruction, full follow-up replay, and source-package verification")
    parser.add_argument("--replay-solvers", action="store_true",
                        help="also regenerate all transport trajectories; requires the recorded numerical backend")
    parser.add_argument("--write-manifest", action="store_true", help="refresh the public-file hash inventory after checks")
    parser.add_argument("--rules-file", help="external JSON array of private literal phrases; matching text is never printed")
    parser.add_argument("--history", action="store_true", help="also audit commits and annotated tags reachable from local branches and tags")
    arguments = parser.parse_args()
    files = check(rules=load_rules(arguments.rules_file), history=arguments.history,
                  allow_manifest_creation=arguments.write_manifest)
    if arguments.full:
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                           PYTHONPATH=str(ROOT / "code/src")+os.pathsep+os.environ.get("PYTHONPATH", ""))
        for command in ([sys.executable, "-m", "pytest", "code/tests", "-p", "no:cacheprovider", "-q"],
                        [sys.executable, "-m", "gcqaoa.verify"],
                        [sys.executable, "code/scripts/run_followup.py", "--verify", "--full"],
                        [sys.executable, "code/scripts/run_followup.py", "--revision", "--verify", "--full"],
                        [sys.executable, "-m", "gcqaoa.mechanism", "--verify", "--full"],
                        [sys.executable, "-m", "gcqaoa.decision", "--verify", "--full"],
                        [sys.executable, "code/scripts/package_arxiv.py", "--check"]):
            if command[1] == "code/scripts/run_followup.py" and "--revision" not in command:
                completed = subprocess.run(command, cwd=ROOT, env=environment, check=True,
                                           text=True, stdout=subprocess.PIPE)
                with gzip.open(ROOT / "code/results/followup.json.gz", "rt") as stream:
                    expected_runs = len(json.load(stream)["runs"])
                result = check_followup_replay(completed.stdout, expected_runs)
                print("Follow-up replay: " + json.dumps(result, sort_keys=True), flush=True)
            else:
                subprocess.run(command, cwd=ROOT, env=environment, check=True)
    if arguments.replay_solvers:
        with gzip.open(ROOT / "code/results/transport.json.gz", "rt") as stream:
            check_transport_diagnostics(json.load(stream),replay=True)
    hashes = {path.relative_to(ROOT).as_posix(): digest(path) for path in files if path != MANIFEST}
    if arguments.write_manifest:
        MANIFEST.write_text(json.dumps({"schema_version": 1, "algorithm": "sha256", "files": hashes}, indent=2) + "\n")
    elif not MANIFEST.is_file() or json.loads(MANIFEST.read_text()).get("files") != hashes:
        raise RuntimeError("Public files differ from the file inventory; regenerate affected outputs and run --write-manifest after reviewing the changes.")
    print(f"PASS: {len(hashes)} public files; four-item root, local page links, manuscript dependencies, and file hashes verified.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Folder check failed: {error}", file=sys.stderr)
        sys.exit(1)
