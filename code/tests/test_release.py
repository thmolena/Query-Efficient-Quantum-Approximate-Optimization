"""Exercise PDF inspection and redacted public-content checks with synthetic data."""
import gzip
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

from pypdf import PdfWriter
from pypdf.generic import ArrayObject, DictionaryObject, NameObject, NumberObject, TextStringObject


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code/scripts"))
SPEC = importlib.util.spec_from_file_location("release_checks", ROOT / "code/scripts/check_release.py")
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)
import build_paper as paper


def pdf_bytes(*, metadata=None, attachment=False, action=None, annotation=None, open_view=False):
    writer = PdfWriter()
    page = writer.add_blank_page(width=100, height=100)
    writer.add_metadata(metadata or {"/Title": release.TITLE, "/Author": release.AUTHOR,
                                    "/Subject": "Quantum approximate optimization", "/Keywords": "QAOA, MaxCut"})
    if attachment:
        writer.add_attachment("example.txt", b"Synthetic attached record")
    if action is not None:
        writer._root_object[NameObject("/OpenAction")] = action
    if open_view:
        destination = ArrayObject([page.indirect_reference, NameObject("/Fit")])
        writer._root_object[NameObject("/OpenAction")] = (DictionaryObject({NameObject("/S"): NameObject("/GoTo"),
                                                                         NameObject("/D"): destination})
                                                         if open_view == "action" else destination)
    if annotation is not None:
        page[NameObject("/Annots")] = writer._add_object(ArrayObject([writer._add_object(annotation)]))
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def link(uri):
    return DictionaryObject({NameObject("/Type"): NameObject("/Annot"), NameObject("/Subtype"): NameObject("/Link"),
                             NameObject("/Rect"): ArrayObject([NumberObject(value) for value in (0, 0, 10, 10)]),
                             NameObject("/A"): DictionaryObject({NameObject("/S"): NameObject("/URI"),
                                                             NameObject("/URI"): TextStringObject(uri)})})


def inline_source(payload=None):
    payload = pdf_bytes() if payload is None else payload
    names = sorted(paper.EMBEDDED_PDF_NAMES)
    source = "\\documentclass{article}\n\\begin{document}\n\\cite{one}\n"
    source += "".join(f"\\includegraphics{{{name}}}\n" for name in names)
    source += "\\begin{thebibliography}{1}\n\\bibitem{one} A result.\n\\end{thebibliography}\n\\end{document}\n"
    return source + "".join(f"%BEGIN_EMBEDDED_PDF {name}\n%{payload.hex()}\n%END_EMBEDDED_PDF\n"
                            for name in names)


class ManuscriptBuildTests(unittest.TestCase):
    def test_embedded_archive_depends_only_on_main_tex(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = inline_source()
            (root / "main.tex").write_text(source)
            (root / "main.bbl").write_text("stale legacy bibliography")
            self.assertEqual(paper.manuscript_files(root), [Path("main.tex")])
            self.assertEqual(set(paper.embedded_pdfs(source)), paper.EMBEDDED_PDF_NAMES)

    def test_embedded_assets_reject_unsafe_duplicate_incomplete_and_invalid_payloads(self):
        source = inline_source()
        first = sorted(paper.EMBEDDED_PDF_NAMES)[0]
        second = sorted(paper.EMBEDDED_PDF_NAMES)[1]
        cases = [source.replace(f"%BEGIN_EMBEDDED_PDF {first}", "%BEGIN_EMBEDDED_PDF ../escape.pdf"),
                 source.replace(f"%BEGIN_EMBEDDED_PDF {first}", "%BEGIN_EMBEDDED_PDF /escape.pdf"),
                 source.replace(f"%BEGIN_EMBEDDED_PDF {first}", f"%BEGIN_EMBEDDED_PDF {second}"),
                 source[:source.rfind("%BEGIN_EMBEDDED_PDF")],
                 source.replace("%END_EMBEDDED_PDF", "", 1),
                 source.replace("%25504446", "%z5504446", 1),
                 source.replace("%25504446", "%5504446", 1),
                 inline_source(b"not a PDF"), inline_source(b"%PDF-1.7\ninvalid\n%%EOF\n")]
        for index, content in enumerate(cases):
            with self.subTest(case=index), self.assertRaises(ValueError):
                paper.embedded_pdfs(content)

    def test_embedded_assets_must_be_used_and_cannot_hide_external_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "extra.tex").write_text("extra content")
            for source in (inline_source().replace("\\includegraphics{qaoa-decision.pdf}", ""),
                           inline_source().replace("\\begin{document}", "\\input{extra}\n\\begin{document}")):
                (root / "main.tex").write_text(source)
                with self.assertRaisesRegex(ValueError, "no external dependencies"):
                    paper.manuscript_files(root)

    def test_legacy_dependencies_and_bbl_remain_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.tex").write_text("\\input{part}\n\\bibliography{refs}")
            (root / "part.tex").write_text("\\includegraphics{plot}")
            (root / "plot.pdf").write_bytes(pdf_bytes())
            (root / "refs.bib").write_text("reference")
            self.assertEqual(paper.manuscript_files(root, include_bbl=False),
                             list(map(Path, ["main.tex", "part.tex", "plot.pdf", "refs.bib"])))
            with self.assertRaisesRegex(ValueError, "main.bbl"):
                paper.manuscript_files(root)
            (root / "main.bbl").write_text("compiled reference")
            self.assertIn(Path("main.bbl"), paper.manuscript_files(root))
            (root / "part.tex").write_text("\\input{../escape}")
            with self.assertRaisesRegex(ValueError, "unsafe manuscript dependency"):
                paper.manuscript_files(root)

    def test_lualatex_runs_twice_without_bibtex_or_shell_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.tex").write_text(inline_source())
            calls = []
            def compile_one(command, **options):
                calls.append(command)
                self.assertEqual(options["cwd"], root)
                self.assertEqual(options["env"]["FORCE_SOURCE_DATE"], "1")
                (root / "main.pdf").write_bytes(b"compiled PDF")
                (root / "main.log").write_text("successful build\n")
                return SimpleNamespace(returncode=0, stdout="")
            with patch.dict(os.environ, {}, clear=True), \
                    patch.object(paper.shutil, "which", return_value="/tools/lualatex"), \
                    patch.object(paper.subprocess, "run", side_effect=compile_one):
                self.assertEqual(paper.compile_manuscript(root), 0)
            self.assertEqual(calls, [["/tools/lualatex", "-no-shell-escape", "-interaction=nonstopmode",
                                      "-halt-on-error", "main.tex"]] * 2)
            self.assertFalse((root / "main.bbl").exists())

    def test_legacy_compiler_requires_a_fresh_bbl(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.tex").write_text("\\bibliography{refs}")
            (root / "refs.bib").write_text("reference")
            calls = []
            def compile_one(command, **_):
                calls.append(command[0])
                (root / "main.pdf").write_bytes(b"compiled PDF")
                (root / "main.log").write_text("successful build\n")
                return SimpleNamespace(returncode=0, stdout="")
            with patch.dict(os.environ, {}, clear=True), \
                    patch.object(paper.shutil, "which", side_effect=lambda name: name if name in ("pdflatex", "bibtex") else None), \
                    patch.object(paper.subprocess, "run", side_effect=compile_one):
                with self.assertRaisesRegex(RuntimeError, "main.bbl"):
                    paper.compile_manuscript(root)
            self.assertEqual(calls, ["pdflatex", "bibtex", "pdflatex", "pdflatex"])

    def test_build_copies_only_source_and_keeps_generated_assets_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.tex").write_text(inline_source())
            def compile_one(work):
                self.assertNotEqual(work, root)
                self.assertEqual([path.name for path in work.iterdir()], ["main.tex"])
                (work / "main.pdf").write_bytes(b"compiled PDF")
                (work / "qaoa-decision.pdf").write_bytes(b"temporary figure")
                return 0
            with patch.object(paper, "SUBMISSION", root), patch.object(paper, "compile_manuscript", side_effect=compile_one):
                paper.main()
            self.assertEqual({path.name for path in root.iterdir()}, {"main.tex", "main.pdf"})


class FolderInventoryTests(unittest.TestCase):
    def test_local_executions_do_not_change_supplied_file_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("README.md", "code/docs/METHOD.md", "code/results/study.json.gz",
                         "code/results/executions/local/study.json.gz", ".git/config"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("synthetic record")
            with patch.object(release, "ROOT", root):
                names = {path.relative_to(root).as_posix() for path in release.public_files()}
            self.assertEqual(names, {"README.md", "code/docs/METHOD.md", "code/results/study.json.gz"})


class FollowupReplayTests(unittest.TestCase):
    def test_silent_or_incomplete_replays_cannot_pass(self):
        invalid = ["", "{}", json.dumps({"runs": 576, "replayed_runs": 0, "checks": 200}),
                   json.dumps({"runs": 575, "replayed_runs": 575, "checks": 200}),
                   json.dumps({"runs": 576, "replayed_runs": 576, "checks": 0})]
        for output in invalid:
            with self.subTest(output=output):
                with self.assertRaises(ValueError):
                    release.check_followup_replay(output, 576)
        report = {"runs": 576, "replayed_runs": 576, "checks": 479011}
        self.assertEqual(release.check_followup_replay(json.dumps(report), 576), report)


class BibliographyTests(unittest.TestCase):
    @staticmethod
    def entry(key, doi="10.1234/example", eprint=""):
        return (f"@article{{{key},\n  author = {{Researcher, A.}},\n  title = {{A result}},\n"
                f"  year = {{2020}},\n  doi = {{{doi}}},\n  annote = {{Verified claim and source}},\n"
                + (f"  eprint = {{{eprint}}},\n" if eprint else "") + "}\n")

    def test_counts_use_cited_entries_and_compiled_bibliography(self):
        result = release.check_bibliography(r"A\cite{one,two}. B\citep[Sec.~2]{one}.",
                    self.entry("one") + self.entry("two", "10.1234/another"),
                    r"\bibitem[A(2020)]{one} A\bibitem[B(2021)]{two} B")
        self.assertEqual(result, {"distinct_references": 2, "citation_commands": 2,
                                  "reference_mentions": 3})
        with self.assertRaisesRegex(ValueError, "Compiled bibliography"):
            release.check_bibliography(r"\cite{one}", self.entry("one"), r"\bibitem{two} X")

    def test_missing_unused_duplicate_and_wildcard_entries_are_rejected(self):
        for tex, bibliography in [(r"\cite{missing}", self.entry("one")),
                                  (r"\cite{one}", self.entry("one") + self.entry("two", "10.1234/two")),
                                  (r"\cite{one}", self.entry("one") * 2),
                                  (r"\cite{one,two}", self.entry("one") + self.entry("two", "https://doi.org/10.1234/EXAMPLE")),
                                  (r"\cite{one,two}", self.entry("one", "a", "1234.56789v1") + self.entry("two", "b", "1234.56789v2")),
                                  (r"\nocite{*}", self.entry("one")),
                                  (r"\cite{one}", self.entry("one").replace("  annote = {Verified claim and source},\n", ""))]:
            with self.subTest(tex=tex, bibliography=bibliography), self.assertRaises(ValueError):
                release.check_bibliography(tex, bibliography)

    def test_inline_bibliography_coverage_is_strict_and_ignores_comments(self):
        source = "\\cite{one}\n\\begin{thebibliography}{1}\n\\bibitem{one} A\n\\end{thebibliography}\n"
        self.assertTrue(paper.inline_bibliography(source))
        self.assertFalse(paper.inline_bibliography("% " + source.replace("\n", "\n% ")))
        result = release.check_bibliography(source, self.entry("one"), source + "% \\bibitem{unused} X\n")
        self.assertEqual(result["distinct_references"], 1)
        for altered in (source.replace("\\bibitem{one}", "\\bibitem{two}"),
                        source.replace("\\bibitem{one}", "\\bibitem{one} X \\bibitem{one}")):
            with self.assertRaisesRegex(ValueError, "Compiled bibliography"):
                release.check_bibliography(altered, self.entry("one"), altered)
        for altered in (source + "\\bibliography{refs}", source + "\\input{main.bbl}",
                        source + "\\bibitem{two} Outside", source.replace("\\end{thebibliography}", "")):
            with self.assertRaises(ValueError):
                paper.inline_bibliography(altered)


class PublicContentTests(unittest.TestCase):
    def test_factual_pdf_and_indirect_web_link_are_accepted(self):
        audit = release.PublicAudit()
        audit.blob("main.pdf", pdf_bytes(annotation=link("https://example.org/paper")))
        audit.finish()

    def test_initial_internal_page_view_is_harmless(self):
        for mode in (True, "action"):
            with self.subTest(mode=mode):
                audit = release.PublicAudit()
                audit.blob("main.pdf", pdf_bytes(open_view=mode))
                audit.finish()

    def test_manuscript_metadata_must_be_factual_and_complete(self):
        good = {"/Title": release.TITLE, "/Author": release.AUTHOR,
                "/Subject": "Quantum approximate optimization", "/Keywords": "QAOA"}
        cases = [{**good, "/Author": "Different author"}, {**good, "/Title": "Different title"},
                 {**good, "/Keywords": ""}, {**good, "/Subject": "Under review"},
                 {**good, "/AdministrativeField": "Synthetic value"}]
        for metadata in cases:
            with self.subTest(fields=tuple(metadata)):
                audit = release.PublicAudit()
                audit.blob("submission/main.pdf", pdf_bytes(metadata=metadata))
                self.assertTrue(audit.errors)

    def test_embedded_attachments_are_rejected(self):
        audit = release.PublicAudit()
        audit.blob("figure.pdf", pdf_bytes(attachment=True))
        self.assertTrue(any("embedded attachments" in error for error in audit.errors))

    def test_automatic_actions_and_non_link_annotations_are_rejected(self):
        action = DictionaryObject({NameObject("/S"): NameObject("/JavaScript"),
                                   NameObject("/JS"): TextStringObject("void(0)")})
        note = DictionaryObject({NameObject("/Type"): NameObject("/Annot"), NameObject("/Subtype"): NameObject("/Text"),
                                 NameObject("/Contents"): TextStringObject("Synthetic annotation")})
        for payload in (pdf_bytes(action=action), pdf_bytes(annotation=note)):
            audit = release.PublicAudit()
            audit.blob("figure.pdf", payload)
            self.assertTrue(audit.errors)

    def test_unsafe_pdf_link_targets_are_rejected(self):
        for uri in ("file:///private/example", "javascript:void(0)", "https://user:pass@example.org", "https://"):
            with self.subTest(scheme=uri.split(":", 1)[0]):
                audit = release.PublicAudit()
                audit.blob("figure.pdf", pdf_bytes(annotation=link(uri)))
                self.assertTrue(any("unsafe URI" in error for error in audit.errors))

    def test_pdf_metadata_is_checked_by_private_literals(self):
        audit = release.PublicAudit(["synthetic marker"])
        audit.blob("example.pdf", pdf_bytes(metadata={"/Subject": "SYNTHETIC marker"}))
        self.assertTrue(any("Private rule 1" in error for error in audit.errors))

    def test_malformed_pdf_diagnostics_do_not_echo_payload(self):
        audit = release.PublicAudit()
        audit.blob("confidential-filename.pdf", b"Synthetic malformed PDF contents")
        with self.assertRaises(RuntimeError) as caught:
            audit.finish()
        self.assertNotIn("Synthetic", str(caught.exception))
        self.assertNotIn("confidential-filename", str(caught.exception))

    def test_generic_class_and_bibliography_must_be_active(self):
        good = b"\\documentclass[11pt]{article}\n\\bibliographystyle{plainnat}\n"
        audit = release.PublicAudit()
        audit.blob("main.tex", good)
        audit.finish()
        for content in (good.replace(b"{article}", b"{report}"), b"% " + good,
                        good.replace(b"plainnat", b"plain")):
            audit = release.PublicAudit()
            audit.blob("main.tex", content)
            self.assertTrue(audit.errors)

    def test_inline_manuscript_and_embedded_pdf_contents_are_audited(self):
        audit = release.PublicAudit()
        audit.blob("main.tex", inline_source().encode())
        audit.finish()
        for payload in (pdf_bytes(attachment=True), pdf_bytes(action=DictionaryObject({
                NameObject("/S"): NameObject("/JavaScript"), NameObject("/JS"): TextStringObject("void(0)")}))):
            audit = release.PublicAudit()
            audit.blob("main.tex", inline_source(payload).encode())
            self.assertTrue(audit.errors)

    def test_private_literals_are_not_regex_and_errors_are_redacted(self):
        phrase = "private [synthetic] marker"
        audit = release.PublicAudit([phrase])
        audit.blob("hidden-name.txt", b"Private synthetic marker")
        audit.finish()
        audit.blob("hidden-name.txt", phrase.upper().encode())
        with self.assertRaises(RuntimeError) as caught:
            audit.finish()
        message = str(caught.exception)
        self.assertIn("Private rule 1 matched; object ", message)
        self.assertNotIn(phrase, message.lower())
        self.assertNotIn("hidden-name", message)

    def test_html_comments_entities_and_whitespace_are_scanned(self):
        audit = release.PublicAudit(["synthetic marker"])
        audit.blob("index.html", b"<!-- synthetic&#32;\nmarker -->")
        self.assertTrue(audit.errors)

    def test_compressed_json_unicode_and_zip_metadata_are_scanned(self):
        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, "w") as archive:
            archive.comment = b"Synthetic archive marker"
            item = zipfile.ZipInfo("neutral.json.gz")
            item.comment = b"Synthetic member marker"
            archive.writestr(item, gzip.compress(b'{"value": "synthetic \\u0070ayload marker"}'))
        audit = release.PublicAudit(["synthetic archive marker", "synthetic member marker", "synthetic payload marker"])
        audit.blob("neutral.zip", archive_bytes.getvalue())
        for index in range(1, 4):
            self.assertTrue(any(f"Private rule {index}" in error for error in audit.errors))

    def test_rules_must_be_external_and_invalid_file_errors_are_redacted(self):
        base = ROOT / "code/.validation"
        base.mkdir(exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(dir=base) as directory:
                parent = Path(directory)
                repository = parent / "repository"
                repository.mkdir()
                external = parent / "external.json"
                external.write_text(json.dumps(["synthetic marker"]))
                self.assertEqual(release.load_rules(external, root=repository), ("synthetic marker",))
                internal = repository / "internal.json"
                internal.write_text("[]")
                with self.assertRaisesRegex(ValueError, "outside the repository"):
                    release.load_rules(internal, root=repository)
                alias = parent / "alias.json"
                alias.symlink_to(internal)
                with self.assertRaisesRegex(ValueError, "outside the repository"):
                    release.load_rules(alias, root=repository)
                external.write_text("invalid-json synthetic marker")
                with self.assertRaises(ValueError) as caught:
                    release.load_rules(external, root=repository)
                self.assertNotIn("synthetic marker", str(caught.exception))
                self.assertNotIn(str(external), str(caught.exception))
        finally:
            if not any(base.iterdir()):
                base.rmdir()

    def test_history_checks_published_refs_and_redacts_messages(self):
        commit, tag, blob = "1" * 40, "2" * 40, "3" * 40
        calls = []

        def git(command, **_):
            arguments = command[2:]
            calls.append(arguments)
            if arguments[0] == "for-each-ref":
                self.assertEqual(arguments[-2:], ["refs/heads", "refs/tags"])
                return f"refs/heads/main {commit} commit\nrefs/tags/paper-v2 {tag} tag\n".encode()
            if arguments[:2] == ["cat-file", "tag"]:
                return f"object {commit}\ntype commit\ntag paper-v2\n\nSynthetic tag marker\n".encode()
            if arguments[0] == "rev-list":
                return (commit + "\n").encode()
            if arguments[:2] == ["cat-file", "commit"]:
                return b"Synthetic commit marker\n"
            if arguments[0] == "ls-tree":
                return f"100644 blob {blob}\tsubmission/main.tex\0".encode()
            if arguments[:2] == ["cat-file", "blob"]:
                return b"\\documentclass{report}\n\\bibliographystyle{plainnat}\n"
            self.fail("Unexpected Git command")

        audit = release.PublicAudit(["synthetic tag marker", "synthetic commit marker"])
        with patch.object(release.subprocess, "check_output", side_effect=git):
            audit.history()
        self.assertTrue(any("Private rule 1" in error for error in audit.errors))
        self.assertTrue(any("Private rule 2" in error for error in audit.errors))
        self.assertTrue(any("generic article" in error for error in audit.errors))
        self.assertNotIn("refs/local-archive", str(calls))


class TransportReplayTests(unittest.TestCase):
    def test_followup_uncertainty_preserves_fixed_bank_weights(self):
        from gcqaoa.report import followup_ci
        rows=[{'graph':f'{panel}:{graph}','panel':panel,'signed_reference_gap':float(value)}
              for panel,value in [(0,1),(1,9)] for graph in range(3) for _ in range(2)]
        # Every graph within a bank has the same outcome. Resampling shots or
        # banks would manufacture uncertainty that the conditional estimand excludes.
        self.assertEqual(followup_ci(rows),(500.0,500.0,500.0))

    def test_saved_convergence_flags_must_follow_the_stopping_diagnostic(self):
        protocol={'outer_limit':1000,'outer_tolerance':1e-9,'fw_limit':200,'fw_tolerance':1e-10}
        records=[{'iterations':1000,'residual':1e-6,'feasible':True,'converged':False},
                 {'iterations':0,'gap':0.0,'trace':[],'feasible':True,'converged':True},
                 {'iterations':5,'residual':1e-12,'feasible':False,'converged':False}]
        for record in records:
            with self.subTest(record=record):
                release.check_transport_status(record,protocol)
                altered={**record,'converged':not record['converged']}
                with self.assertRaisesRegex(ValueError,'convergence flag'):
                    release.check_transport_status(altered,protocol)
        with self.assertRaisesRegex(ValueError,'valid range'):
            release.check_transport_status({**records[0],'iterations':1001},protocol)

    def test_full_replay_checks_unselected_starts_and_discrete_fields(self):
        fresh={'starts':[{'coupling':[[0.5,0.0]],'iterations':2,'residual':1e-10,'converged':True},
                         {'coupling':[[0.0,0.5]],'iterations':7,'residual':1e-7,'converged':False}],
               'selected':0}
        release.compare_transport_replay(fresh,fresh)
        for field,value in [('coupling',[[0.25,0.25]]),('iterations',8),('residual',0.1),('converged',True)]:
            altered=copy.deepcopy(fresh)
            altered['starts'][1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                release.compare_transport_replay(altered,fresh)
        with self.assertRaisesRegex(ValueError,'discrete diagnostics'):
            release.compare_transport_replay({**fresh,'selected':1},fresh)


if __name__ == "__main__":
    unittest.main()
