#!/usr/bin/env python3
"""Compile the manuscript in isolation, including its self-contained LuaLaTeX form."""
import io
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SUBMISSION = ROOT / "submission"
EMBEDDED_PDF_NAMES = frozenset({
    "qaoa-efficiency.pdf", "qaoa-allocation.pdf", "qaoa-components.pdf",
    "qaoa-acceptance-power.pdf", "qaoa-decision.pdf", "qaoa-calibration.pdf",
})


def active_tex(source):
    """Discard TeX comments, including the embedded hexadecimal payloads."""
    return re.sub(r"(?<!\\)%[^\n]*", "", source)


def inline_bibliography(source):
    """Identify one complete inline bibliography, rejecting mixed workflows."""
    active = active_tex(source)
    begins = re.findall(r"\\begin\s*\{thebibliography\}", active)
    ends = re.findall(r"\\end\s*\{thebibliography\}", active)
    if not begins and not ends:
        return False
    blocks = re.findall(r"\\begin\s*\{thebibliography\}\s*\{[^}]+\}(.*?)"
                        r"\\end\s*\{thebibliography\}", active, re.S)
    if len(begins) != 1 or len(ends) != 1 or len(blocks) != 1 or not re.search(
            r"\\bibitem(?:\[[^]]*\])?\s*\{[^}]+\}", blocks[0]):
        raise ValueError("Require one complete, nonempty inline bibliography")
    if re.search(r"\\bibliography\s*\{|\\input\s*\{main\.bbl\}", active):
        raise ValueError("Do not mix inline bibliography and external BibTeX inputs")
    outside = active.replace(blocks[0], "", 1)
    if re.search(r"\\bibitem(?:\[[^]]*\])?\s*\{", outside):
        raise ValueError("Inline bibliography entries must be inside its environment")
    return True


def embedded_pdfs(source):
    """Validate the six named hex-encoded figures without writing any files."""
    from pypdf import PdfReader

    payloads, name, chunks = {}, None, []
    for line in source.splitlines():
        if line.lstrip().startswith("%BEGIN_EMBEDDED_PDF"):
            match = re.fullmatch(r"%BEGIN_EMBEDDED_PDF ([a-z0-9-]+\.pdf)", line)
            if name is not None or match is None or match[1] not in EMBEDDED_PDF_NAMES:
                raise ValueError("Missing or unsafe embedded PDF name")
            name = match[1]
            if name in payloads:
                raise ValueError("Duplicate embedded PDF name")
            chunks = []
        elif line.lstrip().startswith("%END_EMBEDDED_PDF"):
            if line != "%END_EMBEDDED_PDF" or name is None:
                raise ValueError("Unmatched embedded PDF end marker")
            payload = bytes.fromhex("".join(chunks))
            if not payload.startswith(b"%PDF-") or not payload.rstrip().endswith(b"%%EOF"):
                raise ValueError("Invalid embedded PDF payload")
            try:
                reader = PdfReader(io.BytesIO(payload), strict=True)
                if reader.is_encrypted or len(reader.pages) != 1:
                    raise ValueError("Embedded figure must be one unencrypted PDF page")
            except Exception as error:
                raise ValueError("Unreadable embedded PDF payload") from error
            payloads[name] = payload
            name, chunks = None, []
        elif name is not None:
            if not re.fullmatch(r"%(?:[0-9a-fA-F]{2})*", line):
                raise ValueError("Invalid embedded PDF hexadecimal line")
            chunks.append(line[1:])
    if name is not None:
        raise ValueError("Unterminated embedded PDF payload")
    if payloads and set(payloads) != EMBEDDED_PDF_NAMES:
        raise ValueError("Require all six embedded PDF figures")
    return payloads


def manuscript_files(directory=SUBMISSION, include_bbl=True):
    """Resolve the actual TeX dependencies, rejecting paths outside the manuscript."""
    directory = Path(directory)
    main_source = (directory / "main.tex").read_text()
    embedded = embedded_pdfs(main_source)
    inline = inline_bibliography(main_source)
    if embedded and not inline:
        raise ValueError("Embedded manuscript requires an inline bibliography")
    used_embedded = set()
    found, pending = set(), [Path("main.tex")]
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError(f"Missing or unsafe manuscript dependency: {name}")
        found.add(name)
        if name.suffix != ".tex":
            continue
        source = active_tex(path.read_text())
        for command, argument in re.findall(
                r"\\(input|include|includegraphics|bibliography)(?:\[[^\]]*\])?\{([^}]+)\}", source):
            for item in argument.split(","):
                dependency = Path(item.strip())
                if not dependency.suffix:
                    dependency = dependency.with_suffix(".bib" if command == "bibliography" else
                                                        ".pdf" if command == "includegraphics" else ".tex")
                if command == "includegraphics" and dependency.as_posix() in embedded:
                    used_embedded.add(dependency.as_posix())
                    continue
                pending.append(dependency)
    if embedded and (used_embedded != set(embedded) or found != {Path("main.tex")}):
        raise ValueError("Embedded manuscript must use all six figures and have no external dependencies")
    if include_bbl and not inline:
        if not (directory / "main.bbl").is_file():
            raise ValueError("Build the manuscript to generate main.bbl before packaging.")
        found.add(Path("main.bbl"))
    return sorted(found)


def compile_manuscript(directory):
    """Use two LuaLaTeX passes for inline sources, retaining legacy BibTeX builds."""
    directory = Path(directory)
    manuscript_files(directory, include_bbl=False)
    inline = inline_bibliography((directory / "main.tex").read_text())
    environment = dict(os.environ, SOURCE_DATE_EPOCH="1789689600", FORCE_SOURCE_DATE="1")
    tectonic = os.environ.get("TECTONIC", "tectonic")
    if inline:
        requested = os.environ.get("LUALATEX")
        lua = shutil.which(requested or "lualatex")
        tinytex = Path.home() / "Library/TinyTeX/bin/universal-darwin/lualatex"
        if lua is None and requested is None and tinytex.is_file():
            lua = str(tinytex)
        if lua is None:
            raise RuntimeError("Install LuaLaTeX or set LUALATEX to its executable path.")
        tex = [lua, "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
        commands = [tex, tex]
    elif shutil.which(tectonic):
        command = [tectonic, "--reruns", "2", "--keep-logs", "--keep-intermediates"]
        if os.environ.get("TECTONIC_ONLY_CACHED") == "1":
            command.append("--only-cached")
        commands = [command + ["main.tex"]]
    elif shutil.which("pdflatex") and shutil.which("bibtex"):
        tex = ["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
        commands = [tex, ["bibtex", "main"], tex, tex]
    else:
        raise RuntimeError("Install Tectonic or pdfLaTeX with BibTeX and the standard article/natbib packages.")
    for command in commands:
        completed = subprocess.run(command, cwd=directory, env=environment,
                                   text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(completed.stdout[-14000:])
    log = (directory / "main.log").read_text(errors="replace")
    failures = [line for line in log.splitlines() if re.search(
        r"Overfull \\[hv]box|There were undefined|(?:Citation|Reference).*undefined|multiply defined|multiply-defined|^!", line)]
    if failures:
        raise RuntimeError("Unresolved manuscript diagnostics:\n" + "\n".join(failures))
    for name in (("main.pdf",) if inline else ("main.pdf", "main.bbl")):
        if not (directory / name).is_file() or not (directory / name).stat().st_size:
            raise RuntimeError(f"The compiler did not produce {name}.")
    return sum("Underfull" in line for line in log.splitlines())


def main():
    inline = inline_bibliography((SUBMISSION / "main.tex").read_text())
    with tempfile.TemporaryDirectory(prefix=".paper-", dir=SUBMISSION) as temporary:
        work = Path(temporary)
        for name in manuscript_files(SUBMISSION, include_bbl=False):
            (work / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SUBMISSION / name, work / name)
        underfull = compile_manuscript(work)
        for name in (("main.pdf",) if inline else ("main.pdf", "main.bbl")):
            (work / name).replace(SUBMISSION / name)
    outputs = "submission/main.pdf" + ("" if inline else " and fresh main.bbl")
    print(f"Built {outputs}; no undefined references or overfull boxes. "
          f"Underfull-box warnings: {underfull} (inspect layout visually).")


if __name__ == "__main__":
    main()
