#!/usr/bin/env python3
"""Compile the authoritative manuscript; retain only its PDF and fresh BibTeX output."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
SUBMISSION = ROOT / "submission"


def manuscript_files(directory=SUBMISSION, include_bbl=True):
    """Resolve the actual TeX dependencies, rejecting paths outside the manuscript."""
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
        source = re.sub(r"(?<!\\)%[^\n]*", "", path.read_text())
        for command, argument in re.findall(
                r"\\(input|include|includegraphics|bibliography)(?:\[[^\]]*\])?\{([^}]+)\}", source):
            for item in argument.split(","):
                dependency = Path(item.strip())
                if not dependency.suffix:
                    dependency = dependency.with_suffix(".bib" if command == "bibliography" else
                                                        ".pdf" if command == "includegraphics" else ".tex")
                pending.append(dependency)
    if include_bbl:
        if not (directory / "main.bbl").is_file():
            raise ValueError("Build the manuscript to generate main.bbl before packaging.")
        found.add(Path("main.bbl"))
    return sorted(found)


def compile_manuscript(directory):
    """Use Tectonic or the standard pdfLaTeX/BibTeX sequence in an isolated directory."""
    environment = dict(os.environ, SOURCE_DATE_EPOCH="1789689600", FORCE_SOURCE_DATE="1")
    tectonic = os.environ.get("TECTONIC", "tectonic")
    if shutil.which(tectonic):
        command = [tectonic, "--reruns", "2", "--keep-logs", "--keep-intermediates"]
        if os.environ.get("TECTONIC_ONLY_CACHED") == "1":
            command.append("--only-cached")
        commands = [command + ["main.tex"]]
    elif shutil.which("pdflatex") and shutil.which("bibtex"):
        tex = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
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
    for name in ("main.pdf", "main.bbl"):
        if not (directory / name).is_file() or not (directory / name).stat().st_size:
            raise RuntimeError(f"The compiler did not produce {name}.")
    return sum("Underfull" in line for line in log.splitlines())


def main():
    with tempfile.TemporaryDirectory(prefix=".paper-", dir=SUBMISSION) as temporary:
        work = Path(temporary)
        for name in manuscript_files(include_bbl=False):
            (work / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SUBMISSION / name, work / name)
        underfull = compile_manuscript(work)
        for name in ("main.pdf", "main.bbl"):
            (work / name).replace(SUBMISSION / name)
    print(f"Built submission/main.pdf and fresh main.bbl; no undefined references or overfull boxes. "
          f"Underfull-box warnings: {underfull} (inspect layout visually).")


if __name__ == "__main__":
    main()
