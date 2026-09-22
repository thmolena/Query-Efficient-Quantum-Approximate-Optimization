#!/usr/bin/env python3
"""Export only used manuscript dependencies and compile the extracted archive."""
import hashlib
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

from build_paper import SUBMISSION, compile_manuscript, main as build_paper, manuscript_files


def pdf_text(path):
    return subprocess.check_output(["pdftotext", "-layout", str(path), "-"], text=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',
                        help='verify a temporary source archive without retaining an upload ZIP')
    arguments=parser.parse_args(argv)
    if not shutil.which("pdftotext"):
        raise RuntimeError("Install Poppler (pdftotext) to verify the extracted source package.")
    build_paper()
    names = manuscript_files()
    target = SUBMISSION / "dist" / "arxiv-2604.24803-v2-source.zip"
    if not arguments.check:
        target.parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".arxiv-", dir=SUBMISSION) as temporary:
        work = Path(temporary)
        archive = work / target.name
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as output:
            for name in names:
                info = zipfile.ZipInfo(name.as_posix(), date_time=(2026, 9, 18, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                output.writestr(info, (SUBMISSION / name).read_bytes())
        extracted = work / "extracted"
        extracted.mkdir()
        with zipfile.ZipFile(archive) as source:
            source.extractall(extracted)
        compile_manuscript(extracted)
        if pdf_text(extracted / "main.pdf") != pdf_text(SUBMISSION / "main.pdf"):
            raise RuntimeError("The archive build differs from the authoritative manuscript text.")
        if not arguments.check:
            archive.replace(target)
    if arguments.check:
        print(f"Verified temporary source archive: {len(names)} dependencies; extracted build matches manuscript text.")
    else:
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        target.with_suffix(".zip.sha256").write_text(f"{digest}  {target.name}\n")
        print(f"Verified source archive: {target.name}; {len(names)} files; extracted build matches manuscript text.")


if __name__ == "__main__":
    main()
