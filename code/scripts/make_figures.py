"""Render vector manuscript figures and record exact input/output provenance."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gcqaoa.report import main

if __name__ == "__main__":
    main(["--figures-only", *sys.argv[1:]])
