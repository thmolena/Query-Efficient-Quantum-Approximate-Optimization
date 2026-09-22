"""Export one selected canonical run or list available run identifiers."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gcqaoa.artifacts import main

if __name__ == "__main__":
    main(["export", *sys.argv[1:]])
