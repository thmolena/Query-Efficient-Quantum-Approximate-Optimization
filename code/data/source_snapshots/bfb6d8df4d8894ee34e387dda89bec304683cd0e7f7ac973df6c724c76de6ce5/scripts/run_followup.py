"""Run or verify the independent-bank finite-shot follow-up protocol."""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gcqaoa.followup import main

if __name__ == "__main__":
    main()
