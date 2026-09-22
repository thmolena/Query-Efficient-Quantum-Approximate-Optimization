"""Run the small real-circuit smoke protocol without replacing paper results."""
import argparse
from pathlib import Path
import sys
sys.dont_write_bytecode = True
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))
from gcqaoa.experiment import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CODE_ROOT / "configs" / "smoke.json",
                        help="path to the supplied smoke configuration")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.config.resolve() != CODE_ROOT / "configs" / "smoke.json":
        parser.error("run_smoke uses code/configs/smoke.json; use run_study.py for other configurations")
    command = ["--config", str(args.config)]
    if args.output is not None:
        command += ["--output", str(args.output)]
    main(command)
