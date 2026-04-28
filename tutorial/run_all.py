#!/usr/bin/env python3
"""Run all Sketch2 tutorials in sequence.

This script prepares a tutorial environment under /tmp via tutorial_00.py,
sets SKETCH2_LIB and SKETCH2_CONFIG for child processes, then runs all
remaining tutorial scripts one by one.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all tutorial scripts end-to-end.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/tmp") / "sketch2_tutorial",
        help="Tutorial environment root directory (default: /tmp/sketch2_tutorial)",
    )
    parser.add_argument(
        "--config-name",
        default="config.ini",
        help="Config filename under <root>/db (default: config.ini)",
    )
    return parser.parse_args()


def run_cmd(argv: list[str], env: dict[str, str] | None = None) -> None:
    print(f"$ {' '.join(argv)}")
    subprocess.run(argv, check=True, env=env)


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    py = sys.executable

    root = args.root.resolve()
    config_path = root / "db" / args.config_name
    lib_dir = root / "lib"

    # Step 1: Prepare a self-contained tutorial environment.
    run_cmd(
        [
            py,
            str(script_dir / "tutorial_00.py"),
            "--root",
            str(root),
            "--config-name",
            args.config_name,
        ]
    )

    # Step 2: Set local environment variables for all subsequent tutorial runs.
    env = os.environ.copy()
    env["SKETCH2_LIB"] = str(lib_dir)
    env["SKETCH2_CONFIG"] = str(config_path)

    print("\nUsing environment variables:")
    print(f"  SKETCH2_LIB={env['SKETCH2_LIB']}")
    print(f"  SKETCH2_CONFIG={env['SKETCH2_CONFIG']}")
    print("")

    # Step 3: Run tutorial_01.py ... tutorial_08.py sequentially.
    for index in range(1, 9):
        script_name = f"tutorial_{index:02d}.py"
        dataset_name = f"tutorial_{index:02d}_dataset"
        run_cmd([py, str(script_dir / script_name), dataset_name], env=env)

    print("\nAll tutorial scripts finished successfully.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
