#!/usr/bin/env python3
"""Prepare a self-contained Sketch2 tutorial environment.

Creates the directory layout described in Tutorial.md, copies the shared
library plus Python helpers, writes a minimal config.ini, and prints the
environment variable exports that tutorial scripts expect.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def find_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def find_lib_source(repo_root: Path) -> Path:
    """Return the best available libsketch2.so built artifact."""
    candidates = [
        repo_root / "build" / "lib" / "libsketch2.so",
        repo_root / "bin" / "libsketch2.so",
        repo_root / "build-dbg" / "lib" / "libsketch2.so",
        repo_root / "bin-dbg" / "libsketch2.so",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit("libsketch2.so not found. Build the project first (e.g., `cmake --build build`).")


def write_default_config(config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "[log]\n"
        "level=INFO\n"
        "path=\n"
        "\n"
        "[thread_pool]\n"
        "size=4\n"
    )
    config_path.write_text(content, encoding="ascii")


def copy_python_helpers(repo_root: Path, lib_dir: Path) -> None:
    helpers = ["sketch2_wrapper.py", "sketch2_utils.py"]
    src_dir = repo_root / "src" / "pytest"
    for helper in helpers:
        src = src_dir / helper
        if not src.exists():
            raise SystemExit(f"Missing helper script: {src}")
        shutil.copy2(src, lib_dir / helper)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize Sketch2 tutorial environment.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd() / "sketch2",
        help="Root directory for the tutorial environment (default: ./sketch2)",
    )
    parser.add_argument(
        "--config-name",
        default="config.ini",
        help="Config filename to create under the db directory (default: config.ini)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = find_repo_root()

    env_root = args.root.resolve()
    db_dir = env_root / "db"
    lib_dir = env_root / "lib"
    config_path = db_dir / args.config_name

    lib_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)

    # Copy shared library and Python helpers.
    lib_source = find_lib_source(repo_root)
    shutil.copy2(lib_source, lib_dir / "libsketch2.so")
    copy_python_helpers(repo_root, lib_dir)

    # Write a minimal config.ini if it does not exist yet.
    if not config_path.exists():
        write_default_config(config_path)

    print("Tutorial environment prepared.")
    print(f"  root:      {env_root}")
    print(f"  db dir:    {db_dir}")
    print(f"  config:    {config_path}")
    print(f"  lib dir:   {lib_dir}")
    print()
    print("Set these in your shell before running tutorial_01.py:")
    print(f"  export SKETCH2_LIB={lib_dir}")
    print(f"  export SKETCH2_CONFIG={config_path}")


if __name__ == "__main__":
    main()
