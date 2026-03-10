#!/usr/bin/env python3
"""Interactive shell for the Sketch2 Python wrapper."""

from __future__ import annotations

import argparse
import code
from pathlib import Path

from parasol_wrapper import Parasol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start an interactive Sketch2 Python shell")
    parser.add_argument("--db-root", required=True, help="Database root directory passed to sk_connect")
    parser.add_argument("--dataset", help="Dataset name to create/open before entering the shell")
    parser.add_argument("--create", action="store_true", help="Create the dataset before opening it")
    parser.add_argument("--type", default="f32", help="Dataset type for --create")
    parser.add_argument("--dim", type=int, default=4, help="Dataset dimension for --create")
    parser.add_argument("--range-size", type=int, default=1000, help="Dataset range size for --create")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.db_root)
    ps = Parasol(root)
    dataset_name = args.dataset

    try:
        if dataset_name and args.create:
            ps.create(dataset_name, type_name=args.type, dim=args.dim, range_size=args.range_size)
        if dataset_name:
            ps.open(dataset_name)

        banner_lines = [
            "Sketch2 interactive shell",
            f"db_root={root}",
            "Available objects:",
            "  ps           -> Parasol wrapper instance",
            "  root         -> database root Path",
            "  dataset_name -> selected dataset name or None",
        ]
        if dataset_name:
            banner_lines.append(f"Opened dataset: {dataset_name}")

        code.interact(
            banner="\n".join(banner_lines),
            local={
                "ps": ps,
                "root": root,
                "dataset_name": dataset_name,
            },
        )
    finally:
        if dataset_name:
            try:
                ps.close(dataset_name)
            except Exception:
                pass
        ps.close_handle()


if __name__ == "__main__":
    main()
