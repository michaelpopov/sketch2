#!/usr/bin/env python3
"""Continuously query the smoke-test dataset to create sustained read load."""

from __future__ import annotations

import argparse
import time

from common import apply_runtime_env, find_lib_path, load_config, load_sketch2_types, log, query_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test reader process.")
    parser.add_argument(
        "--reader-id",
        default="reader",
        help="Label used in log messages for this reader process",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config()
    apply_runtime_env(config)
    lib_path = find_lib_path()
    Sketch2, _ = load_sketch2_types()

    with Sketch2(config.db_dir, lib_path=lib_path) as sketch2:
        sketch2.open(config.dataset)
        log(args.reader_id, f"opened dataset={config.dataset} repeats={config.repeat}")

        for iteration in range(config.repeat):
            query = query_vector(iteration, config.dims)
            ids = sketch2.knn(query, config.knn_count)
            if not ids:
                raise RuntimeError("knn returned no ids")
            log(
                args.reader_id,
                f"iteration={iteration + 1}/{config.repeat} query_ids={len(ids)} top_id={ids[0]}",
            )
            time.sleep(config.sleep_seconds)

        sketch2.close()

    log(args.reader_id, "completed all iterations")


if __name__ == "__main__":
    main()
