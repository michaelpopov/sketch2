#!/usr/bin/env python3
"""Initializer for the compute performance test."""

from __future__ import annotations

import os

from common import (
    load_config,
    load_sketch2_types,
    log,
    find_lib_path,
    write_config_file,
    write_dataset_metadata,
)

def main() -> None:
    config = load_config()
    single_dist = os.environ.get("COMPUTE_PERF_SINGLE_DIST")
    dist_funcs = [single_dist] if single_dist else config.dist_funcs

    config.db_dir.mkdir(parents=True, exist_ok=True)
    write_config_file(config)
    os.environ["SKETCH2_CONFIG"] = str(config.db_dir / "config.ini")

    lib_path = find_lib_path()
    Sketch2, _ = load_sketch2_types()
    shared_input_path = config.db_dir / "input_perf_shared.bin"

    with Sketch2(config.db_dir, lib_path=lib_path) as sketch2:
        try:
            shared_input_ready = False
            for dist in dist_funcs:
                dataset_name = f"{config.dataset}_{dist}"
                log("initializer", f"creating dataset {dataset_name} with dist={dist}")

                sketch2.create(
                    dataset_name,
                    type_name=config.type_name,
                    dim=config.dims,
                    range_size=config.range_size,
                    dist_func=dist,
                )

                if not shared_input_ready:
                    log("initializer",
                        f"generating {config.count} vectors once using sketch2.generate_test_data (native generator)")
                    # Generate the shared binary corpus once, load it into the
                    # first dataset, then reuse the same file for the remaining
                    # datasets so all metrics ingest identical input.
                    sketch2.generate_test_data(
                        shared_input_path,
                        count=config.count,
                        start_id=0,
                        pattern="perf_test",
                        binary=True,
                    )
                    shared_input_ready = True
                else:
                    log("initializer", f"loading shared perf input into {dataset_name}")
                    sketch2.load_file(shared_input_path)

                log("initializer", f"dataset {dataset_name} is ready")
                sketch2.close()
        finally:
            if shared_input_path.exists():
                shared_input_path.unlink()

    if single_dist is None:
        metadata_path = write_dataset_metadata(config)
        log("initializer", f"wrote dataset metadata to {metadata_path}")

    log("initializer", "initialization complete")

if __name__ == "__main__":
    main()
