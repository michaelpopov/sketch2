#!/usr/bin/env python3
"""Initializer for the compute performance test."""

from __future__ import annotations

import os
import time

from common import (
    load_config,
    load_sketch2_types,
    log,
    find_lib_path,
    write_config_file,
    write_dataset_metadata,
)


def _fmt_secs(elapsed: float) -> str:
    return f"{elapsed:.3f}s"


def _stage_start(name: str) -> float:
    t0 = time.perf_counter()
    log("initializer", f"=== STAGE START: {name} ===")
    return t0


def _stage_end(name: str, start_time: float) -> None:
    elapsed = time.perf_counter() - start_time
    log("initializer", f"=== STAGE END: {name} (elapsed={_fmt_secs(elapsed)}) ===")


def _load_dataset(sketch2, dataset_name: str, shared_input_path, index: int, total: int) -> None:
    stage_name = f"load dataset {index}/{total}: {dataset_name}"
    stage_t0 = _stage_start(stage_name)
    try:
        sketch2.open(dataset_name)
        try:
            log("initializer", f"loading shared input into {dataset_name}")
            load_t0 = time.perf_counter()
            sketch2.load_file(shared_input_path)
            load_elapsed = time.perf_counter() - load_t0
            log("initializer", f"load complete for {dataset_name} (load_time={_fmt_secs(load_elapsed)})")
        finally:
            sketch2.close()
    finally:
        _stage_end(stage_name, stage_t0)


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
    target_datasets = [f"{config.dataset}_{dist}" for dist in dist_funcs]
    generator_dist = dist_funcs[0]
    generator_dataset_name = f"__compute_perf_input_gen_{os.getpid()}"

    total_t0 = _stage_start("initializer total")

    with Sketch2(config.db_dir, lib_path=lib_path) as sketch2:
        try:
            create_stage_t0 = _stage_start("create target datasets")
            for dataset_name, dist in zip(target_datasets, dist_funcs):
                log("initializer", f"creating dataset {dataset_name} with dist={dist}")
                sketch2.create(
                    dataset_name,
                    type_name=config.type_name,
                    dim=config.dims,
                    range_size=config.range_size,
                    dist_func=dist,
                )
                sketch2.close()
            _stage_end("create target datasets", create_stage_t0)

            generate_stage_t0 = _stage_start("generate shared input file")
            generator_created = False
            try:
                log("initializer", f"creating temporary generator dataset {generator_dataset_name} with dist={generator_dist}")
                sketch2.create(
                    generator_dataset_name,
                    type_name=config.type_name,
                    dim=config.dims,
                    range_size=config.range_size,
                    dist_func=generator_dist,
                )
                generator_created = True
                log(
                    "initializer",
                    (
                        f"generating shared input ({config.count} vectors, pattern=perf_test, binary=true) "
                        "using sketch2.generate_test_data"
                    ),
                )
                sketch2.generate_test_data(
                    shared_input_path,
                    count=config.count,
                    start_id=0,
                    pattern="perf_test",
                    binary=True,
                )
                log("initializer", f"shared input generated at {shared_input_path}")
            finally:
                # generate_test_data keeps the temporary dataset loaded; drop it so
                # load timings below represent only benchmark datasets.
                try:
                    sketch2.close()
                except Exception:
                    pass
                if generator_created:
                    sketch2.drop(generator_dataset_name)
                    log("initializer", f"dropped temporary generator dataset {generator_dataset_name}")
            _stage_end("generate shared input file", generate_stage_t0)

            load_all_stage_t0 = _stage_start("load shared input into target datasets")
            for idx, dataset_name in enumerate(target_datasets, start=1):
                _load_dataset(sketch2, dataset_name, shared_input_path, idx, len(target_datasets))
                log("initializer", f"dataset {dataset_name} is ready")
            _stage_end("load shared input into target datasets", load_all_stage_t0)
        finally:
            if shared_input_path.exists():
                cleanup_t0 = _stage_start("cleanup shared input file")
                shared_input_path.unlink()
                _stage_end("cleanup shared input file", cleanup_t0)

    if single_dist is None:
        metadata_t0 = _stage_start("write dataset metadata")
        metadata_path = write_dataset_metadata(config)
        _stage_end("write dataset metadata", metadata_t0)
        log("initializer", f"wrote dataset metadata to {metadata_path}")

    _stage_end("initializer total", total_t0)
    log("initializer", "initialization complete")


if __name__ == "__main__":
    main()
