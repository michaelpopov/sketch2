#!/usr/bin/env python3
"""Initializer for the compute performance test."""

from __future__ import annotations

import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from common import (
    load_config,
    load_sketch2_types,
    log,
    find_lib_path,
    write_config_file,
    cosine_demo_vector,
    generic_demo_vector,
    fmt_typed_vector,
)

def write_chunk(
    chunk_path: str,
    from_id: int,
    count: int,
    dim: int,
    type_name: str,
    dist: str
) -> str:
    chunk_size = 4096
    with Path(chunk_path).open("w", encoding="utf-8") as out:
        chunk: list[str] = []
        for item_id in range(from_id, from_id + count):
            if dist == "cos":
                vals = cosine_demo_vector(item_id, dim, type_name)
            else:
                vals = generic_demo_vector(item_id, dim, type_name)
            chunk.append(f"{item_id} : [ {fmt_typed_vector(vals, type_name)} ]\n")
            if len(chunk) >= chunk_size:
                out.writelines(chunk)
                chunk.clear()
        if chunk:
            out.writelines(chunk)
    return chunk_path

def write_input_file_parallel(path: Path, count: int, dim: int, type_name: str, dist: str) -> None:
    workers = min(os.cpu_count() or 1, max(1, count // 25000))
    if workers <= 1:
        with path.open("w", encoding="utf-8") as out:
            out.write(f"{type_name},{dim}\n")
            chunk_size = 4096
            chunk: list[str] = []
            for item_id in range(count):
                if dist == "cos":
                    vals = cosine_demo_vector(item_id, dim, type_name)
                else:
                    vals = generic_demo_vector(item_id, dim, type_name)
                chunk.append(f"{item_id} : [ {fmt_typed_vector(vals, type_name)} ]\n")
                if len(chunk) >= chunk_size:
                    out.writelines(chunk)
                    chunk.clear()
            if chunk:
                out.writelines(chunk)
        return

    chunk_dir = path.parent / f"chunks_{dist}"
    shutil.rmtree(chunk_dir, ignore_errors=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    rows_per_chunk = (count + workers - 1) // workers
    chunk_specs = []
    chunk_start = 0
    chunk_index = 0
    while chunk_start < count:
        chunk_count = min(rows_per_chunk, count - chunk_start)
        chunk_path = chunk_dir / f"{chunk_index:04d}.part"
        chunk_specs.append((str(chunk_path), chunk_start, chunk_count, dim, type_name, dist))
        chunk_start += chunk_count
        chunk_index += 1

    try:
        from concurrent.futures import as_completed
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(write_chunk, *spec) for spec in chunk_specs]
            completed = 0
            for future in as_completed(futures):
                future.result()  # raise any exceptions
                completed += 1
                if completed % max(1, (len(chunk_specs) // 10)) == 0 or completed == len(chunk_specs):
                    log("initializer", f"  generated {completed}/{len(chunk_specs)} chunks for {dist}")
    except PermissionError:
        log("initializer", "process pool unavailable; falling back to single-process chunk generation")
        for chunk_spec in chunk_specs:
            write_chunk(*chunk_spec)

    with path.open("w", encoding="utf-8") as out:
        out.write(f"{type_name},{dim}\n")
        for chunk_spec in chunk_specs:
            cp = Path(chunk_spec[0])
            with cp.open("r", encoding="utf-8") as chunk_file:
                shutil.copyfileobj(chunk_file, out)

    shutil.rmtree(chunk_dir, ignore_errors=True)

def main() -> None:
    config = load_config()

    # Only wipe roots that look like harness-owned temp state or an existing Sketch2 dir.
    if config.db_dir.exists():
        db_dir_str = str(config.db_dir)
        is_temp = db_dir_str.startswith("/tmp/sketch2_COMPUTE_PERF.")
        is_sketch = (config.db_dir / "config.ini").exists()
        skip_init = os.environ.get("COMPUTE_PERF_SKIP_INIT") == "1"
        if (is_temp or is_sketch) and not skip_init:
            log("initializer", f"cleaning existing db_dir: {config.db_dir}")
            shutil.rmtree(config.db_dir)
        elif skip_init:
            log("initializer", f"skipping wipe of db_dir as requested: {config.db_dir}")
        else:
            log(
                "initializer",
                (
                    f"warning: db_dir {config.db_dir} exists but does not look like a harness temp "
                    "dir or Sketch2 dir; not wiping for safety"
                ),
            )

    config.db_dir.mkdir(parents=True, exist_ok=True)
    
    write_config_file(config)
    os.environ["SKETCH2_CONFIG"] = str(config.db_dir / "config.ini")
    
    from common import (
        get_ground_truth_knn,
        save_ground_truth,
        cosine_demo_query,
        generic_demo_query
    )
    
    lib_path = find_lib_path()
    Sketch2, _ = load_sketch2_types()
    
    with Sketch2(config.db_dir, lib_path=lib_path) as sketch2:
        for dist in config.dist_funcs:
            dataset_name = f"{config.dataset}_{dist}"
            log("initializer", f"creating dataset {dataset_name} with dist={dist}")

            sketch2.create(
                dataset_name,
                type_name=config.type_name,
                dim=config.dims,
                range_size=config.range_size,
                dist_func=dist,
            )

            input_path = config.db_dir / f"input_{dist}.txt"

            log("initializer", f"generating {config.count} vectors using sketch2.generate_test_data (native generator)")
            # native generator both generates and loads; COS uses CosCompatible pattern
            # inside generate_test_data(), and binary I/O accelerates generation.
            sketch2.generate_test_data(input_path, count=config.count, start_id=0, binary=True)

            if input_path.exists():
                input_path.unlink()

            log("initializer", f"calculating ground truth for {dist}")
            if dist == "cos":
                query_vals = cosine_demo_query(config.dims, config.type_name)
            else:
                query_vals = generic_demo_query(config.dims, config.type_name)
            
            if os.environ.get("DUMMY_CALC") == "1":
                log("initializer", f"skipping ground truth calculation for {dist} (DUMMY_CALC=1)")
                save_ground_truth(config, dist, [], [])
            else:
                expected_ids, expected_dists = get_ground_truth_knn(
                    config.count, config.dims, config.type_name, dist, query_vals, config.knn_count
                )
                save_ground_truth(config, dist, expected_ids, expected_dists)
            
            log("initializer", f"dataset {dataset_name} is ready")
            sketch2.close()

    log("initializer", "initialization complete")

if __name__ == "__main__":
    main()
