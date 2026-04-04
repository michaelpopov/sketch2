#!/usr/bin/env python3
"""Create a fresh smoke-test database and load the initial dataset contents."""

from __future__ import annotations

from common import (
    apply_runtime_env,
    dataset_part_dirs,
    load_config,
    load_sketch2_types,
    log,
    prepare_empty_db_dir,
    vector_string,
    find_lib_path,
)


def main() -> None:
    config = load_config()
    prepare_empty_db_dir(config)
    cfg_path = apply_runtime_env(config)
    lib_path = find_lib_path()
    Sketch2, _ = load_sketch2_types()

    log("initializer", f"db_dir={config.db_dir}")
    log("initializer", f"config={cfg_path}")
    log("initializer", f"lib={lib_path}")

    with Sketch2(config.db_dir, lib_path=lib_path) as sketch2:
        sketch2.create(
            config.dataset,
            dirs=dataset_part_dirs(config),
            type_name=config.type_name,
            dim=config.dims,
            range_size=config.range_size,
            dist_func=config.dist_func,
        )
        log(
            "initializer",
            (
                f"created dataset={config.dataset} dims={config.dims} "
                f"count={config.count} type={config.type_name} dist={config.dist_func}"
            ),
        )

        sketch2.start_writing()
        for item_id in range(config.count):
            sketch2.write_vector(item_id, vector_string(item_id, config.dims))
        sketch2.complete_writing()
        sketch2.close()

    log("initializer", f"loaded {config.count} initial vectors")


if __name__ == "__main__":
    main()
