#!/usr/bin/env python3
"""Continuously mutate the smoke-test dataset to create sustained write load."""

from __future__ import annotations

import time

from common import apply_runtime_env, find_lib_path, load_config, load_sketch2_types, log, vector_string


def main() -> None:
    config = load_config()
    apply_runtime_env(config)
    lib_path = find_lib_path()
    Sketch2, _ = load_sketch2_types()

    live_ids = list(range(config.count))
    next_id = config.count
    iterations = config.repeat

    with Sketch2(config.db_dir, lib_path=lib_path) as sketch2:
        sketch2.open(config.dataset)
        log("writer", f"opened dataset={config.dataset} iterations={iterations}")

        for iteration in range(iterations):
            delete_count = max(1, len(live_ids) // 3)
            delete_ids = live_ids[:delete_count]
            update_ids = live_ids[delete_count:delete_count * 2]
            added_ids = list(range(next_id, next_id + delete_count))

            sketch2.start_writing()
            for item_id in delete_ids:
                sketch2.write_deleted(item_id)
            for item_id in update_ids:
                sketch2.write_vector(item_id, vector_string(item_id, config.dims, revision=iteration + 1))
            for item_id in added_ids:
                sketch2.write_vector(item_id, vector_string(item_id, config.dims, revision=0))
            sketch2.complete_writing()

            sketch2.merge_delta()

            live_ids = live_ids[delete_count:] + added_ids
            next_id += delete_count

            log(
                "writer",
                (
                    f"iteration={iteration + 1}/{iterations} "
                    f"deleted={len(delete_ids)} updated={len(update_ids)} added={len(added_ids)} "
                    f"live={len(live_ids)} next_id={next_id}"
                ),
            )
            time.sleep(config.sleep_seconds)

        sketch2.close()

    log("writer", "completed all iterations")


if __name__ == "__main__":
    main()
