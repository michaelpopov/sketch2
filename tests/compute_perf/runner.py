#!/usr/bin/env python3
"""Performance runner for the compute performance test."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

from common import (
    find_binary,
    load_config,
    load_sketch2_types,
    log,
    find_lib_path,
    fmt_typed_vector,
    load_ground_truth,
    query_values_for_dist,
    tree_size_bytes,
    validate_knn_results
)


def engine_label() -> str:
    engine = os.environ.get("SKETCH2_COMPUTE_ENGINE")
    return engine if engine else "auto"


def diag_dir(config) -> Path:
    raw = os.environ.get("COMPUTE_PERF_DIAG_DIR")
    if raw:
        path = Path(raw)
    else:
        path = config.db_dir / "logs" / "diag"
    path.mkdir(parents=True, exist_ok=True)
    return path


def diag_file_path(config, dist: str) -> Path:
    return diag_dir(config) / f"diag_{engine_label()}_{dist}.json"


def repro_script_path(config, dist: str) -> Path:
    return diag_dir(config) / f"repro_{engine_label()}_{dist}.sh"


def repro_loop_script_path(config, dist: str) -> Path:
    return diag_dir(config) / f"repro_loop_{engine_label()}_{dist}.sh"


def write_json(path: Path, data: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def tracked_env(config, dist: str) -> dict[str, str]:
    values = {
        "SKETCH2_CONFIG_ROOT": str(config.db_dir),
        "SKETCH2_CONFIG": str(config.db_dir / "config.ini"),
        "COMPUTE_PERF_TEST_DATASET": config.dataset,
        "COMPUTE_PERF_TEST_DIMS": str(config.dims),
        "COMPUTE_PERF_TEST_COUNT": str(config.count),
        "COMPUTE_PERF_TEST_REPEAT": str(config.repeat),
        "COMPUTE_PERF_TEST_K": str(config.knn_count),
        "COMPUTE_PERF_TEST_TYPE": config.type_name,
        "COMPUTE_PERF_TEST_DIST": ",".join(config.dist_funcs),
        "COMPUTE_PERF_TEST_RANGE_SIZE": str(config.range_size),
        "COMPUTE_PERF_TEST_LOG_LEVEL": config.log_level,
        "COMPUTE_PERF_TEST_THREAD_POOL_SIZE": str(config.thread_pool_size),
        "COMPUTE_PERF_TEST_ENGINES": ",".join(config.compute_engines),
        "COMPUTE_PERF_TEST_BENCHMARKS": ",".join(config.benchmark_layers),
        "COMPUTE_PERF_KERNEL_ITERATIONS": str(config.kernel_iterations),
        "COMPUTE_PERF_KERNEL_WARMUP_ITERATIONS": str(config.kernel_warmup_iterations),
        "COMPUTE_PERF_KERNEL_REPEATS": str(config.kernel_repeats),
        "COMPUTE_PERF_SINGLE_DIST": dist,
    }
    engine = os.environ.get("SKETCH2_COMPUTE_ENGINE")
    if engine:
        values["SKETCH2_COMPUTE_ENGINE"] = engine
    diag_root = os.environ.get("COMPUTE_PERF_DIAG_DIR")
    if diag_root:
        values["COMPUTE_PERF_DIAG_DIR"] = diag_root
    return values


def write_repro_scripts(config, dist: str) -> None:
    env_map = tracked_env(config, dist)
    runner_path = Path(__file__).resolve()
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(runner_path.parent))}",
    ]
    for key, value in env_map.items():
        lines.append(f"export {key}={shlex.quote(value)}")
    lines.append(f"python3 {shlex.quote(runner_path.name)}")
    repro_path = repro_script_path(config, dist)
    repro_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    repro_path.chmod(0o755)

    loop_lines = lines[:-1] + [
        'ITERATIONS="${1:-20}"',
        'i=1',
        'while [ "$i" -le "$ITERATIONS" ]; do',
        '  echo "[repro-loop] run $i/$ITERATIONS"',
        f"  python3 {shlex.quote(runner_path.name)}",
        '  i=$((i+1))',
        'done',
    ]
    loop_path = repro_loop_script_path(config, dist)
    loop_path.write_text("\n".join(loop_lines) + "\n", encoding="utf-8")
    loop_path.chmod(0o755)


def initial_diag_state(config, dist: str, query_vals: list[float | int], query_str: str) -> dict:
    dataset_name = f"{config.dataset}_{dist}"
    dataset_dir = config.db_dir / dataset_name
    return {
        "status": "running",
        "stage": "initialized",
        "engine": engine_label(),
        "metric": dist,
        "pid": os.getpid(),
        "db_dir": str(config.db_dir),
        "config_path": str(config.db_dir / "config.ini"),
        "dataset_name": dataset_name,
        "dataset_dir": str(dataset_dir),
        "dataset_size_bytes": tree_size_bytes(dataset_dir),
        "dataset_ini": str(config.db_dir / dataset_name / f"{dataset_name}.ini"),
        "ground_truth_path": str(config.db_dir / f"ground_truth_{dist}.json"),
        "repeat": config.repeat,
        "knn_count": config.knn_count,
        "dims": config.dims,
        "type_name": config.type_name,
        "query_digest_sha256": hashlib.sha256(query_str.encode("utf-8")).hexdigest(),
        "query_preview": query_str[:160],
        "query_value_count": len(query_vals),
        "repro_script": str(repro_script_path(config, dist)),
        "repro_loop_script": str(repro_loop_script_path(config, dist)),
    }


def update_diag(path: Path, state: dict, **updates) -> None:
    state.update(updates)
    write_json(path, state)


def kernel_bench_env(engine: str) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("SKETCH2_COMPUTE_BACKEND", None)
    if engine == "scalar":
        env["SKETCH2_COMPUTE_BACKEND"] = "scalar"
    return env


def kernel_engine_label(config) -> str:
    engine = engine_label()
    if engine != "auto":
        return engine

    compiled_engine = os.environ.get("COMPUTE_PERF_COMPILED_ENGINE", "").strip().lower()
    if compiled_engine:
        return compiled_engine

    concrete_engines = [candidate for candidate in config.compute_engines if candidate != "auto"]
    if len(concrete_engines) == 1:
        return concrete_engines[0]

    raise RuntimeError(
        "kernel benchmark requires a concrete compute engine; "
        "set COMPUTE_PERF_COMPILED_ENGINE or SKETCH2_COMPUTE_ENGINE"
    )


def run_kernel_benchmark(config, dist: str) -> dict:
    kernel_engine = kernel_engine_label(config)
    bench_path = find_binary("bench_compute")
    cmd = [
        str(bench_path),
        "--engine", kernel_engine,
        "--dist", dist,
        "--type", config.type_name,
        "--dim", str(config.dims),
        "--iterations", str(config.kernel_iterations),
        "--warmup-iterations", str(config.kernel_warmup_iterations),
        "--repeats", str(config.kernel_repeats),
    ]
    result = subprocess.run(
        cmd,
        env=kernel_bench_env(kernel_engine),
        cwd=str(bench_path.parent),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"bench_compute failed for engine={kernel_engine} dist={dist} "
            f"with exit code {result.returncode}: {result.stderr.strip() or result.stdout.strip()}"
        )
    json_start = result.stdout.find("{")
    if json_start == -1:
        raise RuntimeError(f"bench_compute did not emit JSON: {result.stdout}")
    try:
        return json.loads(result.stdout[json_start:])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"bench_compute produced invalid JSON: {result.stdout}") from exc


def print_kernel_report(payload: dict) -> None:
    print("--- KERNEL PERFORMANCE REPORT ---")
    print(f"Compute Engine:    {payload['engine']}")
    print(f"Metric:            {payload['dist']}")
    print(f"Type:              {payload['type']}")
    print(f"Dim:               {payload['dim']}")
    print(f"Warmup Iterations: {payload['warmup_iterations']}")
    print(f"Iterations:        {payload['iterations']}")
    print(f"Repeats:           {payload['repeats']}")
    print(f"Active Backend:    {payload['active_compute_backend']}")
    if "numkong_backend" in payload:
        print(f"NumKong Backend:   {payload['numkong_backend']}")
    for entry in payload["cases"]:
        print(
            "Kernel Case:      "
            f"{entry['name']} avg={entry['avg_ns_per_call']:.3f}ns "
            f"min={entry['min_ns_per_call']:.3f}ns max={entry['max_ns_per_call']:.3f}ns"
        )
    print("-------------------------------")


def bytes_to_gb(size_bytes: int) -> float:
    return size_bytes / 1_000_000_000.0


def run_single_distance(dist: str) -> None:
    config = load_config()
    os.environ["SKETCH2_CONFIG"] = str(config.db_dir / "config.ini")
    write_repro_scripts(config, dist)

    lib_path = find_lib_path()
    Sketch2, _ = load_sketch2_types()

    with Sketch2(config.db_dir, lib_path=lib_path) as sketch2:
        dataset_name = f"{config.dataset}_{dist}"
        log("runner", f"benchmarking dataset {dataset_name} with engine={engine_label()}")

        query_vals = query_values_for_dist(dist, config.dims, config.type_name)

        query_str = fmt_typed_vector(query_vals, config.type_name)
        diag_path = diag_file_path(config, dist)
        diag_state = initial_diag_state(config, dist, query_vals, query_str)
        write_json(diag_path, diag_state)

        log("runner", f"loading ground truth for {dist}...")
        expected_ids, expected_scores = load_ground_truth(config, dist)
        
        update_diag(
            diag_path,
            diag_state,
            stage="ground_truth_loaded",
            expected_ids_preview=expected_ids[: min(5, len(expected_ids))],
            expected_scores_preview=expected_scores[: min(5, len(expected_scores))],
        )

        if "kernel" in config.benchmark_layers:
            update_diag(diag_path, diag_state, stage="kernel_benchmark")
            kernel_payload = run_kernel_benchmark(config, dist)
            update_diag(diag_path, diag_state, kernel_benchmark=kernel_payload)
            print_kernel_report(kernel_payload)

        times = []
        opened = False
        try:
            if "scan" not in config.benchmark_layers:
                update_diag(diag_path, diag_state, status="ok", stage="completed")
                return
            update_diag(diag_path, diag_state, stage="opening_dataset")
            sketch2.open(dataset_name)
            opened = True
            log("runner", f"warm-up iteration for {dist}")
            update_diag(diag_path, diag_state, stage="before_warmup_knn")
            warmup_ids = sketch2.knn(query_str, config.knn_count)
            update_diag(
                diag_path,
                diag_state,
                stage="after_warmup_knn",
                warmup_ids_preview=warmup_ids[: min(5, len(warmup_ids))],
            )
            if not validate_knn_results(
                warmup_ids,
                expected_ids,
                expected_scores,
                query_vals,
                config.dims,
                config.type_name,
                dist,
            ):
                raise AssertionError(f"KNN warm-up mismatch for {dist} under engine={engine_label()}")

            log("runner", f"running {config.repeat} iterations for {dist}")
            for i in range(config.repeat):
                update_diag(diag_path, diag_state, stage="before_timed_knn", iteration=i)
                t0 = time.perf_counter()
                ids = sketch2.knn(query_str, config.knn_count)
                t1 = time.perf_counter()
                update_diag(
                    diag_path,
                    diag_state,
                    stage="after_timed_knn",
                    iteration=i,
                    actual_ids_preview=ids[: min(5, len(ids))],
                )

                if not ids:
                    raise AssertionError(
                        f"KNN returned no results at iteration {i} for {dist} under engine={engine_label()}"
                    )
                
                if not validate_knn_results(
                    ids, expected_ids, expected_scores, query_vals,
                    config.dims, config.type_name, dist
                ):
                    log("runner", f"ERROR: result mismatch at iteration {i}")
                    log("runner", f"  actual   = {ids}")
                    log("runner", f"  expected = {expected_ids}")
                    raise AssertionError(f"KNN result mismatch for {dist} under engine={engine_label()}")

                times.append(t1 - t0)
        except Exception as exc:
            update_diag(
                diag_path,
                diag_state,
                status="failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise
        finally:
            if opened:
                update_diag(diag_path, diag_state, stage="closing_dataset")
                sketch2.close()

        if times:
            min_t = min(times)
            max_t = max(times)
            avg_t = statistics.mean(times)
            update_diag(
                diag_path,
                diag_state,
                status="ok",
                stage="completed",
                min_time_sec=min_t,
                max_time_sec=max_t,
                avg_time_sec=avg_t,
            )

            print(f"--- PERFORMANCE REPORT ---")
            print(f"Compute Engine: {engine_label()}")
            print(f"Metric:         {dist}")
            print(f"Iterations:     {config.repeat}")
            print(f"Min Time:       {min_t:.6f}s")
            print(f"Max Time:       {max_t:.6f}s")
            print(f"Avg Time:       {avg_t:.6f}s")
            print(f"Dataset Size:   {bytes_to_gb(diag_state['dataset_size_bytes']):.6f} GB")
            print(f"--------------------------")


def run_all_distances() -> None:
    config = load_config()
    runner_path = Path(__file__).resolve()
    child_base_env = os.environ.copy()

    for dist in config.dist_funcs:
        child_env = child_base_env.copy()
        child_env["COMPUTE_PERF_SINGLE_DIST"] = dist
        result = subprocess.run([sys.executable, str(runner_path)], env=child_env, cwd=str(runner_path.parent))
        if result.returncode == 0:
            continue
        diag_path = diag_file_path(config, dist)
        if result.returncode < 0:
            raise SystemExit(
                f"runner subprocess died from signal {-result.returncode} while benchmarking dist={dist} "
                f"under engine={engine_label()}; see {diag_path}"
            )
        raise SystemExit(
            f"runner subprocess failed with exit code {result.returncode} while benchmarking dist={dist} "
            f"under engine={engine_label()}; see {diag_path}"
        )

    log("runner", "benchmarking complete")


def main() -> None:
    single_dist = os.environ.get("COMPUTE_PERF_SINGLE_DIST")
    if single_dist:
        run_single_distance(single_dist)
        return
    run_all_distances()

if __name__ == "__main__":
    main()
