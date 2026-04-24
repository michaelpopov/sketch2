#!/usr/bin/env python3
"""Orchestrate the compute performance harness."""

from __future__ import annotations

import argparse
import os
import resource
import shutil
import signal
import subprocess
import sys
from pathlib import Path

from common import dataset_metadata_path, load_config, load_dataset_metadata


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
RUNTIME_DIR = REPO_ROOT / "bin"
RUNTIME_LABEL = "highway"


def release_build_target_for() -> str:
    return "rel"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def ensure_runtime_artifacts(runtime_dir: Path) -> None:
    required = [
        runtime_dir / "libsketch2.so",
        runtime_dir / "bench_compute",
    ]
    for path in required:
        if not path.exists():
            build_target = release_build_target_for()
            raise SystemExit(
                f"[driver] ERROR: required runtime artifact not found: {path}\n"
                f"[driver] Compute perf runs only against release artifacts in {runtime_dir}.\n"
                f"[driver] Build them first with `make {build_target}` before "
                "running compute perf tests."
            )
def build_env(
    runtime_dir: Path,
    runtime_label: str,
) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("SKETCH2_CONFIG_ROOT", "/tmp/sketch2_tests_compute_perf")
    env["SKETCH2_CONFIG"] = str(Path(env["SKETCH2_CONFIG_ROOT"]) / "config.ini")
    env.setdefault("COMPUTE_PERF_SKIP_INIT", "0")
    env.setdefault("COMPUTE_PERF_TEST_DATASET", "perf_test")
    env.setdefault("COMPUTE_PERF_TEST_DIMS", "256")
    env.setdefault("COMPUTE_PERF_TEST_COUNT", "100000")
    env.setdefault("COMPUTE_PERF_TEST_REPEAT", "10")
    env.setdefault("COMPUTE_PERF_TEST_K", "20")
    env.setdefault("COMPUTE_PERF_TEST_TYPE", "f32")
    env.setdefault("COMPUTE_PERF_TEST_DIST", "cos,l2,dot")
    env.setdefault("COMPUTE_PERF_TEST_RANGE_SIZE", "10000")
    env.setdefault("COMPUTE_PERF_TEST_LOG_LEVEL", "ERROR")
    env.setdefault("COMPUTE_PERF_TEST_THREAD_POOL_SIZE", "1")
    env["COMPUTE_PERF_RUNTIME_LABEL"] = runtime_label
    env.setdefault("COMPUTE_PERF_TEST_BENCHMARKS", "scan,kernel")
    env.setdefault("COMPUTE_PERF_KERNEL_ITERATIONS", "200000")
    env.setdefault("COMPUTE_PERF_KERNEL_WARMUP_ITERATIONS", "5000")
    env.setdefault("COMPUTE_PERF_KERNEL_REPEATS", "7")
    env.setdefault("COMPUTE_PERF_TEST_CLEANUP", "0")
    env["COMPUTE_PERF_RUNTIME_DIR"] = str(runtime_dir)
    env["SKETCH2_LIB"] = str(runtime_dir)
    return env


def apply_effective_dataset_config(env: dict[str, str], config) -> None:
    env["SKETCH2_CONFIG_ROOT"] = str(config.db_dir)
    env["SKETCH2_CONFIG"] = str(config.db_dir / "config.ini")
    env["COMPUTE_PERF_TEST_DATASET"] = config.dataset
    env["COMPUTE_PERF_TEST_DIMS"] = str(config.dims)
    env["COMPUTE_PERF_TEST_COUNT"] = str(config.count)
    env["COMPUTE_PERF_TEST_K"] = str(config.knn_count)
    env["COMPUTE_PERF_TEST_TYPE"] = config.type_name
    env["COMPUTE_PERF_TEST_DIST"] = ",".join(config.dist_funcs)
    env["COMPUTE_PERF_TEST_RANGE_SIZE"] = str(config.range_size)


def log(message: str) -> None:
    print(f"[driver] {message}", flush=True)


def format_rlimit(value: int) -> str:
    if value == resource.RLIM_INFINITY:
        return "unlimited"
    return str(value)


def enable_core_dumps() -> str:
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_CORE)
        target_soft = hard if hard != resource.RLIM_INFINITY else resource.RLIM_INFINITY
        resource.setrlimit(resource.RLIMIT_CORE, (target_soft, hard))
        soft, _ = resource.getrlimit(resource.RLIMIT_CORE)
        return format_rlimit(soft)
    except (OSError, ValueError):
        try:
            soft, _ = resource.getrlimit(resource.RLIMIT_CORE)
            return format_rlimit(soft)
        except (OSError, ValueError):
            return "unknown"


def read_core_pattern() -> str | None:
    core_pattern_path = Path("/proc/sys/kernel/core_pattern")
    if not core_pattern_path.is_file():
        return None
    try:
        return core_pattern_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def ensure_log_dirs(env: dict[str, str]) -> tuple[Path, Path]:
    log_dir = Path(env["SKETCH2_CONFIG_ROOT"]) / "logs"
    diag_dir = log_dir / "diag"
    log_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)
    env["COMPUTE_PERF_DIAG_DIR"] = str(diag_dir)
    return log_dir, diag_dir


def write_run_env(env: dict[str, str], log_dir: Path) -> None:
    keys = [
        "SKETCH2_CONFIG_ROOT",
        "SKETCH2_CONFIG",
        "SKETCH2_LIB",
        "COMPUTE_PERF_RUNTIME_DIR",
        "COMPUTE_PERF_RUNTIME_LABEL",
        "COMPUTE_PERF_SKIP_INIT",
        "COMPUTE_PERF_TEST_DATASET",
        "COMPUTE_PERF_TEST_DIMS",
        "COMPUTE_PERF_TEST_COUNT",
        "COMPUTE_PERF_TEST_REPEAT",
        "COMPUTE_PERF_TEST_K",
        "COMPUTE_PERF_TEST_TYPE",
        "COMPUTE_PERF_TEST_DIST",
        "COMPUTE_PERF_TEST_RANGE_SIZE",
        "COMPUTE_PERF_TEST_LOG_LEVEL",
        "COMPUTE_PERF_TEST_THREAD_POOL_SIZE",
        "COMPUTE_PERF_TEST_BENCHMARKS",
        "COMPUTE_PERF_KERNEL_ITERATIONS",
        "COMPUTE_PERF_KERNEL_WARMUP_ITERATIONS",
        "COMPUTE_PERF_KERNEL_REPEATS",
        "COMPUTE_PERF_TEST_CLEANUP",
        "COMPUTE_PERF_DIAG_DIR",
    ]
    lines = [f"{key}={env[key]}" for key in keys if key in env]
    (log_dir / "run_env.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_config(env: dict[str, str], config, runtime_dir: Path, core_limit: str, cache_state: str) -> None:
    log("performance test configuration")
    log(f"  runtime_dir={runtime_dir}")
    log(f"  runtime_label={env['COMPUTE_PERF_RUNTIME_LABEL']}")
    log(f"  config_root={config.db_dir}")
    log(f"  cache_state={cache_state}")
    log(f"  dataset_metadata={dataset_metadata_path(config.db_dir)}")
    log(f"  dataset={config.dataset}")
    log(f"  dims={config.dims}")
    log(f"  count={config.count}")
    log(f"  repeat={config.repeat}")
    log(f"  k={config.knn_count}")
    log(f"  type={config.type_name}")
    log(f"  dist={','.join(config.dist_funcs)}")
    log(f"  range_size={config.range_size}")
    log(f"  benchmarks={env['COMPUTE_PERF_TEST_BENCHMARKS']}")
    log(f"  kernel_iterations={env['COMPUTE_PERF_KERNEL_ITERATIONS']}")
    log(f"  kernel_warmup_iterations={env['COMPUTE_PERF_KERNEL_WARMUP_ITERATIONS']}")
    log(f"  kernel_repeats={env['COMPUTE_PERF_KERNEL_REPEATS']}")
    log(f"  cleanup={env['COMPUTE_PERF_TEST_CLEANUP']}")
    log(f"  diag_dir={env['COMPUTE_PERF_DIAG_DIR']}")
    log(f"  core_limit={core_limit}")
    core_pattern = read_core_pattern()
    if core_pattern is not None:
        log(f"  core_pattern={core_pattern}")


def run_logged(cmd: list[str], env: dict[str, str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            cmd,
            cwd=SCRIPT_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return process.wait()


def verify_dataset(config, dist: str) -> Path:
    root = config.db_dir
    dataset_path = root / f"{config.dataset}_{dist}"
    if not dataset_path.is_dir():
        raise SystemExit(f"[driver] ERROR: expected dataset directory not found: {dataset_path}")
    return dataset_path


def print_diag_paths(diag_dir: Path, runtime_label: str) -> None:
    log(f"diagnostics directory: {diag_dir}")
    diag_paths = sorted(diag_dir.glob(f"diag_{runtime_label}_*.json"))
    if diag_paths:
        log("diagnostic state files:")
        for path in diag_paths:
            print(path, flush=True)
    repro_paths = sorted(diag_dir.glob(f"repro_{runtime_label}_*.sh"))
    if repro_paths:
        log("repro scripts:")
        for path in repro_paths:
            print(path, flush=True)
    loop_paths = sorted(diag_dir.glob(f"repro_loop_{runtime_label}_*.sh"))
    if loop_paths:
        log("repro loop scripts:")
        for path in loop_paths:
            print(path, flush=True)


def install_signal_handlers() -> None:
    def _raise_keyboard_interrupt(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGINT, _raise_keyboard_interrupt)
    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)


def cleanup_root(env: dict[str, str]) -> None:
    root = env["SKETCH2_CONFIG_ROOT"]
    if env["COMPUTE_PERF_TEST_CLEANUP"] == "1":
        log(f"cleaning up {root}...")
        shutil.rmtree(root, ignore_errors=True)
    else:
        log(
            f"preserving {root} "
            "(set COMPUTE_PERF_TEST_CLEANUP=1 to remove it automatically)"
        )


def initialize_dataset_cache(env: dict[str, str], log_dir: Path) -> None:
    init_log = log_dir / "initializer.log"
    init_rc = run_logged([sys.executable, "initializer.py"], env, init_log)
    if init_rc != 0:
        raise SystemExit(
            f"[driver] ERROR: initializer.py failed with exit code {init_rc}. See {init_log}"
        )


def ensure_cache_state(env: dict[str, str]) -> tuple[object, str]:
    config = load_config(env)
    metadata = load_dataset_metadata(config.db_dir)

    if metadata is None:
        if config.db_dir.exists():
            raise SystemExit(
                f"[driver] ERROR: dataset cache directory exists without metadata: "
                f"{config.db_dir}\n"
                f"[driver] Remove {config.db_dir} and rerun to rebuild the cache."
            )
        config.db_dir.mkdir(parents=True, exist_ok=True)
        return config, "initialize"

    return config, "reuse"


def main() -> int:
    parse_args()
    runtime_dir = RUNTIME_DIR
    ensure_runtime_artifacts(runtime_dir)

    runtime_label = RUNTIME_LABEL

    env = build_env(runtime_dir, runtime_label)
    config, cache_state = ensure_cache_state(env)
    apply_effective_dataset_config(env, config)

    install_signal_handlers()
    core_limit = enable_core_dumps()

    try:
        log_dir, diag_dir = ensure_log_dirs(env)
        if cache_state == "initialize":
            log(f"dataset cache does not exist yet; initializing {config.db_dir}")
            initialize_dataset_cache(env, log_dir)
            config = load_config(env)
            apply_effective_dataset_config(env, config)
            cache_state = "initialized"
            log_dir, diag_dir = ensure_log_dirs(env)
        else:
            log(f"reusing existing dataset cache at {config.db_dir}")

        for dist in config.dist_funcs:
            verify_dataset(config, dist)

        write_run_env(env, log_dir)
        print_config(env, config, runtime_dir, core_limit, cache_state)

        for dist in config.dist_funcs:
            log(f"benchmarks runtime={runtime_label} dist={dist}")
            env["COMPUTE_PERF_SINGLE_DIST"] = dist
            runner_log = log_dir / f"runner_{runtime_label}_{dist}.log"
            runner_rc = run_logged([sys.executable, "runner.py"], env, runner_log)
            if runner_rc != 0:
                print_diag_paths(diag_dir, runtime_label)
                raise SystemExit(
                    f"[driver] ERROR: runner.py failed for runtime={runtime_label} "
                    f"dist={dist} with exit code {runner_rc}. See {runner_log}"
                )

        reporter_log = log_dir / "reporter.log"
        reporter_rc = run_logged([sys.executable, "reporter.py"], env, reporter_log)
        if reporter_rc != 0:
            raise SystemExit(
                f"[driver] ERROR: reporter.py failed with exit code {reporter_rc}. "
                f"See {reporter_log}"
            )
        return 0
    except KeyboardInterrupt:
        log("interrupted")
        return 130
    finally:
        cleanup_root(env)


if __name__ == "__main__":
    sys.exit(main())
