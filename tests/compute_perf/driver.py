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
import tempfile
from pathlib import Path

from common import DEFAULT_DB_DIR, dataset_metadata_path, load_config, load_dataset_metadata, load_sketch2_types


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_RUNTIME_DIR = REPO_ROOT / "bin"
ENGINE_RUNTIME_DIRS = {
    "highway": REPO_ROOT / "bin",
    "numkong": REPO_ROOT / "bin-nk",
}
SUPPORTED_ENGINES = frozenset(ENGINE_RUNTIME_DIRS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine",
        choices=sorted(SUPPORTED_ENGINES),
        help=(
            "Select which runtime directory to use. "
            "When omitted, the harness uses REPO_ROOT/bin."
        ),
    )
    return parser.parse_args()


def runtime_dir_for(engine: str | None) -> Path:
    if engine is None:
        return DEFAULT_RUNTIME_DIR
    return ENGINE_RUNTIME_DIRS[engine]


def ensure_runtime_artifacts(runtime_dir: Path) -> None:
    required = [
        runtime_dir / "libsketch2.so",
        runtime_dir / "bench_compute",
    ]
    for path in required:
        if not path.exists():
            raise SystemExit(
                f"[driver] ERROR: required runtime artifact not found: {path}\n"
                f"[driver] Build the runtime outputs into {runtime_dir} before "
                "running compute perf tests."
            )


def default_config_root() -> str:
    return str(DEFAULT_DB_DIR)


def build_env(
    runtime_dir: Path,
    requested_engine: str | None,
    compiled_engine: str,
    config_root: str,
) -> dict[str, str]:
    env = os.environ.copy()
    env["SKETCH2_CONFIG_ROOT"] = config_root
    env["SKETCH2_CONFIG"] = str(Path(config_root) / "config.ini")
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
    env["COMPUTE_PERF_TEST_ENGINES"] = compiled_engine
    env.setdefault("COMPUTE_PERF_TEST_BENCHMARKS", "scan,kernel")
    env.setdefault("COMPUTE_PERF_KERNEL_ITERATIONS", "200000")
    env.setdefault("COMPUTE_PERF_KERNEL_WARMUP_ITERATIONS", "5000")
    env.setdefault("COMPUTE_PERF_KERNEL_REPEATS", "7")
    env.setdefault("COMPUTE_PERF_TEST_CLEANUP", "0")
    env["COMPUTE_PERF_RUNTIME_DIR"] = str(runtime_dir)
    env["SKETCH2_LIB"] = str(runtime_dir)
    env["COMPUTE_PERF_COMPILED_ENGINE"] = compiled_engine
    env["COMPUTE_PERF_REQUESTED_ENGINE"] = requested_engine or ""
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
        "COMPUTE_PERF_REQUESTED_ENGINE",
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
        "COMPUTE_PERF_COMPILED_ENGINE",
        "COMPUTE_PERF_TEST_ENGINES",
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
    log(f"  requested_engine={env['COMPUTE_PERF_REQUESTED_ENGINE'] or 'default(bin)'}")
    log(f"  compiled_engine={env['COMPUTE_PERF_COMPILED_ENGINE']}")
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


def verify_dataset(config, dist: str) -> tuple[Path, Path]:
    root = config.db_dir
    dataset_path = root / f"{config.dataset}_{dist}"
    if not dataset_path.is_dir():
        raise SystemExit(f"[driver] ERROR: expected dataset directory not found: {dataset_path}")
    gt_path = root / f"ground_truth_{dist}.json"
    if not gt_path.is_file():
        raise SystemExit(f"[driver] ERROR: expected ground truth file not found: {gt_path}")
    return dataset_path, gt_path


def print_diag_paths(diag_dir: Path, engine: str) -> None:
    log(f"diagnostics directory: {diag_dir}")
    diag_paths = sorted(diag_dir.glob(f"diag_{engine}_*.json"))
    if diag_paths:
        log("diagnostic state files:")
        for path in diag_paths:
            print(path, flush=True)
    repro_paths = sorted(diag_dir.glob(f"repro_{engine}_*.sh"))
    if repro_paths:
        log("repro scripts:")
        for path in repro_paths:
            print(path, flush=True)
    loop_paths = sorted(diag_dir.glob(f"repro_loop_{engine}_*.sh"))
    if loop_paths:
        log("repro loop scripts:")
        for path in loop_paths:
            print(path, flush=True)


def probe_compiled_engine(runtime_dir: Path, config_root: str) -> str:
    try:
        Sketch2, _ = load_sketch2_types()
        with Sketch2(config_root, lib_path=runtime_dir / "libsketch2.so") as sketch2:
            engine = sketch2.compute_engine().strip().lower()
    except AttributeError as exc:
        raise SystemExit(
            f"[driver] ERROR: {runtime_dir / 'libsketch2.so'} does not export the "
            "new sk_compute_engine() API. Rebuild that runtime directory before "
            "running compute perf tests."
        ) from exc
    if engine not in SUPPORTED_ENGINES:
        raise SystemExit(
            f"[driver] ERROR: libsketch2.so in {runtime_dir} reported unsupported "
            f"compute engine {engine!r}."
        )
    return engine


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
    args = parse_args()
    runtime_dir = runtime_dir_for(args.engine)
    ensure_runtime_artifacts(runtime_dir)

    initial_config_root = os.environ.get("SKETCH2_CONFIG_ROOT") or default_config_root()
    probe_root = tempfile.mkdtemp(prefix="sketch2_compute_perf_probe.", dir="/tmp")
    try:
        compiled_engine = probe_compiled_engine(runtime_dir, probe_root)
    finally:
        shutil.rmtree(probe_root, ignore_errors=True)
    if args.engine is not None and compiled_engine != args.engine:
        raise SystemExit(
            f"[driver] ERROR: requested --engine {args.engine!r}, but "
            f"{runtime_dir / 'libsketch2.so'} reports {compiled_engine!r}."
        )

    env = build_env(runtime_dir, args.engine, compiled_engine, initial_config_root)
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
            log(f"benchmarks engine={compiled_engine} dist={dist}")
            env.pop("SKETCH2_COMPUTE_ENGINE", None)
            env["COMPUTE_PERF_SINGLE_DIST"] = dist
            runner_log = log_dir / f"runner_{compiled_engine}_{dist}.log"
            runner_rc = run_logged([sys.executable, "runner.py"], env, runner_log)
            if runner_rc != 0:
                print_diag_paths(diag_dir, compiled_engine)
                raise SystemExit(
                    f"[driver] ERROR: runner.py failed for engine={compiled_engine} "
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
