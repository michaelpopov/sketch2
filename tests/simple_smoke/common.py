#!/usr/bin/env python3
"""Shared helpers for the simple smoke test harness."""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SmokeConfig:
    db_dir: Path
    dataset: str
    dims: int
    count: int
    sleep_seconds: float
    repeat: int
    readers: int
    knn_count: int
    range_size: int
    type_name: str
    dist_func: str
    log_level: str
    thread_pool_size: int


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_lib_path() -> Path:
    root = repo_root()
    candidates = [
        root / "bin-dbg-hwy" / "libsketch2.so",
        root / "bin-hwy" / "libsketch2.so",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit("libsketch2.so not found under build*/lib or bin* directories")


def wrapper_dir() -> Path:
    path = repo_root() / "src" / "pytest"
    if not path.exists():
        raise SystemExit(f"sketch2_wrapper.py directory is missing: {path}")
    return path


def load_sketch2_types():
    path = str(wrapper_dir())
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        from sketch2_wrapper import Sketch2, Sketch2Error
    except ModuleNotFoundError as exc:
        raise SystemExit(f"Failed to import sketch2_wrapper from {path}: {exc}") from exc
    return Sketch2, Sketch2Error


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got: {raw!r}") from exc
    if value < 1:
        raise SystemExit(f"{name} must be >= 1, got: {value}")
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be a number, got: {raw!r}") from exc
    if value < 0.0:
        raise SystemExit(f"{name} must be >= 0, got: {value}")
    return value


def load_config() -> SmokeConfig:
    raw_db_dir = os.environ.get("SIMPLE_SMOKE_TEST_DB_DIR")
    if not raw_db_dir:
        raise SystemExit("SIMPLE_SMOKE_TEST_DB_DIR must be set")

    dataset = os.environ.get("SIMPLE_SMOKE_TEST_DATASET", "simple_smoke")
    if not dataset:
        raise SystemExit("SIMPLE_SMOKE_TEST_DATASET must not be empty")

    dims = _env_int("SIMPLE_SMOKE_TEST_DIMS", 8)
    count = _env_int("SIMPLE_SMOKE_TEST_COUNT", 900)
    sleep_seconds = _env_float("SIMPLE_SMOKE_TEST_SLEEP", 5.0)
    repeat = _env_int("SIMPLE_SMOKE_TEST_REPEAT", 12)
    readers = _env_int("SIMPLE_SMOKE_TEST_READERS", 3)
    knn_count = _env_int("SIMPLE_SMOKE_TEST_K", 5)
    type_name = os.environ.get("SIMPLE_SMOKE_TEST_TYPE", "f32")
    dist_func = os.environ.get("SIMPLE_SMOKE_TEST_DIST", "l2")
    log_level = os.environ.get("SIMPLE_SMOKE_TEST_LOG_LEVEL", "INFO")
    thread_pool_size = _env_int("SIMPLE_SMOKE_TEST_THREAD_POOL_SIZE", 4)
    range_size = _env_int("SIMPLE_SMOKE_TEST_RANGE_SIZE", 10000)

    return SmokeConfig(
        db_dir=Path(raw_db_dir).resolve(),
        dataset=dataset,
        dims=dims,
        count=count,
        sleep_seconds=sleep_seconds,
        repeat=repeat,
        readers=readers,
        knn_count=knn_count,
        range_size=range_size,
        type_name=type_name,
        dist_func=dist_func,
        log_level=log_level,
        thread_pool_size=thread_pool_size,
    )


def dataset_root(config: SmokeConfig) -> Path:
    return config.db_dir / config.dataset


def dataset_part_dirs(config: SmokeConfig) -> list[Path]:
    return [
        dataset_root(config) / "part_00",
        dataset_root(config) / "part_01",
        dataset_root(config) / "part_02",
    ]


def config_path(config: SmokeConfig) -> Path:
    return config.db_dir / "config.ini"


def write_config_file(config: SmokeConfig) -> Path:
    path = config_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "[log]\n"
        f"level={config.log_level}\n"
        "path=\n"
        "\n"
        "[thread_pool]\n"
        f"size={config.thread_pool_size}\n"
    )
    path.write_text(content, encoding="ascii")
    return path


def prepare_empty_db_dir(config: SmokeConfig) -> None:
    if config.db_dir.exists():
        # Only wipe if it already contains Sketch2 config or if it's explicitly under /tmp.
        is_sketch = (config.db_dir / "config.ini").exists()
        is_temp = str(config.db_dir.resolve()).startswith("/tmp/")
        if is_sketch or is_temp:
            log("initializer", f"cleaning existing db_dir: {config.db_dir}")
            shutil.rmtree(config.db_dir)
        else:
            log("initializer", f"warning: db_dir {config.db_dir} exists and is not a temp dir or known Sketch2 dir; not wiping for safety")

    config.db_dir.mkdir(parents=True, exist_ok=True)


def apply_runtime_env(config: SmokeConfig) -> Path:
    path = write_config_file(config)
    os.environ["SKETCH2_CONFIG"] = str(path)
    os.environ.setdefault("SKETCH2_LOG_LEVEL", config.log_level)
    os.environ.setdefault("SKETCH2_THREAD_POOL_SIZE", str(config.thread_pool_size))
    return path


def vector_values(item_id: int, dims: int, revision: int = 0) -> list[float]:
    base = item_id + 1 + revision * 17
    return [
        ((base * (index + 3) + (index * 11)) % 1000) / 100.0
        for index in range(dims)
    ]


def vector_string(item_id: int, dims: int, revision: int = 0) -> str:
    return ", ".join(f"{value:.2f}" for value in vector_values(item_id, dims, revision))


def query_vector(iteration: int, dims: int) -> str:
    return vector_string(iteration * 13 + 7, dims, revision=iteration % 5)


def log(role: str, message: str) -> None:
    print(f"[{role}] {message}", flush=True)
