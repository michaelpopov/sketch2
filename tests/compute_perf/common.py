#!/usr/bin/env python3
"""Shared helpers for the compute performance test harness."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


DATASET_METADATA_FILENAME = "dataset_metadata.json"


@dataclass(frozen=True)
class DatasetMetadata:
    dataset: str
    dims: int
    count: int
    knn_count: int
    type_name: str
    dist_funcs: list[str]
    range_size: int

    def to_json_dict(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "dataset": self.dataset,
            "dims": self.dims,
            "count": self.count,
            "knn_count": self.knn_count,
            "type_name": self.type_name,
            "dist_funcs": self.dist_funcs,
            "range_size": self.range_size,
        }


@dataclass(frozen=True)
class PerfConfig:
    db_dir: Path
    dataset: str
    dims: int
    count: int
    repeat: int
    knn_count: int
    type_name: str
    dist_funcs: list[str]
    range_size: int
    log_level: str
    thread_pool_size: int
    compute_engines: list[str]
    benchmark_layers: list[str]
    kernel_iterations: int
    kernel_warmup_iterations: int
    kernel_repeats: int


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# Add src/pytest to sys.path
_pytest_dir = str(repo_root() / "src" / "pytest")
if _pytest_dir not in sys.path:
    sys.path.insert(0, _pytest_dir)


from sketch2_test_vectors import (
    cosine_demo_query,
    cosine_demo_vector,
    cosine_distance,
    dot_distance,
    find_library,
    fmt_typed_vector,
    generic_demo_query,
    generic_demo_vector,
    l2_distance_sq,
    native_sequential_vector,
    repo_root as shared_repo_root,
)


def find_lib_path() -> Path:
    runtime_dir = configured_runtime_dir()
    if runtime_dir is not None:
        lib_path = runtime_dir / "libsketch2.so"
        if lib_path.exists():
            return lib_path
        raise FileNotFoundError(f"libsketch2.so not found in COMPUTE_PERF_RUNTIME_DIR: {runtime_dir}")
    return find_library()


def find_binary(name: str) -> Path:
    runtime_dir = configured_runtime_dir()
    if runtime_dir is not None:
        candidate = runtime_dir / name
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"{name} not found in COMPUTE_PERF_RUNTIME_DIR: {runtime_dir}")

    root = repo_root()
    candidates = [
        root / "bin" / name,
        root / "bin-dbg" / name,
        root / "bin-san" / name,
        root / "build" / "bin" / name,
        root / "build-dbg" / "bin" / name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{name} not found in expected output directories: {candidates}")


def configured_runtime_dir() -> Path | None:
    raw = os.environ.get("COMPUTE_PERF_RUNTIME_DIR")
    if not raw:
        return None
    path = Path(raw).resolve()
    if not path.exists():
        raise FileNotFoundError(f"COMPUTE_PERF_RUNTIME_DIR does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"COMPUTE_PERF_RUNTIME_DIR is not a directory: {path}")
    return path


def wrapper_dir() -> Path:
    path = shared_repo_root() / "src" / "pytest"
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
    return _env_int_from(os.environ, name, default)


def _env_required_str_from(env: Mapping[str, str], name: str) -> str:
    raw = env.get(name)
    if raw is None or raw == "":
        raise SystemExit(f"{name} must be set by the caller")
    return raw


def _env_required_int_from(env: Mapping[str, str], name: str) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        raise SystemExit(f"{name} must be set by the caller")
    return _env_int_from(env, name, 0)


def _env_int_from(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got: {raw!r}") from exc
    if value < 1:
        raise SystemExit(f"{name} must be >= 1, got: {value}")
    return value


def dataset_metadata_path(db_dir: Path) -> Path:
    return db_dir / DATASET_METADATA_FILENAME


def load_dataset_metadata(db_dir: Path) -> DatasetMetadata | None:
    path = dataset_metadata_path(db_dir)
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    format_version = data.get("format_version", 1)
    if format_version != 1:
        raise SystemExit(f"unsupported dataset metadata format_version={format_version} in {path}")

    try:
        dist_funcs = [str(item).strip().lower() for item in data["dist_funcs"] if str(item).strip()]
        return DatasetMetadata(
            dataset=str(data["dataset"]),
            dims=int(data["dims"]),
            count=int(data["count"]),
            knn_count=int(data["knn_count"]),
            type_name=str(data["type_name"]),
            dist_funcs=dist_funcs,
            range_size=int(data["range_size"]),
        )
    except KeyError as exc:
        raise SystemExit(f"dataset metadata is missing required field {exc!s} in {path}") from exc
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"dataset metadata is invalid in {path}: {exc}") from exc


def write_dataset_metadata(config: PerfConfig) -> Path:
    path = dataset_metadata_path(config.db_dir)
    metadata = DatasetMetadata(
        dataset=config.dataset,
        dims=config.dims,
        count=config.count,
        knn_count=config.knn_count,
        type_name=config.type_name,
        dist_funcs=config.dist_funcs,
        range_size=config.range_size,
    )
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(metadata.to_json_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)
    return path


def load_config(env: Mapping[str, str] | None = None) -> PerfConfig:
    env = os.environ if env is None else env
    raw_db_dir = _env_required_str_from(env, "SKETCH2_CONFIG_ROOT")
    db_dir = Path(raw_db_dir).resolve()
    metadata = load_dataset_metadata(db_dir)

    dataset = _env_required_str_from(env, "COMPUTE_PERF_TEST_DATASET")
    dims = _env_required_int_from(env, "COMPUTE_PERF_TEST_DIMS")
    count = _env_required_int_from(env, "COMPUTE_PERF_TEST_COUNT")
    repeat = _env_required_int_from(env, "COMPUTE_PERF_TEST_REPEAT")
    knn_count = _env_required_int_from(env, "COMPUTE_PERF_TEST_K")
    type_name = _env_required_str_from(env, "COMPUTE_PERF_TEST_TYPE")

    dist_str = _env_required_str_from(env, "COMPUTE_PERF_TEST_DIST")
    dist_funcs = [d.strip().lower() for d in dist_str.split(",") if d.strip()]

    range_size = _env_required_int_from(env, "COMPUTE_PERF_TEST_RANGE_SIZE")
    log_level = _env_required_str_from(env, "COMPUTE_PERF_TEST_LOG_LEVEL")
    thread_pool_size = _env_required_int_from(env, "COMPUTE_PERF_TEST_THREAD_POOL_SIZE")

    if metadata is not None:
        dataset = metadata.dataset
        dims = metadata.dims
        count = metadata.count
        knn_count = metadata.knn_count
        type_name = metadata.type_name
        dist_funcs = metadata.dist_funcs
        range_size = metadata.range_size

    engine_str = _env_required_str_from(env, "COMPUTE_PERF_TEST_ENGINES")
    compute_engines = [e.strip().lower() for e in engine_str.split(",") if e.strip()]
    benchmark_str = _env_required_str_from(env, "COMPUTE_PERF_TEST_BENCHMARKS")
    benchmark_layers = [layer.strip().lower() for layer in benchmark_str.split(",") if layer.strip()]
    kernel_iterations = _env_required_int_from(env, "COMPUTE_PERF_KERNEL_ITERATIONS")
    kernel_warmup_iterations = _env_required_int_from(env, "COMPUTE_PERF_KERNEL_WARMUP_ITERATIONS")
    kernel_repeats = _env_required_int_from(env, "COMPUTE_PERF_KERNEL_REPEATS")

    return PerfConfig(
        db_dir=db_dir,
        dataset=dataset,
        dims=dims,
        count=count,
        repeat=repeat,
        knn_count=knn_count,
        type_name=type_name,
        dist_funcs=dist_funcs,
        range_size=range_size,
        log_level=log_level,
        thread_pool_size=thread_pool_size,
        compute_engines=compute_engines,
        benchmark_layers=benchmark_layers,
        kernel_iterations=kernel_iterations,
        kernel_warmup_iterations=kernel_warmup_iterations,
        kernel_repeats=kernel_repeats,
    )


def config_path(config: PerfConfig) -> Path:
    return config.db_dir / "config.ini"


def write_config_file(config: PerfConfig) -> Path:
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


def ground_truth_path(config: PerfConfig, dist_func: str) -> Path:
    return config.db_dir / f"ground_truth_{dist_func}.json"


def save_ground_truth(config: PerfConfig, dist_func: str, ids: list[int], scores: list[float]) -> None:
    path = ground_truth_path(config, dist_func)
    data = {
        "ids": ids,
        "scores": scores
    }
    with path.open("w", encoding="utf-8") as out:
        json.dump(data, out)


def load_ground_truth(config: PerfConfig, dist_func: str) -> tuple[list[int], list[float]]:
    path = ground_truth_path(config, dist_func)
    if not path.exists():
        raise FileNotFoundError(f"ground truth file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    score_values = data.get("scores")
    if score_values is None:
        score_values = data["dists"]
    return data["ids"], score_values


def tree_size_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def distance_helpers(
    dist_func: str,
) -> tuple[
    callable[[list[float | int], list[float | int]], float],
    callable[[int, int, str], list[float | int]],
]:
    if dist_func == "cos":
        return cosine_distance, cosine_demo_vector
    if dist_func == "l2":
        return l2_distance_sq, native_sequential_vector
    if dist_func == "dot":
        return dot_distance, native_sequential_vector
    raise ValueError(f"unsupported score function: {dist_func}")


def query_values_for_dist(dist_func: str, dim: int, type_name: str) -> list[float | int]:
    if dist_func == "cos":
        return cosine_demo_query(dim, type_name)
    if dist_func in ("l2", "dot"):
        return generic_demo_query(dim, type_name)
    raise ValueError(f"unsupported score function: {dist_func}")


def smaller_score_is_better(dist_func: str) -> bool:
    if dist_func in ("cos", "l2"):
        return True
    if dist_func == "dot":
        return False
    raise ValueError(f"unsupported score function: {dist_func}")


def sort_metric_values(values: list[float], dist_func: str) -> list[float]:
    return sorted(values, reverse=not smaller_score_is_better(dist_func))


def expected_dists_for_ids(
    item_ids: list[int],
    dim: int,
    type_name: str,
    dist_func: str,
    query_vals: list[float | int],
) -> list[float]:
    dist_calc, vector_gen = distance_helpers(dist_func)
    scores = []
    for item_id in item_ids:
        vec = vector_gen(item_id, dim, type_name)
        scores.append(dist_calc(query_vals, vec))
    return sort_metric_values(scores, dist_func)


def get_ground_truth_knn(
    count: int, dim: int, type_name: str, dist_func: str, query_vals: list[float | int], k: int
) -> tuple[list[int], list[float]]:
    # cosine_demo_vector: LCM(17, 11, 7, 5) = 6545
    # native_sequential_vector: not periodic

    dist_calc = None
    vector_gen = None
    period = 1  # 1 means no period optimization
    if dist_func == "cos":
        dist_calc = cosine_distance
        vector_gen = cosine_demo_vector
        period = 6545
    else:
        dist_calc, vector_gen = distance_helpers(dist_func)

    # Calculate scores
    if period > 1:
        period_distances = []
        for m in range(min(period, count)):
            vec = vector_gen(m, dim, type_name)
            d = dist_calc(query_vals, vec)
            period_distances.append(d)

        candidates = []
        for item_id in range(count):
            d = period_distances[item_id % period]
            candidates.append((d, item_id))
    else:
        # Non-periodic or very large period: calculate all
        candidates = []
        for item_id in range(count):
            vec = vector_gen(item_id, dim, type_name)
            d = dist_calc(query_vals, vec)
            candidates.append((d, item_id))
    
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=not smaller_score_is_better(dist_func))
    top_k = candidates[:k]
    return [item_id for _, item_id in top_k], [d for d, _ in top_k]


def validate_knn_results(
    actual_ids: list[int],
    expected_ids: list[int],
    expected_scores: list[float],
    query_vals: list[float | int],
    dim: int,
    type_name: str,
    dist_func: str
) -> bool:
    if len(actual_ids) != len(expected_ids):
        return False

    if len(set(actual_ids)) != len(actual_ids):
        return False

    if actual_ids == expected_ids:
        return True

    # If IDs differ, check if it's just a tie-breaking difference.
    # We calculate scores for the actual IDs and compare them with expected scores.
    try:
        dist_calc, vector_gen = distance_helpers(dist_func)
    except ValueError:
        return False

    actual_scores = []
    for aid in actual_ids:
        vec = vector_gen(aid, dim, type_name)
        actual_scores.append(dist_calc(query_vals, vec))

    # Compare the score multisets so engines can break ties differently.
    eps = 1e-9
    for actual_score, expected_score in zip(sort_metric_values(actual_scores, dist_func), expected_scores):
        if abs(actual_score - expected_score) > eps:
            return False

    return True


def log(role: str, message: str) -> None:
    print(f"[{role}] {message}", flush=True)
