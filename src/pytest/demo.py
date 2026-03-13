#!/usr/bin/env python3
"""Demo: write vectors through Parasol, then read KNN results through SQLite."""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path

from parasol_wrapper import Parasol


def log_step(message: str) -> None:
    print(f"[demo] {message}", flush=True)


def fmt_scalar_vector(value: float, dim: int) -> str:
    return ", ".join(f"{value:.6f}" for _ in range(dim))


def fmt_values(values: list[float]) -> str:
    return ", ".join(f"{value:.6f}" for value in values)


def cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a)
    norm_b = sum(y * y for y in b)
    if norm_a == 0.0 and norm_b == 0.0:
        return 0.0
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    cosine = dot / ((norm_a * norm_b) ** 0.5)
    cosine = max(-1.0, min(1.0, cosine))
    return 1.0 - cosine


def cosine_demo_vector(item_id: int, dim: int) -> list[float]:
    values = [0.0] * dim
    values[0] = float((item_id % 17) + 1)
    values[1] = float(((item_id * 3) % 11) - 5)
    values[2] = float(((item_id * 5) % 7) - 3)
    for index in range(3, dim):
        values[index] = float(((item_id + index) % 5) - 2)
    return values


def cosine_demo_query(dim: int) -> list[float]:
    values = [0.0] * dim
    values[0] = 1.0
    values[1] = -0.5
    values[2] = 0.25
    for index in range(3, dim):
        values[index] = 0.1 * (1 if index % 2 == 0 else -1)
    return values


def expected_topk(from_id: int, count: int, query_value: float, k: int, dist_func: str, dim: int) -> list[int]:
    if dist_func == "cos":
        query_vec = cosine_demo_query(dim)
        scored = [
            (cosine_distance(cosine_demo_vector(item_id, dim), query_vec), item_id)
            for item_id in range(from_id, from_id + count)
        ]
        scored.sort()
        return [item_id for _, item_id in scored[:k]]

    scored = [(abs((item_id + 0.1) - query_value), item_id) for item_id in range(from_id, from_id + count)]
    scored.sort()
    return [item_id for _, item_id in scored[:k]]


def default_vlite_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "build" / "lib" / "libvlite.so",
        repo_root / "bin" / "libvlite.so",
        repo_root / "build-dbg" / "lib" / "libvlite.so",
        repo_root / "bin-dbg" / "libvlite.so",
        repo_root / "build-san" / "lib" / "libvlite.so",
        repo_root / "bin-san" / "libvlite.so",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("libvlite.so not found in build or bin directories")


def dataset_ini_path(root: Path, dataset_name: str) -> Path:
    return root / f"{dataset_name}.ini"


def fill_dataset(ps: Parasol, from_id: int, count: int, dim: int, dist_func: str) -> None:
    log_step(f"writing {count} vectors into the Parasol dataset using dist_func={dist_func}")
    if dist_func == "cos":
        for item_id in range(from_id, from_id + count):
            ps.upsert(item_id, fmt_values(cosine_demo_vector(item_id, dim)))
        return

    for item_id in range(from_id, from_id + count):
        ps.ups2(item_id, item_id + 0.1)


def sqlite_knn(dataset_ini: Path, vlite_lib: Path, query_vec: str, k: int) -> list[int]:
    log_step(f"opening in-memory SQLite and loading extension: {vlite_lib}")
    con = sqlite3.connect(":memory:")
    try:
        con.enable_load_extension(True)
        con.load_extension(str(vlite_lib))
        ini_sql = str(dataset_ini).replace("'", "''")
        create_sql = f"CREATE VIRTUAL TABLE nn USING vlite('{ini_sql}')"
        query_sql = "SELECT id FROM nn WHERE query = ? AND k = ? ORDER BY distance"

        log_step(f"executing SQL: {create_sql}")
        con.execute(create_sql)
        log_step(f"executing SQL: {query_sql}")
        log_step(f"SQLite bindings: query={query_vec!r}, k={k}")
        rows = con.execute(query_sql, (query_vec, k)).fetchall()
        return [int(row[0]) for row in rows]
    finally:
        con.close()


def run_demo(
    count: int,
    dim: int,
    k: int,
    keep: bool,
    dist_func: str,
    parasol_lib: Path | None,
    vlite_lib: Path | None,
) -> None:
    root = Path(tempfile.mkdtemp(prefix="sketch2_py_demo_"))
    dataset_name = "dataset"
    from_id = 0
    vlite_path = vlite_lib if vlite_lib is not None else default_vlite_path()
    dataset_ini = dataset_ini_path(root, dataset_name)

    try:
        log_step(f"created temporary workspace: {root}")
        if parasol_lib is not None:
            log_step(f"using Parasol library override: {parasol_lib}")
        log_step(f"using vlite extension: {vlite_path}")
        with Parasol(root, lib_path=parasol_lib) as ps:
            log_step(f"connected to libparasol: {ps.lib_path}")
            log_step(f"creating dataset '{dataset_name}' (dim={dim}, dist_func={dist_func})")
            ps.create(dataset_name, type_name="f32", dim=dim, range_size=1000, dist_func=dist_func)

            t0 = time.perf_counter()
            fill_dataset(ps, from_id=from_id, count=count, dim=dim, dist_func=dist_func)

            # SQLite reads only the persisted dataset state, so the writer must
            # flush the accumulator before the virtual table opens the dataset.
            log_step("merging the writer-side accumulator into persisted dataset files")
            ps.merge_accumulator()
            t1 = time.perf_counter()

            log_step("closing the Parasol writer handle before opening the SQLite reader")
            ps.close(dataset_name)

            query_value = count * 0.631 + 0.123
            query_vec = fmt_values(cosine_demo_query(dim)) if dist_func == "cos" else fmt_scalar_vector(query_value, dim)
            log_step("computing the expected top-k result in Python for comparison")
            expected = expected_topk(from_id, count, query_value, k, dist_func, dim)
            actual = sqlite_knn(dataset_ini, vlite_path, query_vec, k)

            print(f"load+merge time: {t1 - t0:.3f}s")
            print(f"dist_func={dist_func}")
            print(f"query={query_vec}")
            print(f"k={k}")
            print(f"actual   = {actual}")
            print(f"expected = {expected}")

            if actual != expected:
                raise AssertionError("SQLite KNN result mismatch")

            print("SQLite KNN check passed")
            log_step(f"dropping dataset '{dataset_name}'")
            ps.drop(dataset_name)
    finally:
        if keep:
            log_step(f"dataset preserved at: {root}")
        else:
            log_step(f"removing temporary workspace: {root}")
            shutil.rmtree(root, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parasol write + SQLite read demo")
    parser.add_argument("--count", type=int, default=20000, help="Number of vectors to load")
    parser.add_argument("--dim", type=int, default=4, help="Vector dimension (>=4)")
    parser.add_argument("--k", type=int, default=10, help="Top-K neighbors to query")
    parser.add_argument(
        "--dist-func",
        default="l1",
        choices=("l1", "l2", "cos"),
        help="Distance function used when creating the dataset",
    )
    parser.add_argument("--parasol-lib", type=Path, help="Path to libparasol.so")
    parser.add_argument("--vlite-lib", type=Path, help="Path to libvlite.so")
    parser.add_argument("--keep", action="store_true", help="Keep generated dataset directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise ValueError("--count must be >= 1")
    if args.dim < 4:
        raise ValueError("--dim must be >= 4")
    if args.k < 1:
        raise ValueError("--k must be >= 1")
    if args.k > args.count:
        raise ValueError("--k must be <= --count")

    run_demo(
        count=args.count,
        dim=args.dim,
        k=args.k,
        keep=args.keep,
        dist_func=args.dist_func,
        parasol_lib=args.parasol_lib,
        vlite_lib=args.vlite_lib,
    )


if __name__ == "__main__":
    main()
