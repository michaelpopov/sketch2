#!/usr/bin/env python3
"""Demo: bulk-load vectors into libparasol dataset and validate KNN output."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

from parasol_wrapper import Parasol


def fmt_vec(value: float, dim: int) -> str:
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


def expected_topk_with_metric(from_id: int, count: int, query: float, k: int, dist_func: str,
                              dim: int) -> list[int]:
    if dist_func == "cos":
        query_vec = cosine_demo_query(dim)
        scored = [
            (cosine_distance(cosine_demo_vector(item_id, dim), query_vec), item_id)
            for item_id in range(from_id, from_id + count)
        ]
        scored.sort()
        return [item_id for _, item_id in scored[:k]]

    # Sequential generator writes each vector as id+0.1 repeated by dim.
    scored = [(abs((i + 0.1) - query), i) for i in range(from_id, from_id + count)]
    scored.sort()
    return [idx for _, idx in scored[:k]]


def load_cosine_demo_vectors(ps: Parasol, from_id: int, count: int, dim: int) -> None:
    for item_id in range(from_id, from_id + count):
        ps.upsert(item_id, fmt_values(cosine_demo_vector(item_id, dim)))
    ps.merge_accumulator()


def run_demo(count: int, dim: int, k: int, keep: bool, dist_func: str) -> None:
    root = Path(tempfile.mkdtemp(prefix="sketch2_py_demo_"))
    dataset_name = "dataset"
    dataset_dir = root / dataset_name
    from_id = 0

    try:
        with Parasol(root) as ps:
            ps.create(dataset_name, type_name="f32", dim=dim, range_size=1000, dist_func=dist_func)

            t0 = time.perf_counter()
            if dist_func == "cos":
                load_cosine_demo_vectors(ps, from_id, count, dim)
                print(f"generated {count} vectors via sk_upsert for cosine demo")
            else:
                ps.generate(count=count, start_id=from_id, pattern=0)
                print(f"generated {count} vectors via sk_generate")
            t1 = time.perf_counter()

            query = count * 0.631 + 0.123
            query_vec = fmt_values(cosine_demo_query(dim)) if dist_func == "cos" else fmt_vec(query, dim)
            actual = ps.knn(query_vec, k)
            expected = expected_topk_with_metric(from_id, count, query, k, dist_func, dim)

            print(f"load+store time: {t1 - t0:.3f}s")
            print(f"dist_func={dist_func}")
            print(f"query={query:.6f}, k={k}")
            print(f"actual   = {actual}")
            print(f"expected = {expected}")

            if actual != expected:
                raise AssertionError("KNN result mismatch")

            print("KNN check passed")
            ps.close(dataset_name)
            ps.drop(dataset_name)
    finally:
        if keep:
            print(f"dataset preserved at: {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parasol bulk-load and KNN validation demo")
    parser.add_argument("--count", type=int, default=20000, help="Number of vectors to load")
    parser.add_argument("--dim", type=int, default=4, help="Vector dimension (>=4)")
    parser.add_argument("--k", type=int, default=10, help="Top-K neighbors to query")
    parser.add_argument("--dist-func", default="l1", choices=("l1", "l2", "cos"),
                        help="Distance function used when creating the dataset")
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

    run_demo(count=args.count, dim=args.dim, k=args.k, keep=args.keep, dist_func=args.dist_func)


if __name__ == "__main__":
    main()
