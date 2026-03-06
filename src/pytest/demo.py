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


def expected_topk(from_id: int, count: int, query: float, k: int) -> list[int]:
    # Sequential generator writes each vector as id+0.1 repeated by dim.
    scored = [(abs((i + 0.1) - query), i) for i in range(from_id, from_id + count)]
    scored.sort()
    return [idx for _, idx in scored[:k]]


def run_demo(count: int, dim: int, k: int, keep: bool) -> None:
    root = Path(tempfile.mkdtemp(prefix="sketch2_py_demo_"))
    dataset_dir = root / "dataset"
    from_id = 0

    try:
        with Parasol() as ps:
            ps.create(dataset_dir, type_name="f32", dim=dim, range_size=1000, data_merge_ratio=2)
            ps.open(dataset_dir)

            t0 = time.perf_counter()
            ps.generate(from_id=from_id, count=count, pattern=0, every_n_deleted=0)
            print(f"generated {count} vectors via sk_generate")
            ps.load()
            t1 = time.perf_counter()

            query = count * 0.631 + 0.123
            query_vec = fmt_vec(query, dim)
            actual = ps.knn(query_vec, k)
            expected = expected_topk(from_id, count, query, k)

            print(f"load+store time: {t1 - t0:.3f}s")
            print(f"query={query:.6f}, k={k}")
            print(f"actual   = {actual}")
            print(f"expected = {expected}")

            if actual != expected:
                raise AssertionError("KNN result mismatch")

            print("KNN check passed")
            ps.drop()
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

    run_demo(count=args.count, dim=args.dim, k=args.k, keep=args.keep)


if __name__ == "__main__":
    main()
