#!/usr/bin/env python3
"""Demo: write vectors through Sketch2, then read KNN results through SQLite."""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from sketch2_test_vectors import (
    F16_MAX,
    I16_MAX,
    cosine_demo_query,
    cosine_demo_vector,
    demo_query_scalar,
    find_library,
    fmt_typed_vector,
    quantize_value,
    quantize_values,
    repo_root,
)
from sketch2_wrapper import Sketch2


def log_step(message: str) -> None:
    print(f"[demo] {message}", flush=True)


def parse_size_arg(value: str) -> int:
    text = value.strip().upper()
    multipliers = {
        "K": 1_000,
        "M": 1_000_000,
    }

    if not text:
        raise argparse.ArgumentTypeError("size value must not be empty")

    suffix = text[-1]
    if suffix in multipliers:
        number_part = text[:-1]
        if not number_part.isdigit():
            raise argparse.ArgumentTypeError(f"invalid size value: {value}")
        return int(number_part) * multipliers[suffix]

    if not text.isdigit():
        raise argparse.ArgumentTypeError(f"invalid size value: {value}")
    return int(text)


def dataset_ini_path(root: Path, dataset_name: str) -> Path:
    return root / dataset_name / f"{dataset_name}.ini"


def write_input_chunk(
    chunk_path: str,
    from_id: int,
    count: int,
    dim: int,
    type_name: str,
) -> str:
    chunk_size = 4096
    with Path(chunk_path).open("w", encoding="utf-8") as out:
        chunk: list[str] = []
        for item_id in range(from_id, from_id + count):
            values = cosine_demo_vector(item_id, dim, type_name)
            chunk.append(f"{item_id} : [ {fmt_typed_vector(values, type_name)} ]\n")
            if len(chunk) >= chunk_size:
                out.writelines(chunk)
                chunk.clear()
        if chunk:
            out.writelines(chunk)
    return chunk_path


def write_input_chunk_star(args: tuple[str, int, int, int, str]) -> str:
    return write_input_chunk(*args)


def write_input_file(path: Path, from_id: int, count: int, dim: int, type_name: str) -> None:
    workers = min(os.cpu_count() or 1, max(1, count // 50000))
    if workers <= 1:
        with path.open("w", encoding="utf-8") as out:
            out.write(f"{type_name},{dim}\n")
            chunk_size = 4096
            chunk: list[str] = []
            for item_id in range(from_id, from_id + count):
                values = cosine_demo_vector(item_id, dim, type_name)
                chunk.append(f"{item_id} : [ {fmt_typed_vector(values, type_name)} ]\n")
                if len(chunk) >= chunk_size:
                    out.writelines(chunk)
                    chunk.clear()
            if chunk:
                out.writelines(chunk)
        return

    chunk_dir = path.parent / "demo.input.parts"
    shutil.rmtree(chunk_dir, ignore_errors=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    rows_per_chunk = (count + workers - 1) // workers
    chunk_specs: list[tuple[str, int, int, int, str]] = []
    chunk_start = from_id
    chunk_index = 0
    while chunk_start < from_id + count:
        chunk_count = min(rows_per_chunk, from_id + count - chunk_start)
        chunk_path = chunk_dir / f"{chunk_index:04d}.part"
        chunk_specs.append((str(chunk_path), chunk_start, chunk_count, dim, type_name))
        chunk_start += chunk_count
        chunk_index += 1

    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(pool.map(write_input_chunk_star, chunk_specs))
    except PermissionError:
        log_step("process pool is unavailable in this runtime; falling back to single-process chunk generation")
        for chunk_spec in chunk_specs:
            write_input_chunk_star(chunk_spec)

    with path.open("w", encoding="utf-8") as out:
        out.write(f"{type_name},{dim}\n")
        for chunk_path, _, _, _, _ in chunk_specs:
            with Path(chunk_path).open("r", encoding="utf-8") as chunk_file:
                shutil.copyfileobj(chunk_file, out)

    shutil.rmtree(chunk_dir, ignore_errors=True)


def load_dataset_from_python_input_file(
    ps: Sketch2,
    input_path: Path,
    from_id: int,
    count: int,
    dim: int,
    type_name: str,
) -> tuple[float, float]:
    log_step(f"writing {count} Python-generated vectors to temporary text input file: {input_path}")
    t0 = time.perf_counter()
    write_input_file(input_path, from_id=from_id, count=count, dim=dim, type_name=type_name)
    t1 = time.perf_counter()
    log_step(f"bulk-loading vectors from input file through libsketch2: {input_path}")
    t2 = time.perf_counter()
    ps.load_file(input_path)
    t3 = time.perf_counter()
    return t1 - t0, t3 - t2


def fill_dataset(
    ps: Sketch2,
    input_path: Path,
    from_id: int,
    count: int,
    dim: int,
    type_name: str,
    binary: bool,
    dist_func: str,
) -> tuple[float, float]:
    log_step(f"writing {count} vectors into the Sketch2 dataset using dist_func={dist_func}")
    log_step(f"generating and loading {count} vectors using sketch2.generate_test_data (native generator)")
    t0 = time.perf_counter()
    # Use native binary generation for speed; generate_test_data auto-selects
    # CosCompatible pattern when the dataset dist_func is COS.
    ps.generate_test_data(input_path, count=count, start_id=from_id, binary=True)
    t1 = time.perf_counter()
    return t1 - t0, 0.0


def sqlite_knn(dataset_ini: Path, extension_lib: Path, query_vec: str, k: int) -> tuple[list[int], float]:
    log_step(f"opening in-memory SQLite and loading extension: {extension_lib}")
    con = sqlite3.connect(":memory:")
    try:
        con.enable_load_extension(True)
        con.load_extension(str(extension_lib))
        ini_sql = str(dataset_ini).replace("'", "''")
        create_sql = f"CREATE VIRTUAL TABLE nn USING vlite('{ini_sql}')"
        query_sql = "SELECT id FROM nn WHERE query = ? AND k = ? ORDER BY score"

        log_step(f"executing SQL: {create_sql}")
        con.execute(create_sql)
        log_step(f"executing SQL: {query_sql}")
        log_step(f"SQLite bindings: k={k}")
        t0 = time.perf_counter()
        rows = con.execute(query_sql, (query_vec, k)).fetchall()
        t1 = time.perf_counter()
        return [int(row[0]) for row in rows], t1 - t0
    finally:
        con.close()


def run_demo(
    count: int,
    dim: int,
    k: int,
    range_size: int,
    type_name: str,
    binary: bool,
    keep: bool,
    dist_func: str,
    sketch2_lib: Path | None,
    extension_lib: Path | None,
) -> None:
    root = Path(tempfile.mkdtemp(prefix="sketch2_py_demo_"))
    dataset_name = "dataset"
    from_id = 0
    extension_path = extension_lib if extension_lib is not None else find_library()
    dataset_ini = dataset_ini_path(root, dataset_name)
    input_path = root / "demo.input"

    try:
        log_step(f"created temporary workspace: {root}")
        if sketch2_lib is not None:
            log_step(f"using Sketch2 library override: {sketch2_lib}")
        log_step(f"using SQLite extension: {extension_path}")
        with Sketch2(root, lib_path=sketch2_lib) as ps:
            log_step(f"connected to libsketch2: {ps.lib_path}")
            log_step(
                f"creating dataset '{dataset_name}' "
                f"(type={type_name}, dim={dim}, range_size={range_size}, dist_func={dist_func})"
            )
            ps.create(dataset_name, type_name=type_name, dim=dim, range_size=range_size, dist_func=dist_func.lower())

            generate_time, load_time = fill_dataset(
                ps, input_path=input_path, from_id=from_id, count=count, dim=dim, type_name=type_name, binary=binary, dist_func=dist_func
            )

            # SQLite reads only the persisted dataset state, so the virtual table
            # should wait until the writer has finished loading data.
            log_step("writer finished loading persisted dataset files")

            query_value = demo_query_scalar(count, type_name)
            query_vec = (
                fmt_typed_vector(cosine_demo_query(dim, type_name), type_name)
                if dist_func == "COS"
                else fmt_typed_vector([quantize_value(type_name, query_value)] * dim, type_name)
            )
            log_step("computing the expected top-k result through Sketch2 for comparison")
            expected = ps.knn(query_vec, k)

            log_step("closing the Sketch2 writer handle before opening the SQLite reader")
            ps.close()
            actual, query_time = sqlite_knn(dataset_ini, extension_path, query_vec, k)
            if dist_func == "DOT":
                # DOT is similarity: larger score is better.
                actual = list(reversed(actual))

            print(f"generate input time: {generate_time:.3f}s")
            print(f"load data time: {load_time:.3f}s")
            print(f"sqlite query time: {query_time:.3f}s")
            print(f"type={type_name}")
            print(f"input_format=binary")
            print(f"dist_func={dist_func}")
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
    parser = argparse.ArgumentParser(description="Sketch2 write + SQLite read demo")
    parser.add_argument(
        "--count",
        type=parse_size_arg,
        default=parse_size_arg("20000"),
        help="Number of vectors to load; accepts suffixes like 10K or 10M",
    )
    parser.add_argument("--dim", type=int, default=4, help="Vector dimension (>=4)")
    parser.add_argument("--k", type=int, default=10, help="Top-K neighbors to query")
    parser.add_argument(
        "--range-size",
        type=parse_size_arg,
        default=parse_size_arg("1000"),
        help="Dataset range size; accepts suffixes like 10K or 10M",
    )
    parser.add_argument("--type", default="f16", choices=("f32", "f16", "i16"), help="Dataset element type")
    parser.add_argument(
        "--dist-func",
        default="COS",
        choices=("DOT", "L2", "COS"),
        help="Distance function used when creating the dataset",
    )
    parser.add_argument(
        "--sketch2-lib",
        dest="sketch2_lib",
        type=Path,
        help="Path to libsketch2.so (provides the Sketch2api entry points)",
    )
    parser.add_argument(
        "--extension-lib",
        "--vlite-lib",
        dest="extension_lib",
        type=Path,
        help="Path to SQLite extension library (libsketch2.so; legacy alias: --vlite-lib)",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Use libsketch2 binary generation instead of the Python text input file path",
    )
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
    if args.range_size < 1:
        raise ValueError("--range-size must be >= 1")

    run_demo(
        count=args.count,
        dim=args.dim,
        k=args.k,
        range_size=args.range_size,
        type_name=args.type,
        binary=args.binary,
        keep=args.keep,
        dist_func=args.dist_func,
        sketch2_lib=args.sketch2_lib,
        extension_lib=args.extension_lib,
    )


if __name__ == "__main__":
    main()
