#!/usr/bin/env python3
"""Demo: write vectors through Parasol, then read KNN results through SQLite."""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import struct
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from math import isfinite
from pathlib import Path

from parasol_wrapper import Parasol

F16_MAX = 65504.0
I16_MIN = -32768
I16_MAX = 32767


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


def quantize_value(type_name: str, value: float) -> float | int:
    if type_name == "f32":
        return struct.unpack("f", struct.pack("f", value))[0]
    if type_name == "f16":
        return struct.unpack("e", struct.pack("e", value))[0]
    if type_name == "i16":
        return int(value)
    raise ValueError(f"unsupported type: {type_name}")


def quantize_values(type_name: str, values: list[float]) -> list[float | int]:
    return [quantize_value(type_name, value) for value in values]


def demo_query_scalar(count: int, type_name: str) -> float | int:
    raw_value = count * 0.631 + 0.123
    if type_name == "f32":
        return quantize_value(type_name, raw_value)
    if type_name == "f16":
        bounded = max(-F16_MAX, min(F16_MAX, raw_value))
        quantized = quantize_value(type_name, bounded)
        if not isfinite(float(quantized)):
            raise ValueError("demo f16 query value must remain finite")
        return quantized
    if type_name == "i16":
        bounded = max(I16_MIN, min(I16_MAX, int(raw_value)))
        return quantize_value(type_name, float(bounded))
    raise ValueError(f"unsupported type: {type_name}")


def fmt_typed_vector(values: list[float | int], type_name: str) -> str:
    if type_name == "i16":
        return ", ".join(str(int(value)) for value in values)
    return ", ".join(f"{float(value):.3f}" for value in values)


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


def cosine_demo_vector(item_id: int, dim: int, type_name: str) -> list[float | int]:
    values = [0.0] * dim
    values[0] = float((item_id % 17) + 1)
    values[1] = float(((item_id * 3) % 11) - 5)
    values[2] = float(((item_id * 5) % 7) - 3)
    for index in range(3, dim):
        values[index] = float(((item_id + index) % 5) - 2)
    return quantize_values(type_name, values)


def cosine_demo_query(dim: int, type_name: str) -> list[float | int]:
    values = [0.0] * dim
    values[0] = 1.0
    values[1] = -0.5
    values[2] = 0.25
    for index in range(3, dim):
        values[index] = 0.1 * (1 if index % 2 == 0 else -1)
    return quantize_values(type_name, values)


def l1_distance(a: list[float | int], b: list[float | int]) -> float:
    return sum(abs(float(x) - float(y)) for x, y in zip(a, b))


def l2_distance_sq(a: list[float | int], b: list[float | int]) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))


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


def load_dataset_with_binary_generator(ps: Parasol, from_id: int, count: int) -> tuple[float, float]:
    log_step(
        f"loading {count} generated binary vectors through libparasol "
        f"(pattern=sequential, start_id={from_id})"
    )
    t0 = time.perf_counter()
    ps.generate_bin(count=count, start_id=from_id, pattern=0)
    t1 = time.perf_counter()
    return 0.0, t1 - t0


def effective_input_format(binary: bool, dist_func: str) -> str:
    if dist_func == "COS":
        return "text"
    return "binary" if binary else "text"


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
    ps: Parasol,
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
    log_step(f"bulk-loading vectors from input file through libparasol: {input_path}")
    t2 = time.perf_counter()
    ps.load_file(input_path)
    t3 = time.perf_counter()
    return t1 - t0, t3 - t2


def fill_dataset(
    ps: Parasol,
    input_path: Path,
    from_id: int,
    count: int,
    dim: int,
    type_name: str,
    binary: bool,
    dist_func: str,
) -> tuple[float, float]:
    log_step(f"writing {count} vectors into the Parasol dataset using dist_func={dist_func}")
    if dist_func == "COS":
        log_step("cosine demo keeps the original Python-generated input path")
        return load_dataset_from_python_input_file(
            ps, input_path=input_path, from_id=from_id, count=count, dim=dim, type_name=type_name
        )

    if binary:
        log_step("binary demo uses libparasol binary generation instead of Python-side input file generation")
        return load_dataset_with_binary_generator(ps, from_id=from_id, count=count)

    return load_dataset_from_python_input_file(
        ps, input_path=input_path, from_id=from_id, count=count, dim=dim, type_name=type_name
    )


def sqlite_knn(dataset_ini: Path, vlite_lib: Path, query_vec: str, k: int) -> tuple[list[int], float]:
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
    parasol_lib: Path | None,
    vlite_lib: Path | None,
) -> None:
    root = Path(tempfile.mkdtemp(prefix="sketch2_py_demo_"))
    dataset_name = "dataset"
    from_id = 0
    vlite_path = vlite_lib if vlite_lib is not None else default_vlite_path()
    dataset_ini = dataset_ini_path(root, dataset_name)
    input_path = root / "demo.input"

    try:
        log_step(f"created temporary workspace: {root}")
        if parasol_lib is not None:
            log_step(f"using Parasol library override: {parasol_lib}")
        log_step(f"using vlite extension: {vlite_path}")
        with Parasol(root, lib_path=parasol_lib) as ps:
            log_step(f"connected to libparasol: {ps.lib_path}")
            log_step(
                f"creating dataset '{dataset_name}' "
                f"(type={type_name}, dim={dim}, range_size={range_size}, dist_func={dist_func})"
            )
            ps.create(dataset_name, type_name=type_name, dim=dim, range_size=range_size, dist_func=dist_func.lower())

            generate_time, load_time = fill_dataset(
                ps, input_path=input_path, from_id=from_id, count=count, dim=dim, type_name=type_name, binary=binary, dist_func=dist_func
            )

            # SQLite reads only the persisted dataset state, so the writer must
            # flush the accumulator before the virtual table opens the dataset.
            log_step("merging the writer-side accumulator into persisted dataset files")
            t0 = time.perf_counter()
            ps.merge_accumulator()
            t1 = time.perf_counter()
            merge_time = t1 - t0

            query_value = demo_query_scalar(count, type_name)
            query_vec = (
                fmt_typed_vector(cosine_demo_query(dim, type_name), type_name)
                if dist_func == "COS"
                else fmt_typed_vector([quantize_value(type_name, query_value)] * dim, type_name)
            )
            log_step("computing the expected top-k result through Parasol for comparison")
            expected = ps.knn(query_vec, k)

            log_step("closing the Parasol writer handle before opening the SQLite reader")
            ps.close(dataset_name)
            actual, query_time = sqlite_knn(dataset_ini, vlite_path, query_vec, k)

            print(f"generate input time: {generate_time:.3f}s")
            print(f"load data time: {load_time:.3f}s")
            print(f"merge time: {merge_time:.3f}s")
            print(f"sqlite query time: {query_time:.3f}s")
            print(f"type={type_name}")
            print(f"input_format={effective_input_format(binary, dist_func)}")
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
    parser = argparse.ArgumentParser(description="Parasol write + SQLite read demo")
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
    parser.add_argument("--type", default="f32", choices=("f32", "f16", "i16"), help="Dataset element type")
    parser.add_argument(
        "--dist-func",
        default="L1",
        choices=("L1", "L2", "COS"),
        help="Distance function used when creating the dataset",
    )
    parser.add_argument("--parasol-lib", type=Path, help="Path to libparasol.so")
    parser.add_argument("--vlite-lib", type=Path, help="Path to libvlite.so")
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Use libparasol binary generation instead of the Python text input file path",
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
        parasol_lib=args.parasol_lib,
        vlite_lib=args.vlite_lib,
    )


if __name__ == "__main__":
    main()
