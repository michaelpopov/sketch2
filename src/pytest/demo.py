#!/usr/bin/env python3
"""Demo: write vectors through Sketch2, then read KNN results through SQLite."""

from __future__ import annotations

import argparse
from array import array
import heapq
import math
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from sketch2_test_vectors import (
    F16_MAX,
    F8_MAX,
    I16_MAX,
    cosine_distance,
    cosine_demo_query,
    cosine_demo_vector,
    demo_query_scalar,
    dot_distance,
    find_library,
    fmt_typed_vector,
    generic_demo_vector,
    l2_distance_sq,
    native_sequential_vector,
    quantize_value,
)
from sketch2_wrapper import Sketch2


# Stored i16 L2 scores are float-backed. Keeping each query component small
# avoids cancellation at the largest supported dimension while still producing
# a nontrivial deterministic ranking against the bounded native test payload.
I16_L2_QUERY_LIMIT = 500

# Highway accumulates floating-point query norms and dot products in f32 lanes.
# These constants bound the remaining error after the persisted vector norm has
# been modelled exactly below.
_F32_UNIT_ROUNDOFF = 2.0 ** -24
_F32_MIN_NORMAL = 2.0 ** -126
_F32_MIN_SUBNORMAL = 2.0 ** -149


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


def default_demo_count() -> int:
    return parse_size_arg("10M") if os.environ.get("SKETCH2_BIG_DEMO") else parse_size_arg("100K")


def dataset_ini_path(root: Path, dataset_name: str) -> Path:
    return root / dataset_name / f"{dataset_name}.ini"


def fill_dataset(
    ps: Sketch2,
    input_path: Path,
    from_id: int,
    count: int,
    dist_func: str,
) -> tuple[float, float]:
    log_step(f"writing {count} vectors into the Sketch2 dataset using dist_func={dist_func}")
    log_step(f"generating and loading {count} vectors using sketch2.generate_test_data (native generator)")
    t0 = time.perf_counter()
    # Use native binary generation for speed and let the API choose the default
    # pattern for the active dataset metric.
    ps.generate_test_data(input_path, count=count, start_id=from_id, pattern="auto", binary=True)
    t1 = time.perf_counter()
    return t1 - t0, 0.0


def native_demo_vector(
    item_id: int,
    from_id: int,
    dim: int,
    type_name: str,
    dist_func: str,
) -> list[float | int]:
    """Mirror sk_generate_test_data(..., pattern="auto") with decoded values."""
    if type_name == "f8":
        # C++ uses the bounded range-relative base-72 mapping for all three
        # auto-selected f8 patterns (COS, DOT, and sequential/L2).
        return native_sequential_vector(item_id, dim, type_name, min_id=from_id)
    if dist_func == "COS":
        return cosine_demo_vector(item_id, dim, type_name)
    return generic_demo_vector(item_id, dim, type_name)


def demo_query_values(count: int, dim: int, type_name: str, dist_func: str) -> list[float | int]:
    if dist_func == "COS":
        return cosine_demo_query(dim, type_name)
    query_value = demo_query_scalar(count, type_name)
    if type_name == "i16" and dist_func == "L2":
        query_value = max(-I16_L2_QUERY_LIMIT, min(I16_L2_QUERY_LIMIT, int(query_value)))
    return [quantize_value(type_name, float(query_value))] * dim


def _round_to_f32(value: float) -> float:
    return float(quantize_value("f32", value))


def _f32_ulp(value: float) -> float:
    """Return a conservative spacing for a finite f32 value."""
    magnitude = abs(value)
    if magnitude < _F32_MIN_NORMAL:
        return _F32_MIN_SUBNORMAL
    if not math.isfinite(magnitude):
        return math.inf
    return math.ldexp(1.0, math.floor(math.log2(magnitude)) - 23)


def l2_score_with_stored_norm(
    query: list[float | int],
    vector: list[float | int],
) -> float:
    """Mirror L2's float-backed persisted vector norm before scanner scoring."""
    stored_vector_norm_sq = _round_to_f32(
        sum(float(value) * float(value) for value in vector))
    query_norm_sq = sum(float(value) * float(value) for value in query)
    dot = dot_distance(query, vector)
    return max(0.0, stored_vector_norm_sq + query_norm_sq - 2.0 * dot)


def l2_simd_score_error_bound(
    query: list[float | int],
    vector: list[float | int],
    expected_score: float,
    type_name: str,
) -> float:
    """Bound L2 score drift from f32 SIMD query-norm and dot accumulation.

    The writer calculates a vector norm in scalar double and persists its f32
    narrowing, which `l2_score_with_stored_norm()` reproduces exactly.  The
    scanner then accumulates the query norm and dot product in f32 lanes for
    f32/f16/f8 values.  A conservative gamma(2n) bound covers those FMA and
    reduction roundings; the final term covers the local f32 result score.
    """
    if type_name == "i16":
        # i16 query norms and dots use widened integer accumulation. Only the
        # scanner's final local f32 score can differ from this double oracle.
        return _f32_ulp(expected_score)
    if type_name not in ("f32", "f16", "f8"):
        raise ValueError(f"unsupported type: {type_name}")
    if len(query) != len(vector):
        raise ValueError("query and vector dimensions must match")

    rounding_steps = max(1, 2 * len(query))
    scaled_unit_roundoff = rounding_steps * _F32_UNIT_ROUNDOFF
    if scaled_unit_roundoff >= 1.0:
        return math.inf
    gamma = scaled_unit_roundoff / (1.0 - scaled_unit_roundoff)
    query_norm_sq = sum(float(value) * float(value) for value in query)
    dot_magnitude = sum(abs(float(left) * float(right)) for left, right in zip(query, vector))
    simd_error = gamma * (query_norm_sq + 2.0 * dot_magnitude)

    stored_vector_norm_sq = _round_to_f32(
        sum(float(value) * float(value) for value in vector))
    final_expression_magnitude = (
        abs(stored_vector_norm_sq) + abs(query_norm_sq) + 2.0 * dot_magnitude)
    double_roundoff = 4.0 * sys.float_info.epsilon * final_expression_magnitude
    return simd_error + double_roundoff + _f32_ulp(expected_score)


def metric_score(
    query: list[float | int],
    vector: list[float | int],
    dist_func: str,
    *,
    type_name: str | None = None,
) -> float:
    if dist_func == "DOT":
        return dot_distance(query, vector)
    if dist_func == "L2":
        if type_name is not None:
            return l2_score_with_stored_norm(query, vector)
        return l2_distance_sq(query, vector)
    if dist_func == "COS":
        return cosine_distance(
            [float(value) for value in query], [float(value) for value in vector])
    raise ValueError(f"unsupported distance function: {dist_func}")


# This independent decoded oracle is kept for focused tests. The end-to-end
# demo compares the two public query surfaces through Sketch2.knn_items().
@dataclass(frozen=True)
class DecodedKnnReference:
    from_id: int
    dim: int
    type_name: str
    dist_func: str
    query: tuple[float | int, ...]
    cutoff_score: float
    strictly_better_ids: frozenset[int]
    diagnostic_ids: tuple[int, ...]
    scores_by_offset: array

    def contains_id(self, item_id: int) -> bool:
        return self.from_id <= item_id < self.from_id + len(self.scores_by_offset)

    def score_for_id(self, item_id: int) -> float:
        return self.scores_by_offset[item_id - self.from_id]

    def is_cutoff_tie(self, item_id: int) -> bool:
        return self.score_for_id(item_id) == self.cutoff_score

    def l2_score_tolerance_for_id(self, item_id: int) -> float:
        vector = native_demo_vector(item_id, self.from_id, self.dim, self.type_name, self.dist_func)
        return l2_simd_score_error_bound(
            list(self.query), vector, self.score_for_id(item_id), self.type_name)


def decoded_knn_reference(
    *,
    count: int,
    from_id: int,
    dim: int,
    type_name: str,
    dist_func: str,
    query: list[float | int],
    k: int,
) -> DecodedKnnReference:
    """Compute the top-k boundary from decoded native-generator vectors.

    Scores are cached compactly by ID offset during the single decoded-vector
    pass. This lets result validation identify exact cutoff ties without
    decoding and scoring the full dataset a second time.
    """
    if not 1 <= k <= count:
        raise ValueError("k must be in [1, count]")

    scores_by_offset = array("d")

    def scored_items():
        for item_id in range(from_id, from_id + count):
            vector = native_demo_vector(item_id, from_id, dim, type_name, dist_func)
            score = metric_score(query, vector, dist_func, type_name=type_name)
            scores_by_offset.append(score)
            yield score, item_id

    if dist_func == "DOT":
        selected = heapq.nlargest(k, scored_items(), key=lambda row: row[0])
        selected.sort(key=lambda row: (-row[0], row[1]))
    else:
        selected = heapq.nsmallest(k, scored_items(), key=lambda row: row[0])
        selected.sort(key=lambda row: (row[0], row[1]))

    cutoff_score = selected[-1][0]
    # Every score strictly better than the cutoff is necessarily in the top-k
    # selection. Tie membership is resolved from the compact cache during
    # result validation.
    strictly_better_ids = frozenset(
        item_id
        for score, item_id in selected
        if (
            (dist_func == "DOT" and score > cutoff_score)
            or (dist_func != "DOT" and score < cutoff_score)
        )
    )

    return DecodedKnnReference(
        from_id=from_id,
        dim=dim,
        type_name=type_name,
        dist_func=dist_func,
        query=tuple(query),
        cutoff_score=cutoff_score,
        strictly_better_ids=strictly_better_ids,
        diagnostic_ids=tuple(item_id for _, item_id in selected),
        scores_by_offset=scores_by_offset,
    )


def assert_sqlite_rows_match_decoded_reference(
    rows: list[tuple[int, float]],
    *,
    reference: DecodedKnnReference,
    dist_func: str,
    k: int,
) -> None:
    """Check scores and membership while allowing only genuine cutoff ties."""
    if len(rows) != k:
        raise AssertionError(f"SQLite returned {len(rows)} rows, expected {k}")

    actual_ids = [item_id for item_id, _ in rows]
    if len(set(actual_ids)) != len(actual_ids):
        raise AssertionError(f"SQLite returned duplicate IDs: {actual_ids}")

    actual_id_set = set(actual_ids)
    missing_strict_ids = reference.strictly_better_ids - actual_id_set
    if missing_strict_ids:
        raise AssertionError(
            f"SQLite omitted IDs strictly better than the cutoff: {sorted(missing_strict_ids)}")

    previous_score: float | None = None
    previous_tolerance = 0.0
    for item_id, actual_score in rows:
        if not reference.contains_id(item_id):
            raise AssertionError(
                f"SQLite returned ID {item_id} outside the generated range "
                f"[{reference.from_id}, {reference.from_id + len(reference.scores_by_offset)})")

        expected_score = reference.score_for_id(item_id)
        if item_id not in reference.strictly_better_ids and expected_score != reference.cutoff_score:
            raise AssertionError(
                f"SQLite returned ID {item_id} outside the decoded top-k cutoff tie set")

        score_tolerance = 1e-5
        if dist_func == "L2":
            score_tolerance = reference.l2_score_tolerance_for_id(item_id)
            if abs(actual_score - expected_score) > score_tolerance:
                raise AssertionError(
                    f"SQLite score for ID {item_id} was {actual_score}, expected {expected_score} "
                    f"within {score_tolerance}")
        elif not math.isclose(actual_score, expected_score, rel_tol=1e-5, abs_tol=1e-5):
            raise AssertionError(
                f"SQLite score for ID {item_id} was {actual_score}, expected {expected_score}")

        if previous_score is not None:
            if dist_func == "DOT" and actual_score > previous_score + 1e-5:
                raise AssertionError("SQLite DOT rows are not descending by score")
            if dist_func != "DOT" and actual_score + score_tolerance + previous_tolerance < previous_score:
                raise AssertionError("SQLite distance rows are not ascending by score")
        previous_score = actual_score
        previous_tolerance = score_tolerance


def assert_sqlite_rows_match_sketch2(
    rows: list[tuple[int, float]],
    expected_rows: list[tuple[int, float]],
) -> None:
    """Check that SQLite exposes the same KNN IDs and scores as Sketch2."""
    if rows != expected_rows:
        raise AssertionError(
            f"SQLite KNN rows differ from Sketch2: SQLite={rows}, Sketch2={expected_rows}")


def sqlite_knn(
    dataset_ini: Path,
    extension_lib: Path,
    query_vec: str,
    k: int,
    dist_func: str,
) -> tuple[list[tuple[int, float]], float]:
    log_step(f"opening in-memory SQLite and loading extension: {extension_lib}")
    con = sqlite3.connect(":memory:")
    try:
        con.enable_load_extension(True)
        con.load_extension(str(extension_lib))
        dataset_name = dataset_ini.stem
        db_path = dataset_ini.parent.parent
        db_path_sql = str(db_path).replace("'", "''")
        dataset_name_sql = dataset_name.replace("'", "''")
        create_sql = f"CREATE VIRTUAL TABLE nn USING vlite('{db_path_sql}', '{dataset_name_sql}')"
        order = "DESC" if dist_func == "DOT" else "ASC"
        query_sql = f"SELECT id, score FROM nn WHERE query = ? AND k = ? ORDER BY score {order}"

        log_step(f"executing SQL: {create_sql}")
        con.execute(create_sql)
        log_step(f"executing SQL: {query_sql}")
        log_step(f"SQLite bindings: k={k}")
        t0 = time.perf_counter()
        rows = con.execute(query_sql, (query_vec, k)).fetchall()
        t1 = time.perf_counter()
        return [(int(row[0]), float(row[1])) for row in rows], t1 - t0
    finally:
        con.close()


def run_demo(
    count: int,
    dim: int,
    k: int,
    range_size: int,
    type_name: str,
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
                ps,
                input_path=input_path,
                from_id=from_id,
                count=count,
                dist_func=dist_func,
            )

            # SQLite reads only the persisted dataset state, so the virtual table
            # should wait until the writer has finished loading data.
            log_step("writer finished loading persisted dataset files")

            query_values = demo_query_values(count, dim, type_name, dist_func)
            query_vec = fmt_typed_vector(query_values, type_name)
            log_step("computing expected KNN IDs and scores through Sketch2 for comparison")
            expected_rows = ps.knn_items(query_vec, k)

            log_step("closing the Sketch2 writer handle before opening the SQLite reader")
            ps.close()
            actual_rows, query_time = sqlite_knn(
                dataset_ini, extension_path, query_vec, k, dist_func)
            assert_sqlite_rows_match_sketch2(actual_rows, expected_rows)
            actual = [item_id for item_id, _ in actual_rows]
            expected = [item_id for item_id, _ in expected_rows]

            print(f"generate input time: {generate_time:.3f}s")
            print(f"load data time: {load_time:.3f}s")
            print(f"sqlite query time: {query_time:.3f}s")
            print(f"type={type_name}")
            print(f"input_format=binary")
            print(f"dist_func={dist_func}")
            print(f"k={k}")
            print(f"actual   = {actual}")
            print(f"expected = {expected}")

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
        default=default_demo_count(),
        help="Number of vectors to load; defaults to 100K, or 10M when SKETCH2_BIG_DEMO is set",
    )
    parser.add_argument("--dim", type=int, default=4, help="Vector dimension (>=4)")
    parser.add_argument("--k", type=int, default=10, help="Top-K neighbors to query")
    parser.add_argument(
        "--range-size",
        type=parse_size_arg,
        default=parse_size_arg("1000"),
        help="Dataset range size; accepts suffixes like 10K or 10M",
    )
    parser.add_argument("--type", default="f16", choices=("f32", "f16", "f8", "i16"), help="Dataset element type")
    parser.add_argument(
        "--dist-func",
        default="COS",
        choices=("DOT", "L2", "COS"),
        help="Score function used when creating the dataset",
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
        keep=args.keep,
        dist_func=args.dist_func,
        sketch2_lib=args.sketch2_lib,
        extension_lib=args.extension_lib,
    )


if __name__ == "__main__":
    main()
