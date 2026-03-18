"""Integration test 5: Distance Function Correctness at Scale.

For each distance function (L1, L2, COS), creates a dataset with
geometrically placed vectors where the correct KNN ordering is
analytically known, then verifies at scale.  Uses both the C API
wrapper and the SQLite virtual table.
"""

from __future__ import annotations

import math
import sqlite3
import unittest
from typing import Callable

from sketch2_wrapper import Sketch2

from integ_helpers import IntegTestBase, lib_path

VECTOR_COUNT = 1000
DIM = 4
RANGE_SIZE = 10000
NUM_QUERIES = 50


def _l1_distance(index: float, query: float) -> float:
    return abs(index - query) * DIM


def _l2_distance(index: float, query: float) -> float:
    return math.sqrt(DIM) * abs(index - query)


def _populate_linear_dataset(ps: Sketch2, vector_count: int = VECTOR_COUNT) -> None:
    for i in range(vector_count):
        val = f"{float(i)}"
        ps.upsert(i, f"{val}, {val}, {val}, {val}")
    ps.merge_accumulator()


def _assert_ordering(test_case: unittest.TestCase, ps: Sketch2, metric_fn: Callable[[float, float], float],
        vector_count: int = VECTOR_COUNT) -> None:
    for q_idx in range(NUM_QUERIES):
        q = q_idx * (vector_count / NUM_QUERIES)
        q_str = f"{q}, {q}, {q}, {q}"
        ids = ps.knn(q_str, 5)
        dists = [metric_fn(float(i), q) for i in ids]
        test_case.assertEqual(dists, sorted(dists),
            f"ordering violated for query {q}: ids={ids}")


class DistanceFunctionL1Test(IntegTestBase):
    """L1 (Manhattan) distance: sum of |a_i - b_i|."""

    _tmpdir_prefix = "sketch2_integ_dist_l1_"

    def test_l1_ordering_at_scale(self) -> None:
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dim=DIM, range_size=RANGE_SIZE, dist_func="l1")
            _populate_linear_dataset(ps)
            self.progress(f"Loaded {VECTOR_COUNT} vectors, running {NUM_QUERIES} L1 queries...")
            _assert_ordering(self, ps, _l1_distance)
            self.progress(f"All {NUM_QUERIES} L1 ordering checks passed")

            ps.close(self.dataset_name)

    def test_l1_update_changes_ranking(self) -> None:
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dim=DIM, range_size=RANGE_SIZE, dist_func="l1")

            for i in range(100):
                val = f"{float(i)}"
                ps.upsert(i, f"{val}, {val}, {val}, {val}")
            ps.merge_accumulator()

            ids = ps.knn("50.0, 50.0, 50.0, 50.0", 1)
            self.assertEqual([50], ids)
            self.progress(f"Before update: nearest to 50 is {ids}")

            ps.upsert(50, "9999.0, 9999.0, 9999.0, 9999.0")
            ps.merge_accumulator()

            ids = ps.knn("50.0, 50.0, 50.0, 50.0", 1)
            self.assertNotEqual([50], ids)
            self.assertIn(ids[0], [49, 51])
            self.progress(f"After moving id=50 away: nearest is now {ids}")

            ps.close(self.dataset_name)


class DistanceFunctionL2Test(IntegTestBase):
    """L2 (Euclidean) distance: sqrt(sum of (a_i - b_i)^2)."""

    _tmpdir_prefix = "sketch2_integ_dist_l2_"

    def test_l2_ordering_at_scale(self) -> None:
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dim=DIM, range_size=RANGE_SIZE, dist_func="l2")
            _populate_linear_dataset(ps)
            self.progress(f"Loaded {VECTOR_COUNT} vectors, running {NUM_QUERIES} L2 queries...")
            _assert_ordering(self, ps, _l2_distance)
            self.progress(f"All {NUM_QUERIES} L2 ordering checks passed")

            ps.close(self.dataset_name)

    def test_l2_via_sqlite_matches_api(self) -> None:
        ini_path = self.ini_path()

        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dim=DIM, range_size=RANGE_SIZE, dist_func="l2")
            for i in range(200):
                val = f"{float(i)}"
                ps.upsert(i, f"{val}, {val}, {val}, {val}")
            ps.merge_accumulator()

            api_ids = ps.knn("50.0, 50.0, 50.0, 50.0", 10)
            ps.close(self.dataset_name)
        self.progress(f"API KNN (L2): {api_ids}")

        with sqlite3.connect(":memory:") as conn:
            conn.enable_load_extension(True)
            conn.load_extension(lib_path())
            conn.execute(f"CREATE VIRTUAL TABLE nn USING vlite('{ini_path}')")

            rows = conn.execute(
                "SELECT id FROM nn "
                "WHERE query = '50.0, 50.0, 50.0, 50.0' AND k = 10 "
                "ORDER BY distance"
            ).fetchall()

        sql_ids = [row[0] for row in rows]
        self.assertEqual(api_ids, sql_ids)
        self.progress(f"SQLite KNN matches API: {sql_ids}")


class DistanceFunctionCosTest(IntegTestBase):
    """Cosine distance: 1 - cos(a, b)."""

    _tmpdir_prefix = "sketch2_integ_dist_cos_"

    def _cosine_distance(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 1.0
        return 1.0 - dot / (norm_a * norm_b)

    def test_cos_ordering_known_geometry(self) -> None:
        """Vectors placed along known directions; cosine ordering is predictable."""
        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dim=DIM, range_size=RANGE_SIZE, dist_func="cos")

            vectors: dict[int, list[float]] = {}
            for i in range(100):
                theta = i * math.pi / 200
                v = [math.cos(theta), math.sin(theta), 0.0, 0.0]
                vectors[i] = v
                ps.upsert(i, f"{v[0]}, {v[1]}, {v[2]}, {v[3]}")
            ps.merge_accumulator()
            self.progress("Loaded 100 angular vectors for cosine test")

            query = [1.0, 0.0, 0.0, 0.0]
            ids = ps.knn("1.0, 0.0, 0.0, 0.0", 10)

            expected_dists = [self._cosine_distance(query, vectors[i]) for i in ids]
            self.assertEqual(expected_dists, sorted(expected_dists),
                f"Cosine ordering violated: ids={ids}")
            self.assertEqual(0, ids[0])
            self.progress(f"Cosine KNN ordering correct, nearest ids: {ids[:5]}")

            ps.close(self.dataset_name)

    def test_cos_via_sqlite_returns_distances(self) -> None:
        ini_path = self.ini_path()

        with Sketch2(self.root) as ps:
            ps.create(self.dataset_name, dim=DIM, range_size=RANGE_SIZE, dist_func="cos")

            ps.upsert(1, "1.0, 0.0, 0.0, 0.0")
            ps.upsert(2, "0.707, 0.707, 0.0, 0.0")  # 45 degrees
            ps.upsert(3, "0.0, 1.0, 0.0, 0.0")       # 90 degrees
            ps.merge_accumulator()
            ps.close(self.dataset_name)

        with sqlite3.connect(":memory:") as conn:
            conn.enable_load_extension(True)
            conn.load_extension(lib_path())
            conn.execute(f"CREATE VIRTUAL TABLE nn USING vlite('{ini_path}')")

            rows = conn.execute(
                "SELECT id, distance FROM nn "
                "WHERE query = '1.0, 0.0, 0.0, 0.0' AND k = 3 "
                "ORDER BY distance"
            ).fetchall()

        ids = [row[0] for row in rows]
        dists = [row[1] for row in rows]

        self.assertEqual([1, 2, 3], ids)
        self.assertAlmostEqual(0.0, dists[0], places=3)
        self.assertAlmostEqual(1.0, dists[2], places=1)
        self.assertLess(dists[0], dists[1])
        self.assertLess(dists[1], dists[2])
        self.progress(f"SQLite cosine distances: {[f'{d:.4f}' for d in dists]}")


if __name__ == "__main__":
    unittest.main()
