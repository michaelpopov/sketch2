from __future__ import annotations

import math
import sys
import unittest
from unittest.mock import patch

from demo import (
    F16_MAX,
    F8_MAX,
    I16_MAX,
    assert_sqlite_rows_match_decoded_reference,
    decoded_knn_reference,
    demo_query_values,
    demo_query_scalar,
    metric_score,
    native_demo_vector,
    parse_args,
    run_demo,
)
from sketch2_test_vectors import (
    bounded_demo_vector,
    f8_encode_bits,
    f8_ordinal_bytes,
    find_library,
)


class DemoQueryScalarTest(unittest.TestCase):
    def test_demo_query_scalar_keeps_f16_in_range(self) -> None:
        value = demo_query_scalar(10_000_000, "f16")
        self.assertTrue(math.isfinite(float(value)))
        self.assertLessEqual(abs(float(value)), F16_MAX)

    def test_demo_query_scalar_keeps_f8_in_range(self) -> None:
        value = demo_query_scalar(10_000_000, "f8")
        self.assertTrue(math.isfinite(float(value)))
        self.assertLessEqual(abs(float(value)), F8_MAX)

    def test_demo_query_scalar_keeps_i16_in_range(self) -> None:
        value = demo_query_scalar(10_000_000, "i16")
        self.assertIsInstance(value, int)
        self.assertLessEqual(abs(value), I16_MAX)

    def test_demo_query_scalar_preserves_small_f32_values(self) -> None:
        value = demo_query_scalar(10, "f32")
        self.assertAlmostEqual(float(value), 10 * 0.631 + 0.123, places=6)

    def test_i16_l2_query_stays_in_the_bounded_scoring_range(self) -> None:
        self.assertEqual([500, 500, 500, 500], demo_query_values(10_000_000, 4, "i16", "L2"))


class DemoFloat8SupportTest(unittest.TestCase):
    def test_parse_args_accepts_f8(self) -> None:
        with patch.object(sys, "argv", ["demo.py", "--type", "f8"]):
            self.assertEqual("f8", parse_args().type)

    def test_native_f8_generator_uses_range_relative_multidimensional_bytes(self) -> None:
        min_id = 1_000_000
        vector = native_demo_vector(min_id + 72, min_id, 3, "f8", "DOT")
        l2_vector = native_demo_vector(min_id + 72, min_id, 3, "f8", "L2")
        self.assertEqual([0xCF, 0xCE, 0xCF], [f8_encode_bits(value) for value in vector])
        self.assertEqual(f8_ordinal_bytes(72, 3), [f8_encode_bits(value) for value in vector])
        self.assertEqual([f8_encode_bits(value) for value in vector], [f8_encode_bits(value) for value in l2_vector])
        self.assertTrue(all(math.isfinite(float(value)) for value in vector))

    def test_decoded_reference_allows_only_real_cutoff_ties(self) -> None:
        query = [0.0, 0.0, 0.0, 0.0]
        reference = decoded_knn_reference(
            count=4,
            from_id=100,
            dim=4,
            type_name="f8",
            dist_func="DOT",
            query=query,
            k=2,
        )
        self.assertEqual(0.0, reference.cutoff_score)
        self.assertEqual(frozenset(), reference.strictly_better_ids)
        self.assertTrue(all(reference.is_cutoff_tie(item_id) for item_id in range(100, 104)))
        # Every generated vector has score zero for this query, so either pair
        # from the scanned range is valid.
        assert_sqlite_rows_match_decoded_reference(
            [(103, 0.0), (102, 0.0)],
            reference=reference,
            dist_func="DOT",
            k=2,
        )
        with self.assertRaisesRegex(AssertionError, "outside the generated range"):
            assert_sqlite_rows_match_decoded_reference(
                [(999, 0.0), (998, 0.0)],
                reference=reference,
                dist_func="DOT",
                k=2,
            )

    def test_decoded_reference_scores_each_vector_once(self) -> None:
        with patch("demo.native_demo_vector", return_value=[1.0, 1.0, 1.0, 1.0]) as vector_oracle:
            reference = decoded_knn_reference(
                count=4,
                from_id=0,
                dim=4,
                type_name="f32",
                dist_func="DOT",
                query=[1.0, 1.0, 1.0, 1.0],
                k=2,
            )
            assert_sqlite_rows_match_decoded_reference(
                [(0, 4.0), (1, 4.0)],
                reference=reference,
                dist_func="DOT",
                k=2,
            )

        self.assertEqual(4, vector_oracle.call_count)

    def test_decoded_reference_rejects_near_but_not_equal_cutoff_score(self) -> None:
        # At this f8 L2 boundary, IDs 665 and 1071 differ by one E5M2-grid
        # contribution (0.00390625). That is close under math.isclose's
        # default relative tolerance at this magnitude, but it is not a tie.
        count = 1_000
        from_id = 100
        dim = 16
        k = 157
        query = demo_query_values(count, dim, "f8", "L2")
        reference = decoded_knn_reference(
            count=count,
            from_id=from_id,
            dim=dim,
            type_name="f8",
            dist_func="L2",
            query=query,
            k=k,
        )

        def score(item_id: int) -> float:
            return metric_score(
                query,
                native_demo_vector(item_id, from_id, dim, "f8", "L2"),
                "L2",
            )

        near_id = 1071
        near_score = score(near_id)
        self.assertNotEqual(reference.cutoff_score, near_score)
        self.assertTrue(math.isclose(reference.cutoff_score, near_score, abs_tol=1e-12))
        self.assertFalse(reference.is_cutoff_tie(near_id))

        rows = sorted(
            ((item_id, score(item_id)) for item_id in reference.strictly_better_ids),
            key=lambda row: row[1],
        )
        rows.append((near_id, near_score))
        self.assertEqual(k, len(rows))
        with self.assertRaisesRegex(AssertionError, "outside the decoded top-k cutoff tie set"):
            assert_sqlite_rows_match_decoded_reference(
                rows,
                reference=reference,
                dist_func="L2",
                k=k,
            )

    def test_small_f8_demo_path_uses_decoded_scores(self) -> None:
        library = find_library()
        run_demo(
            count=4,
            dim=4,
            k=2,
            range_size=16,
            type_name="f8",
            keep=False,
            dist_func="L2",
            sketch2_lib=library,
            extension_lib=library,
        )


class DemoBoundedNativePayloadTest(unittest.TestCase):
    def test_oracle_uses_bounded_payload_past_f16_and_i16_scalar_limits(self) -> None:
        self.assertEqual(
            [3.0, -4.0, -3.0, 1.0],
            bounded_demo_vector(65520, 4, "f16"),
        )
        self.assertEqual(
            [10, 3, 2, -1],
            bounded_demo_vector(32768, 4, "i16"),
        )
        self.assertEqual(
            bounded_demo_vector(65520, 4, "f16"),
            native_demo_vector(65520, 0, 4, "f16", "DOT"),
        )
        self.assertEqual(
            bounded_demo_vector(32768, 4, "i16"),
            native_demo_vector(32768, 0, 4, "i16", "L2"),
        )

    def test_f16_dot_and_l2_demos_cross_the_scalar_overflow_boundary(self) -> None:
        library = find_library()
        for dist_func in ("DOT", "L2"):
            with self.subTest(dist_func=dist_func):
                run_demo(
                    count=65_522,
                    dim=4,
                    k=10,
                    range_size=1_000,
                    type_name="f16",
                    keep=False,
                    dist_func=dist_func,
                    sketch2_lib=library,
                    extension_lib=library,
                )

    def test_i16_dot_and_l2_demos_cross_the_narrowing_boundary(self) -> None:
        library = find_library()
        for dist_func in ("DOT", "L2"):
            with self.subTest(dist_func=dist_func):
                run_demo(
                    count=32_770,
                    dim=4,
                    k=10,
                    range_size=1_000,
                    type_name="i16",
                    keep=False,
                    dist_func=dist_func,
                    sketch2_lib=library,
                    extension_lib=library,
                )


if __name__ == "__main__":
    unittest.main()
