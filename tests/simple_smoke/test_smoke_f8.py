#!/usr/bin/env python3
"""Focused behavioral coverage for the f8 simple-smoke path."""

from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from unittest.mock import patch

import common
import reader


class _FakeSketch2:
    instance: "_FakeSketch2 | None" = None

    def __init__(self, *_args, **_kwargs) -> None:
        self.queries: list[tuple[str, int]] = []
        _FakeSketch2.instance = self

    def __enter__(self) -> "_FakeSketch2":
        return self

    def __exit__(self, *_args) -> None:
        return None

    def open(self, _dataset: str) -> None:
        return None

    def knn(self, query: str, count: int) -> list[int]:
        self.queries.append((query, count))
        return [7]

    def close(self) -> None:
        return None


class SimpleSmokeF8BehaviorTest(unittest.TestCase):
    def test_f8_uses_the_common_two_decimal_numeric_text_pattern(self) -> None:
        self.assertEqual(
            "0.03, 0.15, 0.27, 0.39",
            common.vector_string(0, 4),
        )

    def test_f8_reader_requires_only_a_nonempty_knn_result(self) -> None:
        config = common.SmokeConfig(
            db_dir=Path("/tmp/sketch2_simple_smoke_test"),
            dataset="f8_smoke",
            dims=4,
            count=2,
            sleep_seconds=0.0,
            repeat=1,
            readers=1,
            knn_count=1,
            range_size=16,
            type_name="f8",
            dist_func="l2",
            log_level="ERROR",
            thread_pool_size=1,
        )
        args = argparse.Namespace(reader_id="test-reader")

        with (
            patch.object(reader, "parse_args", return_value=args),
            patch.object(reader, "load_config", return_value=config),
            patch.object(reader, "apply_runtime_env"),
            patch.object(reader, "find_lib_path", return_value=Path("libsketch2.so")),
            patch.object(reader, "load_sketch2_types", return_value=(_FakeSketch2, object)),
            patch.object(reader.time, "sleep"),
        ):
            reader.main()

        self.assertIsNotNone(_FakeSketch2.instance)
        self.assertEqual(
            [(common.query_vector(0, config.dims), config.knn_count)],
            _FakeSketch2.instance.queries,
        )


if __name__ == "__main__":
    unittest.main()
