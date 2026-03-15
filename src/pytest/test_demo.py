from __future__ import annotations

import math
import unittest

from demo import F16_MAX, I16_MAX, demo_query_scalar


class DemoQueryScalarTest(unittest.TestCase):
    def test_demo_query_scalar_keeps_f16_in_range(self) -> None:
        value = demo_query_scalar(10_000_000, "f16")
        self.assertTrue(math.isfinite(float(value)))
        self.assertLessEqual(abs(float(value)), F16_MAX)

    def test_demo_query_scalar_keeps_i16_in_range(self) -> None:
        value = demo_query_scalar(10_000_000, "i16")
        self.assertIsInstance(value, int)
        self.assertLessEqual(abs(value), I16_MAX)

    def test_demo_query_scalar_preserves_small_f32_values(self) -> None:
        value = demo_query_scalar(10, "f32")
        self.assertAlmostEqual(float(value), 10 * 0.631 + 0.123, places=6)


if __name__ == "__main__":
    unittest.main()
