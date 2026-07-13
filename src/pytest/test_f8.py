"""Cross-language E5M2 fixture, generator, and ctypes parity coverage."""

from __future__ import annotations

import csv
import math
import struct
import tempfile
import unittest
from pathlib import Path

from sketch2_test_vectors import (
    F8_CODEBOOK_BITS,
    F8_MAX,
    dot_distance,
    f8_capacity,
    f8_codebook_values,
    f8_decode_bits,
    f8_encode_bits,
    f8_encode_parts,
    f8_ordinal_bytes,
    f8_ordinal_fits,
    f8_ordinal_vector,
    f8_range_fits,
    find_library,
    fmt_typed_vector,
    native_sequential_vector,
    quantize_value,
    repo_root,
    try_encode_f8,
)
from sketch2_wrapper import Sketch2


def fixture_rows(filename: str) -> list[dict[str, str]]:
    fixture_path = repo_root() / "src" / "core" / "utils" / "testdata" / filename
    with fixture_path.open("r", encoding="utf-8", newline="") as source:
        return list(csv.DictReader(line for line in source if not line.startswith("#")))


def f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def read_f8_binary_records(path, dim: int) -> tuple[str, list[tuple[int, list[int]]]]:
    with path.open("rb") as source:
        header = source.readline().decode("ascii").rstrip("\n")
        records: list[tuple[int, list[int]]] = []
        while record_id := source.read(8):
            if len(record_id) != 8:
                raise AssertionError("truncated f8 generator record ID")
            payload = list(source.read(dim))
            if len(payload) != dim:
                raise AssertionError("truncated f8 generator record payload")
            records.append((struct.unpack("<Q", record_id)[0], payload))
    return header, records


class Float8FixtureParityTest(unittest.TestCase):
    def test_decode_fixture_matches_every_f8_byte(self) -> None:
        rows = fixture_rows("float8_decode_v1.csv")
        self.assertEqual(256, len(rows))

        for row in rows:
            bits = int(row["f8_bits_hex"], 16)
            f16_bits = int(row["f16_bits_hex"], 16)
            decoded = f8_decode_bits(bits)
            expected_sign = -1.0 if row["sign"] == "negative" else 1.0

            self.assertEqual(bits << 8, f16_bits)
            self.assertEqual(expected_sign, math.copysign(1.0, decoded))
            classification = row["classification"]
            if classification == "nan":
                self.assertTrue(math.isnan(decoded), f"byte 0x{bits:02x}")
            elif classification == "inf":
                self.assertTrue(math.isinf(decoded), f"byte 0x{bits:02x}")
            else:
                self.assertTrue(math.isfinite(decoded), f"byte 0x{bits:02x}")
                self.assertEqual(
                    int(row["finite_value_f32_bits_hex"], 16),
                    f32_bits(decoded),
                    f"byte 0x{bits:02x}",
                )
                self.assertEqual(bits, f8_encode_bits(decoded), f"byte 0x{bits:02x}")

    def test_encode_fixture_matches_both_rne_stages(self) -> None:
        rows = fixture_rows("float8_encode_v1.csv")
        self.assertGreater(len(rows), 1000)

        for row in rows:
            input_bits = int(row["f32_bits_hex"], 16)
            input_value = f32_from_bits(input_bits)
            v32, f16_bits, f8_bits = f8_encode_parts(input_value)

            self.assertEqual(input_bits, f32_bits(v32), row["f32_bits_hex"])
            self.assertEqual(int(row["f16_bits_hex"], 16), f16_bits, row["f32_bits_hex"])
            self.assertEqual(int(row["f8_bits_hex"], 16), f8_bits, row["f32_bits_hex"])
            if row["checked"] == "accept":
                self.assertEqual(f8_bits, try_encode_f8(input_value), row["f32_bits_hex"])
            else:
                self.assertIsNone(try_encode_f8(input_value), row["f32_bits_hex"])

    def test_f32_first_boundary_differs_from_direct_binary64_to_f16(self) -> None:
        # One binary64 ULP above the exact f16 midpoint rounds back to that
        # midpoint at the f32 boundary.  The f32-first path therefore uses
        # f16 0x0080 (then f8 0x00), while direct binary64-to-f16 incorrectly
        # chooses 0x0081 (then f8 0x01).
        midpoint = 128.5 * 2.0 ** -24
        value = math.nextafter(midpoint, math.inf)

        _, f16_bits, f8_bits = f8_encode_parts(value)
        direct_f16_bits = struct.unpack("<H", struct.pack("<e", value))[0]
        direct_f8_bits = (direct_f16_bits + 0x7F + ((direct_f16_bits >> 8) & 1)) >> 8

        self.assertEqual(0x0080, f16_bits)
        self.assertEqual(0x00, f8_bits)
        self.assertEqual(0x0081, direct_f16_bits)
        self.assertEqual(0x01, direct_f8_bits & 0xFF)
        self.assertNotEqual(f8_bits, direct_f8_bits & 0xFF)

    def test_checked_and_bounded_paths_preserve_contract(self) -> None:
        self.assertEqual(0x80, f8_encode_bits(-0.0))
        self.assertEqual(-1.0, math.copysign(1.0, f8_decode_bits(0x80)))
        self.assertIsNone(try_encode_f8(math.inf))
        self.assertIsNone(try_encode_f8(-math.inf))
        self.assertIsNone(try_encode_f8(math.nan))
        self.assertIsNone(try_encode_f8(61440.0))
        self.assertEqual(F8_MAX, quantize_value("f8", 1e300))
        with self.assertRaises(ValueError):
            quantize_value("f8", math.nan)


class Float8GeneratorParityTest(unittest.TestCase):
    def test_full_codebook_is_the_canonical_sorted_byte_sequence(self) -> None:
        expected_bits = bytes.fromhex(
            "cf ce cd cc cb ca c9 c8 c7 c6 c5 c4 c3 c2 c1 c0 "
            "bf be bd bc bb ba b9 b8 b7 b6 b5 b4 b3 b2 b1 b0 "
            "af ae ad ac 2c 2d 2e 2f 30 31 32 33 34 35 36 37 "
            "38 39 3a 3b 3c 3d 3e 3f 40 41 42 43 44 45 46 47 "
            "48 49 4a 4b 4c 4d 4e 4f"
        )
        self.assertEqual(expected_bits, bytes(F8_CODEBOOK_BITS))

        values = f8_codebook_values()
        self.assertEqual(sorted(values), list(values))
        self.assertEqual(-28.0, values[0])
        self.assertEqual(-0.0625, values[35])
        self.assertEqual(0.0625, values[36])
        self.assertEqual(28.0, values[-1])
        self.assertEqual(list(F8_CODEBOOK_BITS), [f8_encode_bits(value) for value in values])

    def test_base72_ordinal_bytes_and_capacity_match_cpp(self) -> None:
        self.assertEqual([0xCF], f8_ordinal_bytes(0, 1))
        self.assertEqual([0x4F], f8_ordinal_bytes(71, 1))
        self.assertEqual([0xCF, 0xCE], f8_ordinal_bytes(72, 2))
        self.assertEqual([0x4F, 0x4F, 0xCF], f8_ordinal_bytes(72 * 72 - 1, 3))
        self.assertEqual([0x4F, 0x4F, 0x4F], f8_ordinal_bytes(72 ** 3 - 1, 3))
        with self.assertRaises(ValueError):
            f8_ordinal_bytes(72, 1)

        self.assertEqual(1, f8_capacity(0))
        self.assertEqual(72, f8_capacity(1))
        self.assertEqual(72 ** 2, f8_capacity(2))
        self.assertEqual(72 ** 10, f8_capacity(10))
        self.assertIsNone(f8_capacity(11))
        self.assertTrue(f8_range_fits(2, 72 ** 2))
        self.assertFalse(f8_range_fits(2, 72 ** 2 + 1))
        self.assertTrue(f8_range_fits(11, (1 << 64) - 1))
        self.assertFalse(f8_ordinal_fits(12, 1 << 64))
        self.assertFalse(f8_range_fits(12, 1 << 64))

    def test_native_vector_uses_explicit_or_range_relative_ordinal(self) -> None:
        min_id = 9_000_000
        item_id = min_id + 72
        expected = [0xCF, 0xCE, 0xCF]

        from_range = native_sequential_vector(item_id, 3, "f8", min_id=min_id)
        explicit = native_sequential_vector(item_id, 3, "f8", ordinal=72)
        self.assertEqual(expected, [f8_encode_bits(value) for value in from_range])
        self.assertEqual(expected, [f8_encode_bits(value) for value in explicit])
        self.assertEqual(expected, [f8_encode_bits(value) for value in f8_ordinal_vector(72, 3)])
        with self.assertRaises(ValueError):
            native_sequential_vector(item_id, 3, "f8")


class Float8CtypesEndToEndTest(unittest.TestCase):
    def test_native_generator_bytes_match_python_codebook_and_ordinals(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sketch2_f8_generator_") as root:
            root_path = Path(root)
            input_path = root_path / "generated.bin"
            library = find_library()
            start_id = 1_000_000
            with Sketch2(root_path, lib_path=library) as ps:
                ps.create("f8gen", type_name="f8", dim=4, range_size=128, dist_func="dot")
                ps.generate_test_data(input_path, count=73, start_id=start_id, pattern="auto", binary=True)

                header, records = read_f8_binary_records(input_path, 4)
                self.assertEqual("f8,4,bin", header)
                self.assertEqual(73, len(records))
                for ordinal, (item_id, payload) in enumerate(records):
                    self.assertEqual(start_id + ordinal, item_id)
                    self.assertEqual(f8_ordinal_bytes(ordinal, 4), payload)

                # The least-significant base-72 digit walks every canonical
                # codebook byte before ordinal 72 carries into dimension 1.
                self.assertEqual(list(F8_CODEBOOK_BITS), [payload[0] for _, payload in records[:72]])
                self.assertEqual(f8_ordinal_bytes(72, 4), records[72][1])
                ps.close()
                ps.drop("f8gen")

    def test_f8_staged_write_query_and_two_decimal_readback(self) -> None:
        with tempfile.TemporaryDirectory(prefix="sketch2_f8_ctypes_") as root:
            library = find_library()
            item_ids = (100, 101, 102)
            ordinals = (0, 35, 71)
            vectors = {
                item_id: f8_ordinal_vector(ordinal, 4)
                for item_id, ordinal in zip(item_ids, ordinals)
            }
            query = [1.0, 1.0, 1.0, 1.0]
            expected_rows = sorted(
                ((item_id, dot_distance(query, vector)) for item_id, vector in vectors.items()),
                key=lambda row: (-row[1], row[0]),
            )
            self.assertEqual([(102, -56.0), (101, -84.0625), (100, -112.0)], expected_rows)

            with Sketch2(root, lib_path=library) as ps:
                ps.create("f8ds", type_name="f8", dim=4, range_size=16, dist_func="dot")
                ps.start_writing()
                for item_id, vector in vectors.items():
                    ps.write_vector(item_id, fmt_typed_vector(vector, "f8"))
                ps.complete_writing()

                returned_rows = ps.knn_items(fmt_typed_vector(query, "f8"), len(item_ids))
                self.assertEqual(
                    [item_id for item_id, _ in expected_rows],
                    [item_id for item_id, _ in returned_rows],
                )
                for (_, expected_score), (_, returned_score) in zip(expected_rows, returned_rows):
                    self.assertAlmostEqual(expected_score, returned_score, places=12)
                self.assertEqual(
                    "[ -0.06, -28.00, -28.00, -28.00 ]",
                    ps.get(101),
                )
                # sk_get delegates to the shared vector printer, so this
                # readback also verifies its documented two-decimal f8
                # presentation.
                ps.close()
                ps.drop("f8ds")


if __name__ == "__main__":
    unittest.main()
