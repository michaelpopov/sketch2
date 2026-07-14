#!/usr/bin/env python3
"""Shared vector generation and metric-scoring helpers for Sketch2 tests and demos."""

from __future__ import annotations

import math
import os
import struct
from pathlib import Path


F16_MAX = 65504.0
F8_MAX = 57344.0
I16_MIN = -32768
I16_MAX = 32767

F8_CODEBOOK_SIZE = 72
_UINT64_MAX = (1 << 64) - 1

# This is the same byte-built, numerically sorted E5M2 codebook as
# float8_codebook::kBits in core/utils/float8.h.  Building it from sign,
# exponent, and mantissa fields keeps the Python generator on the f8 grid
# without relying on decimal rounding.
F8_CODEBOOK_BITS = tuple(
    [0x80 | (exponent << 2) | mantissa
     for exponent in range(19, 10, -1)
     for mantissa in range(3, -1, -1)]
    + [(exponent << 2) | mantissa
       for exponent in range(11, 20)
       for mantissa in range(4)]
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_library() -> Path:
    configured_dir = os.environ.get("SKETCH2_LIB")
    if configured_dir:
        configured_path = Path(configured_dir).resolve() / "libsketch2.so"
        if configured_path.exists():
            return configured_path

    root = repo_root()
    candidates = [
        root / "bin-hwy" / "libsketch2.so",
        root / "bin-dbg-hwy" / "libsketch2.so",
        root / "build" / "lib" / "libsketch2.so",
        root / "build-dbg" / "lib" / "libsketch2.so",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("libsketch2.so not found under bin* or build*/lib directories")


def f8_decode_bits(bits: int) -> float:
    """Decode an E5M2 byte through its exact binary16 high-byte alias."""
    if not 0 <= bits <= 0xFF:
        raise ValueError(f"f8 byte out of range: {bits}")
    return struct.unpack("<e", struct.pack("<H", bits << 8))[0]


def f16_bits_to_f8_rne(f16_bits: int) -> int:
    """Apply the normative unchecked f16-bit-pattern to E5M2 RNE stage."""
    if not 0 <= f16_bits <= 0xFFFF:
        raise ValueError(f"f16 bits out of range: {f16_bits}")
    return ((f16_bits + 0x7F + ((f16_bits >> 8) & 1)) >> 8) & 0xFF


def f8_encode_parts(value: float) -> tuple[float, int, int]:
    """Return the normative f32 value, f16 bits, and f8 byte for a value.

    This is intentionally the unchecked scalar conversion: callers that model
    numeric ingest should use try_encode_f8(), while bounded demo generation
    should use quantize_value("f8", ...).  The first pack imposes the f32 input
    contract before Python's binary64 value is packed to f16.
    """
    try:
        v32 = struct.unpack("<f", struct.pack("<f", value))[0]
        h = struct.unpack("<H", struct.pack("<e", v32))[0]
    except (OverflowError, struct.error) as exc:
        raise OverflowError(f"value cannot be represented at the f32/f16 boundary: {value}") from exc
    return v32, h, f16_bits_to_f8_rne(h)


def f8_encode_bits(value: float) -> int:
    """Encode a finite f32-contract value to its E5M2 byte, unchecked."""
    return f8_encode_parts(value)[2]


def try_encode_f8(value: float) -> int | None:
    """Mirror checked C++ numeric ingest: reject non-finite and f8-overflow."""
    if not math.isfinite(value):
        return None
    try:
        bits = f8_encode_bits(value)
    except OverflowError:
        return None
    return bits if (bits & 0x7F) < 0x7C else None


def f8_codebook_values() -> tuple[float, ...]:
    return tuple(f8_decode_bits(bits) for bits in F8_CODEBOOK_BITS)


def f8_ordinal_fits(dim: int, ordinal: int) -> bool:
    """Return whether ordinal fits in dim little-endian base-72 digits."""
    if dim < 0 or not 0 <= ordinal <= _UINT64_MAX:
        return False
    remaining = ordinal
    while dim and remaining:
        remaining //= F8_CODEBOOK_SIZE
        dim -= 1
    return remaining == 0


def f8_capacity(dim: int) -> int | None:
    """Match the C++ uint64_t capacity helper (None means overflow)."""
    if dim < 0:
        raise ValueError("f8 dimension must be non-negative")
    capacity = 1
    for _ in range(dim):
        if capacity > _UINT64_MAX // F8_CODEBOOK_SIZE:
            return None
        capacity *= F8_CODEBOOK_SIZE
    return capacity


def f8_range_fits(dim: int, item_count: int) -> bool:
    """Match float8_codebook::range_fits for range-relative item counts."""
    return (
        dim >= 0
        and 0 <= item_count <= _UINT64_MAX
        and (item_count == 0 or f8_ordinal_fits(dim, item_count - 1))
    )


def f8_ordinal_bytes(ordinal: int, dim: int) -> list[int]:
    """Map an ordinal to C++-matching little-endian base-72 codebook bytes."""
    if not f8_ordinal_fits(dim, ordinal):
        raise ValueError(f"f8 ordinal {ordinal} exceeds 72^{dim} capacity")
    remaining = ordinal
    bits: list[int] = []
    for _ in range(dim):
        bits.append(F8_CODEBOOK_BITS[remaining % F8_CODEBOOK_SIZE])
        remaining //= F8_CODEBOOK_SIZE
    return bits


def f8_ordinal_vector(ordinal: int, dim: int) -> list[float]:
    return [f8_decode_bits(bits) for bits in f8_ordinal_bytes(ordinal, dim)]


def quantize_value(type_name: str, value: float) -> float | int:
    if type_name == "f32":
        return struct.unpack("f", struct.pack("f", value))[0]
    if type_name == "f16":
        bounded = max(-F16_MAX, min(F16_MAX, value))
        return struct.unpack("e", struct.pack("e", bounded))[0]
    if type_name == "f8":
        if not math.isfinite(value):
            raise ValueError("f8 demo quantization requires a finite value")
        # Unlike checked ingest, the demo path deliberately bounds finite
        # values before the f32/f16/f8 pipeline, matching the f16 helper's
        # existing bounded role.  Comparisons preserve a signed zero.
        bounded = value
        if bounded > F8_MAX:
            bounded = F8_MAX
        elif bounded < -F8_MAX:
            bounded = -F8_MAX
        bits = try_encode_f8(bounded)
        if bits is None:
            raise ValueError("bounded f8 demo value must encode as finite")
        return f8_decode_bits(bits)
    if type_name == "i16":
        return int(max(I16_MIN, min(I16_MAX, value)))
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
        if not math.isfinite(float(quantized)):
            raise ValueError("demo f16 query value must remain finite")
        return quantized
    if type_name == "f8":
        bounded = max(-F8_MAX, min(F8_MAX, raw_value))
        quantized = quantize_value(type_name, bounded)
        if not math.isfinite(float(quantized)):
            raise ValueError("demo f8 query value must remain finite")
        return quantized
    if type_name == "i16":
        bounded = max(I16_MIN, min(I16_MAX, int(raw_value)))
        return quantize_value(type_name, float(bounded))
    raise ValueError(f"unsupported type: {type_name}")


def fmt_typed_vector(values: list[float | int], type_name: str) -> str:
    if type_name == "i16":
        return ", ".join(str(int(value)) for value in values)
    if type_name == "f8":
        # All canonical f8 values round-trip through this lossless formatting.
        return ", ".join(format(float(value), ".9g") for value in values)
    return ", ".join(f"{float(value):.6f}" for value in values)


def bounded_demo_vector(
    item_id: int,
    dim: int,
    type_name: str,
    *,
    ordinal: int | None = None,
    min_id: int | None = None,
) -> list[float | int]:
    """Match the native bounded multidimensional test payload."""
    if type_name == "f8":
        # Native f8 generation uses the range-relative ordinal mapping rather
        # than this numeric shape.
        return native_sequential_vector(
            item_id, dim, type_name, ordinal=ordinal, min_id=min_id)
    values = [0.0] * dim
    if dim == 0:
        return quantize_values(type_name, values)
    values[0] = float((item_id % 17) + 1)
    if dim == 1:
        return quantize_values(type_name, values)
    values[1] = float((((item_id % 11) * 3) % 11) - 5)
    if dim == 2:
        return quantize_values(type_name, values)
    values[2] = float((((item_id % 7) * 5) % 7) - 3)
    for index in range(3, dim):
        values[index] = float((((item_id % 5) + (index % 5)) % 5) - 2)
    return quantize_values(type_name, values)


def cosine_demo_vector(
    item_id: int,
    dim: int,
    type_name: str,
    *,
    ordinal: int | None = None,
    min_id: int | None = None,
) -> list[float | int]:
    return bounded_demo_vector(
        item_id, dim, type_name, ordinal=ordinal, min_id=min_id)


def cosine_demo_query(dim: int, type_name: str) -> list[float | int]:
    values = [0.0] * dim
    values[0] = 1.0
    values[1] = -0.5
    values[2] = 0.25
    for index in range(3, dim):
        values[index] = 0.1 * (1 if index % 2 == 0 else -1)
    return quantize_values(type_name, values)


def generic_demo_vector(
    item_id: int,
    dim: int,
    type_name: str,
    *,
    ordinal: int | None = None,
    min_id: int | None = None,
) -> list[float | int]:
    # Use the same logic as the native sequential generator to ensure uniqueness.
    return native_sequential_vector(item_id, dim, type_name, ordinal=ordinal, min_id=min_id)


def generic_demo_query(dim: int, type_name: str) -> list[float | int]:
    values = [0.5] * dim
    return quantize_values(type_name, values)


def native_sequential_vector(
    item_id: int,
    dim: int,
    type_name: str,
    *,
    ordinal: int | None = None,
    min_id: int | None = None,
) -> list[float | int]:
    # Matches generate_sequential_input_file in input_generator.cpp.
    if type_name == "f8":
        if ordinal is None:
            if min_id is None:
                raise ValueError("f8 sequential vectors require an explicit ordinal or min_id")
            ordinal = item_id - min_id
        return f8_ordinal_vector(ordinal, dim)
    if type_name in ("f16", "i16"):
        return bounded_demo_vector(item_id, dim, type_name)
    value = float(item_id) + 0.1
    return quantize_values(type_name, [value] * dim)


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


def dot_distance(a: list[float | int], b: list[float | int]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def l2_distance_sq(a: list[float | int], b: list[float | int]) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))
