#!/usr/bin/env python3
"""Shared vector generation and distance helpers for Sketch2 tests and demos."""

from __future__ import annotations

import math
import struct
from pathlib import Path


F16_MAX = 65504.0
I16_MIN = -32768
I16_MAX = 32767


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_library() -> Path:
    root = repo_root()
    candidates = [
        root / "bin" / "libsketch2.so",
        root / "bin-dbg" / "libsketch2.so",
        root / "bin-san" / "libsketch2.so",
        root / "build" / "lib" / "libsketch2.so",
        root / "build-dbg" / "lib" / "libsketch2.so",
        root / "build-san" / "lib" / "libsketch2.so",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("libsketch2.so not found under bin* or build*/lib directories")


def quantize_value(type_name: str, value: float) -> float | int:
    if type_name == "f32":
        return struct.unpack("f", struct.pack("f", value))[0]
    if type_name == "f16":
        bounded = max(-F16_MAX, min(F16_MAX, value))
        return struct.unpack("e", struct.pack("e", bounded))[0]
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
    if type_name == "i16":
        bounded = max(I16_MIN, min(I16_MAX, int(raw_value)))
        return quantize_value(type_name, float(bounded))
    raise ValueError(f"unsupported type: {type_name}")


def fmt_typed_vector(values: list[float | int], type_name: str) -> str:
    if type_name == "i16":
        return ", ".join(str(int(value)) for value in values)
    return ", ".join(f"{float(value):.6f}" for value in values)


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


def generic_demo_vector(item_id: int, dim: int, type_name: str) -> list[float | int]:
    # Use the same logic as the native sequential generator to ensure uniqueness.
    return native_sequential_vector(item_id, dim, type_name)


def generic_demo_query(dim: int, type_name: str) -> list[float | int]:
    values = [0.5] * dim
    return quantize_values(type_name, values)


def native_sequential_vector(item_id: int, dim: int, type_name: str) -> list[float | int]:
    # Matches generate_sequential_input_file in input_generator.cpp
    if type_name == "i16":
        value = int(item_id)
    else:
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
