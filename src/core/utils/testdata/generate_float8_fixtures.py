#!/usr/bin/env python3
"""Generate the shared E5M2 fixture corpus.

This generator intentionally has its own integer/bit-level IEEE reference.  It
does not import production code or call a C++ conversion helper: the checked-in
fixtures are independent expectations for both C++ and future Python tests.
"""

from __future__ import annotations

import argparse
import struct
import sys
from fractions import Fraction
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
DECODE_FIXTURE = THIS_DIR / "float8_decode_v1.csv"
ENCODE_FIXTURE = THIS_DIR / "float8_encode_v1.csv"


def round_shift_right_rne(value: int, shift: int) -> int:
    """Integer round-to-nearest-even of value / 2**shift."""
    if shift == 0:
        return value
    quotient = value >> shift
    remainder = value & ((1 << shift) - 1)
    halfway = 1 << (shift - 1)
    return quotient + int(remainder > halfway or (remainder == halfway and quotient & 1))


def f32_to_f16_bits_reference(bits: int) -> int:
    """Independent scalar IEEE f32 -> f16 RNE reference over raw bits."""
    sign = (bits >> 16) & 0x8000
    raw_exponent = (bits >> 23) & 0xFF
    fraction = bits & 0x7FFFFF

    if raw_exponent == 0xFF:
        if fraction == 0:
            return sign | 0x7C00
        # The checked fixture corpus only contains finite f32 values.  Keep the
        # reference complete and ensure an unchecked NaN stays a NaN, however.
        return sign | 0x7C00 | ((fraction >> 13) | 0x0200)

    if raw_exponent == 0:
        # All f32 subnormals underflow to signed f16 zero.
        return sign

    exponent = raw_exponent - 127
    if exponent >= 16:
        return sign | 0x7C00

    if exponent >= -14:
        output_exponent = exponent + 15
        output_fraction = round_shift_right_rne(fraction, 13)
        if output_fraction == 0x400:
            output_fraction = 0
            output_exponent += 1
            if output_exponent == 0x1F:
                return sign | 0x7C00
        return sign | (output_exponent << 10) | output_fraction

    significand = 0x800000 | fraction
    output_fraction = round_shift_right_rne(significand, -(exponent + 1))
    if output_fraction >= 0x400:
        return sign | 0x0400
    return sign | output_fraction


def f16_to_f8_bits_reference(f16_bits: int) -> int:
    """Normative second RNE stage, expressed independently in Python ints."""
    return (f16_bits + 0x7F + ((f16_bits >> 8) & 1)) >> 8 & 0xFF


def f16_to_f32_bits_reference(f16_bits: int) -> int:
    """Exact f16 -> f32 conversion over IEEE bit fields."""
    sign = (f16_bits & 0x8000) << 16
    raw_exponent = (f16_bits >> 10) & 0x1F
    fraction = f16_bits & 0x03FF

    if raw_exponent == 0:
        if fraction == 0:
            return sign
        exponent = -14
        while (fraction & 0x0400) == 0:
            fraction <<= 1
            exponent -= 1
        return sign | ((exponent + 127) << 23) | ((fraction & 0x03FF) << 13)

    if raw_exponent == 0x1F:
        return sign | 0x7F800000 | (fraction << 13)

    return sign | ((raw_exponent - 15 + 127) << 23) | (fraction << 13)


def f16_finite_fraction(f16_bits: int) -> Fraction:
    """Return an exact rational for a finite f16 bit pattern."""
    sign = -1 if f16_bits & 0x8000 else 1
    raw_exponent = (f16_bits >> 10) & 0x1F
    fraction = f16_bits & 0x03FF
    assert raw_exponent != 0x1F

    if raw_exponent == 0:
        magnitude = Fraction(fraction, 1 << 24)
    else:
        significand = 0x0400 | fraction
        power = raw_exponent - 25
        magnitude = Fraction(significand << power) if power >= 0 else Fraction(significand, 1 << -power)
    return sign * magnitude


def exact_fraction_to_f32_bits(value: Fraction) -> int:
    """Encode an exactly f32-representable dyadic Fraction without float math."""
    if value == 0:
        return 0

    sign = 0x80000000 if value < 0 else 0
    numerator = abs(value.numerator)
    denominator = value.denominator
    assert denominator & (denominator - 1) == 0

    binary_exponent = -(denominator.bit_length() - 1)
    significand_top = numerator.bit_length() - 1
    unbiased_exponent = binary_exponent + significand_top
    assert -126 <= unbiased_exponent <= 127

    shift = 23 - significand_top
    assert shift >= 0
    significand = numerator << shift
    assert 0x800000 <= significand < 0x1000000
    return sign | ((unbiased_exponent + 127) << 23) | (significand & 0x7FFFFF)


def f16_midpoint_to_f32_bits(first: int, second: int) -> int:
    return exact_fraction_to_f32_bits((f16_finite_fraction(first) + f16_finite_fraction(second)) / 2)


def next_f32_up(bits: int) -> int:
    """Raw-bit nextafter(bits, +infinity) for finite non-NaN f32 values."""
    if (bits & 0x7FFFFFFF) == 0:
        return 0x00000001
    return bits - 1 if bits & 0x80000000 else bits + 1


def next_f32_down(bits: int) -> int:
    """Raw-bit nextafter(bits, -infinity) for finite non-NaN f32 values."""
    if (bits & 0x7FFFFFFF) == 0:
        return 0x80000001
    return bits + 1 if bits & 0x80000000 else bits - 1


def f16_is_finite(bits: int) -> bool:
    return ((bits >> 10) & 0x1F) != 0x1F


def f32_is_finite(bits: int) -> bool:
    return ((bits >> 23) & 0xFF) != 0xFF


def add_case(cases: dict[int, set[str]], bits: int, *tags: str) -> None:
    assert 0 <= bits <= 0xFFFFFFFF
    assert f32_is_finite(bits)
    cases.setdefault(bits, set()).update(tags)


def add_f32_neighborhood(cases: dict[int, set[str]], bits: int, *tags: str) -> None:
    add_case(cases, next_f32_down(bits), *tags)
    add_case(cases, bits, *tags)
    add_case(cases, next_f32_up(bits), *tags)


def make_encode_cases() -> dict[int, set[str]]:
    cases: dict[int, set[str]] = {}

    # Every finite f16 pattern whose low byte is exactly the stage-two tie.
    # This includes positive/negative subnormals, normals, binade carries, and
    # the finite-to-infinity boundary at 0x7b80 / 0xfb80.
    tie_high_bytes = list(range(0x00, 0x7C)) + list(range(0x80, 0xFC))
    for high_byte in tie_high_bytes:
        f16_bits = (high_byte << 8) | 0x80
        assert f16_is_finite(f16_bits)
        tag_set = {"stage2_tie"}
        if (high_byte & 0x03) == 0x03:
            tag_set.add("binade_carry")
        if (high_byte & 0x7F) == 0x7B:
            tag_set.add("max_finite_to_inf")
        add_case(cases, f16_to_f32_bits_reference(f16_bits), *sorted(tag_set))

        output = f16_to_f8_bits_reference(f16_bits)
        changing_neighbor = None
        for candidate in (f16_bits - 1, f16_bits + 1):
            if f16_is_finite(candidate) and f16_to_f8_bits_reference(candidate) != output:
                changing_neighbor = candidate
                break
        assert changing_neighbor is not None

        # The f32 values immediately around the relevant f16 midpoint expose
        # the rare two-stage double-rounding outcome that a direct f32->f8
        # conversion would not reproduce.
        midpoint = f16_midpoint_to_f32_bits(f16_bits, changing_neighbor)
        midpoint_tags = {"f32_f16_double_round_boundary", "stage2_tie"}
        if (high_byte & 0x7F) == 0x7B:
            midpoint_tags.add("max_finite_to_inf")
        add_f32_neighborhood(cases, midpoint, *sorted(midpoint_tags))

    # Explicit named values make the signed-zero, underflow, f16 normal, and
    # f8 subnormal/normal transitions discoverable without reverse-engineering
    # the broad tie corpus above.
    special_positive_f16 = (
        0x0000, 0x0001, 0x007F, 0x0080, 0x0081, 0x00FF, 0x0100,
        0x02FF, 0x0300, 0x037F, 0x0380, 0x0381, 0x03FF, 0x0400,
        0x0401, 0x7AFF, 0x7B00, 0x7B7F, 0x7B80, 0x7B81,
    )
    for f16_bits in special_positive_f16:
        tags = {"named_boundary"}
        if f16_bits in (0x0000,):
            tags.add("signed_zero")
        if 0x0001 <= f16_bits <= 0x0401:
            tags.add("subnormal_normal_transition")
        if f16_bits >= 0x7AFF:
            tags.add("max_finite_to_inf")
        add_case(cases, f16_to_f32_bits_reference(f16_bits), *sorted(tags))
        if f16_bits != 0:
            add_case(cases, f16_to_f32_bits_reference(f16_bits | 0x8000), *sorted(tags))

    # Signed zero, both f32 subnormal signs, and the first-stage f32/f16
    # underflow/normal thresholds are distinct from f8's stage-two ties.
    add_case(cases, 0x00000000, "signed_zero")
    add_case(cases, 0x80000000, "signed_zero")
    add_case(cases, 0x00000001, "f32_subnormal")
    add_case(cases, 0x80000001, "f32_subnormal")
    for exponent in (-25, -24, -16, -14, 15):
        positive = (exponent + 127) << 23
        for sign in (0, 0x80000000):
            add_f32_neighborhood(cases, sign | positive, "first_stage_boundary")

    return cases


def decode_classification(f16_bits: int) -> str:
    raw_exponent = (f16_bits >> 10) & 0x1F
    fraction = f16_bits & 0x03FF
    if raw_exponent == 0:
        return "zero" if fraction == 0 else "subnormal"
    if raw_exponent == 0x1F:
        return "inf" if fraction == 0 else "nan"
    return "normal"


def render_decode_fixture() -> str:
    lines = [
        "# Generated by generate_float8_fixtures.py; do not edit by hand.",
        "# schema=v1; f8_bits_hex,f16_bits_hex,sign,classification,finite_value_f32_bits_hex,finite_value_hex",
        "f8_bits_hex,f16_bits_hex,sign,classification,finite_value_f32_bits_hex,finite_value_hex",
    ]
    for f8_bits in range(256):
        f16_bits = f8_bits << 8
        classification = decode_classification(f16_bits)
        sign = "negative" if f8_bits & 0x80 else "positive"
        if classification in ("zero", "subnormal", "normal"):
            f32_bits = f16_to_f32_bits_reference(f16_bits)
            value = struct.unpack("<f", struct.pack("<I", f32_bits))[0]
            finite_bits = f"0x{f32_bits:08x}"
            finite_value = value.hex()
        else:
            finite_bits = "-"
            finite_value = "-"
        lines.append(
            f"0x{f8_bits:02x},0x{f16_bits:04x},{sign},{classification},{finite_bits},{finite_value}")
    return "\n".join(lines) + "\n"


def render_encode_fixture() -> str:
    lines = [
        "# Generated by generate_float8_fixtures.py; do not edit by hand.",
        "# schema=v1; f32_bits_hex,f16_bits_hex,f8_bits_hex,checked,tags",
        "f32_bits_hex,f16_bits_hex,f8_bits_hex,checked,tags",
    ]
    for f32_bits, tags in sorted(make_encode_cases().items()):
        f16_bits = f32_to_f16_bits_reference(f32_bits)
        f8_bits = f16_to_f8_bits_reference(f16_bits)
        checked = "reject" if (f8_bits & 0x7F) >= 0x7C else "accept"
        tag_set = set(tags)
        if checked == "reject":
            tag_set.add("rounds_to_inf")
        lines.append(
            f"0x{f32_bits:08x},0x{f16_bits:04x},0x{f8_bits:02x},{checked},{'|'.join(sorted(tag_set))}")
    return "\n".join(lines) + "\n"


def update_or_check(path: Path, content: str, check: bool) -> bool:
    if check:
        try:
            existing = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            print(f"missing fixture: {path}", file=sys.stderr)
            return False
        if existing != content:
            print(f"fixture differs: {path}", file=sys.stderr)
            return False
        return True

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail instead of rewriting differing fixtures")
    args = parser.parse_args()

    decode_ok = update_or_check(DECODE_FIXTURE, render_decode_fixture(), args.check)
    encode_ok = update_or_check(ENCODE_FIXTURE, render_encode_fixture(), args.check)
    return 0 if decode_ok and encode_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
