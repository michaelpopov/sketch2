# E5M2 golden fixtures

`generate_float8_fixtures.py` is the sole generator for this directory.  It
uses a standalone integer/bit-level IEEE reference and does not import or call
the production C++ encoder.  Regenerate with:

```sh
python3 core/utils/testdata/generate_float8_fixtures.py
python3 core/utils/testdata/generate_float8_fixtures.py --check
```

The `check_float8_fixtures` CTest runs the `--check` form automatically so CI
detects missing or stale generated files.

Both files are UTF-8 CSV with comment lines beginning with `#` and a stable
v1 header row.  Hex bit-pattern fields always use a `0x` prefix; `-` means the
field is not applicable.

`float8_decode_v1.csv` columns:

1. `f8_bits_hex` — E5M2 byte;
2. `f16_bits_hex` — exact E5M2-as-f16 bit pattern (`byte << 8`);
3. `sign` — `positive` or `negative`;
4. `classification` — `zero`, `subnormal`, `normal`, `inf`, or `nan`;
5. `finite_value_f32_bits_hex` — exact binary32 value for finite rows;
6. `finite_value_hex` — the same finite value in hexadecimal floating syntax.

`float8_encode_v1.csv` columns:

1. `f32_bits_hex` — the normative binary32 input boundary;
2. `f16_bits_hex` — independently expected first-stage RNE result;
3. `f8_bits_hex` — independently expected second-stage E5M2 result;
4. `checked` — `accept` or `reject` under `try_encode_float8` policy;
5. `tags` — deterministic `|`-separated coverage labels.

The encode corpus contains every finite f16 stage-two tie, the f32 values
immediately surrounding each f32-to-f16 boundary that can alter that tie,
signed zero, f32/f16 underflow boundaries, subnormal/normal transitions,
binade carries, and the max-finite-to-Inf transition for both signs.
