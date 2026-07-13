// Unit tests for the scalar E5M2 float8 contract.

#include "utils/float8.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cfenv>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

namespace sketch2 {
namespace {

template <typename T>
concept HasDoubleConversionOperator = requires {
    static_cast<double (T::*)() const>(&T::operator double);
};

template <typename T>
concept HasCompoundAdditionOperator = requires(T value) {
    value += value;
};

static_assert(sizeof(float8) == 1);
static_assert(alignof(float8) == 1);
static_assert(std::is_trivially_copyable_v<float8>);
static_assert(std::is_standard_layout_v<float8>);
static_assert(std::is_trivially_default_constructible_v<float8>);
static_assert(std::is_convertible_v<float8, float>);
static_assert(!std::is_convertible_v<float, float8>);
static_assert(!HasDoubleConversionOperator<float8>);
static_assert(!HasCompoundAdditionOperator<float8>);
static_assert(sizeof(float16) == 2);
static_assert(__FLT16_MANT_DIG__ == 11);
static_assert(__FLT16_MAX_EXP__ == 16);

struct DecodeRow {
    uint8_t f8_bits;
    uint16_t f16_bits;
    bool negative;
    std::string classification;
    uint32_t finite_value_f32_bits;
    bool has_finite_value;
};

struct EncodeRow {
    uint32_t f32_bits;
    uint16_t f16_bits;
    uint8_t f8_bits;
    bool checked_accepts;
    std::vector<std::string> tags;
};

constexpr char kDecodeFixtureHeader[] =
    "f8_bits_hex,f16_bits_hex,sign,classification,finite_value_f32_bits_hex,finite_value_hex";
constexpr char kEncodeFixtureHeader[] = "f32_bits_hex,f16_bits_hex,f8_bits_hex,checked,tags";
constexpr size_t kExpectedDecodeFixtureRows = 256;

std::filesystem::path fixture_path(const char* filename) {
    return std::filesystem::path(SKETCH2_UTILS_TESTDATA_DIR) / filename;
}

std::string hex_bits(uint32_t bits, int width) {
    std::ostringstream stream;
    stream << "0x" << std::hex << std::setfill('0') << std::setw(width) << bits;
    return stream.str();
}

[[noreturn]] void fixture_error(const std::string& source, size_t line_number, const std::string& detail) {
    throw std::runtime_error(source + ":" + std::to_string(line_number) + ": " + detail);
}

[[noreturn]] void fixture_error(const std::string& source, const std::string& detail) {
    throw std::runtime_error(source + ": " + detail);
}

std::vector<std::string> split_fields(const std::string& line, char delimiter) {
    std::vector<std::string> fields;
    size_t start = 0;
    while (true) {
        const size_t end = line.find(delimiter, start);
        fields.emplace_back(line.substr(start, end - start));
        if (end == std::string::npos) {
            return fields;
        }
        start = end + 1;
    }
}

uint32_t parse_hex_exact(
    const std::string& value,
    size_t digits,
    const std::string& source,
    size_t line_number,
    const char* field_name) {
    if (value.size() != digits + 2 || value[0] != '0' || value[1] != 'x') {
        fixture_error(source, line_number, std::string("invalid ") + field_name + " width: " + value);
    }

    uint32_t parsed = 0;
    for (size_t i = 2; i < value.size(); ++i) {
        const char c = value[i];
        uint32_t digit = 0;
        if (c >= '0' && c <= '9') {
            digit = static_cast<uint32_t>(c - '0');
        } else if (c >= 'a' && c <= 'f') {
            digit = static_cast<uint32_t>(c - 'a' + 10);
        } else if (c >= 'A' && c <= 'F') {
            digit = static_cast<uint32_t>(c - 'A' + 10);
        } else {
            fixture_error(source, line_number, std::string("invalid ") + field_name + " digit: " + value);
        }
        parsed = (parsed << 4) | digit;
    }
    return parsed;
}

std::string decode_classification(uint16_t f16_bits) {
    const uint16_t exponent = static_cast<uint16_t>((f16_bits >> 10) & 0x1fU);
    const uint16_t fraction = static_cast<uint16_t>(f16_bits & 0x03ffU);
    if (exponent == 0) {
        return fraction == 0 ? "zero" : "subnormal";
    }
    if (exponent == 0x1fU) {
        return fraction == 0 ? "inf" : "nan";
    }
    return "normal";
}

uint32_t parse_finite_hex_float_bits(
    const std::string& value,
    const std::string& source,
    size_t line_number) {
    errno = 0;
    char* end = nullptr;
    const float parsed = std::strtof(value.c_str(), &end);
    if (end != value.c_str() + value.size() || errno != 0 || !std::isfinite(parsed)) {
        fixture_error(source, line_number, "invalid finite_value_hex: " + value);
    }
    return std::bit_cast<uint32_t>(parsed);
}

const std::set<std::string>& known_encode_tags() {
    static const std::set<std::string> tags = {
        "binade_carry",
        "f32_f16_double_round_boundary",
        "f32_subnormal",
        "first_stage_boundary",
        "max_finite_to_inf",
        "named_boundary",
        "rounds_to_inf",
        "signed_zero",
        "stage2_tie",
        "subnormal_normal_transition",
    };
    return tags;
}

std::vector<std::string> parse_encode_tags(
    const std::string& value,
    const std::string& source,
    size_t line_number) {
    if (value.empty()) {
        fixture_error(source, line_number, "empty tags field");
    }

    std::set<std::string> unique_tags;
    std::vector<std::string> tags = split_fields(value, '|');
    for (const std::string& tag : tags) {
        if (!known_encode_tags().contains(tag)) {
            fixture_error(source, line_number, "unknown tag: " + tag);
        }
        if (!unique_tags.insert(tag).second) {
            fixture_error(source, line_number, "duplicate tag: " + tag);
        }
    }
    return tags;
}

bool has_tag(const EncodeRow& row, std::string_view tag) {
    return std::find(row.tags.begin(), row.tags.end(), tag) != row.tags.end();
}

uint8_t f16_bits_to_f8_reference(uint16_t f16_bits) {
    return static_cast<uint8_t>(
        (static_cast<uint32_t>(f16_bits) + 0x7fU + ((f16_bits >> 8) & 1U)) >> 8);
}

void require_encode_sentinel(
    const std::vector<EncodeRow>& rows,
    const std::string& source,
    uint32_t f32_bits,
    uint16_t f16_bits,
    uint8_t f8_bits,
    bool checked_accepts,
    std::initializer_list<std::string_view> required_tags) {
    const auto it = std::find_if(rows.begin(), rows.end(), [f32_bits](const EncodeRow& row) {
        return row.f32_bits == f32_bits;
    });
    if (it == rows.end()) {
        fixture_error(source, "missing required encode sentinel " + hex_bits(f32_bits, 8));
    }
    if (it->f16_bits != f16_bits || it->f8_bits != f8_bits ||
        it->checked_accepts != checked_accepts) {
        fixture_error(source, "invalid required encode sentinel " + hex_bits(f32_bits, 8));
    }
    for (const std::string_view tag : required_tags) {
        if (!has_tag(*it, tag)) {
            fixture_error(source, "encode sentinel " + hex_bits(f32_bits, 8) +
                " is missing tag " + std::string(tag));
        }
    }
}

// The fixture generator owns exhaustive corpus construction and is checked by
// the check_float8_fixtures CTest. Keep only compact semantic sentinels here so
// accidental corpus shrinkage cannot remove the format's load-bearing cases.
void validate_encode_sentinels(const std::vector<EncodeRow>& rows, const std::string& source) {
    require_encode_sentinel(rows, source, 0x00000000U, 0x0000U, 0x00U, true, {"signed_zero"});
    require_encode_sentinel(rows, source, 0x80000000U, 0x8000U, 0x80U, true, {"signed_zero"});
    require_encode_sentinel(rows, source, 0x00000001U, 0x0000U, 0x00U, true, {"f32_subnormal"});
    require_encode_sentinel(rows, source, 0x80000001U, 0x8000U, 0x80U, true, {"f32_subnormal"});

    require_encode_sentinel(
        rows, source, 0x33800000U, 0x0001U, 0x00U, true, {"subnormal_normal_transition"});
    require_encode_sentinel(
        rows, source, 0xb3800000U, 0x8001U, 0x80U, true, {"subnormal_normal_transition"});
    require_encode_sentinel(
        rows, source, 0x38800000U, 0x0400U, 0x04U, true, {"subnormal_normal_transition"});
    require_encode_sentinel(
        rows, source, 0xb8800000U, 0x8400U, 0x84U, true, {"subnormal_normal_transition"});

    require_encode_sentinel(rows, source, 0x3f900000U, 0x3c80U, 0x3cU, true, {"stage2_tie"});
    require_encode_sentinel(rows, source, 0x3fb00000U, 0x3d80U, 0x3eU, true, {"stage2_tie"});
    require_encode_sentinel(
        rows, source, 0x3ff00000U, 0x3f80U, 0x40U, true, {"stage2_tie", "binade_carry"});

    require_encode_sentinel(
        rows, source, 0x3f900fffU, 0x3c80U, 0x3cU, true, {"f32_f16_double_round_boundary"});
    require_encode_sentinel(
        rows, source, 0x3f901001U, 0x3c81U, 0x3dU, true, {"f32_f16_double_round_boundary"});
    require_encode_sentinel(
        rows, source, 0xbf900fffU, 0xbc80U, 0xbcU, true, {"f32_f16_double_round_boundary"});
    require_encode_sentinel(
        rows, source, 0xbf901001U, 0xbc81U, 0xbdU, true, {"f32_f16_double_round_boundary"});

    require_encode_sentinel(rows, source, 0x47600000U, 0x7b00U, 0x7bU, true, {"max_finite_to_inf"});
    require_encode_sentinel(
        rows, source, 0x47700000U, 0x7b80U, 0x7cU, false, {"max_finite_to_inf", "rounds_to_inf"});
    require_encode_sentinel(
        rows, source, 0xc7700000U, 0xfb80U, 0xfcU, false, {"max_finite_to_inf", "rounds_to_inf"});
}

std::vector<DecodeRow> load_decode_rows(std::istream& input, const std::string& source) {
    std::vector<DecodeRow> parsed;
    std::array<bool, kExpectedDecodeFixtureRows> seen {};
    bool saw_header = false;
    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (!saw_header) {
            if (line != kDecodeFixtureHeader) {
                fixture_error(source, line_number, "invalid decode fixture header");
            }
            saw_header = true;
            continue;
        }

        const std::vector<std::string> fields = split_fields(line, ',');
        if (fields.size() != 6) {
            fixture_error(source, line_number, "invalid float8 decode fixture row");
        }
        const uint8_t f8_bits = static_cast<uint8_t>(
            parse_hex_exact(fields[0], 2, source, line_number, "f8_bits_hex"));
        const uint16_t f16_bits = static_cast<uint16_t>(
            parse_hex_exact(fields[1], 4, source, line_number, "f16_bits_hex"));
        if (seen[f8_bits]) {
            fixture_error(source, line_number, "duplicate f8 byte " + hex_bits(f8_bits, 2));
        }
        seen[f8_bits] = true;
        if (f16_bits != static_cast<uint16_t>(f8_bits) << 8) {
            fixture_error(source, line_number, "f16_bits_hex does not equal f8_bits_hex << 8");
        }

        bool negative = false;
        if (fields[2] == "positive") {
            negative = false;
        } else if (fields[2] == "negative") {
            negative = true;
        } else {
            fixture_error(source, line_number, "invalid sign: " + fields[2]);
        }
        if (negative != ((f8_bits & 0x80U) != 0)) {
            fixture_error(source, line_number, "sign does not match f8_bits_hex");
        }

        const std::string expected_classification = decode_classification(f16_bits);
        if (fields[3] != expected_classification) {
            fixture_error(source, line_number, "invalid classification: " + fields[3]);
        }
        const bool finite = expected_classification == "zero" ||
            expected_classification == "subnormal" || expected_classification == "normal";
        uint32_t finite_value_f32_bits = 0;
        if (finite) {
            finite_value_f32_bits = parse_hex_exact(
                fields[4], 8, source, line_number, "finite_value_f32_bits_hex");
            if (parse_finite_hex_float_bits(fields[5], source, line_number) != finite_value_f32_bits) {
                fixture_error(source, line_number, "finite_value_hex does not match finite_value_f32_bits_hex");
            }
        } else if (fields[4] != "-" || fields[5] != "-") {
            fixture_error(source, line_number, "non-finite decode rows must use '-' finite-value fields");
        }

        parsed.push_back({f8_bits, f16_bits, negative, expected_classification, finite_value_f32_bits, finite});
    }

    if (!saw_header) {
        fixture_error(source, "missing decode fixture header");
    }
    for (size_t bits = 0; bits < seen.size(); ++bits) {
        if (!seen[bits]) {
            fixture_error(source, "missing f8 byte " + hex_bits(static_cast<uint32_t>(bits), 2));
        }
    }
    return parsed;
}

std::vector<EncodeRow> load_encode_rows(std::istream& input, const std::string& source) {
    std::vector<EncodeRow> parsed;
    std::set<uint32_t> seen_inputs;
    bool saw_header = false;
    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (!saw_header) {
            if (line != kEncodeFixtureHeader) {
                fixture_error(source, line_number, "invalid encode fixture header");
            }
            saw_header = true;
            continue;
        }

        const std::vector<std::string> fields = split_fields(line, ',');
        if (fields.size() != 5) {
            fixture_error(source, line_number, "invalid float8 encode fixture row");
        }
        const uint32_t f32_bits = parse_hex_exact(fields[0], 8, source, line_number, "f32_bits_hex");
        const uint16_t f16_bits = static_cast<uint16_t>(
            parse_hex_exact(fields[1], 4, source, line_number, "f16_bits_hex"));
        const uint8_t f8_bits = static_cast<uint8_t>(
            parse_hex_exact(fields[2], 2, source, line_number, "f8_bits_hex"));
        if (!std::isfinite(std::bit_cast<float>(f32_bits))) {
            fixture_error(source, line_number, "f32_bits_hex must be finite");
        }
        if (!seen_inputs.insert(f32_bits).second) {
            fixture_error(source, line_number, "duplicate f32 input " + hex_bits(f32_bits, 8));
        }
        if (f16_bits_to_f8_reference(f16_bits) != f8_bits) {
            fixture_error(source, line_number, "f8_bits_hex does not match f16_bits_hex RNE result");
        }

        bool checked_accepts = false;
        if (fields[3] == "accept") {
            checked_accepts = true;
        } else if (fields[3] == "reject") {
            checked_accepts = false;
        } else {
            fixture_error(source, line_number, "invalid checked policy: " + fields[3]);
        }
        if (checked_accepts != ((f8_bits & 0x7fU) < 0x7cU)) {
            fixture_error(source, line_number, "checked policy does not match f8_bits_hex");
        }

        parsed.push_back({f32_bits, f16_bits, f8_bits, checked_accepts,
            parse_encode_tags(fields[4], source, line_number)});
    }

    if (!saw_header) {
        fixture_error(source, "missing encode fixture header");
    }
    validate_encode_sentinels(parsed, source);
    return parsed;
}

std::string fixture_contents(const char* filename) {
    const std::filesystem::path path = fixture_path(filename);
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open fixture: " + path.string());
    }
    std::ostringstream content;
    content << input.rdbuf();
    return content.str();
}

std::string replace_once(std::string value, const std::string& old_text, const std::string& new_text) {
    const size_t offset = value.find(old_text);
    if (offset == std::string::npos) {
        throw std::runtime_error("test fixture mutation target was not found");
    }
    value.replace(offset, old_text.size(), new_text);
    return value;
}

template <typename Loader>
void expect_fixture_rejected(Loader&& loader, std::string_view expected_detail) {
    try {
        loader();
        ADD_FAILURE() << "fixture loader accepted malformed input";
    } catch (const std::runtime_error& error) {
        EXPECT_NE(std::string::npos, std::string(error.what()).find(expected_detail));
    }
}

const std::vector<DecodeRow>& decode_rows() {
    static const std::vector<DecodeRow> rows = [] {
        const std::filesystem::path path = fixture_path("float8_decode_v1.csv");
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("failed to open float8 decode fixture");
        }
        return load_decode_rows(input, path.string());
    }();
    return rows;
}

const std::vector<EncodeRow>& encode_rows() {
    static const std::vector<EncodeRow> rows = [] {
        const std::filesystem::path path = fixture_path("float8_encode_v1.csv");
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error("failed to open float8 encode fixture");
        }
        return load_encode_rows(input, path.string());
    }();
    return rows;
}

uint16_t float16_bits_for_test(float16 value) {
    uint16_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

class RoundingModeRestore {
public:
    RoundingModeRestore() : original_(std::fegetround()) {}

    ~RoundingModeRestore() {
        if (original_ != -1) {
            (void)std::fesetround(original_);
        }
    }

    bool valid() const {
        return original_ != -1;
    }

private:
    int original_;
};

} // namespace

TEST(float8, representation_and_value_initialization_contract) {
    const float8 value {};
    EXPECT_EQ(0x00, value.to_bits());
    EXPECT_EQ(0x00000000U, std::bit_cast<uint32_t>(static_cast<float>(value)));
    EXPECT_EQ(57344.0f, kFloat8Max);
}

TEST(float8, fixture_loaders_reject_duplicate_and_missing_rows) {
    const std::string decode_fixture = fixture_contents("float8_decode_v1.csv");
    const std::string decode_zero =
        "0x00,0x0000,positive,zero,0x00000000,0x0.0p+0\n";
    const std::string decode_one =
        "0x01,0x0100,positive,subnormal,0x37800000,0x1.0000000000000p-16\n";

    const std::string duplicate_decode = replace_once(decode_fixture, decode_one, decode_zero);
    expect_fixture_rejected([&] {
        std::istringstream input(duplicate_decode);
        (void)load_decode_rows(input, "duplicate decode fixture");
    }, "duplicate f8 byte");

    const std::string missing_decode = replace_once(decode_fixture, decode_one, "");
    expect_fixture_rejected([&] {
        std::istringstream input(missing_decode);
        (void)load_decode_rows(input, "missing decode fixture");
    }, "missing f8 byte");

    const std::string encode_fixture = fixture_contents("float8_encode_v1.csv");
    const std::string encode_zero =
        "0x00000000,0x0000,0x00,accept,named_boundary|signed_zero\n";
    const std::string encode_subnormal =
        "0x00000001,0x0000,0x00,accept,f32_subnormal\n";

    const std::string duplicate_encode = replace_once(encode_fixture, encode_subnormal, encode_zero);
    expect_fixture_rejected([&] {
        std::istringstream input(duplicate_encode);
        (void)load_encode_rows(input, "duplicate encode fixture");
    }, "duplicate f32 input");

    const std::string missing_encode = replace_once(encode_fixture, encode_subnormal, "");
    expect_fixture_rejected([&] {
        std::istringstream input(missing_encode);
        (void)load_encode_rows(input, "missing encode fixture");
    }, "missing required encode sentinel 0x00000001");
}

TEST(float8, fixture_loaders_reject_invalid_tokens_and_out_of_range_fields) {
    const std::string decode_fixture = fixture_contents("float8_decode_v1.csv");
    const std::string invalid_decode = replace_once(decode_fixture, "positive,zero", "sideways,zero");
    expect_fixture_rejected([&] {
        std::istringstream input(invalid_decode);
        (void)load_decode_rows(input, "invalid decode fixture");
    }, "invalid sign");

    const std::string out_of_range_decode = replace_once(
        decode_fixture,
        "0x00,0x0000,positive,zero,0x00000000,0x0.0p+0",
        "0x100,0x0000,positive,zero,0x00000000,0x0.0p+0");
    expect_fixture_rejected([&] {
        std::istringstream input(out_of_range_decode);
        (void)load_decode_rows(input, "out-of-range decode fixture");
    }, "invalid f8_bits_hex width");

    const std::string encode_fixture = fixture_contents("float8_encode_v1.csv");
    const std::string invalid_encode = replace_once(
        encode_fixture,
        "0x00000000,0x0000,0x00,accept,named_boundary|signed_zero",
        "0x00000000,0x0000,0x00,unknown,named_boundary|signed_zero");
    expect_fixture_rejected([&] {
        std::istringstream input(invalid_encode);
        (void)load_encode_rows(input, "invalid encode fixture");
    }, "invalid checked policy");

    const std::string out_of_range_encode = replace_once(
        encode_fixture,
        "0x00000000,0x0000,0x00,accept,named_boundary|signed_zero",
        "0x00000000,0x0000,0x100,accept,named_boundary|signed_zero");
    expect_fixture_rejected([&] {
        std::istringstream input(out_of_range_encode);
        (void)load_encode_rows(input, "out-of-range encode fixture");
    }, "invalid f8_bits_hex width");
}

TEST(float8, exhaustive_unchecked_decode_matches_fixture) {
    const std::vector<DecodeRow>& rows = decode_rows();
    ASSERT_EQ(kExpectedDecodeFixtureRows, rows.size());

    for (const DecodeRow& row : rows) {
        SCOPED_TRACE(hex_bits(row.f8_bits, 2));
        const float8 value = float8::from_bits(row.f8_bits);
        const float decoded = value;

        EXPECT_EQ(row.f8_bits, value.to_bits());
        EXPECT_EQ(row.f16_bits, static_cast<uint16_t>(row.f8_bits) << 8);
        EXPECT_EQ(row.negative, std::signbit(decoded));

        if (row.has_finite_value) {
            EXPECT_TRUE(std::isfinite(decoded));
            EXPECT_EQ(row.finite_value_f32_bits, std::bit_cast<uint32_t>(decoded));
            if (row.classification == "zero") {
                EXPECT_EQ(FP_ZERO, std::fpclassify(decoded));
            } else if (row.classification == "subnormal") {
                // This is an E5M2/f16 subnormal; it is still a normal f32
                // after exact promotion, which is the intended decode path.
                EXPECT_EQ(0u, (row.f16_bits >> 10) & 0x1fU);
                EXPECT_NE(0u, row.f16_bits & 0x03ffU);
                EXPECT_TRUE(std::isnormal(decoded));
            } else {
                EXPECT_EQ(FP_NORMAL, std::fpclassify(decoded));
            }
        } else if (row.classification == "inf") {
            EXPECT_TRUE(std::isinf(decoded));
        } else {
            EXPECT_EQ("nan", row.classification);
            EXPECT_TRUE(std::isnan(decoded));
        }
    }
}

TEST(float8, encode_fixture_matches_both_rne_stages) {
    const std::vector<EncodeRow>& rows = encode_rows();
    ASSERT_FALSE(rows.empty());

    for (const EncodeRow& row : rows) {
        SCOPED_TRACE(hex_bits(row.f32_bits, 8));
        const float input = std::bit_cast<float>(row.f32_bits);
        ASSERT_TRUE(std::isfinite(input));

        EXPECT_EQ(row.f16_bits, float16_bits_for_test(f32_to_f16_rne(input)));

        const float8 unchecked(input);
        EXPECT_EQ(row.f8_bits, unchecked.to_bits());

        float8 checked = float8::from_bits(0x55);
        EXPECT_EQ(row.checked_accepts, try_encode_float8(input, checked));
        if (row.checked_accepts) {
            EXPECT_EQ(row.f8_bits, checked.to_bits());
        } else {
            EXPECT_EQ(0x55, checked.to_bits());
        }
    }
}

TEST(float8, conversion_is_independent_of_ambient_rounding_mode) {
    RoundingModeRestore restore;
    ASSERT_TRUE(restore.valid());
    const std::vector<EncodeRow>& rows = encode_rows();
    const int modes[] = {FE_DOWNWARD, FE_UPWARD, FE_TOWARDZERO};

    for (const int mode : modes) {
        ASSERT_EQ(0, std::fesetround(mode));
        for (const EncodeRow& row : rows) {
            const float input = std::bit_cast<float>(row.f32_bits);
            SCOPED_TRACE(hex_bits(row.f32_bits, 8));
            EXPECT_EQ(row.f16_bits, float16_bits_for_test(f32_to_f16_rne(input)));
            EXPECT_EQ(row.f8_bits, float8(input).to_bits());
        }
    }
}

TEST(float8, rne_ties_and_binade_carries_are_exact) {
    EXPECT_EQ(0x3c, float8(1.125f).to_bits());       // even lower f8 LSB
    EXPECT_EQ(0x3e, float8(1.375f).to_bits());       // odd lower f8 LSB
    EXPECT_EQ(0x40, float8(1.875f).to_bits());       // carry into next binade
    EXPECT_EQ(0xbc, float8(-1.125f).to_bits());
    EXPECT_EQ(0xbe, float8(-1.375f).to_bits());
    EXPECT_EQ(0xc0, float8(-1.875f).to_bits());
}

TEST(float8, signed_zero_and_subnormal_normal_boundaries_are_preserved) {
    const float positive_zero = float8::from_bits(0x00);
    const float negative_zero = float8::from_bits(0x80);
    EXPECT_EQ(0x00000000U, std::bit_cast<uint32_t>(positive_zero));
    EXPECT_EQ(0x80000000U, std::bit_cast<uint32_t>(negative_zero));

    EXPECT_EQ(0x37800000U, std::bit_cast<uint32_t>(static_cast<float>(float8::from_bits(0x01))));
    EXPECT_EQ(0x38400000U, std::bit_cast<uint32_t>(static_cast<float>(float8::from_bits(0x03))));
    EXPECT_EQ(0x38800000U, std::bit_cast<uint32_t>(static_cast<float>(float8::from_bits(0x04))));
    EXPECT_EQ(0xb7800000U, std::bit_cast<uint32_t>(static_cast<float>(float8::from_bits(0x81))));
    EXPECT_EQ(0xb8800000U, std::bit_cast<uint32_t>(static_cast<float>(float8::from_bits(0x84))));
}

TEST(float8, checked_encode_rejects_nonfinite_and_finite_overflow) {
    float8 output = float8::from_bits(0x55);
    EXPECT_FALSE(try_encode_float8(std::numeric_limits<float>::quiet_NaN(), output));
    EXPECT_EQ(0x55, output.to_bits());
    EXPECT_FALSE(try_encode_float8(std::numeric_limits<float>::infinity(), output));
    EXPECT_EQ(0x55, output.to_bits());
    EXPECT_FALSE(try_encode_float8(-std::numeric_limits<float>::infinity(), output));
    EXPECT_EQ(0x55, output.to_bits());

    EXPECT_TRUE(try_encode_float8(kFloat8Max, output));
    EXPECT_EQ(0x7b, output.to_bits());
    EXPECT_TRUE(try_encode_float8(-kFloat8Max, output));
    EXPECT_EQ(0xfb, output.to_bits());

    EXPECT_EQ(0x7c, float8(61424.0f).to_bits());
    EXPECT_FALSE(try_encode_float8(61424.0f, output));
    EXPECT_EQ(0xfb, output.to_bits());
    EXPECT_EQ(0xfc, float8(-61424.0f).to_bits());
    EXPECT_FALSE(try_encode_float8(-61424.0f, output));
    EXPECT_EQ(0xfb, output.to_bits());
}

} // namespace sketch2
