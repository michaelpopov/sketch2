// Defines shared enums, return types, and common utility conversions.

#pragma once
#include <string>
#include <stdexcept>
#include <cstdint>

// NEON intrinsics are only used by the aarch64 compute path.
#if defined(__aarch64__)
#include <arm_neon.h> 
#endif

namespace sketch2 {

#if !defined(__FLT16_MANT_DIG__)
#error "f16 is not supported on this platform: _Float16 type is unavailable"
#endif
using float16 = _Float16;

enum class DataType {
    f16,
    f32,
    i16,
};

enum class DistFunc {
    L1,
    L2,
    COS,
};

inline void validate_type(DataType) {}

inline const char* data_type_to_string(DataType type) {
    switch (type) {
        case DataType::f16: return "f16";
        case DataType::f32: return "f32";
        case DataType::i16: return "i16";
        default: return "unknown";
    }
}

inline size_t data_type_size(DataType type) {
    switch (type) {
        case DataType::f16: return 2;
        case DataType::f32: return 4;
        case DataType::i16: return 2;
        default: return 0;
    }
}

inline DataType data_type_from_int(int t)
{
    switch (t) {
        case 0: return DataType::f16;
        case 1: return DataType::f32;
        case 2: return DataType::i16;
        default: throw std::runtime_error("Invalid data type number.");
    }
}

inline DataType data_type_from_string(const std::string &type_str) {
    if (type_str == "f32") return DataType::f32;
    if (type_str == "f16") return DataType::f16;
    if (type_str == "i16") return DataType::i16;
    throw std::runtime_error("Invalid data type string.");
}

inline int data_type_to_int(DataType type)
{
    switch (type)
    {
        case DataType::f16: return 0;
        case DataType::f32: return 1;
        case DataType::i16: return 2;
        default: throw std::runtime_error("Invalid data type.");
    }
}

inline const char* dist_func_to_string(DistFunc func) {
    switch (func) {
        case DistFunc::L1: return "l1";
        case DistFunc::L2: return "l2";
        case DistFunc::COS: return "cos";
        default: throw std::runtime_error("Invalid distance function.");
    }
}

inline DistFunc dist_func_from_string(const std::string& func_str) {
    if (func_str == "l1") return DistFunc::L1;
    if (func_str == "l2") return DistFunc::L2;
    if (func_str == "cos") return DistFunc::COS;
    throw std::runtime_error("Invalid distance function string.");
}

inline void validate_dist_func(DistFunc func) {
    (void)dist_func_to_string(func);
}

// Ret exists as the project's lightweight result type for functions that need
// to return either success or an error without throwing. It carries the status
// code, message, and an optional content flag used by higher-level callers.
class Ret
{
public:
    Ret(int code) : code_(code) {}
    Ret(const std::string& message) : code_(-1), message_(message) {}
    Ret(const char* message) : code_(-1), message_(message) {}
    Ret(int code, const std::string& message, bool is_content = false)
      : code_(code), message_(message), is_content_(is_content) {}
    Ret(const Ret& ret) : code_(ret.code_), message_(ret.message_) {}
    Ret& operator=(const Ret&) = default;
    int code() const { return code_; }
    const std::string& message() const { return message_; }
    bool is_content() const { return is_content_; }
    
private:
    int code_ = 0;
    std::string message_;
    bool is_content_ = false;
};

#define CHECK(ret) \
    do { \
        const auto check_ret = (ret); \
        if (check_ret.code() != 0) { return check_ret; } \
    } while (0)

} // namespace sketch2
