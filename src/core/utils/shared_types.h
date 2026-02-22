#pragma once
#include <string>
#include <stdexcept>
#include <cstdint>

// Support for f16 on aarch64
#if defined(__aarch64__)
#include <arm_neon.h> 
#endif

namespace sketch2 {

#if defined(__aarch64__)
using float16 = _Float16;
#else
  struct float16_t {                 // stub that can't be used as a number
      uint16_t bits;
  };
#endif

enum class DataType {
    f16,
    f32,
    i32,
};

static inline constexpr bool supports_f16() {
#if defined(__aarch64__)
    return true;
#else
    return false;
#endif
}

static inline void validate_type(DataType t) {
    if (t == DataType::f16 && !supports_f16()) {
        throw std::runtime_error("f16 is not supported.");
    }
}

static inline const char* data_type_to_string(DataType type) {
    switch (type) {
        case DataType::f16: return "f16";
        case DataType::f32: return "f32";
        case DataType::i32: return "i32";
        default: return "unknown";
    }
}

static inline size_t data_type_size(DataType type) {
    switch (type) {
        case DataType::f16: return 2;
        case DataType::f32: return 4;
        case DataType::i32: return 4;
        default: return 0;
    }
}

static inline DataType data_type_from_int(int t)
{
    switch (t) {
        case 0: return DataType::f16;
        case 1: return DataType::f32;
        case 2: return DataType::i32;
        default: throw std::runtime_error("Invalid data type number.");
    }
}

static inline DataType data_type_from_string(const std::string &type_str) {
    if (type_str == "f32") return DataType::f32;
    if (type_str == "f16") return DataType::f16;
    if (type_str == "i32") return DataType::i32;
    throw std::runtime_error("Invalid data type string.");
}

static inline int data_type_to_int(DataType type)
{
    switch (type)
    {
        case DataType::f16: return 0;
        case DataType::f32: return 1;
        case DataType::i32: return 2;
        default: throw std::runtime_error("Invalid data type.");
    }
}

class Ret
{
public:
    Ret(int code) : code_(code) {}
    Ret(const std::string& message) : code_(-1), message_(message) {}
    Ret(const char* message) : code_(-1), message_(message) {}
    Ret(int code, const std::string& message, bool is_content = false)
      : code_(code), message_(message), is_content_(is_content) {}
    operator int() const { return code_; } // Automatic conversion to int
    int code() const { return code_; }
    const std::string& message() const { return message_; }
    bool is_content() const { return is_content_; }
    
private:
    int code_ = 0;
    std::string message_;
    bool is_content_ = false;
};

#define CHECK(ret) \
    if (ret != 0) { \
        return ret; \
    }

}
