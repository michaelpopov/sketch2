#pragma once
#include <string>
#include <stdexcept>

namespace sketch2 {

enum class DataType {
    f16,
    f32,
    i32,
};

static inline const char* to_string(DataType type) {
    switch (type) {
        case DataType::f16: return "f16";
        case DataType::f32: return "f32";
        case DataType::i32: return "i32";
        default: return "unknown";
    }
}

static inline size_t to_size(DataType type) {
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