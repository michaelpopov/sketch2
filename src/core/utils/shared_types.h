#pragma once
#include <string>

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

class Ret {
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