#include "input_reader.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <stdexcept>

namespace sketch2 {

InputReader::~InputReader() {
    if (map_) {
        munmap(const_cast<uint8_t*>(map_), map_len_);
    }
}

Ret InputReader::init(const std::string& path) {
    try {
        return init_(path);
    } catch (const std::exception& e) {
        return Ret(e.what());
    }
}

Ret InputReader::init_(const std::string& path) {
    if (map_) {
        return Ret("Input reader is initialized already.");
    }
    // Map file into memory
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return Ret("Failed to open file: " + path);
    }
    struct stat st;
    if (fstat(fd, &st) < 0) {
        close(fd);
        return Ret("Failed to stat file: " + path);
    }
    map_len_ = static_cast<size_t>(st.st_size);
    if (map_len_ == 0) {
        close(fd);
        return Ret("File is empty: " + path);
    }
    void* m = mmap(nullptr, map_len_, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m == MAP_FAILED) {
        return Ret("Failed to mmap file: " + path);
    }
    map_ = static_cast<const uint8_t*>(m);

    const char* p   = reinterpret_cast<const char*>(map_);
    const char* end = p + map_len_;

    // Parse the first line: "{type},{dim}\n"
    const char* comma = static_cast<const char*>(memchr(p, ',', static_cast<size_t>(end - p)));
    if (!comma) {
        return Ret("Invalid header: missing comma");
    }
    std::string type_str(p, static_cast<size_t>(comma - p));
    type_ = data_type_from_string(type_str);

    char* dim_end;
    dim_ = static_cast<size_t>(strtoull(comma + 1, &dim_end, 10));
    if (dim_end == comma + 1) {
        return Ret("Invalid header: missing dimension");
    }

    if (size() < sizeof(uint64_t)) {
        return Ret("Invalid header: vector data size is too small");
    }

    buf_.resize(size()); // pre-allocate buffer for one vector, used for parsing

    // Advance past the header newline
    const char* line = dim_end;
    while (line < end && *line != '\n') ++line;
    if (line < end) ++line;

    uint64_t prev_id = 0;
    bool once = true;

    // Parse each vector line: "{id} : [ {data...} ]\n"
    while (line < end) {
        char* id_end;
        uint64_t id = strtoull(line, &id_end, 10);
        if (id_end == line) break; // blank or trailing content

        if (once) {
            once = false;
        } else {
            if (prev_id >= id) {
                return Ret("Invalid order of ids");
            }
        }
        prev_id = id;

        const char* bracket = static_cast<const char*>(
            memchr(id_end, '[', static_cast<size_t>(end - id_end)));
        if (!bracket) {
            return Ret("Invalid line: missing '['");
        }

        // offset points to the character after "[" (first number)
        uint64_t offset = static_cast<uint64_t>(bracket + 1 - p);

        lines_.push_back({id, offset});

        const char* nl = static_cast<const char*>(
            memchr(id_end, '\n', static_cast<size_t>(end - id_end)));
        if (!nl) break;
        line = nl + 1;
    }

    return Ret(0);
}

size_t InputReader::count() const {
    return lines_.size();
}

DataType InputReader::type() const {
    return type_;
}

size_t InputReader::dim() const {
    return dim_;
}

size_t InputReader::size() const {
    return dim_ * data_type_size(type_);
}

uint64_t InputReader::id(size_t index) const {
    if (index >= lines_.size()) {
        throw std::out_of_range("InputReader::id: index out of range");
    }
    return lines_[index].id;
}

const uint8_t* InputReader::data(size_t index) const {
    if (index >= lines_.size()) {
        throw std::out_of_range("InputReader::data: index out of range");
    }
    const char* p = reinterpret_cast<const char*>(map_) + lines_[index].offset;
    const char* end = reinterpret_cast<const char*>(map_) + map_len_;

    if (type_ == DataType::f32) {
        float* out = reinterpret_cast<float*>(buf_.data());
        for (size_t d = 0; d < dim_; ++d) {
            if (p >= end) {
                throw std::runtime_error("InputReader::data: truncated vector payload");
            }
            char* next;
            out[d] = strtof(p, &next);
            if (next == p) {
                throw std::runtime_error("InputReader::data: invalid f32 token");
            }
            p = next;
            while (p < end && (*p == ',' || *p == ' ')) ++p;
        }
    } else if (type_ == DataType::i16) {
        int16_t* out = reinterpret_cast<int16_t*>(buf_.data());
        for (size_t d = 0; d < dim_; ++d) {
            if (p >= end) {
                throw std::runtime_error("InputReader::data: truncated vector payload");
            }
            char* next;
            out[d] = static_cast<int16_t>(strtol(p, &next, 10));
            if (next == p) {
                throw std::runtime_error("InputReader::data: invalid i16 token");
            }
            p = next;
            while (p < end && (*p == ',' || *p == ' ')) ++p;
        }
    } else if (type_ == DataType::f16) {
        float16* out = reinterpret_cast<float16*>(buf_.data());
        for (size_t d = 0; d < dim_; ++d) {
            if (p >= end) {
                throw std::runtime_error("InputReader::data: truncated vector payload");
            }
            char* next;
            float f = strtof(p, &next);
            if (next == p) {
                throw std::runtime_error("InputReader::data: invalid f16 token");
            }
            out[d] = static_cast<float16>(f);
            p = next;
            while (p < end && (*p == ',' || *p == ' ')) ++p;
        }
    }

    return buf_.data();
}

bool InputReader::is_no_data(size_t index) const {
    if (index >= lines_.size()) {
        throw std::out_of_range("InputReader::is_no_data: index out of range");
    }
    const char* p = reinterpret_cast<const char*>(map_) + lines_[index].offset;
    return *p == ']';
}

// Instructions:
// Return true if there is an overlap between range and ids, assuming ids are sorted.
bool InputReader::is_range_present(uint64_t start_range, uint64_t end_range) const {
    if (start_range >= end_range || lines_.empty()) {
        return false;
    }

    const uint64_t min_id = lines_.front().id;
    const uint64_t max_id = lines_.back().id;
    if (end_range <= min_id || start_range > max_id) {
        return false;
    }

    auto it = std::lower_bound(
        lines_.begin(), lines_.end(), start_range,
        [](const LineInfo& line, uint64_t value) { return line.id < value; });
    return it != lines_.end() && it->id < end_range;
}

/***********************************************
 *   InputReaderView
 */

InputReaderView::InputReaderView(const InputReader& reader, uint64_t start, uint64_t end)
    : reader_(reader), view_index_(0), count_(0) {

    if (start > end) {
        throw std::invalid_argument("InputReaderView: start must be <= end");
    }

    if (start == 0 && end == 0) {
        // special case: view the whole reader
        view_index_ = 0;
        count_ = reader_.count();
        return;
    }

    bool once = true;
    for (size_t i = 0; i < reader_.count(); ++i) {
        uint64_t id = reader_.id(i);
        if (id >= end) {
            break; // ids are sorted, we can stop here
        }
        if (id >= start) {
            if (once) {
                view_index_ = i;
                once = false;
            }
            ++count_;
        }
    }
}

size_t InputReaderView::count() const {
    return static_cast<size_t>(count_);
}

DataType InputReaderView::type() const {
    return reader_.type();
}

size_t InputReaderView::dim() const {
    return reader_.dim();
}

size_t InputReaderView::size() const {
    return reader_.size();
}

uint64_t InputReaderView::id(size_t index) const {
    if (index >= count_) {
        throw std::out_of_range("InputReaderView::id: index out of range");
    }
    return reader_.id(view_index_ + index);
}

const uint8_t* InputReaderView::data(size_t index) const {
    if (index >= count_) {
        throw std::out_of_range("InputReaderView::data: index out of range");
    }
    return reader_.data(view_index_ + index);
}

bool InputReaderView::is_no_data(size_t index) const {
    if (index >= count_) {
        throw std::out_of_range("InputReaderView::is_no_data: index out of range");
    }
    return reader_.is_no_data(view_index_ + index);
}

} // namespace sketch2
