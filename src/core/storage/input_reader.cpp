#include "input_reader.h"
#include "utils/string_utils.h"
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
    madvise(m, map_len_, MADV_SEQUENTIAL);
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
    validate_type(type_);

    char* dim_end;
    dim_ = static_cast<size_t>(strtoull(comma + 1, &dim_end, 10));
    if (dim_end == comma + 1) {
        return Ret("Invalid header: missing dimension");
    }

    if (dim_ < 4 || dim_ > 4096) {
        return Ret("Invalid header: dimension out of range");
    }

    if (size() < sizeof(uint64_t)) {
        return Ret("Invalid header: vector data size is too small");
    }

    // Advance past the header newline
    const char* line = dim_end;
    while (line < end && *line != '\n') ++line;
    if (line < end) ++line;

    // Parse each vector line: "{id} : [ {data...} ]\n"
    while (line < end) {
        const char* next_nl = static_cast<const char*>(memchr(line, '\n', static_cast<size_t>(end - line)));
        const char* line_limit = next_nl ? next_nl : end;

        char* id_end;
        uint64_t id = strtoull(line, &id_end, 10);
        if (id_end == line) {
            // Skip empty lines or trailing whitespace
            if (next_nl) {
                line = next_nl + 1;
                continue;
            } else {
                break;
            }
        }

        const char* bracket = static_cast<const char*>(
            memchr(id_end, '[', static_cast<size_t>(line_limit - id_end)));
        if (!bracket) {
            return Ret("Invalid line: missing '['");
        }
        const char* close = static_cast<const char*>(
            memchr(bracket + 1, ']', static_cast<size_t>(line_limit - (bracket + 1))));
        if (!close) {
            return Ret("Invalid line: missing ']'");
        }

        // offset points to the character after "[" (first number)
        uint64_t offset = static_cast<uint64_t>(bracket + 1 - p);
        uint64_t end_offset = static_cast<uint64_t>(close - p);
        lines_.push_back({id, offset, end_offset});

        line = next_nl ? next_nl + 1 : end;
    }

    std::sort(lines_.begin(), lines_.end(),
        [](const LineInfo& lhs, const LineInfo& rhs) {
            return lhs.id < rhs.id;
        });

    for (size_t i = 1; i < lines_.size(); ++i) {
        if (lines_[i - 1].id == lines_[i].id) {
            return Ret("Duplicate ids");
        }
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

Ret InputReader::data(size_t index, uint8_t* buf, size_t size) const {
    if (index >= lines_.size()) {
        return Ret("InputReader::data: index out of range");
    }

    const char* p = reinterpret_cast<const char*>(map_) + lines_[index].offset;
    const char* vec_end = reinterpret_cast<const char*>(map_) + lines_[index].end;

    return parse_vector(buf, size, type_, dim_, p, vec_end);
}

bool InputReader::is_no_data(size_t index) const {
    if (index >= lines_.size()) {
        throw std::out_of_range("InputReader::is_no_data: index out of range");
    }
    const char* p = reinterpret_cast<const char*>(map_) + lines_[index].offset;
    return *p == ']';
}

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

Ret InputReaderView::data(size_t index, uint8_t* buf, size_t size) const {
    if (index >= count_) {
        return Ret("InputReaderView::data: index out of range");
    }
    return reader_.data(view_index_ + index, buf, size);
}

bool InputReaderView::is_no_data(size_t index) const {
    if (index >= count_) {
        throw std::out_of_range("InputReaderView::is_no_data: index out of range");
    }
    return reader_.is_no_data(view_index_ + index);
}

} // namespace sketch2
