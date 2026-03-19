// Implements parsing and range views over textual and binary input files.

#include "input_reader.h"
#include "utils/log.h"
#include "utils/shared_consts.h"
#include "utils/string_utils.h"
#include "utils/timer.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <stdexcept>

namespace sketch2 {

namespace {

Ret parse_input_header(const char* begin, const char* end, DataType* type, size_t* dim, bool* binary) {
    if (begin == nullptr || end == nullptr || type == nullptr || dim == nullptr || binary == nullptr || begin >= end) {
        return Ret("Invalid header");
    }

    const std::string header(begin, static_cast<size_t>(end - begin));
    const size_t first_comma = header.find(',');
    if (first_comma == std::string::npos) {
        return Ret("Invalid header: missing comma");
    }

    try {
        *type = data_type_from_string(header.substr(0, first_comma));
    } catch (const std::exception& e) {
        return Ret(e.what());
    }

    const size_t second_comma = header.find(',', first_comma + 1);
    const std::string dim_part = second_comma == std::string::npos
        ? header.substr(first_comma + 1)
        : header.substr(first_comma + 1, second_comma - first_comma - 1);
    if (dim_part.empty()) {
        return Ret("Invalid header: missing dimension");
    }

    char* dim_end = nullptr;
    *dim = static_cast<size_t>(strtoull(dim_part.c_str(), &dim_end, 10));
    if (dim_end == dim_part.c_str() || *dim_end != '\0') {
        return Ret("Invalid header: invalid dimension");
    }

    *binary = false;
    if (second_comma != std::string::npos) {
        const std::string mode = header.substr(second_comma + 1);
        if (mode != "bin") {
            return Ret("Invalid header: unsupported mode");
        }
        *binary = true;
    }

    return Ret(0);
}

} // namespace

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

// Memory-maps the input file, parses its header and record boundaries, then
// stores sorted metadata so later reads can parse text vectors on demand or
// memcpy binary payloads without rescanning the whole file.
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
    auto fail = [this](const std::string& message) -> Ret {
        munmap(const_cast<uint8_t*>(map_), map_len_);
        map_ = nullptr;
        map_len_ = 0;
        type_ = DataType::f32;
        dim_ = 0;
        binary_ = false;
        lines_.clear();
        return Ret(message);
    };

    const char* p   = reinterpret_cast<const char*>(map_);
    const char* end = p + map_len_;

    const char* header_end = static_cast<const char*>(memchr(p, '\n', static_cast<size_t>(end - p)));
    if (!header_end) {
        return fail("Invalid header: missing newline");
    }
    Ret header_ret = parse_input_header(p, header_end, &type_, &dim_, &binary_);
    if (header_ret.code() != 0) {
        return fail(header_ret.message());
    }

    if (dim_ < kMinDimension || dim_ > kMaxDimension) {
        return fail("Invalid header: dimension out of range");
    }

    if (size() < sizeof(uint64_t)) {
        return fail("Invalid header: vector data size is too small");
    }

    const char* record_begin = header_end + 1;
    if (binary_) {
        const size_t record_size = sizeof(uint64_t) + size();
        const size_t payload_bytes = static_cast<size_t>(end - record_begin);
        if (payload_bytes % record_size != 0) {
            return fail("Invalid binary payload size");
        }

        for (const char* record = record_begin; record < end; record += record_size) {
            uint64_t id = 0;
            std::memcpy(&id, record, sizeof(id));
            const uint64_t offset = static_cast<uint64_t>((record + sizeof(id)) - p);
            const uint64_t end_offset = offset + size();
            lines_.push_back({id, offset, end_offset});
        }
    } else {
        const char* line = record_begin;
        bool once = true;

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
                return fail("Invalid line: missing '['");
            }
            const char* close = static_cast<const char*>(
                memchr(bracket + 1, ']', static_cast<size_t>(line_limit - (bracket + 1))));
            if (!close) {
                return fail("Invalid line: missing ']'");
            }

            // offset points to the character after "[" (first number)
            uint64_t offset = static_cast<uint64_t>(bracket + 1 - p);
            uint64_t end_offset = static_cast<uint64_t>(close - p);
            lines_.push_back({id, offset, end_offset});

            if (once && bracket[1] != ']') { // skip checking "delete" vectors
                once = false;
                const char* p = reinterpret_cast<const char*>(map_) + offset;
                const char* vec_end = reinterpret_cast<const char*>(map_) + end_offset;
                is_comma_delimited_ = check_comma_format(p, vec_end);
            }

            line = next_nl ? next_nl + 1 : end;
        }
    }

    const auto by_id = [](const LineInfo& lhs, const LineInfo& rhs) {
        return lhs.id < rhs.id;
    };
    if (!std::is_sorted(lines_.begin(), lines_.end(), by_id)) {
        std::sort(lines_.begin(), lines_.end(),
            by_id);
    }

    for (size_t i = 1; i < lines_.size(); ++i) {
        if (lines_[i - 1].id == lines_[i].id) {
            return fail("Duplicate ids");
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

bool InputReader::is_binary() const {
    return binary_;
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
    if (size < this->size()) {
        return Ret("InputReader::data: invalid input buffer size");
    }

    if (binary_) {
        std::memcpy(buf, map_ + lines_[index].offset, this->size());
        return Ret(0);
    }

    const char* p = reinterpret_cast<const char*>(map_) + lines_[index].offset;
    const char* vec_end = reinterpret_cast<const char*>(map_) + lines_[index].end;

    return is_comma_delimited_ ? parse_vector(buf, size, type_, dim_, p, vec_end) :
        parse_vector_spaces(buf, size, type_, dim_, p, vec_end);
}

Ret InputReader::raw_data(size_t index, const uint8_t** data) const {
    if (index >= lines_.size()) {
        return Ret("InputReader::raw_data: index out of range");
    }
    if (data == nullptr) {
        return Ret("InputReader::raw_data: data pointer is null");
    }
    if (!binary_) {
        return Ret("InputReader::raw_data: raw access is only available in binary mode");
    }

    *data = map_ + lines_[index].offset;
    return Ret(0);
}

bool InputReader::is_no_data(size_t index) const {
    if (index >= lines_.size()) {
        throw std::out_of_range("InputReader::is_no_data: index out of range");
    }
    if (binary_) {
        return false;
    }
    const char* p = reinterpret_cast<const char*>(map_) + lines_[index].offset;
    return *p == ']';
}

// Checks whether any parsed id falls into [start_range, end_range) using the
// sorted line index instead of rescanning the mapped text.
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

// Finds the first parsed line in [start, end) and the number of contiguous
// entries in that range so InputReaderView can expose a cheap subrange.
std::pair<size_t, size_t> InputReader::find_index_range(uint64_t start, uint64_t end) const {
    auto first = std::lower_bound(
        lines_.begin(), lines_.end(), start,
        [](const LineInfo& line, uint64_t value) { return line.id < value; });

    auto last = std::lower_bound(
        first, lines_.end(), end,
        [](const LineInfo& line, uint64_t value) { return line.id < value; });

    return {
        static_cast<size_t>(first - lines_.begin()),
        static_cast<size_t>(last - first)
    };
}

/***********************************************
 *   InputReaderView
 */

// Creates a logical view over either the whole reader or the ids that fall
// inside [start, end), keeping only offsets into the original reader state.
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

    const auto [index, count] = reader_.find_index_range(start, end);
    view_index_ = index;
    count_ = count;
}

size_t InputReaderView::count() const {
    return count_;
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

bool InputReaderView::is_binary() const {
    return reader_.is_binary();
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

Ret InputReaderView::raw_data(size_t index, const uint8_t** data) const {
    if (index >= count_) {
        return Ret("InputReaderView::raw_data: index out of range");
    }
    return reader_.raw_data(view_index_ + index, data);
}

bool InputReaderView::is_no_data(size_t index) const {
    if (index >= count_) {
        throw std::out_of_range("InputReaderView::is_no_data: index out of range");
    }
    return reader_.is_no_data(view_index_ + index);
}

} // namespace sketch2
