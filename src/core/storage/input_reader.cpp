// Implements parsing and range views over textual and binary input files.

#include "input_reader.h"
#include "utils/log.h"
#include "utils/shared_consts.h"
#include "utils/shared_types.h"
#include "utils/string_utils.h"
#include "utils/timer.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <charconv>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <system_error>

namespace sketch2 {

namespace {

bool is_bit_set(uint64_t word, size_t bit_index) {
    return (word & (uint64_t{1} << bit_index)) != 0;
}

} // namespace

void LinesInfo::clear() {
    ids_.clear();
    offsets32_.clear();
    offsets64_.clear();
}

void LinesInfo::reserve(size_t count) {
    ids_.reserve(count);
    if (is_u64_offsets_) {
        offsets64_.reserve(count);
    } else {
        offsets32_.reserve(count);
    }
}

void LinesInfo::set_u64_offsets(bool enabled) {
    if (!empty()) {
        throw std::logic_error("LinesInfo::set_u64_offsets requires empty container");
    }
    is_u64_offsets_ = enabled;
}

void LinesInfo::add(uint64_t id, uint64_t offset) {
    ids_.push_back(id);
    if (is_u64_offsets_) {
        offsets64_.push_back(offset);
        return;
    }
    if (offset > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        throw std::overflow_error("LinesInfo::add: offset exceeds uint32_t range");
    }
    offsets32_.push_back(static_cast<uint32_t>(offset));
}

uint64_t LinesInfo::id(size_t index) const {
    check_index(index);
    return ids_[index];
}

uint64_t LinesInfo::offset(size_t index) const {
    check_index(index);
    return is_u64_offsets_ ? offsets64_[index] : static_cast<uint64_t>(offsets32_[index]);
}

void LinesInfo::sort() {
    if (size() < 2 || std::is_sorted(ids_.begin(), ids_.end())) {
        return;
    }

    std::vector<size_t> order(size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::sort(order.begin(), order.end(), [this](size_t lhs, size_t rhs) {
        return ids_[lhs] < ids_[rhs];
    });

    std::vector<uint64_t> sorted_ids;
    sorted_ids.reserve(ids_.size());
    if (is_u64_offsets_) {
        std::vector<uint64_t> sorted_offsets;
        sorted_offsets.reserve(offsets64_.size());
        for (size_t index : order) {
            sorted_ids.push_back(ids_[index]);
            sorted_offsets.push_back(offsets64_[index]);
        }
        ids_ = std::move(sorted_ids);
        offsets64_ = std::move(sorted_offsets);
        return;
    }

    std::vector<uint32_t> sorted_offsets;
    sorted_offsets.reserve(offsets32_.size());
    for (size_t index : order) {
        sorted_ids.push_back(ids_[index]);
        sorted_offsets.push_back(offsets32_[index]);
    }
    ids_ = std::move(sorted_ids);
    offsets32_ = std::move(sorted_offsets);
}

size_t LinesInfo::lower_bound_index(uint64_t value) const {
    return lower_bound_index(0, value);
}

size_t LinesInfo::lower_bound_index(size_t first, uint64_t value) const {
    if (first > size()) {
        throw std::out_of_range("LinesInfo::lower_bound_index: first out of range");
    }
    auto it = std::lower_bound(ids_.begin() + static_cast<std::ptrdiff_t>(first), ids_.end(), value);
    return static_cast<size_t>(it - ids_.begin());
}

void LinesInfo::check_index(size_t index) const {
    if (index >= size()) {
        throw std::out_of_range("LinesInfo: index out of range");
    }
}

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

    lines_.set_u64_offsets(
        static_cast<uint64_t>(map_len_ - 1) > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()));

    const char* p   = reinterpret_cast<const char*>(map_);
    const char* end = p + map_len_;

    const char* header_end = static_cast<const char*>(memchr(p, '\n', static_cast<size_t>(end - p)));
    if (!header_end) {
        return fail("Invalid header: missing newline");
    }
    Ret header_ret = parse_input_header(p, header_end);
    if (header_ret.code() != 0) {
        return fail(header_ret.message());
    }

    if (dim_ < kMinDimension || dim_ > kMaxDimension) {
        return fail("Invalid header: dimension out of range");
    }

    if (size() < sizeof(uint64_t)) {
        return fail("Invalid header: vector data size is too small");
    }

    Ret ret_lines_process{0};
    const char* record_begin = header_end + 1;
    if (binary_) {
        ret_lines_process = bit_indexed_ ? process_binary_indexed_data(record_begin, end) :
             process_binary_data(record_begin, end);
    } else {
        ret_lines_process = process_text_data(record_begin, end);
    }

    if (ret_lines_process.code() != 0) {
        return fail(ret_lines_process.message());
    }

    lines_.sort();

    for (size_t i = 1; i < lines_.size(); ++i) {
        if (lines_.id(i - 1) == lines_.id(i)) {
            return fail("Duplicate ids");
        }
    }

    return Ret(0);
}

Ret InputReader::process_binary_indexed_data(const char* record_begin, const char* end) {
    const char* p   = reinterpret_cast<const char*>(map_);
    const size_t record_size = sizeof(uint64_t) + size();
    size_t record_counter = 0;

    for (const char* record = record_begin; record < end;) {
        const char* const block_begin = record;
        if (record + sizeof(uint64_t) > end) {
            return Ret("Corrupted indexed binary data: missing block bitset");
        }

        uint64_t word = 0;
        std::memcpy(&word, record, sizeof(word));
        record += sizeof(uint64_t);

        for (size_t block_index = 0; block_index < kIndexedBinaryBlockItems; ++block_index, ++record_counter) {
            const bool is_deleted = is_bit_set(word, block_index);
            const size_t item_size = is_deleted ? sizeof(uint64_t) : record_size;
            if (record + item_size > end) {
                return Ret("Corrupted indexed binary data: truncated record");
            }

            uint64_t id = 0;
            uint64_t offset = 0;
            std::memcpy(&id, record, sizeof(id));
            if (!is_deleted) {
                offset = static_cast<uint64_t>((record + sizeof(id)) - p);
            }

            record += item_size;
            lines_.add(id, offset);

            if (record == end) {
                if (block_index + 1 == kIndexedBinaryBlockItems) {
                    return Ret("Corrupted indexed binary data: missing full-block footer");
                }
                return Ret(0);
            }
        }

        if (record + sizeof(IndexedBlockFooter) > end) {
            return Ret("Corrupted indexed binary data: truncated full-block footer");
        }

        IndexedBlockFooter footer{};
        std::memcpy(&footer, record, sizeof(footer));
        if (footer.count != record_counter) {
            return Ret("Corrupted indexed binary data: mismatching footer counter");
        }

        const uint32_t block_crc = crc32_update(
            0, reinterpret_cast<const uint8_t*>(block_begin), static_cast<size_t>(record - block_begin));
        if (footer.crc32 != block_crc) {
            return Ret("Corrupted indexed binary data: mismatching footer checksum");
        }

        record += sizeof(footer);
    }

    return Ret(0);
}

Ret InputReader::process_binary_data(const char* record_begin, const char* end) {
    const char* p   = reinterpret_cast<const char*>(map_);
    const size_t record_size = sizeof(uint64_t) + size();
    const size_t payload_bytes = static_cast<size_t>(end - record_begin);
    if (payload_bytes % record_size != 0) {
        return Ret("Invalid binary payload size");
    }

    for (const char* record = record_begin; record < end; record += record_size) {
        uint64_t id = 0;
        std::memcpy(&id, record, sizeof(id));
        const uint64_t offset = static_cast<uint64_t>((record + sizeof(id)) - p);
        lines_.add(id, offset);
    }
    
    return Ret(0);
}

Ret InputReader::process_text_data(const char* record_begin, const char* end) {
    const char* p   = reinterpret_cast<const char*>(map_);
    const char* line = record_begin;
    bool once = true;

    // Parse each vector line: "{id} : [ {data...} ]\n"
    while (line < end) {
        const char* next_nl = static_cast<const char*>(memchr(line, '\n', static_cast<size_t>(end - line)));
        const char* line_limit = next_nl ? next_nl : end;

        const char* id_begin = line;
        while (id_begin < line_limit && std::isspace(static_cast<unsigned char>(*id_begin))) {
            ++id_begin;
        }

        uint64_t id = 0;
        const auto id_result = std::from_chars(id_begin, line_limit, id, 10);
        if (id_result.ptr == id_begin) {
            // Skip empty lines or trailing whitespace
            if (next_nl) {
                line = next_nl + 1;
                continue;
            } else {
                break;
            }
        }
        if (id_result.ec == std::errc::result_out_of_range) {
            return Ret("Invalid line: id out of range");
        }
        if (id_result.ec != std::errc{}) {
            return Ret("Invalid line: invalid id");
        }

        const char* bracket = static_cast<const char*>(
            memchr(id_result.ptr, '[', static_cast<size_t>(line_limit - id_result.ptr)));
        if (!bracket) {
            return Ret("Invalid line: missing '['");
        }
        const char* close = static_cast<const char*>(
            memchr(bracket + 1, ']', static_cast<size_t>(line_limit - (bracket + 1))));
        if (!close) {
            return Ret("Invalid line: missing ']'");
        }

        const bool is_deleted = close == bracket + 1;
        if (!is_deleted) {
            const char* first_payload = bracket + 1;
            while (first_payload < close && std::isspace(static_cast<unsigned char>(*first_payload))) {
                ++first_payload;
            }
            if (first_payload == close) {
                return Ret("Invalid line: deleted vector must be []");
            }
        }

        // offset points to the character after "[" (first number)
        uint64_t offset = static_cast<uint64_t>(bracket + 1 - p);
        lines_.add(id, offset);

        if (once && !is_deleted) {
            once = false;
            const char* p = reinterpret_cast<const char*>(map_) + offset;
            is_comma_delimited_ = check_comma_format(p, close);
        }

        line = next_nl ? next_nl + 1 : end;
    }

    return Ret{0};
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

bool InputReader::is_comma_delimited() const {
    return is_comma_delimited_;
}

uint64_t InputReader::id(size_t index) const {
    if (index >= lines_.size()) {
        throw std::out_of_range("InputReader::id: index out of range");
    }
    return lines_.id(index);
}

Ret InputReader::data(size_t index, uint8_t* buf, size_t size) const {
    if (index >= lines_.size()) {
        return Ret("InputReader::data: index out of range");
    }
    if (size < this->size()) {
        return Ret("InputReader::data: invalid input buffer size");
    }

    if (binary_) {
        if (lines_.offset(index) == 0) {
            return Ret("InputReader::data: vector is deleted");
        }

        std::memcpy(buf, map_ + lines_.offset(index), this->size());
        return Ret(0);
    }

    if (is_no_data(index)) {
        return Ret("InputReader::data: vector is deleted");
    }

    const char* p = reinterpret_cast<const char*>(map_) + lines_.offset(index);
    const char* vec_end = nullptr;
    CHECK(find_text_vector_end(index, &vec_end));

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
    if (lines_.offset(index) == 0) {
        return Ret("InputReader::raw_data: vector is deleted");
    }

    *data = map_ + lines_.offset(index);
    return Ret(0);
}

Ret InputReader::text_data_range(size_t index, const char** begin, const char** end) const {
    if (index >= lines_.size()) {
        return Ret("InputReader::text_data_range: index out of range");
    }
    if (begin == nullptr || end == nullptr) {
        return Ret("InputReader::text_data_range: invalid range pointers");
    }
    if (binary_) {
        return Ret("InputReader::text_data_range: text access is only available in text mode");
    }
    if (is_no_data(index)) {
        return Ret("InputReader::text_data_range: vector is deleted");
    }

    *begin = reinterpret_cast<const char*>(map_) + lines_.offset(index);
    return find_text_vector_end(index, end);
}

Ret InputReader::find_text_vector_end(size_t index, const char** vec_end) const {
    if (index >= lines_.size()) {
        return Ret("InputReader::find_text_vector_end: index out of range");
    }
    if (vec_end == nullptr) {
        return Ret("InputReader::find_text_vector_end: output pointer is null");
    }
    if (binary_) {
        return Ret("InputReader::find_text_vector_end: text access is only available in text mode");
    }

    const char* start = reinterpret_cast<const char*>(map_) + lines_.offset(index);
    const char* map_end = reinterpret_cast<const char*>(map_) + map_len_;
    const char* line_end = static_cast<const char*>(memchr(start, '\n', static_cast<size_t>(map_end - start)));
    if (line_end == nullptr) {
        line_end = map_end;
    }
    const char* close = static_cast<const char*>(memchr(start, ']', static_cast<size_t>(line_end - start)));
    if (close == nullptr) {
        return Ret("InputReader::find_text_vector_end: invalid text payload");
    }

    *vec_end = close;
    return Ret(0);
}

bool InputReader::is_no_data(size_t index) const {
    if (index >= lines_.size()) {
        throw std::out_of_range("InputReader::is_no_data: index out of range");
    }
    if (binary_) {
        return bit_indexed_ ? (lines_.offset(index) == 0) : false;
    }
    const char* p = reinterpret_cast<const char*>(map_) + lines_.offset(index);
    return *p == ']';
}

// Checks whether any parsed id falls into [start_range, end_range) using the
// sorted line index instead of rescanning the mapped text.
bool InputReader::is_range_present(uint64_t start_range, uint64_t end_range) const {
    if (start_range >= end_range || lines_.empty()) {
        return false;
    }

    const uint64_t min_id = lines_.id(0);
    const uint64_t max_id = lines_.id(lines_.size() - 1);
    if (end_range <= min_id || start_range > max_id) {
        return false;
    }

    const size_t index = lines_.lower_bound_index(start_range);
    return index != lines_.size() && lines_.id(index) < end_range;
}

// Finds the first parsed line in [start, end) and the number of contiguous
// entries in that range so InputReaderView can expose a cheap subrange.
std::pair<size_t, size_t> InputReader::find_index_range(uint64_t start, uint64_t end) const {
    const size_t first = lines_.lower_bound_index(start);
    const size_t last = lines_.lower_bound_index(first, end);

    return {
        first,
        last - first
    };
}

Ret InputReader::parse_input_header(const char* begin, const char* end) {
    if (begin == nullptr || end == nullptr || begin >= end) {
        return Ret("Invalid header");
    }

    const std::string header(begin, static_cast<size_t>(end - begin));
    const size_t first_comma = header.find(',');
    if (first_comma == std::string::npos) {
        return Ret("Invalid header: missing comma");
    }

    try {
        type_ = data_type_from_string(header.substr(0, first_comma));
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
    dim_ = static_cast<size_t>(strtoull(dim_part.c_str(), &dim_end, 10));
    if (dim_end == dim_part.c_str() || *dim_end != '\0') {
        return Ret("Invalid header: invalid dimension");
    }

    if (second_comma != std::string::npos) {
        const std::string mode = header.substr(second_comma + 1);
        if (mode != BinFileMarker && mode != BinIndexedFileMarker) {
            return Ret("Invalid header: unsupported mode");
        }

        binary_ = true;

        if (mode == BinIndexedFileMarker) {
            bit_indexed_ = true;
        }
    }

    return Ret(0);
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

bool InputReaderView::is_comma_delimited() const {
    return reader_.is_comma_delimited();
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

Ret InputReaderView::text_data_range(size_t index, const char** begin, const char** end) const {
    if (index >= count_) {
        return Ret("InputReaderView::text_data_range: index out of range");
    }
    return reader_.text_data_range(view_index_ + index, begin, end);
}

bool InputReaderView::is_no_data(size_t index) const {
    if (index >= count_) {
        throw std::out_of_range("InputReaderView::is_no_data: index out of range");
    }
    return reader_.is_no_data(view_index_ + index);
}

const uint64_t* InputReaderView::ids_data() const {
    if (count_ == 0) {
        return nullptr;
    }
    return reader_.lines_.ids_data() + view_index_;
}

} // namespace sketch2
