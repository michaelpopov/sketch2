// Declares the accumulator write-ahead log interface.

#pragma once

#include "core/storage/data_file.h"
#include "utils/shared_types.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace sketch2 {

class Accumulator;

enum class WalOp : uint8_t {
    AddVector = 1,
    DeleteVector = 2,
};

// AccumulatorWal exists to persist accumulator mutations in append-only form so
// owner-mode datasets can recover pending updates after a crash. It manages the
// WAL file header, record appends, replay, and log reset.
class AccumulatorWal {
public:
    AccumulatorWal() = default;
    ~AccumulatorWal();

    Ret init(const std::string& path, DataType type, uint64_t dim);
    Ret replay(Accumulator* accumulator);
    Ret append_add_vector(uint64_t id, const uint8_t* data, size_t size);
    Ret append_delete_vector(uint64_t id);
    Ret reset();

private:
    std::string path_;
    int fd_ = -1;
    DataType type_ = DataType::f32;
    uint64_t dim_ = 0;
    size_t vector_size_ = 0;

    Ret open_file_();
    Ret load_or_create_header_();
    Ret write_header_();
    Ret append_record_(WalOp op, uint64_t id, const uint8_t* payload, size_t payload_size);
    Ret replay_(Accumulator* accumulator);
};

} // namespace sketch2
