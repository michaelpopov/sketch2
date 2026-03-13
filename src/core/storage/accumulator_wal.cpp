#include "core/storage/accumulator_wal.h"

#include "core/storage/accumulator.h"
#include "core/utils/shared_consts.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

namespace sketch2 {

namespace {

uint32_t checksum_bytes(uint32_t checksum, const uint8_t* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        checksum ^= data[i];
        checksum *= kFnvPrime;
    }
    return checksum;
}

uint32_t checksum_record(WalOp op, uint64_t id, const uint8_t* payload, size_t payload_size) {
    uint32_t checksum = kFnvOffsetBasis;
    const uint8_t op_byte = static_cast<uint8_t>(op);
    checksum = checksum_bytes(checksum, &op_byte, sizeof(op_byte));
    checksum = checksum_bytes(checksum, reinterpret_cast<const uint8_t*>(&id), sizeof(id));
    checksum = checksum_bytes(checksum, payload, payload_size);
    return checksum;
}

// Retries short writes and EINTR so callers can treat a successful return as a
// fully persisted byte range.
Ret write_all(int fd, const uint8_t* data, size_t size, const std::string& context) {
    size_t written = 0;
    while (written < size) {
        const ssize_t rc = write(fd, data + written, size - written);
        if (rc < 0) {
            if (errno == EINTR) {
                continue;
            }
            return Ret(context + ": " + std::strerror(errno));
        }
        written += static_cast<size_t>(rc);
    }
    return Ret(0);
}

// Reads up to size bytes from a fixed offset while handling EINTR, reporting
// how much data was actually available so WAL replay can trim torn records.
Ret pread_all(int fd, void* data, size_t size, off_t offset, size_t* bytes_read, const std::string& context) {
    uint8_t* out = static_cast<uint8_t*>(data);
    size_t total = 0;
    while (total < size) {
        const ssize_t rc = pread(fd, out + total, size - total, offset + static_cast<off_t>(total));
        if (rc < 0) {
            if (errno == EINTR) {
                continue;
            }
            return Ret(context + ": " + std::strerror(errno));
        }
        if (rc == 0) {
            break;
        }
        total += static_cast<size_t>(rc);
    }
    *bytes_read = total;
    return Ret(0);
}

} // namespace

AccumulatorWal::~AccumulatorWal() {
    if (fd_ >= 0) {
        (void)close(fd_);
    }
}

Ret AccumulatorWal::init(const std::string& path, DataType type, uint64_t dim) {
    if (fd_ >= 0) {
        return Ret("AccumulatorWal: already initialized");
    }

    type_ = type;
    dim_ = dim;
    vector_size_ = static_cast<size_t>(dim) * data_type_size(type);
    path_ = path;

    CHECK(open_file_());
    return load_or_create_header_();
}

// Replays WAL records into the accumulator on startup. The function validates
// record sizes and checksums, truncates incomplete tails left by crashes, and
// then re-applies each logical add/delete in order.
Ret AccumulatorWal::replay(Accumulator* accumulator) {
    if (!accumulator) {
        return Ret("AccumulatorWal::replay: accumulator is null");
    }

    off_t offset = static_cast<off_t>(sizeof(WalFileHeader));
    while (true) {
        WalRecordHeader header{};
        size_t bytes_read = 0;
        CHECK(pread_all(fd_, &header, sizeof(header), offset, &bytes_read,
            "AccumulatorWal::replay: failed to read record header"));
        if (bytes_read == 0) {
            return Ret(0);
        }
        if (bytes_read < sizeof(header)) {
            if (0 != ftruncate(fd_, offset)) {
                return Ret("AccumulatorWal::replay: failed to truncate partial tail: " +
                    std::string(std::strerror(errno)));
            }
            if (lseek(fd_, 0, SEEK_END) < 0) {
                return Ret("AccumulatorWal::replay: failed to seek wal: " +
                    std::string(std::strerror(errno)));
            }
            return Ret(0);
        }
        if (header.size < sizeof(WalRecordHeader)) {
            return Ret("AccumulatorWal::replay: invalid record size");
        }

        const size_t payload_size = header.size - sizeof(WalRecordHeader);
        if (header.op == static_cast<uint8_t>(WalOp::AddVector)) {
            if (payload_size != vector_size_) {
                return Ret("AccumulatorWal::replay: invalid add_vector payload size");
            }
        } else if (header.op == static_cast<uint8_t>(WalOp::DeleteVector)) {
            if (payload_size != 0) {
                return Ret("AccumulatorWal::replay: invalid delete_vector payload size");
            }
        } else {
            return Ret("AccumulatorWal::replay: invalid operation");
        }

        std::vector<uint8_t> payload(payload_size);
        if (payload_size != 0) {
            CHECK(pread_all(fd_, payload.data(), payload_size, offset + static_cast<off_t>(sizeof(header)),
                &bytes_read, "AccumulatorWal::replay: failed to read record payload"));
            if (bytes_read < payload_size) {
                if (0 != ftruncate(fd_, offset)) {
                    return Ret("AccumulatorWal::replay: failed to truncate partial payload: " +
                        std::string(std::strerror(errno)));
                }
                if (lseek(fd_, 0, SEEK_END) < 0) {
                    return Ret("AccumulatorWal::replay: failed to seek wal: " +
                        std::string(std::strerror(errno)));
                }
                return Ret(0);
            }
        }

        const WalOp op = static_cast<WalOp>(header.op);
        if (header.checksum != checksum_record(op, header.id, payload.data(), payload.size())) {
            return Ret("AccumulatorWal::replay: checksum mismatch");
        }

        if (op == WalOp::AddVector) {
            CHECK(accumulator->apply_add_vector_(header.id, payload.data()));
        } else {
            CHECK(accumulator->apply_delete_vector_(header.id));
        }

        offset += static_cast<off_t>(header.size);
    }
}

Ret AccumulatorWal::append_add_vector(uint64_t id, const uint8_t* data, size_t size) {
    if (!data) {
        return Ret("AccumulatorWal::append_add_vector: data is null");
    }
    if (size != vector_size_) {
        return Ret("AccumulatorWal::append_add_vector: invalid vector size");
    }
    return append_record_(WalOp::AddVector, id, data, size);
}

Ret AccumulatorWal::append_delete_vector(uint64_t id) {
    return append_record_(WalOp::DeleteVector, id, nullptr, 0);
}

Ret AccumulatorWal::reset() {
    if (fd_ < 0) {
        return Ret("AccumulatorWal::reset: not initialized");
    }
    if (0 != ftruncate(fd_, static_cast<off_t>(sizeof(WalFileHeader)))) {
        return Ret("AccumulatorWal::reset: failed to truncate wal: " + std::string(std::strerror(errno)));
    }
    if (lseek(fd_, 0, SEEK_END) < 0) {
        return Ret("AccumulatorWal::reset: failed to seek wal: " + std::string(std::strerror(errno)));
    }
    if (0 != fsync(fd_)) {
        return Ret("AccumulatorWal::reset: failed to sync wal: " + std::string(std::strerror(errno)));
    }
    return Ret(0);
}

Ret AccumulatorWal::open_file_() {
    fd_ = open(path_.c_str(), O_RDWR | O_CREAT, 0666);
    if (fd_ < 0) {
        return Ret("AccumulatorWal: failed to open wal file " + path_ + ": " + std::strerror(errno));
    }
    return Ret(0);
}

// Opens an existing WAL header or creates a new one. Existing files are
// validated against the accumulator's type and dimension before appends resume.
Ret AccumulatorWal::load_or_create_header_() {
    const off_t file_size = lseek(fd_, 0, SEEK_END);
    if (file_size < 0) {
        return Ret("AccumulatorWal: failed to stat wal file " + path_ + ": " + std::strerror(errno));
    }

    if (file_size == 0) {
        return write_header_();
    }
    if (file_size < static_cast<off_t>(sizeof(WalFileHeader))) {
        return Ret("AccumulatorWal: wal file is too small");
    }

    WalFileHeader header{};
    size_t bytes_read = 0;
    CHECK(pread_all(fd_, &header, sizeof(header), 0, &bytes_read,
        "AccumulatorWal: failed to read wal header"));
    if (bytes_read != sizeof(header)) {
        return Ret("AccumulatorWal: failed to read wal header");
    }
    if (header.base.magic != kMagic) {
        return Ret("AccumulatorWal: invalid wal magic");
    }
    if (header.base.kind != static_cast<uint16_t>(FileType::Wal)) {
        return Ret("AccumulatorWal: invalid wal kind");
    }
    if (header.base.version != kVersion) {
        return Ret("AccumulatorWal: invalid wal version");
    }
    if (header.type != static_cast<uint16_t>(data_type_to_int(type_))) {
        return Ret("AccumulatorWal: wal type mismatch");
    }
    if (header.dim != dim_) {
        return Ret("AccumulatorWal: wal dim mismatch");
    }
    if (lseek(fd_, 0, SEEK_END) < 0) {
        return Ret("AccumulatorWal: failed to seek wal file " + path_ + ": " + std::strerror(errno));
    }
    return Ret(0);
}

Ret AccumulatorWal::write_header_() {
    WalFileHeader header{};
    header.base.magic = kMagic;
    header.base.kind = static_cast<uint16_t>(FileType::Wal);
    header.base.version = kVersion;
    header.type = static_cast<uint16_t>(data_type_to_int(type_));
    header.dim = static_cast<uint16_t>(dim_);
    header.reserved = 0;

    if (lseek(fd_, 0, SEEK_SET) < 0) {
        return Ret("AccumulatorWal: failed to seek wal file " + path_ + ": " + std::strerror(errno));
    }
    CHECK(write_all(fd_, reinterpret_cast<const uint8_t*>(&header), sizeof(header),
        "AccumulatorWal: failed to write wal header"));
    if (0 != fsync(fd_)) {
        return Ret("AccumulatorWal: failed to sync wal header: " + std::string(std::strerror(errno)));
    }
    if (lseek(fd_, 0, SEEK_END) < 0) {
        return Ret("AccumulatorWal: failed to seek wal file " + path_ + ": " + std::strerror(errno));
    }
    return Ret(0);
}

// Appends a checksummed WAL record and forces it to stable storage so the
// accumulator can recover every acknowledged mutation after a crash.
Ret AccumulatorWal::append_record_(WalOp op, uint64_t id, const uint8_t* payload, size_t payload_size) {
    if (fd_ < 0) {
        return Ret("AccumulatorWal: not initialized");
    }

    WalRecordHeader header{};
    header.size = static_cast<uint32_t>(sizeof(WalRecordHeader) + payload_size);
    header.op = static_cast<uint8_t>(op);
    header.id = id;
    header.checksum = checksum_record(op, id, payload, payload_size);
    header.reserved2 = 0;

    CHECK(write_all(fd_, reinterpret_cast<const uint8_t*>(&header), sizeof(header),
        "AccumulatorWal: failed to append record header"));
    if (payload_size != 0) {
        CHECK(write_all(fd_, payload, payload_size, "AccumulatorWal: failed to append record payload"));
    }
    if (0 != fdatasync(fd_)) {
        return Ret("AccumulatorWal: failed to sync wal record: " + std::string(std::strerror(errno)));
    }
    return Ret(0);
}

} // namespace sketch2
