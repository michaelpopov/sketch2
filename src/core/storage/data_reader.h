#pragma once
#include "utils/shared_types.h"
#include "core/storage/data_file.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sketch2 {

enum class ReaderMode {
    InPlace,   // modified bit set means vector is deleted
    Reference, // modified bit set means first 8 bytes are a pointer (nullptr = deleted)
};

class DataReader {
public:
    class Iterator {
    public:
        void           next();
        bool           eof()  const;
        const uint8_t* data() const;
        uint64_t       id()   const;

    private:
        friend class DataReader;
        Iterator(const DataReader* reader, size_t index, const uint64_t* ids);

        const DataReader*  reader_ = nullptr;
        size_t             index_  = 0;
        const uint64_t*    ids_    = nullptr; // cached pointer to the id array
    };

    ~DataReader();

    Ret init(const std::string& path,
            ReaderMode mode = ReaderMode::InPlace,
            std::unique_ptr<DataReader> delta = nullptr);

    DataType type()  const;
    size_t   dim()   const;
    size_t   size()  const; // size of one vector in bytes
    size_t   count() const; // number of vectors

    Iterator       begin() const;
    uint64_t       id(size_t index) const;
    const uint8_t* get(uint64_t id) const;   // lookup by vector id
    const uint8_t* at(size_t index) const;   // lookup by position
    bool           is_deleted(size_t index) const;

    size_t deleted_count() const { return hdr_->deleted_count; }
    uint64_t deleted_id(size_t index) const;

    bool check_consistency() const;

private:
    uint8_t*                 map_     = nullptr;
    size_t                   map_len_ = 0;
    const DataFileHeader*    hdr_     = nullptr;
    const uint64_t*          ids_     = nullptr; // cached pointer to the ids section
    const uint64_t*          deleted_ids_ = nullptr; // cached pointer to the deleted ids section
    DataType                 type_    = DataType::f32;
    size_t                   size_    = 0;        // size of one vector in bytes
    ReaderMode               mode_    = ReaderMode::InPlace;
    std::vector<bool>  bitset_;
    std::unique_ptr<DataReader> delta_;

    Ret init_(const std::string &path, ReaderMode mode, std::unique_ptr<DataReader> delta);
    Ret init_deleted_bitset();
    Ret init_updates_override();
};

} // namespace sketch2
