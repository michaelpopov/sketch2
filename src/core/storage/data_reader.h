#pragma once
#include "utils/shared_types.h"
#include "core/storage/data_file.h"
#include <cstdint>
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
        void           dont_need() const;

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
             const std::vector<bool>* bitset = nullptr);

    DataType type()  const;
    size_t   dim()   const;
    size_t   size()  const; // size of one vector in bytes
    size_t   count() const; // number of vectors

    Iterator       begin() const;
    uint64_t       id(size_t index) const;
    const uint8_t* get(uint64_t id) const;   // lookup by vector id
    const uint8_t* at(size_t index) const;   // lookup by position
    void           dont_need(size_t index) const;
    bool           is_deleted(size_t index) const;
    size_t deleted_count() const { return hdr_->deleted_count; }
    uint64_t deleted_id(size_t index) const;

private:
    const uint8_t*           map_     = nullptr;
    size_t                   map_len_ = 0;
    const DataFileHeader*    hdr_     = nullptr;
    const uint64_t*          ids_     = nullptr; // cached pointer to the ids section
    const uint64_t*          deleted_ids_ = nullptr; // cached pointer to the deleted ids section
    DataType                 type_    = DataType::f32;
    size_t                   size_    = 0;        // size of one vector in bytes
    ReaderMode               mode_    = ReaderMode::InPlace;
    const std::vector<bool>* bitset_  = nullptr; // optional, not owned

    Ret init_(const std::string &path, ReaderMode mode, const std::vector<bool> *bitset);
};

} // namespace sketch2
