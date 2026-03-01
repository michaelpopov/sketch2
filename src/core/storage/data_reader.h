#pragma once
#include "utils/shared_types.h"
#include "core/storage/data_file.h"
#include <cstdint>
#include <unordered_map>
#include <memory>
#include <string>
#include <vector>

namespace sketch2 {

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

    Ret init(const std::string& path, std::unique_ptr<DataReader> delta = nullptr);

    DataType type()  const;
    uint16_t dim()   const;
    uint16_t size()  const; // size of one vector in bytes
    size_t   count() const; // number of vectors

    Iterator       begin() const;
    uint64_t       id(size_t index) const;
    const uint8_t* get(uint64_t id) const;   // lookup by vector id
    const uint8_t* at(size_t index) const;   // lookup by position; might return nullptr if the vector is deleted
    bool           is_deleted(size_t index) const;

    size_t deleted_count() const { return hdr_->deleted_count; }
    uint64_t deleted_id(size_t index) const;

    bool check_consistency() const;
    bool has_delta() const { return delta_ != nullptr; }

private:
    const uint8_t*           map_     = nullptr;
    size_t                   map_len_ = 0;
    const DataFileHeader*    hdr_     = nullptr;
    const uint64_t*          ids_     = nullptr; // cached pointer to the ids section
    const uint64_t*          deleted_ids_ = nullptr; // cached pointer to the deleted ids section
    DataType                 type_    = DataType::f32;
    uint16_t                 size_    = 0;        // size of one vector in bytes

    std::vector<bool>  bitset_;
    std::unordered_map<uint64_t, uint32_t> mods_;
    std::unique_ptr<DataReader> delta_;

    Ret init_(const std::string &path, std::unique_ptr<DataReader> delta);
    Ret init_dels();
    Ret init_mods();
    const uint8_t* get_by_pos(uint32_t pos) { return map_ + sizeof(DataFileHeader) + size_ * pos; }
};

} // namespace sketch2
