#include "dataset.h"
#include "core/storage/data_file_layout.h"
#include "core/storage/data_reader.h"
#include "core/storage/data_merger.h"
#include "core/storage/data_writer.h"
#include "core/storage/input_reader.h"
#include "core/utils/file_lock.h"
#include "utils/ini_reader.h"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <filesystem>
#include <experimental/scope>
#include <limits>
#include <unordered_map>

namespace sketch2 {

void Dataset::AccumulatorIterator::next() {
    iterator_.next();
}

bool Dataset::AccumulatorIterator::eof() const {
    return iterator_.eof();
}

uint64_t Dataset::AccumulatorIterator::id() const {
    return iterator_.id();
}

const uint8_t* Dataset::AccumulatorIterator::data() const {
    return iterator_.data();
}

namespace {

Ret get_non_negative_ini_u64(const IniReader& cfg, const std::string& name, int def, uint64_t* out) {
    const int value = cfg.get_int(name, def);
    if (value < 0) {
        return Ret("Dataset: " + name + " must be >= 0");
    }
    *out = static_cast<uint64_t>(value);
    return Ret(0);
}

bool parse_dataset_file_id(const std::string& name, const std::string& ext, uint64_t* out) {
    if (name.size() <= ext.size() || name.rfind(ext) != name.size() - ext.size()) {
        return false;
    }

    const std::string id_part = name.substr(0, name.size() - ext.size());
    if (id_part.empty()) {
        return false;
    }

    for (char c : id_part) {
        if (!std::isdigit(static_cast<unsigned char>(c))) {
            return false;
        }
    }

    *out = std::stoull(id_part);
    return true;
}

Ret collect_dataset_items(const DatasetMetadata& metadata, std::vector<DatasetItem>* items) {
    if (metadata.dirs.empty()) {
        return Ret("DatasetReader::init: dirs are not set");
    }

    std::unordered_map<uint64_t, DatasetItem> items_map;

    for (const std::string& dir : metadata.dirs) {
        std::error_code ec;
        if (!std::filesystem::exists(dir, ec) || !std::filesystem::is_directory(dir, ec)) {
            return Ret("DatasetReader::init: invalid directory: " + dir);
        }

        auto dir_iter = std::filesystem::directory_iterator(dir, ec);
        for (; dir_iter != std::filesystem::directory_iterator(); dir_iter.increment(ec)) {
            if (ec) {
                return Ret("DatasetReader::init: failed to iterate directory: " + dir);
            }

            const auto& entry = *dir_iter;
            if (!entry.is_regular_file(ec)) {
                continue;
            }

            const std::string file_name = entry.path().filename().string();
            const std::string file_path = entry.path().string();

            uint64_t file_id = 0;
            if (parse_dataset_file_id(file_name, kDataExt, &file_id)) {
                DatasetItem& item = items_map[file_id];
                item.id = file_id;
                if (!item.data_file_path.empty()) {
                    return Ret("DatasetReader::init: duplicate data file id " + std::to_string(file_id));
                }
                item.data_file_path = file_path;
                continue;
            }

            if (parse_dataset_file_id(file_name, kDeltaExt, &file_id)) {
                DatasetItem& item = items_map[file_id];
                item.id = file_id;
                if (!item.delta_file_path.empty()) {
                    return Ret("DatasetReader::init: duplicate delta file id " + std::to_string(file_id));
                }
                item.delta_file_path = file_path;
                continue;
            }
        }
    }

    std::vector<DatasetItem> sorted_items;
    sorted_items.reserve(items_map.size());
    for (const auto& [file_id, item] : items_map) {
        if (item.data_file_path.empty()) {
            return Ret("DatasetReader::init: missing data file for id " + std::to_string(file_id));
        }
        sorted_items.push_back(item);
    }

    std::sort(sorted_items.begin(), sorted_items.end(),
        [](const DatasetItem& lhs, const DatasetItem& rhs) {
            return lhs.id < rhs.id;
        });

    *items = std::move(sorted_items);
    return Ret(0);
}

std::string dataset_owner_lock_path(const DatasetMetadata& metadata) {
    return metadata.dirs.front() + "/" + kOwnerLockFileName;
}

std::string dataset_accumulator_wal_path(const DatasetMetadata& metadata) {
    return metadata.dirs.front() + "/" + kAccumulatorWalFileName;
}

} // namespace

Dataset::~Dataset() = default;

Ret Dataset::init(const DatasetMetadata& metadata) {
    if (!metadata_.dirs.empty()) {
        return Ret("Dataset is already initialized.");
    }
    if (metadata.dirs.empty()) {
        return Ret("Dataset: dirs must not be empty.");
    }
    if (metadata.range_size == 0) {
        return Ret("Dataset: range_size must be > 0.");
    }
    if (metadata.dim < kMinDimension || metadata.dim > kMaxDimension) {
        return Ret("Dataset: dim must be in range [" +
            std::to_string(kMinDimension) + ", " + std::to_string(kMaxDimension) + "].");
    }
    try {
        validate_type(metadata.type);
        validate_dist_func(metadata.dist_func);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
    metadata_ = metadata;
    invalidate_data_caches_();
    return Ret(0);
}

Ret Dataset::init(const std::vector<std::string>& dirs, uint64_t range_size,
        DataType type, uint64_t dim, uint64_t accumulator_size, DistFunc dist_func) {
    DatasetMetadata metadata;
    metadata.dirs             = dirs;
    metadata.range_size       = range_size;
    metadata.type             = type;
    metadata.dist_func        = dist_func;
    metadata.dim              = dim;
    metadata.accumulator_size = accumulator_size;
    return init(metadata);
}

// Initialize with values from ini file.
Ret Dataset::init(const std::string& path) {
    try {
        return init_(path);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

Ret Dataset::init_(const std::string& path) {
    if (!metadata_.dirs.empty()) {
        return Ret("Dataset is already initialized.");
    }

    IniReader cfg;
    CHECK(cfg.init(path));

    DatasetMetadata metadata;
    metadata.dirs             = cfg.get_str_list("dataset.dirs");
    CHECK(get_non_negative_ini_u64(cfg, "dataset.dim", 0, &metadata.dim));
    CHECK(get_non_negative_ini_u64(cfg, "dataset.range_size", kRangeSize, &metadata.range_size));
    CHECK(get_non_negative_ini_u64(cfg, "dataset.accumulator_size", kAccumulatorBufferSize, &metadata.accumulator_size));

    std::string type_str = cfg.get_str("dataset.type", "f32");
    metadata.type = data_type_from_string(type_str);
    metadata.dist_func = dist_func_from_string(cfg.get_str("dataset.dist_func", "l1"));

    return init(metadata);
}

Ret Dataset::set_guest_mode() {
    if (accumulator_ &&
            (accumulator_->vectors_count() != 0 || accumulator_->deleted_count() != 0)) {
        return Ret("Dataset: cannot switch to guest mode with non-empty accumulator");
    }

    accumulator_.reset();
    owner_lock_.reset();
    mode_ = DatasetMode::Guest;
    return Ret(0);
}

Ret Dataset::store(const std::string& input_path) {
    try {
        CHECK(require_owner_());
        CHECK(ensure_owner_lock_());
        const Ret ret = store_(input_path);
        if (ret.code() == 0) {
            invalidate_data_caches_();
        }
        return ret;
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

Ret Dataset::store_accumulator() {
    try {
        CHECK(require_owner_());
        CHECK(ensure_owner_lock_());
        const Ret ret = store_accumulator_();
        if (ret.code() == 0) {
            invalidate_data_caches_();
        }
        return ret;
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

Ret Dataset::merge() {
    try {
        CHECK(require_owner_());
        CHECK(ensure_owner_lock_());
        const Ret ret = merge_();
        if (ret.code() == 0) {
            invalidate_data_caches_();
        }
        return ret;
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

Ret Dataset::add_vector(uint64_t id, const uint8_t* data) {
    CHECK(require_owner_());
    CHECK(ensure_owner_lock_());
    if (!data) {
        return Ret("Dataset: invalid data argument");
    }
    CHECK(init_accumulator_());
    if (!accumulator_->can_add_vector(id)) {
        CHECK(store_accumulator_());
    }
    return accumulator_->add_vector(id, data);
}

Ret Dataset::delete_vector(uint64_t id) {
    CHECK(require_owner_());
    CHECK(ensure_owner_lock_());
    CHECK(init_accumulator_());
    if (!accumulator_->can_delete_vector(id)) {
        CHECK(store_accumulator_());
    }
    return accumulator_->delete_vector(id);
}

bool Dataset::is_deleted(uint64_t id) const {
    return accumulator_ && accumulator_->is_deleted(id);
}

bool Dataset::is_modified_in_accumulator(uint64_t id) const {
    return accumulator_ && (accumulator_->is_deleted(id) || accumulator_->is_updated(id));
}

Dataset::AccumulatorIterator Dataset::accumulator_begin() const {
    const Ret ret = prepare_read_state();
    if (ret.code() != 0) {
        throw std::runtime_error(ret.message());
    }
    if (!accumulator_) {
        return AccumulatorIterator(Accumulator::Iterator());
    }
    return AccumulatorIterator(accumulator_->begin());
}

Ret Dataset::init_accumulator_() {
    if (accumulator_) {
        return Ret(0);
    }
    if (metadata_.dirs.empty() || metadata_.range_size == 0) {
        return Ret("Dataset: not initialized.");
    }

    accumulator_ = std::make_unique<Accumulator>();
    CHECK(accumulator_->init(metadata_.accumulator_size, metadata_.type, metadata_.dim));
    return accumulator_->attach_wal(dataset_accumulator_wal_path(metadata_));
}

Ret Dataset::store_(const std::string& input_path) {
    if (metadata_.dirs.empty() || metadata_.range_size == 0) {
        return Ret("Dataset: not initialized.");
    }

    InputReader reader;
    CHECK(reader.init(input_path));

    if (metadata_.dim != reader.dim()) {
        return Ret("Dataset: mismatched dim");
    }

    if (metadata_.type != reader.type()) {
        return Ret("Dataset: mismatched type");
    }

    if (reader.count() == 0) {
        return Ret(0);
    }

    const uint64_t min_id = reader.id(0);
    const uint64_t max_id = reader.id(reader.count() - 1);

    const uint64_t first_file = min_id / metadata_.range_size;
    const uint64_t last_file  = max_id / metadata_.range_size;

    for (uint64_t file_id = first_file; file_id <= last_file; ++file_id) {
        const uint64_t range_start = file_id * metadata_.range_size;
        const uint64_t range_end   = range_start + metadata_.range_size;

        if (reader.is_range_present(range_start, range_end)) {
            CHECK(store_and_merge(reader, file_id, range_start, range_end));
        }
    }

    return Ret(0);
}

Ret Dataset::store_accumulator_() {
    if (metadata_.dirs.empty() || metadata_.range_size == 0) {
        return Ret("Dataset: not initialized.");
    }
    CHECK(init_accumulator_());

    const std::vector<uint64_t> vector_ids = accumulator_->get_vector_ids();
    const std::vector<uint64_t> deleted_ids = accumulator_->get_deleted_ids();
    if (vector_ids.empty() && deleted_ids.empty()) {
        accumulator_->clear();
        return Ret(0);
    }

    std::vector<uint64_t> affected_file_ids;
    affected_file_ids.reserve(vector_ids.size() + deleted_ids.size());
    const auto append_file_ids = [this, &affected_file_ids](const std::vector<uint64_t>& ids) {
        for (uint64_t id : ids) {
            const uint64_t file_id = id / metadata_.range_size;
            if (affected_file_ids.empty() || affected_file_ids.back() != file_id) {
                affected_file_ids.push_back(file_id);
            }
        }
    };

    append_file_ids(vector_ids);
    append_file_ids(deleted_ids);
    std::sort(affected_file_ids.begin(), affected_file_ids.end());
    affected_file_ids.erase(std::unique(affected_file_ids.begin(), affected_file_ids.end()), affected_file_ids.end());

    for (uint64_t file_id : affected_file_ids) {
        const uint64_t range_start = file_id * metadata_.range_size;
        const uint64_t range_end = range_start + metadata_.range_size;
        const auto vectors_begin = std::lower_bound(vector_ids.begin(), vector_ids.end(), range_start);
        const auto vectors_end = std::lower_bound(vector_ids.begin(), vector_ids.end(), range_end);
        const auto deleted_begin = std::lower_bound(deleted_ids.begin(), deleted_ids.end(), range_start);
        const auto deleted_end = std::lower_bound(deleted_ids.begin(), deleted_ids.end(), range_end);

        const std::vector<uint64_t> range_ids(vectors_begin, vectors_end);
        const std::vector<uint64_t> range_deleted_ids(deleted_begin, deleted_end);
        CHECK(store_and_merge_accumulator(file_id, range_ids, range_deleted_ids));
    }

    CHECK(accumulator_->reset_wal());
    accumulator_->clear();
    return Ret(0);
}

Ret Dataset::merge_() {
    if (metadata_.dirs.empty() || metadata_.range_size == 0) {
        return Ret("Dataset: not initialized.");
    }

    std::vector<DatasetItem> items;
    CHECK(collect_dataset_items(metadata_, &items));

    for (const DatasetItem& item : items) {
        if (item.delta_file_path.empty()) {
            continue;
        }

        const size_t dir_id = item.id % metadata_.dirs.size();
        const std::string& dir = metadata_.dirs[dir_id];
        const std::string output_path_base = dir + "/" + std::to_string(item.id);
        FileLockGuard file_lock;
        CHECK(file_lock.lock(output_path_base + kLockExt));

        DataReader data_reader;
        CHECK(data_reader.init(item.data_file_path));

        DataReader delta_reader;
        CHECK(delta_reader.init(item.delta_file_path));

        CHECK(merge_data_file(data_reader, delta_reader, output_path_base, kDeltaExt));
    }

    return Ret(0);
}

Ret Dataset::store_and_merge(const InputReader& reader, uint64_t file_id, uint64_t range_start, uint64_t range_end) {
    const size_t dir_id = file_id % metadata_.dirs.size();
    const std::string& dir = metadata_.dirs[dir_id];
    const std::string output_path_base = dir + "/" + std::to_string(file_id);
    FileLockGuard file_lock;
    CHECK(file_lock.lock(output_path_base + kLockExt));

    const std::string output_path = output_path_base + kTempExt;

    // Clear temporary file on exit
    std::experimental::scope_exit file_guard([output_path]() {
        if (std::filesystem::exists(output_path)) {
            std::filesystem::remove(output_path);
        }
    });

    {
        InputReaderView view(reader, range_start, range_end);
        DataWriter writer;
        CHECK(writer.load(view, output_path));
    }

    DataReader output_reader;
    CHECK(output_reader.init(output_path));

    // If data file doesn't exist, then the system in in the "initial stage".
    // Just "promote" the temp file to become a data file.
    const std::string data_path = output_path_base + kDataExt;
    if (!std::filesystem::exists(data_path)) {
        if (output_reader.deleted_count() != 0) {
            return Ret("Dataset::store_and_merge: invalid deleted items");
        }
        std::filesystem::rename(output_path, data_path);
        return Ret(0);
    }

    const std::string delta_path = output_path_base + kDeltaExt;
    if (!std::filesystem::exists(delta_path)) {
        // If data file exists but delta file doesn't exist, check if the temp file
        // contains enough data to justify a merge with the data file. If that's the
        // case, then merge temp file into a data file.
        {
            DataReader data_reader;
            CHECK(data_reader.init(data_path));

            const bool is_merge = check_data_file_merge(data_reader, output_reader);
            if (is_merge) {
                CHECK(merge_data_file(data_reader, output_reader, output_path_base, kTempExt));
                return Ret(0);
            }
        }

        // If temp file doesn't contain enough data to justify merging with the
        // data file, then just "propmote" the temp file to become a delta file.
        std::filesystem::rename(output_path, delta_path);
        return Ret(0);
    }

    // If delta file exists, merge the temp file into the delta file.
    {
        DataReader delta_reader;
        CHECK(delta_reader.init(delta_path));
        CHECK(merge_delta_file(delta_reader, output_reader, output_path_base));
    }

    // Check if the delta file becomes large enough after the merge to justify
    // merging into a data file, do the merge.
    {
        DataReader data_reader;
        CHECK(data_reader.init(data_path));
        DataReader delta_reader;
        CHECK(delta_reader.init(delta_path));

        const bool is_data_delta_merge = check_data_delta_merge(data_reader, delta_reader);
        if (is_data_delta_merge) {
            CHECK(merge_data_file(data_reader, delta_reader, output_path_base, kDeltaExt));
        }
    }

    return Ret(0);
}

Ret Dataset::store_and_merge_accumulator(uint64_t file_id, const std::vector<uint64_t>& ids, const std::vector<uint64_t>& deleted_ids) {
    const size_t dir_id = file_id % metadata_.dirs.size();
    const std::string& dir = metadata_.dirs[dir_id];
    const std::string output_path_base = dir + "/" + std::to_string(file_id);
    FileLockGuard file_lock;
    CHECK(file_lock.lock(output_path_base + kLockExt));
    if (ids.empty() && deleted_ids.empty()) {
        return Ret(0);
    }

    const auto write_from_accumulator = [this, &ids, &deleted_ids](const std::string& path) -> Ret {
        uint64_t min_id = ids.empty() ? 0 : ids.front();
        uint64_t max_id = ids.empty() ? 0 : ids.back();

        DataFileHeader hdr = make_data_header(
            min_id,
            max_id,
            static_cast<uint32_t>(ids.size()),
            static_cast<uint32_t>(deleted_ids.size()),
            metadata_.type,
            static_cast<uint16_t>(metadata_.dim));

        FILE* f = fopen(path.c_str(), "wb");
        if (!f) {
            return Ret("Dataset::store_and_merge_accumulator: failed to open output file: " + path);
        }

        std::vector<char> file_buffer(kFileBufferSize);
        (void)setvbuf(f, file_buffer.data(), _IOFBF, file_buffer.size());

        std::experimental::scope_exit file_guard([&f]() {
            if (f) fclose(f);
        });

        CHECK(write_header_and_data_padding(f, hdr, "Dataset::store_and_merge_accumulator"));

        const size_t vec_size = static_cast<size_t>(metadata_.dim) * data_type_size(metadata_.type);
        const IdsLayout ids_layout = compute_ids_layout(hdr, ids.size(), vec_size);
        for (uint64_t id : ids) {
            const uint8_t* data = accumulator_->get_vector(id);
            if (!data) {
                return Ret("Dataset::store_and_merge_accumulator: missing vector for id " + std::to_string(id));
            }
            if (fwrite(data, vec_size, 1, f) != 1) {
                return Ret("Dataset::store_and_merge_accumulator: failed to write vector data for id " + std::to_string(id));
            }
        }

        CHECK(write_zero_padding(f, ids_layout.ids_padding,
            "Dataset::store_and_merge_accumulator: failed to write id alignment padding"));
        CHECK(write_u64_array(f, ids, "Dataset::store_and_merge_accumulator: failed to write ids"));
        CHECK(write_u64_array(f, deleted_ids, "Dataset::store_and_merge_accumulator: failed to write deleted_ids"));

        const int n1 = fflush(f);
        const int n2 = fsync(fileno(f));
        const int n3 = fclose(f);
        f = nullptr;
        if (n1 != 0 || n2 != 0 || n3 != 0) {
            return Ret("Dataset::store_and_merge_accumulator: failed to flush and close file");
        }

        return Ret(0);
    };

    const std::string data_path = output_path_base + kDataExt;
    if (!std::filesystem::exists(data_path)) {
        if (ids.empty()) {
            return Ret(0);
        }
        CHECK(write_from_accumulator(data_path));
        return Ret(0);
    }

    const std::string delta_path = output_path_base + kDeltaExt;
    if (!std::filesystem::exists(delta_path)) {
        DataReader data_reader;
        CHECK(data_reader.init(data_path));

        const uint64_t output_count = static_cast<uint64_t>(ids.size() + deleted_ids.size());
        const bool is_merge = (data_reader.count() < output_count * metadata_.data_merge_ratio);
        if (is_merge) {
            DataMerger processor;
            const std::string merge_path = output_path_base + kMergeExt;
            CHECK(processor.merge_data_file(data_reader, *accumulator_, ids, deleted_ids, merge_path));
            std::filesystem::rename(merge_path, data_path);
            return Ret(0);
        }

        CHECK(write_from_accumulator(delta_path));
        return Ret(0);
    }

    {
        DataReader delta_reader;
        CHECK(delta_reader.init(delta_path));
        DataMerger processor;
        const std::string merge_path = output_path_base + kMergeExt;
        CHECK(processor.merge_delta_file(delta_reader, *accumulator_, ids, deleted_ids, merge_path));
        std::filesystem::rename(merge_path, delta_path);
    }

    {
        DataReader data_reader;
        CHECK(data_reader.init(data_path));
        DataReader delta_reader;
        CHECK(delta_reader.init(delta_path));

        const bool is_data_delta_merge = check_data_delta_merge(data_reader, delta_reader);
        if (is_data_delta_merge) {
            CHECK(merge_data_file(data_reader, delta_reader, output_path_base, kDeltaExt));
        }
    }

    return Ret(0);
}

bool Dataset::check_data_file_merge(const DataReader& data_reader, const DataReader& output_reader) {
    const uint64_t output_count = output_reader.count() + output_reader.deleted_count();
    return (data_reader.count() < output_count * metadata_.data_merge_ratio);
}

bool Dataset::check_data_delta_merge(const DataReader& data_reader, const DataReader& delta_reader) {
    const uint64_t delta_count = delta_reader.count() + delta_reader.deleted_count();
    return (data_reader.count() < delta_count * metadata_.data_merge_ratio);
}

Ret Dataset::require_owner_() const {
    if (mode_ == DatasetMode::Guest) {
        return Ret("Dataset: guest mode is read-only");
    }
    return Ret(0);
}

Ret Dataset::ensure_owner_lock_() {
    if (owner_lock_ || metadata_.dirs.empty()) {
        return Ret(0);
    }

    owner_lock_ = std::make_unique<FileLockGuard>();
    return owner_lock_->lock(dataset_owner_lock_path(metadata_));
}

Ret Dataset::merge_data_file(const DataReader& data_reader, const DataReader& output_reader,
        const std::string& output_path_base, const std::string& ext) {

    const std::string source_path = output_path_base + ext;
    std::experimental::scope_exit file_guard([source_path]() {
        if (std::filesystem::exists(source_path)) {
            std::filesystem::remove(source_path);
        }
    });

    DataMerger processor;
    const std::string merge_path = output_path_base + kMergeExt;
    CHECK(processor.merge_data_file(data_reader, output_reader, merge_path));

    const std::string data_path = output_path_base + kDataExt;
    std::filesystem::rename(merge_path, data_path);

    return Ret(0);
}

Ret  Dataset::merge_delta_file(const DataReader& delta_reader, const DataReader& output_reader, const std::string& output_path_base) {
    const std::string source_path = output_path_base + kTempExt;
    std::experimental::scope_exit file_guard([source_path]() {
        if (std::filesystem::exists(source_path)) {
            std::filesystem::remove(source_path);
        }
    });

    DataMerger processor;
    const std::string merge_path = output_path_base + kMergeExt;
    CHECK(processor.merge_delta_file(delta_reader, output_reader, merge_path));
    
    const std::string delta_path = output_path_base + kDeltaExt;
    std::filesystem::rename(merge_path, delta_path);

    return Ret(0);
}

DatasetReaderPtr Dataset::reader() const {
    DatasetReaderPtr result = std::make_unique<DatasetReader>();
    const Ret cache_ret = ensure_items_cache_();
    if (cache_ret.code() != 0) {
        throw std::runtime_error(cache_ret.message());
    }
    const auto ret = result->init(this, items_cache_);
    if (ret.code() != 0) {
        throw std::runtime_error(ret.message());
    }
    return result;
}

std::pair<DataReaderPtr, Ret> Dataset::get(uint64_t id) const {
    const Ret ret = ensure_items_cache_();
    if (ret.code() != 0) {
        return {nullptr, ret};
    }
    const DatasetItem* item = find_item_(id / metadata_.range_size);
    if (!item) {
        return {nullptr, Ret(0)};
    }
    return get_cached_reader_(*item);
}

Ret Dataset::prepare_read_state() const {
    if (mode_ == DatasetMode::Guest || accumulator_) {
        return Ret(0);
    }

    auto* self = const_cast<Dataset*>(this);
    CHECK(self->ensure_owner_lock_());
    return self->init_accumulator_();
}

std::pair<const uint8_t*, Ret> Dataset::get_vector(uint64_t id) const {
    const Ret read_state_ret = prepare_read_state();
    if (read_state_ret.code() != 0) {
        return {nullptr, read_state_ret};
    }

    if (accumulator_) {
        if (accumulator_->is_deleted(id)) {
            return {nullptr, Ret(0)};
        }
        const uint8_t* data = accumulator_->get_vector(id);
        if (data) {
            return {data, Ret(0)};
        }
    }

    auto [reader, ret] = get(id);
    if (ret.code() != 0) {
        return {nullptr, ret};
    }
    if (!reader) {
        return {nullptr, Ret(0)};
    }

    return {reader->get(id), Ret(0)};
}

/***********************************************************
 *  DatasetReader 
 */
Ret DatasetReader::init(const Dataset* dataset, std::vector<DatasetItem> items) {
    if (!dataset) {
        return Ret("DatasetReader::init: dataset is null");
    }
    dataset_ = dataset;
    items_ = std::move(items);
    current_ = -1;
    return Ret(0);
}

std::pair<DataReaderPtr, Ret> DatasetReader::next() {
    ++current_;
    if (static_cast<size_t>(current_) >= items_.size()) {
        return {nullptr, Ret(0)};
    }

    return dataset_->get_cached_reader_(items_[current_]);
}
    
std::pair<DataReaderPtr, Ret> DatasetReader::get(uint64_t id) {
    if (!dataset_) {
        return {nullptr, Ret("DatasetReader::get: reader is not initialized")};
    }

    const uint64_t file_id = id / dataset_->metadata_.range_size;
    auto it = std::lower_bound(items_.begin(), items_.end(), file_id,
        [](const DatasetItem& item, uint64_t value) {
            return item.id < value;
        });

    if (it == items_.end() || it->id != file_id) {
        return {nullptr, Ret(0)};
    }

    return dataset_->get_cached_reader_(*it);
}

Ret Dataset::ensure_items_cache_() const {
    if (items_cache_valid_) {
        return Ret(0);
    }
    CHECK(collect_dataset_items(metadata_, &items_cache_));
    items_cache_valid_ = true;
    return Ret(0);
}

const DatasetItem* Dataset::find_item_(uint64_t file_id) const {
    auto it = std::lower_bound(items_cache_.begin(), items_cache_.end(), file_id,
        [](const DatasetItem& item, uint64_t value) {
            return item.id < value;
        });
    if (it == items_cache_.end() || it->id != file_id) {
        return nullptr;
    }
    return &(*it);
}

std::pair<DataReaderPtr, Ret> Dataset::get_cached_reader_(const DatasetItem& item) const {
    const auto cache_it = reader_cache_.find(item.id);
    if (cache_it != reader_cache_.end()) {
        return {cache_it->second, Ret(0)};
    }

    DataReaderPtr reader = std::make_shared<DataReader>();
    Ret ret(0);

    if (item.delta_file_path.empty()) {
        ret = reader->init(item.data_file_path);
    } else {
        auto delta_reader = std::make_unique<DataReader>();
        ret = delta_reader->init(item.delta_file_path);
        if (ret.code() != 0) {
            return {nullptr, ret};
        }
        ret = reader->init(item.data_file_path, std::move(delta_reader));
    }

    if (ret.code() != 0) {
        return {nullptr, ret};
    }

    reader_cache_[item.id] = reader;
    return {reader, Ret(0)};
}

void Dataset::invalidate_data_caches_() {
    items_cache_valid_ = false;
    items_cache_.clear();
    reader_cache_.clear();
}

} // namespace sketch2
