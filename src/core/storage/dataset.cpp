#include "dataset.h"
#include "core/storage/data_reader.h"
#include "core/storage/data_merger.h"
#include "core/storage/data_writer.h"
#include "core/storage/input_reader.h"
#include "utils/ini_reader.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <filesystem>
#include <experimental/scope>
#include <unordered_map>

namespace sketch2 {

static const std::string kTempExt = ".temp";
static const std::string kDataExt = ".data";
static const std::string kDeltaExt = ".delta";
static const std::string kMergeExt = ".merge";

Ret Dataset::init(const std::vector<std::string>& dirs, uint64_t range_size, DataType type, uint64_t dim) {
    if (!dirs_.empty()) {
        return Ret("Dataset is already initialized.");
    }
    if (dirs.empty()) {
        return Ret("Dataset: dirs must not be empty.");
    }
    if (range_size == 0) {
        return Ret("Dataset: range_size must be > 0.");
    }
    if (dim < 4) {
        return Ret("Dataset: dim must be >= 4.");
    }
    dirs_       = dirs;
    range_size_ = range_size;
    type_ = type;
    dim_ = dim;
    return Ret(0);
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
    if (!dirs_.empty()) {
        return Ret("Dataset is already initialized.");
    }

    IniReader cfg;
    CHECK(cfg.init(path));

    auto dirs = cfg.get_str_list("dataset.dirs");
    if (dirs.empty()) {
        dirs = cfg.get_str_list("dirs");
    }

    static const int kRangeSize = 10'000;
    int range_size = cfg.get_int("dataset.range_size", kRangeSize);

    int dim = cfg.get_int("dataset.dim", 0);

    std::string type_str = cfg.get_str("dataset.type", "f32");
    DataType type = data_type_from_string(type_str);
    validate_type(type);

    return init(dirs, static_cast<uint64_t>(range_size), type, static_cast<uint64_t>(dim));
}

Ret Dataset::store(const std::string& input_path) {
    try {
        return store_(input_path);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

Ret Dataset::store_(const std::string& input_path) {
    if (dirs_.empty() || range_size_ == 0) {
        return Ret("Dataset: not initialized.");
    }

    InputReader reader;
    CHECK(reader.init(input_path));

    if (dim_ != reader.dim()) {
        return Ret("Dataset: mismatched dim");
    }

    if (type_ != reader.type()) {
        return Ret("Dataset: mismatched type");
    }

    if (reader.count() == 0) {
        return Ret(0);
    }

    const uint64_t min_id = reader.id(0);
    const uint64_t max_id = reader.id(reader.count() - 1);

    const uint64_t first_file = min_id / range_size_;
    const uint64_t last_file  = max_id / range_size_;

    for (uint64_t file_id = first_file; file_id <= last_file; ++file_id) {
        const uint64_t range_start = file_id * range_size_;
        const uint64_t range_end   = range_start + range_size_;

        if (reader.is_range_present(range_start, range_end)) {
            CHECK(store_and_merge(reader, file_id, range_start, range_end));
        }

    }

    return Ret(0);
}

Ret Dataset::store_and_merge(const InputReader& reader, uint64_t file_id, uint64_t range_start, uint64_t range_end) {
    const size_t dir_id = file_id % dirs_.size();
    const std::string& dir = dirs_[dir_id];
    const std::string output_path_base = dir + "/" + std::to_string(file_id);
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

bool Dataset::check_data_file_merge(const DataReader& data_reader, const DataReader& output_reader) {
    constexpr uint64_t kDataMergeRatio = 2;
    return (data_reader.count() < (output_reader.count() + output_reader.deleted_count()) * kDataMergeRatio);
}

bool Dataset::check_data_delta_merge(const DataReader& data_reader, const DataReader& delta_reader) {
    constexpr uint64_t kDataMergeRatio = 2;
    return (data_reader.count() < (delta_reader.count() + delta_reader.deleted_count()) * kDataMergeRatio);
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
    const auto ret = result->init(dirs_);
    if (ret != 0) {
        throw std::runtime_error(ret.message());
    }
    return result;
}

/***********************************************************
 *  DatasetReader 
 */
Ret DatasetReader::init(const std::vector<std::string>& dirs) {
    if (dirs.empty()) {
        return Ret("DatasetReader::init: dirs are not set");
    }
    if (!dirs_.empty()) {
        return Ret("DatasetReader::init: already initialized");
    }

    dirs_ = dirs;

    // Query all directories defined in dirs_.
    // File all files with names that match pattern <id>.data and <id>.delta.
    // Create Item for each pair of such files with matching id.
    // Add this Item to items_.
    auto parse_id = [](const std::string& name, const std::string& ext, uint64_t* out) -> bool {
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
    };

    std::unordered_map<uint64_t, Item> items_map;

    for (const std::string& dir : dirs_) {
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
            if (parse_id(file_name, kDataExt, &file_id)) {
                Item& item = items_map[file_id];
                item.id = file_id;
                if (!item.data_file_path.empty()) {
                    return Ret("DatasetReader::init: duplicate data file id " + std::to_string(file_id));
                }
                item.data_file_path = file_path;
                continue;
            }

            if (parse_id(file_name, kDeltaExt, &file_id)) {
                Item& item = items_map[file_id];
                item.id = file_id;
                if (!item.delta_file_path.empty()) {
                    return Ret("DatasetReader::init: duplicate delta file id " + std::to_string(file_id));
                }
                item.delta_file_path = file_path;
                continue;
            }
        }
    }

    std::vector<Item> sorted_items;
    sorted_items.reserve(items_map.size());
    for (const auto& [file_id, item] : items_map) {
        if (item.data_file_path.empty()) {
            return Ret("DatasetReader::init: missing data file for id " + std::to_string(file_id));
        }
        sorted_items.push_back(item);
    }

    std::sort(sorted_items.begin(), sorted_items.end(),
        [](const Item& lhs, const Item& rhs) {
            return lhs.id < rhs.id;
        });

    items_ = std::move(sorted_items);

    return 0;
}

std::pair<DataReaderPtr, Ret> DatasetReader::next() {
    ++current_;
    if (static_cast<size_t>(current_) >= items_.size()) {
        return {nullptr, Ret(0)};
    }

    const Item& item = items_[current_];
    DataReaderPtr reader = std::make_unique<DataReader>();
    Ret ret(0);

    if (item.delta_file_path.empty()) {
        ret = reader->init(item.data_file_path);
    } else {
        DataReaderPtr delta_reader = std::make_unique<DataReader>();
        ret = delta_reader->init(item.delta_file_path);
        if (ret != 0) {
            return {nullptr, ret};
        }
        ret = reader->init(item.data_file_path, std::move(delta_reader));
    }

    if (ret != 0) {
        return {nullptr, ret};
    }

    return {std::move(reader), Ret(0)};
}
    
} // namespace sketch2
