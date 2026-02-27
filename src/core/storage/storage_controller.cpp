#include "storage_controller.h"
#include "core/storage/data_reader.h"
#include "core/storage/data_merger.h"
#include "core/storage/data_writer.h"
#include "core/storage/input_reader.h"
#include <cassert>
#include <filesystem>
#include <experimental/scope>

namespace sketch2 {

static const std::string kTempExt = ".temp";
static const std::string kDataExt = ".data";
static const std::string kDeltaExt = ".delta";
static const std::string kMergeExt = ".merge";

Ret StorageController::init(const std::vector<std::string>& dirs, uint64_t range_size) {
    if (!dirs_.empty()) {
        return Ret("StorageController is already initialized.");
    }
    if (dirs.empty()) {
        return Ret("StorageController: dirs must not be empty.");
    }
    if (range_size == 0) {
        return Ret("StorageController: range_size must be > 0.");
    }
    dirs_       = dirs;
    range_size_ = range_size;
    return Ret(0);
}

Ret StorageController::load(const std::string& input_path) {
    try {
        return load_(input_path);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
}

Ret StorageController::load_(const std::string& input_path) {
    if (dirs_.empty() || range_size_ == 0) {
        return Ret("StorageController: not initialized.");
    }

    InputReader reader;
    CHECK(reader.init(input_path));

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
            CHECK(load_and_merge(reader, file_id, range_start, range_end));
        }

    }

    return Ret(0);
}

Ret StorageController::load_and_merge(const InputReader& reader, uint64_t file_id, uint64_t range_start, uint64_t range_end) {
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
            return Ret("StorageController::load_and_merge: invalide deleted items");
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

bool StorageController::check_data_file_merge(const DataReader& data_reader, const DataReader& output_reader) {
    constexpr uint64_t kDataMergeRatio = 2;
    return (data_reader.count() < (output_reader.count() + output_reader.deleted_count()) * kDataMergeRatio);
}

bool StorageController::check_data_delta_merge(const DataReader& data_reader, const DataReader& delta_reader) {
    constexpr uint64_t kDataMergeRatio = 2;
    return (data_reader.count() < (delta_reader.count() + delta_reader.deleted_count()) * kDataMergeRatio);
}

Ret StorageController::merge_data_file(const DataReader& data_reader, const DataReader& output_reader,
        const std::string& output_path_base, const std::string& ext) {
    const std::string source_path = output_path_base + ext;
    if (std::filesystem::exists(source_path)) {
        std::filesystem::remove(source_path);
    }

    DataMerger processor;
    const std::string merge_path = output_path_base + kMergeExt;
    CHECK(processor.merge_data_file(data_reader, output_reader, merge_path));

    const std::string data_path = output_path_base + kDataExt;
    std::filesystem::rename(merge_path, data_path);

    return Ret(0);
}

Ret  StorageController::merge_delta_file(const DataReader& delta_reader, const DataReader& output_reader, const std::string& output_path_base) {
    const std::string source_path = output_path_base + kTempExt;
    if (std::filesystem::exists(source_path)) {
        std::filesystem::remove(source_path);
    }

    DataMerger processor;
    const std::string merge_path = output_path_base + kMergeExt;
    CHECK(processor.merge_delta_file(delta_reader, output_reader, merge_path));
    
    const std::string delta_path = output_path_base + kDeltaExt;
    std::filesystem::rename(merge_path, delta_path);

    return Ret(0);
}

} // namespace sketch2
