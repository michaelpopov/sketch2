#include "storage_controller.h"
#include "core/storage/data_reader.h"
#include "core/storage/data_merger.h"
#include "core/storage/data_writer.h"
#include "core/storage/input_reader.h"
#include "filesystem"

namespace sketch2 {

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

        if (!reader.is_range_present(range_start, range_end)) {
            continue;
        }

        const size_t dir_id = file_id % dirs_.size();
        const std::string& dir = dirs_[dir_id];
        const std::string output_path_base = dir + "/" + std::to_string(file_id);
        const std::string output_path = output_path_base + ".temp";

        InputReaderView view(reader, range_start, range_end);
        DataWriter writer;
        CHECK(writer.load(view, output_path));

        const std::string data_path = dir + "/" + std::to_string(file_id) + ".data";
        if (!std::filesystem::exists(data_path)) {
            CHECK(valid_data_file(output_path));
            std::filesystem::rename(output_path, data_path);
            continue;
        }

        const std::string delta_path = dir + "/" + std::to_string(file_id) + ".delta";
        if (!std::filesystem::exists(delta_path)) {
            const bool is_merge = check_data_file_merge(data_path, output_path);
            if (is_merge) {
                CHECK(merge_data_file(data_path, output_path));
                continue;
            }

            std::filesystem::rename(output_path, delta_path);
            continue;
        }

        CHECK(merge_delta_file(delta_path, output_path));

        const bool is_data_delta_merge = check_data_delta_merge(data_path, delta_path);
        if (is_data_delta_merge) {
            CHECK(merge_data_delta_file(data_path, delta_path));
        }
    }

    return Ret(0);
}

bool StorageController::valid_data_file(const std::string& output_path) {
    DataReader reader;
    const auto ret = reader.init(output_path);
    if (ret != 0) {
        throw std::runtime_error(ret.message());
    }

    if (reader.deleted_count() > 0) {
        return false;
    }

    return true;
}

bool StorageController::check_data_file_merge(const std::string& data_path, const std::string& output_path) {
    constexpr uint64_t kDataMergeRatio = 2;
    return check_merge(data_path, output_path, kDataMergeRatio);
}

Ret  StorageController::merge_data_file(const std::string& data_path, const std::string& output_path) {
    DataMerger processor;
    const auto ret = processor.init(data_path, output_path);
    if (ret != 0) {
        return ret;
    }

    return processor.merge_data_files("");
}

Ret  StorageController::merge_delta_file(const std::string& delta_path, const std::string& output_path) {
    DataMerger processor;
    const auto ret = processor.init(delta_path, output_path);
    if (ret != 0) {
        return ret;
    }

    return processor.merge_delta_file("");
}

bool StorageController::check_data_delta_merge(const std::string& data_path, const std::string& delta_path) {
    constexpr uint64_t kDeltaMergeRatio = 2;
    return check_merge(data_path, delta_path, kDeltaMergeRatio);
}

Ret  StorageController::merge_data_delta_file(const std::string& data_path, const std::string& delta_path) {
    DataMerger processor;
    const auto ret = processor.init(data_path, delta_path);
    if (ret != 0) {
        return ret;
    }

    return processor.merge_data_delta_files("");
}

bool StorageController::check_merge(const std::string& source_path, const std::string& update_path, uint64_t ratio) {
    DataReader source_reader;
    const auto source_ret = source_reader.init(source_path);
    if (source_ret != 0) {
        throw std::runtime_error(source_ret.message());
    }

    DataReader update_reader;
    const auto update_ret = update_reader.init(update_path);
    if (update_ret != 0) {
        throw std::runtime_error(update_ret.message());
    }

    return (source_reader.count() < (update_reader.count() + update_reader.deleted_count()) * ratio);
}


} // namespace sketch2
