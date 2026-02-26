#include "storage_controller.h"
#include "core/storage/input_reader.h"
#include "core/storage/data_writer.h"

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
    if (dirs_.empty() || range_size_ == 0) {
        return Ret("StorageController: not initialized.");
    }

    InputReader reader;
    CHECK(reader.init(input_path));

    if (reader.count() == 0) {
        return Ret(0);
    }

    uint64_t min_id = reader.id(0);
    uint64_t max_id = reader.id(reader.count() - 1);

    uint64_t first_file = min_id / range_size_;
    uint64_t last_file  = max_id / range_size_;

    for (uint64_t file_id = first_file; file_id <= last_file; ++file_id) {
        uint64_t range_start = file_id * range_size_;
        uint64_t range_end   = range_start + range_size_;

        if (!reader.is_range_present(range_start, range_end)) {
            continue;
        }

        const std::string& dir = dirs_[static_cast<size_t>(file_id % dirs_.size())];
        std::string output_path = dir + "/" + std::to_string(file_id) + ".data";

        InputReaderView view(reader, range_start, range_end);
        DataWriter writer;
        CHECK(writer.load(view, output_path));
    }

    return Ret(0);
}

} // namespace sketch2
