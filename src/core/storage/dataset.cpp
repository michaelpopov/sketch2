// Implements the lean Dataset base: metadata initialization only.

#include "dataset.h"
#include "utils/ini_reader.h"

namespace sketch2 {

namespace {

Ret get_non_negative_ini_u64(const IniReader& cfg, const std::string& name, int def, uint64_t* out) {
    const int value = cfg.get_int(name, def);
    if (value < 0) {
        return Ret("Dataset: " + name + " must be >= 0");
    }
    *out = static_cast<uint64_t>(value);
    return Ret(0);
}

} // namespace

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
        validate_dist_func(metadata.dist_func);
    } catch (const std::exception& ex) {
        return Ret(ex.what());
    }
    metadata_ = metadata;
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

std::string Dataset::item_path_base(uint64_t file_id) const {
    return metadata_.dirs[file_id % metadata_.dirs.size()] + "/" + std::to_string(file_id);
}

} // namespace sketch2
