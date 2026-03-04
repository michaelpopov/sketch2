#include "parasol.h"
#include "core/storage/dataset.h"
#include "core/utils/shared_types.h"
#include <filesystem>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace sketch2;

static const char* kInputFileName = "data.input";
static const char* kManagedMarkerFileName = ".sketch2.managed";

#define ERR(x) { \
    ret.code = -1;  \
    strncpy(ret.message, x, sizeof(ret.message)-1); \
    ret.message[sizeof(ret.message)-1] = '\0'; \
    return ret; \
}

#define DECL(x) \
    sk_ret_t x { .handle = nullptr, .code = 0, .message = "" };

struct sk_handle {
    Dataset* ds = nullptr;
    FILE* input = nullptr;
    std::string dir;
};

sk_ret_t sk_create(sk_dataset_metadata_t metadata) {
    DECL(ret)

    std::filesystem::path dir_path = metadata.dir;
    if (dir_path.empty()) {
        ERR("Invalid metadata dir")
    }

    if (!std::filesystem::exists(dir_path)) {
        std::filesystem::create_directories(dir_path);
    }

    std::filesystem::path file_path = dir_path / kMetadataFileName;
    if (std::filesystem::exists(file_path)) {
        ERR("Metadata file already exists")
    }

    // Validate type string
    try {
        (void)data_type_from_string(metadata.type);
    } catch (...) {
        ERR("Invalid metadata type")
    }

    FILE* fp = fopen(file_path.c_str(), "w");
    if (fp == nullptr) {
        ERR("Failed to open metadata file")
    }

    // Keep keys aligned with Dataset::init_(...) expectations.
    const int written = fprintf(fp,
        "[dataset]\n"
        "dirs=%s\n"
        "range_size=%u\n"
        "dim=%u\n"
        "type=%s\n",
        dir_path.string().c_str(),
        metadata.range_size,
        metadata.dim,
        metadata.type);

    const int close_rc = fclose(fp);
    if (written < 0 || close_rc != 0) {
        std::error_code remove_ec;
        std::filesystem::remove(file_path, remove_ec);
        ERR("Failed to write metadata file")
    }

    std::filesystem::path marker_path = dir_path / kManagedMarkerFileName;
    FILE* marker = fopen(marker_path.c_str(), "w");
    if (marker == nullptr) {
        std::error_code remove_ec;
        std::filesystem::remove(file_path, remove_ec);
        ERR("Failed to create managed marker file")
    }

    const int marker_written = fprintf(marker, "managed=1\n");
    const int marker_close_rc = fclose(marker);
    if (marker_written < 0 || marker_close_rc != 0) {
        std::error_code remove_ec;
        std::filesystem::remove(marker_path, remove_ec);
        std::filesystem::remove(file_path, remove_ec);
        ERR("Failed to write managed marker file")
    }

    return ret;
}

sk_ret_t sk_drop(const char* dir) {
    DECL(ret)

    if (dir == nullptr) {
        ERR("Invalid dir parameter")
    }

    std::filesystem::path dir_path = dir;
    if (dir_path.empty() || !std::filesystem::exists(dir_path)) {
        ERR("Invalid metadata dir")
    }

    std::filesystem::path file_path = dir_path / kMetadataFileName;
    if (!std::filesystem::exists(file_path)) {
        ERR("Metadata file is not present")
    }

    std::filesystem::path marker_path = dir_path / kManagedMarkerFileName;
    if (!std::filesystem::exists(marker_path)) {
        ERR("Managed marker file is not present")
    }

    std::error_code ec;
    std::filesystem::remove_all(dir_path, ec);
    if (ec) {
        ERR("Failed to remove directory")
    }

    return ret;
}

sk_ret_t sk_open(const char *path) {
    DECL(ret)

    if (path == nullptr) {
        ERR("Invalid path parameter")
    }

    std::filesystem::path dir_path = path;
    std::filesystem::path file_path = dir_path / kMetadataFileName;
    if (!std::filesystem::exists(file_path)) {
        ERR("Metadata file is not present")
    }

    Dataset* ds = new Dataset();
    Ret ds_ret = ds->init(file_path.string());
    if (ds_ret != 0) {
        delete ds;
        ERR(ds_ret.message().c_str())
    }

    ret.handle = new sk_handle_t;
    ret.handle->ds = ds;
    ret.handle->dir = path;

    return ret;
}

sk_ret_t sk_close(sk_handle_t* handle) {
    DECL(ret)

    if (handle == nullptr || handle->ds == nullptr)  {
        ERR("Invalid handle");
    }

    delete handle->ds;
    if (handle->input) {
        fclose(handle->input);
    }

    delete handle;

    return ret;
}

FILE* open_input_file(sk_handle_t* handle) {
    std::filesystem::path dir_path = handle->dir;
    std::filesystem::path file_path = dir_path / kInputFileName;
    FILE* f = fopen(file_path.c_str(), "w");
    if (f == nullptr) {
        return nullptr;
    }
    ssize_t n = fprintf(f, "%s,%lu\n", data_type_to_string(handle->ds->type()), handle->ds->dim());
    if (n <= 0) {
        fclose(f);
        return nullptr;
    }
    return f;
}

sk_ret_t sk_add(sk_handle_t* handle, uint64_t id, const char *value) {
    DECL(ret)

    if (handle == nullptr || handle->ds == nullptr) {
        ERR("Invalid handle");
    }
    if (value == nullptr) {
        ERR("Invalid value parameter");
    }

    if (handle->input == nullptr) {
        handle->input = open_input_file(handle);
        if (handle->input == nullptr) {
            ERR("Failed to open input file")
        }
    }

    ssize_t n = fprintf(handle->input, "%lu : [ %s ]\n", id, value);
    if (n <= 0) {
        ERR("Failed to write value to input file");
    }

    return ret;
}

sk_ret_t sk_delete(sk_handle_t* handle, uint64_t id) {
    DECL(ret)

    if (handle == nullptr || handle->ds == nullptr) {
        ERR("Invalid handle");
    }

    if (handle->input == nullptr) {
        handle->input = open_input_file(handle);
        if (handle->input == nullptr) {
            ERR("Failed to open input file")
        }
    }

    ssize_t n = fprintf(handle->input, "%lu : []\n", id);
    if (n <= 0) {
        ERR("Failed to write delete marker to input file");
    }

    return ret;
}

sk_ret_t sk_load(sk_handle_t* handle) {
    DECL(ret)

    if (handle == nullptr || handle->ds == nullptr) {
        ERR("Invalid handle");
    }

    if (handle->input) {
        fclose(handle->input);
        handle->input = nullptr;
    }

    std::filesystem::path dir_path = handle->dir;
    std::filesystem::path file_path = dir_path / kInputFileName;
    if (!std::filesystem::exists(file_path)) {
        ERR("Input file is not present")
    }

    Ret store_ret = handle->ds->store(file_path.string());
    if (store_ret != 0) {
        ERR(store_ret.message().c_str())
    }

    std::error_code ec;
    std::filesystem::remove(file_path, ec);
    if (ec) {
        ERR("Failed to remove input file")
    }

    return ret;
}
