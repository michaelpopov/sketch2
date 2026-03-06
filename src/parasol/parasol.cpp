#include "parasol.h"
#include "core/compute/scanner.h"
#include "core/storage/dataset.h"
#include "core/utils/shared_types.h"
#include "core/utils/string_utils.h"
#include <algorithm>
#include <filesystem>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace sketch2;

static const char* kInputFileName = "data.input";
static const char* kManagedMarkerFileName = ".sketch2.managed";

#define ERR(x) { \
    handle->error = -1;  \
    strncpy(handle->message, x, sizeof(handle->message)-1); \
    handle->message[sizeof(handle->message)-1] = '\0'; \
    return -1; \
}

#define DECL \
    if (handle == nullptr) { \
        return -1; \
    } \
    handle->error = 0;  \
    handle->message[0] = '\0';

struct sk_handle {
    sk_handle() {
        memset(message, 0, sizeof(message));
    }

    ~sk_handle() {
        delete ds;
        if (input) fclose(input);
    }

    Dataset* ds = nullptr;
    FILE* input = nullptr;
    std::string dir;
    int error = 0;
    char message[256];
};

sk_handle_t* connect() {
    return new sk_handle;
}

void disconnect(sk_handle_t* handle) {
    delete handle;
}

static int sk_create_(sk_handle_t* handle, sk_dataset_metadata_t metadata);
int sk_create(sk_handle_t* handle, sk_dataset_metadata_t metadata) {
    try {
        return sk_create_(handle, metadata);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}
static int sk_create_(sk_handle_t* handle, sk_dataset_metadata_t metadata) {
    DECL

    if (handle->ds) {
        ERR("Handle already initialized");
    }

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

    handle->dir = dir_path;

    return 0;
}

static int sk_drop_(sk_handle_t* handle);
int sk_drop(sk_handle_t* handle) {
    try {
        return sk_drop_(handle);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}
static int sk_drop_(sk_handle_t* handle) {
    DECL

    const std::string dir = handle->dir;
    (void)sk_close(handle);

    if (dir.empty()) {
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

    return 0;
}

static int sk_open_(sk_handle_t* handle, const char *path);
int sk_open(sk_handle_t* handle, const char *path) {
    try {
        return sk_open_(handle, path);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}
static int sk_open_(sk_handle_t* handle, const char *path) {
    DECL

    if (handle->ds) {
        ERR("Handle already initialized");
    }

    if (path == nullptr) {
        ERR("Invalid path parameter")
    }

    std::filesystem::path dir_path = path;
    std::filesystem::path file_path = dir_path / kMetadataFileName;
    if (!std::filesystem::exists(file_path)) {
        ERR("Metadata file is not present")
    }

    auto ds = std::make_unique<Dataset>();
    Ret ds_ret = ds->init(file_path.string());
    if (ds_ret != 0) {
        ERR(ds_ret.message().c_str())
    }

    handle->ds = ds.release();
    handle->dir = path;

    return 0;
}

int sk_close_(sk_handle_t* handle);
int sk_close(sk_handle_t* handle) {
    try {
        return sk_close_(handle);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}
int sk_close_(sk_handle_t* handle) {
    DECL

    delete handle->ds;
    if (handle->input) {
        fclose(handle->input);
    }

    handle->ds = nullptr;
    handle->input = nullptr;
    handle->dir = "";

    return 0;
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

int sk_add_(sk_handle_t* handle, uint64_t id, const char *value);
int sk_add(sk_handle_t* handle, uint64_t id, const char *value) {
    try {
        return sk_add_(handle, id, value);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}
int sk_add_(sk_handle_t* handle, uint64_t id, const char *value) {
    DECL

    if (handle->ds == nullptr) {
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

    return 0;
}

int sk_delete_(sk_handle_t* handle, uint64_t id);
int sk_delete(sk_handle_t* handle, uint64_t id) {
    try {
        return sk_delete_(handle, id);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}
int sk_delete_(sk_handle_t* handle, uint64_t id) {
    DECL

    if (handle->ds == nullptr) {
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

    return 0;
}

static int sk_load_(sk_handle_t* handle);
int sk_load(sk_handle_t* handle) {
    try {
        return sk_load_(handle);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}
static int sk_load_(sk_handle_t* handle) {
    DECL

    if (handle->ds == nullptr) {
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

    return 0;
}

int sk_knn_(sk_handle_t* handle, const char* vec, uint64_t* ids, uint64_t* ids_count);
int sk_knn(sk_handle_t* handle, const char* vec, uint64_t* ids, uint64_t* ids_count) {
    try {
        return sk_knn_(handle, vec, ids, ids_count);
    } catch (const std::exception& ex) {
        ERR(ex.what())
    }
}
int sk_knn_(sk_handle_t* handle, const char* vec, uint64_t* ids, uint64_t* ids_count) {
    DECL

    if (handle == nullptr || handle->ds == nullptr ||
        vec == nullptr || ids == nullptr ||
        ids_count == nullptr || *ids_count < 1) {
        ERR("Invalid arguments");
    }

    uint64_t count = *ids_count;
    Dataset* ds = handle->ds;
    std::vector<uint8_t> buf(data_type_size(ds->type()) * ds->dim());
    
    Ret convert_ret = parse_vector(buf.data(), buf.size(), ds->type(), ds->dim(), vec);
    if (convert_ret != 0) {
        ERR(convert_ret.message().c_str());
    }

    std::vector<uint64_t> result;
    Scanner scanner;
    Ret scanner_ret = scanner.find(*ds, DistFunc::L1, count, buf.data(), result);
    if (scanner_ret != 0) {
        ERR(scanner_ret.message().c_str());
    }

    for (size_t i = 0; i < count && i < result.size(); i++) {
        ids[i] = result[i];
    }

    *ids_count = std::min(count, result.size());

    return 0;
}

int sk_error(sk_handle_t* handle) {
    if (handle == nullptr) return -1;
    return handle->error;
}

const char* sk_error_message(sk_handle_t* handle) {
    if (handle == nullptr) return "";
    return handle->message;
}
