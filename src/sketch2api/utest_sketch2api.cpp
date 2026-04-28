// Unit tests for the public sketch2api API.

#include "sketch2.h"
#include "internal.h"
#include "sketch2api_testing.h"

#include "core/bitset/bitset_filter_control.h"
#include "core/bitset/chunked_bits.h"
#include "core/utils/singleton.h"
#include "core/bitset/utest_chunked_bits_helpers.h"
#include "storage/input_generator.h"

#include <chrono>
#include <cstdlib>
#include <experimental/scope>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace {

#define ASSERT_OK(handle, call_expr) ASSERT_EQ(0, (call_expr)) << sk_error_message(handle)
#define EXPECT_OK(handle, call_expr) EXPECT_EQ(0, (call_expr)) << sk_error_message(handle)

std::filesystem::path make_temp_dir() {
    const std::filesystem::path base = std::filesystem::temp_directory_path();
    std::filesystem::create_directories(base);
    std::string pattern = (base / "sketch2_parasol_ut_XXXXXX").string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* const dir = mkdtemp(writable.data());
    EXPECT_NE(nullptr, dir);
    return dir == nullptr ? base / "sketch2_parasol_ut_fallback" : std::filesystem::path(dir);
}

std::filesystem::path make_test_data_path() {
    const std::filesystem::path base("/tmp");
    std::string pattern = (base / "sketch2_test_data_XXXXXX").string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');

    const int fd = mkstemp(writable.data());
    EXPECT_NE(-1, fd);
    if (fd != -1) {
        close(fd);
    }

    return fd == -1 ? base / "sketch2_test_data_fallback" : std::filesystem::path(writable.data());
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    return std::string((std::istreambuf_iterator<char>(in)),
        std::istreambuf_iterator<char>());
}

using sketch2::test::unique_filter_name;

std::vector<std::filesystem::path> find_staged_input_files(const std::filesystem::path& dataset_dir) {
    std::vector<std::filesystem::path> paths;
    if (!std::filesystem::exists(dataset_dir)) {
        return paths;
    }

    for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator(dataset_dir)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        const std::string name = entry.path().filename().string();
        if (name.rfind("ds.input.", 0) == 0) {
            paths.push_back(entry.path());
        }
    }

    return paths;
}

std::string api_get(sk_handle_t* handle, uint64_t id) {
    char* value = nullptr;
    EXPECT_EQ(0, sk_get(handle, id, &value)) << sk_error_message(handle);
    if (value == nullptr) {
        return "";
    }

    std::string out(value);
    sk_free(value);
    return out;
}

std::vector<uint64_t> api_knn(sk_handle_t* handle, const char* vec, unsigned int k) {
    uint64_t* ids = nullptr;
    size_t count = 0;
    EXPECT_EQ(0, sk_knn(handle, vec, k, &ids, &count)) << sk_error_message(handle);

    std::vector<uint64_t> out;
    if (ids != nullptr) {
        out.assign(ids, ids + count);
        sk_free(ids);
    }
    return out;
}

std::vector<uint64_t> api_knn_vector(sk_handle_t* handle, const std::vector<float>& vec, unsigned int k) {
    uint64_t* ids = nullptr;
    double* scores = nullptr;
    size_t count = 0;
    EXPECT_EQ(0, sk_knn_vector_items(
        handle, vec.data(), vec.size(), k, nullptr, 0, &ids, &scores, &count))
        << sk_error_message(handle);

    std::vector<uint64_t> out;
    if (ids != nullptr) {
        out.assign(ids, ids + count);
    }
    sk_free(ids);
    sk_free(scores);
    return out;
}

struct AlignedBlob {
    void* data = nullptr;
    size_t size = 0;
    size_t allocation_size = 0;

    AlignedBlob() = default;
    AlignedBlob(const AlignedBlob&) = delete;
    AlignedBlob& operator=(const AlignedBlob&) = delete;
    AlignedBlob(AlignedBlob&& other) noexcept
        : data(std::exchange(other.data, nullptr)),
          size(std::exchange(other.size, 0)),
          allocation_size(std::exchange(other.allocation_size, 0)) {}
    AlignedBlob& operator=(AlignedBlob&& other) noexcept {
        if (this != &other) {
            std::free(data);
            data = std::exchange(other.data, nullptr);
            size = std::exchange(other.size, 0);
            allocation_size = std::exchange(other.allocation_size, 0);
        }
        return *this;
    }
    ~AlignedBlob() {
        std::free(data);
    }
};

size_t align_up(size_t value, size_t alignment) {
    const size_t mask = alignment - 1u;
    return (value + mask) & ~mask;
}

void build_allowed_ids_blob(const std::vector<uint64_t>& ids, AlignedBlob* blob) {
    ASSERT_NE(nullptr, blob);

    sketch2::ChunkedBits bits;
    for (uint64_t id : ids) {
        EXPECT_EQ(0, bits.add(id).code());
    }
    EXPECT_EQ(0, bits.finish().code());

    blob->size = bits.serialized_size_bytes();
    blob->allocation_size = align_up(blob->size, sketch2::kChunkedBitsBlobAlignment);
    blob->data = std::aligned_alloc(sketch2::kChunkedBitsBlobAlignment, blob->allocation_size);
    ASSERT_NE(nullptr, blob->data);
    EXPECT_EQ(0, bits.serialize(blob->data, blob->size).code());
}

void* build_opaque_bitset_filter(const std::vector<uint64_t>& ids) {
    void* state = nullptr;
    bool out_of_memory = false;
    const char* error_message = nullptr;
    for (uint64_t id : ids) {
        EXPECT_EQ(0, sk_bitset_filter_builder_add(&state, id, &out_of_memory, &error_message, nullptr))
            << (error_message != nullptr ? error_message : "");
        if (::testing::Test::HasFailure()) {
            delete static_cast<sketch2::ChunkedBits*>(state);
            return nullptr;
        }
    }

    void* out = nullptr;
    EXPECT_EQ(0, sk_bitset_filter_builder_finish(&state, &out, &out_of_memory, &error_message))
        << (error_message != nullptr ? error_message : "");
    EXPECT_EQ(nullptr, state);
    return out;
}

std::vector<uint64_t> make_spill_bitset_filter_ids() {
    std::vector<uint64_t> ids;
    ids.reserve(50002);
    ids.push_back(20);
    ids.push_back(40);
    for (uint64_t i = 0; i < 50000; ++i) {
        ids.push_back((i + 1) << sketch2::kChunkBits);
    }
    return ids;
}

} // namespace

TEST(sketch2api, create_open_close_drop_lifecycle) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);

    ASSERT_OK(handle, sk_create(handle, "dataset", nullptr, 4, "f32", 1000, "dot"));
    const std::filesystem::path dataset_dir = root / "dataset";
    EXPECT_TRUE(std::filesystem::exists(dataset_dir));
    EXPECT_TRUE(std::filesystem::exists(dataset_dir / "dataset.ini"));
    EXPECT_TRUE(std::filesystem::exists(dataset_dir / "dataset.lock"));

    const std::string ini = read_file(dataset_dir / "dataset.ini");
    EXPECT_NE(ini.find("[dataset]\n"), std::string::npos);
    EXPECT_NE(ini.find("dirs=" + (root / "dataset").string() + "\n"), std::string::npos);
    EXPECT_NE(ini.find("range_size=1000\n"), std::string::npos);
    EXPECT_NE(ini.find("dim=4\n"), std::string::npos);
    EXPECT_NE(ini.find("type=f32\n"), std::string::npos);
    EXPECT_NE(ini.find("dist_func=dot\n"), std::string::npos);

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "dataset"));

    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, reopen_restores_pending_wal_for_get_and_knn) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);

    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "dot"));
    const std::filesystem::path input_path = root / "vectors.txt";
    {
        std::ofstream out(input_path);
        out << "f32,4\n";
        out << "1 : [ 0.0, 0.0, 0.0, 0.0 ]\n";
        out << "2 : [ 10.0, 10.0, 10.0, 10.0 ]\n";
    }
    ASSERT_OK(handle, sk_load_file(handle, input_path.string().c_str()));
    ASSERT_OK(handle, sk_close(handle));

    ASSERT_OK(handle, sk_open(handle, "ds"));

    EXPECT_EQ("[ 0.00, 0.00, 0.00, 0.00 ]", api_get(handle, 1));

    const std::vector<uint64_t> ids = api_knn(handle, "0.0, 0.0, 0.0, 0.0", 2);
    ASSERT_EQ(2u, ids.size());
    EXPECT_EQ(1u, ids[0]);
    EXPECT_EQ(2u, ids[1]);

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));

    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, generate_stats_and_print_smoke) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path test_data_path = make_test_data_path();
    const std::filesystem::path stats_output = root / "stats.txt";
    std::experimental::scope_exit cleanup([&]() {
        std::filesystem::remove(test_data_path);
        std::filesystem::remove(stats_output);
    });

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);

    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "dot"));
    ASSERT_OK(handle, sk_generate_test_data(handle, test_data_path.c_str(), 8, 10, nullptr, false));

    ASSERT_OK(handle, sk_stats(handle, stats_output.string().c_str()));
    const std::string stats_out = read_file(stats_output);
    EXPECT_NE(stats_out.find("Name: ds"), std::string::npos);
    EXPECT_NE(stats_out.find("Type: f32"), std::string::npos);
    EXPECT_NE(stats_out.find("Dist: dot"), std::string::npos);
    EXPECT_NE(stats_out.find("Dim: 4"), std::string::npos);
    EXPECT_NE(stats_out.find("Range: 1000"), std::string::npos);
    EXPECT_NE(stats_out.find(".data:"), std::string::npos);

    testing::internal::CaptureStdout();
    ASSERT_OK(handle, sk_print(handle));
    const std::string print_out = testing::internal::GetCapturedStdout();
    EXPECT_NE(print_out.find("10 : ["), std::string::npos);

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));

    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, generate_bin_creates_and_loads_input) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path test_data_path = make_test_data_path();
    std::experimental::scope_exit cleanup([&]() { std::filesystem::remove(test_data_path); });

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);

    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "dot"));
    ASSERT_OK(handle, sk_generate_test_data(handle, test_data_path.c_str(), 8, 10, nullptr, true));

    EXPECT_NE(api_get(handle, 10).find("[ 10.1"), std::string::npos);

    const std::vector<uint64_t> ids = api_knn(handle, "10.0, 10.0, 10.0, 10.0", 1);
    ASSERT_EQ(1u, ids.size());
    EXPECT_EQ(17u, ids[0]);

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));

    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, load_file_accepts_binary_input) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path input_path = root / "input.bin";

    sketch2::GeneratorConfig cfg;
    cfg.pattern_type = sketch2::PatternType::Sequential;
    cfg.count = 3;
    cfg.min_id = 20;
    cfg.type = sketch2::DataType::f32;
    cfg.dim = 4;
    cfg.max_val = 1000;
    cfg.binary = true;
    ASSERT_EQ(0, sketch2::generate_input_file(input_path.string(), cfg).code());

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);

    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "dot"));
    ASSERT_OK(handle, sk_load_file(handle, input_path.string().c_str()));

    EXPECT_NE(api_get(handle, 20).find("[ 20.1"), std::string::npos);

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));

    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, close_requires_open_dataset) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);

    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "dot"));
    EXPECT_OK(handle, sk_close(handle)); // sk_create() creates AND opens a dataset
    ASSERT_OK(handle, sk_open(handle, "ds"));
    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));

    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, create_rejects_invalid_distance_function) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);

    EXPECT_NE(0, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "cosine"));

    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, drop_waits_for_dataset_owner_lock) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "dot"));

    int pipefd[2];
    ASSERT_EQ(0, pipe(pipefd));

    const pid_t pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        const std::filesystem::path test_data_path = make_test_data_path();
        std::experimental::scope_exit cleanup([&]() { std::filesystem::remove(test_data_path); });

        close(pipefd[0]);
        sk_handle_t* child = sk_new_handle(root.string().c_str());
        if (child == nullptr) {
            _exit(10);
        }
        if (sk_open(child, "ds") != 0) {
            _exit(11);
        }
        if (sk_generate_test_data(child, test_data_path.c_str(), 1, 0, nullptr, false) != 0) {
            _exit(12);
        }
        const char ready = '1';
        if (write(pipefd[1], &ready, 1) != 1) {
            _exit(13);
        }
        usleep(300000);
        if (sk_close(child) != 0) {
            _exit(14);
        }
        sk_release_handle(child);
        close(pipefd[1]);
        _exit(0);
    }

    close(pipefd[1]);
    char ready = 0;
    ASSERT_EQ(1, read(pipefd[0], &ready, 1));
    close(pipefd[0]);

    const auto started = std::chrono::steady_clock::now();
    ASSERT_OK(handle, sk_drop(handle, "ds"));
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started);
    EXPECT_GE(elapsed.count(), 200);

    int status = 0;
    ASSERT_EQ(pid, waitpid(pid, &status, 0));
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(0, WEXITSTATUS(status));

    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, get_rejects_null_output_parameter) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "dot"));
    EXPECT_NE(0, sk_get(handle, 1, nullptr));

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));
    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, knn_rejects_null_output_parameters) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "dot"));
    EXPECT_NE(0, sk_knn(handle, "0.0, 0.0, 0.0, 0.0", 1, nullptr, nullptr));

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));
    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, knn_vector_items_matches_text_knn_and_checks_size) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "dot"));

    ASSERT_OK(handle, sk_start_writing(handle));
    ASSERT_OK(handle, sk_write_vector(handle, 10, "10.0, 10.0, 10.0, 10.0"));
    ASSERT_OK(handle, sk_write_vector(handle, 20, "20.0, 20.0, 20.0, 20.0"));
    ASSERT_OK(handle, sk_write_vector(handle, 30, "30.0, 30.0, 30.0, 30.0"));
    ASSERT_OK(handle, sk_complete_writing(handle));

    const std::vector<uint64_t> text_ids = api_knn(handle, "10.0, 10.0, 10.0, 10.0", 3);
    const std::vector<uint64_t> vector_ids = api_knn_vector(handle, {10.0f, 10.0f, 10.0f, 10.0f}, 3);
    EXPECT_EQ(text_ids, vector_ids);

    const std::vector<float> short_query = {10.0f, 10.0f, 10.0f};
    uint64_t* ids = nullptr;
    double* scores = nullptr;
    size_t count = 0;
    EXPECT_NE(0, sk_knn_vector_items(
        handle, short_query.data(), short_query.size(), 1, nullptr, 0, &ids, &scores, &count));
    EXPECT_EQ(nullptr, ids);
    EXPECT_EQ(nullptr, scores);
    EXPECT_EQ(0u, count);

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));
    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, knn_vector_items_matches_text_knn_items_with_bitset_filter) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "dot"));

    ASSERT_OK(handle, sk_start_writing(handle));
    ASSERT_OK(handle, sk_write_vector(handle, 10, "10.0, 10.0, 10.0, 10.0"));
    ASSERT_OK(handle, sk_write_vector(handle, 20, "20.0, 20.0, 20.0, 20.0"));
    ASSERT_OK(handle, sk_write_vector(handle, 30, "30.0, 30.0, 30.0, 30.0"));
    ASSERT_OK(handle, sk_complete_writing(handle));

    AlignedBlob allowed;
    ASSERT_NO_FATAL_FAILURE(build_allowed_ids_blob({20, 30}, &allowed));

    uint64_t* text_ids = nullptr;
    double* text_scores = nullptr;
    size_t text_count = 0;
    ASSERT_EQ(0, sk_knn_items(handle, "10.0, 10.0, 10.0, 10.0", 3,
        allowed.data, allowed.size, &text_ids, &text_scores, &text_count))
        << sk_error_message(handle);

    const std::vector<float> vector_query = {10.0f, 10.0f, 10.0f, 10.0f};
    uint64_t* vector_ids = nullptr;
    double* vector_scores = nullptr;
    size_t vector_count = 0;
    ASSERT_EQ(0, sk_knn_vector_items(handle, vector_query.data(), vector_query.size(), 3,
        allowed.data, allowed.size, &vector_ids, &vector_scores, &vector_count))
        << sk_error_message(handle);

    ASSERT_EQ(text_count, vector_count);
    for (size_t i = 0; i < text_count; ++i) {
        EXPECT_EQ(text_ids[i], vector_ids[i]);
        EXPECT_DOUBLE_EQ(text_scores[i], vector_scores[i]);
    }

    sk_free(text_ids);
    sk_free(text_scores);
    sk_free(vector_ids);
    sk_free(vector_scores);

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));
    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, bitset_filter_builder_empty_releases_cleanly) {
    void* bitset_filter = build_opaque_bitset_filter({});
    ASSERT_NE(nullptr, bitset_filter);
    EXPECT_EQ(0, sk_bitset_filter_storage_kind_for_testing(bitset_filter));
    sk_release_bitset_filter(bitset_filter);
}

TEST(sketch2api, bitset_filter_builder_named_empty_publishes_file) {
    const std::string name = unique_filter_name("api_empty_filter");

    void* state = nullptr;
    bool out_of_memory = false;
    const char* error_message = nullptr;
    ASSERT_EQ(0, sk_bitset_filter_builder_set_name(
        &state, &out_of_memory, &error_message, name.c_str()))
        << (error_message != nullptr ? error_message : "");

    void* bitset_filter = nullptr;
    ASSERT_EQ(0, sk_bitset_filter_builder_finish(
        &state, &bitset_filter, &out_of_memory, &error_message))
        << (error_message != nullptr ? error_message : "");
    EXPECT_EQ(nullptr, state);
    ASSERT_NE(nullptr, bitset_filter);

    const std::filesystem::path expected_path = sketch2::named_bitset_filter_path(name);
    EXPECT_TRUE(std::filesystem::exists(expected_path));
    EXPECT_GT(std::filesystem::file_size(expected_path), 0u);

    sk_release_bitset_filter(bitset_filter);
    std::filesystem::remove(expected_path);
}

TEST(sketch2api, bitset_filter_drop_removes_named_file_and_is_idempotent) {
    const std::string name = unique_filter_name("api_drop_filter");

    void* state = nullptr;
    bool out_of_memory = false;
    const char* error_message = nullptr;
    ASSERT_EQ(0, sk_bitset_filter_builder_add(
        &state, 20, &out_of_memory, &error_message, name.c_str()))
        << (error_message != nullptr ? error_message : "");

    void* bitset_filter = nullptr;
    ASSERT_EQ(0, sk_bitset_filter_builder_finish(
        &state, &bitset_filter, &out_of_memory, &error_message))
        << (error_message != nullptr ? error_message : "");
    ASSERT_NE(nullptr, bitset_filter);
    sk_release_bitset_filter(bitset_filter);

    const std::filesystem::path expected_path = sketch2::named_bitset_filter_path(name);
    ASSERT_TRUE(std::filesystem::exists(expected_path));

    int removed = -1;
    EXPECT_EQ(0, sk_bitset_filter_drop(name.c_str(), &removed, &out_of_memory, &error_message))
        << (error_message != nullptr ? error_message : "");
    EXPECT_EQ(1, removed);
    EXPECT_FALSE(std::filesystem::exists(expected_path));

    removed = -1;
    EXPECT_EQ(0, sk_bitset_filter_drop(name.c_str(), &removed, &out_of_memory, &error_message))
        << (error_message != nullptr ? error_message : "");
    EXPECT_EQ(0, removed);
}

TEST(sketch2api, bitset_filter_load_maps_named_file) {
    const std::string name = unique_filter_name("api_load_filter");
    const std::filesystem::path expected_path = sketch2::named_bitset_filter_path(name);
    std::experimental::scope_exit cleanup([&]() { std::filesystem::remove(expected_path); });

    void* state = nullptr;
    bool out_of_memory = false;
    const char* error_message = nullptr;
    ASSERT_EQ(0, sk_bitset_filter_builder_add(
        &state, 20, &out_of_memory, &error_message, name.c_str()))
        << (error_message != nullptr ? error_message : "");
    ASSERT_EQ(0, sk_bitset_filter_builder_add(
        &state, 40, &out_of_memory, &error_message, name.c_str()))
        << (error_message != nullptr ? error_message : "");

    void* built_filter = nullptr;
    ASSERT_EQ(0, sk_bitset_filter_builder_finish(
        &state, &built_filter, &out_of_memory, &error_message))
        << (error_message != nullptr ? error_message : "");
    ASSERT_NE(nullptr, built_filter);
    sk_release_bitset_filter(built_filter);

    void* loaded_filter = nullptr;
    ASSERT_EQ(0, sk_bitset_filter_load(
        name.c_str(), &loaded_filter, &out_of_memory, &error_message))
        << (error_message != nullptr ? error_message : "");
    ASSERT_NE(nullptr, loaded_filter);
    EXPECT_EQ(1, sk_bitset_filter_storage_kind_for_testing(loaded_filter));
    sk_release_bitset_filter(loaded_filter);
}

TEST(sketch2api, bitset_filter_load_rejects_null_name) {
    void* bitset_filter = reinterpret_cast<void*>(0x1);
    bool out_of_memory = true;
    const char* error_message = nullptr;

    EXPECT_NE(0, sk_bitset_filter_load(
        nullptr, &bitset_filter, &out_of_memory, &error_message));
    EXPECT_FALSE(out_of_memory);
    EXPECT_EQ(nullptr, bitset_filter);
    ASSERT_NE(nullptr, error_message);
    EXPECT_NE(
        std::string(error_message).find("bitset_load: name must not be null"),
        std::string::npos);
}

TEST(sketch2api, bitset_filter_load_rejects_null_output) {
    bool out_of_memory = true;
    const char* error_message = nullptr;

    EXPECT_NE(0, sk_bitset_filter_load(
        "some_filter", nullptr, &out_of_memory, &error_message));
    EXPECT_FALSE(out_of_memory);
    ASSERT_NE(nullptr, error_message);
    EXPECT_NE(
        std::string(error_message).find("bitset_load: invalid control output"),
        std::string::npos);
}

TEST(sketch2api, bitset_filter_builder_rejects_empty_name) {
    void* state = nullptr;
    bool out_of_memory = false;
    const char* error_message = nullptr;
    EXPECT_NE(0, sk_bitset_filter_builder_set_name(
        &state, &out_of_memory, &error_message, ""));
    EXPECT_FALSE(out_of_memory);
    EXPECT_NE(nullptr, error_message);
    EXPECT_EQ(nullptr, state);
}

TEST(sketch2api, bitset_filter_drop_rejects_invalid_names) {
    int removed = -1;
    bool out_of_memory = false;
    const char* error_message = nullptr;

    EXPECT_NE(0, sk_bitset_filter_drop(nullptr, &removed, &out_of_memory, &error_message));
    EXPECT_FALSE(out_of_memory);
    EXPECT_EQ(0, removed);
    ASSERT_NE(nullptr, error_message);
    EXPECT_NE(
        std::string(error_message).find("bitset_drop: name must not be null"),
        std::string::npos);

    removed = -1;
    error_message = nullptr;
    EXPECT_NE(0, sk_bitset_filter_drop("", &removed, &out_of_memory, &error_message));
    EXPECT_FALSE(out_of_memory);
    EXPECT_EQ(0, removed);
    ASSERT_NE(nullptr, error_message);
    EXPECT_NE(
        std::string(error_message).find("bitset_drop: invalid bitset filter name"),
        std::string::npos);

    removed = -1;
    error_message = nullptr;
    EXPECT_NE(0, sk_bitset_filter_drop("bad-name", &removed, &out_of_memory, &error_message));
    EXPECT_FALSE(out_of_memory);
    EXPECT_EQ(0, removed);
    ASSERT_NE(nullptr, error_message);
    EXPECT_NE(
        std::string(error_message).find("bitset_drop: invalid bitset filter name"),
        std::string::npos);
}

TEST(sketch2api, knn_items_bitset_filter_filters_with_spilled_bitset_filter) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "dot"));

    ASSERT_OK(handle, sk_start_writing(handle));
    ASSERT_OK(handle, sk_write_vector(handle, 10, "10.0, 10.0, 10.0, 10.0"));
    ASSERT_OK(handle, sk_write_vector(handle, 20, "20.0, 20.0, 20.0, 20.0"));
    ASSERT_OK(handle, sk_write_vector(handle, 30, "30.0, 30.0, 30.0, 30.0"));
    ASSERT_OK(handle, sk_write_vector(handle, 40, "40.0, 40.0, 40.0, 40.0"));
    ASSERT_OK(handle, sk_complete_writing(handle));

    void* bitset_filter = build_opaque_bitset_filter(make_spill_bitset_filter_ids());
    ASSERT_NE(nullptr, bitset_filter);
    ASSERT_EQ(2, sk_bitset_filter_storage_kind_for_testing(bitset_filter));

    uint64_t* ids = nullptr;
    double* scores = nullptr;
    size_t count = 0;
    ASSERT_EQ(0, sk_knn_items_bitset_filter(
        handle, "25.0, 25.0, 25.0, 25.0", 4, bitset_filter, &ids, &scores, &count))
        << sk_error_message(handle);

    ASSERT_EQ(2u, count);
    EXPECT_EQ(40u, ids[0]);
    EXPECT_EQ(20u, ids[1]);

    sk_free(ids);
    sk_free(scores);
    sk_release_bitset_filter(bitset_filter);

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));
    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, staged_write_creates_data_and_removes_input_file) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dataset_dir = root / "ds";

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "dot"));

    ASSERT_OK(handle, sk_start_writing(handle));
    const std::vector<std::filesystem::path> staged_inputs = find_staged_input_files(dataset_dir);
    ASSERT_EQ(1u, staged_inputs.size());
    const std::filesystem::path input_path = staged_inputs.front();
    EXPECT_TRUE(std::filesystem::exists(input_path));
    ASSERT_OK(handle, sk_write_vector(handle, 10, "10.1, 10.1, 10.1, 10.1"));
    ASSERT_OK(handle, sk_write_vector(handle, 11, "11.1 11.1 11.1 11.1"));
    ASSERT_OK(handle, sk_complete_writing(handle));

    EXPECT_TRUE(find_staged_input_files(dataset_dir).empty());
    EXPECT_NE(api_get(handle, 10).find("[ 10.1"), std::string::npos);
    EXPECT_NE(api_get(handle, 11).find("[ 11.1"), std::string::npos);

    const std::vector<uint64_t> ids = api_knn(handle, "10.0, 10.0, 10.0, 10.0", 2);
    ASSERT_EQ(2u, ids.size());
    EXPECT_EQ(11u, ids[0]);
    EXPECT_EQ(10u, ids[1]);

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));
    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, staged_write_delete_hides_existing_item) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "dot"));

    ASSERT_OK(handle, sk_start_writing(handle));
    ASSERT_OK(handle, sk_write_vector(handle, 10, "10.1, 10.1, 10.1, 10.1"));
    ASSERT_OK(handle, sk_complete_writing(handle));

    ASSERT_OK(handle, sk_start_writing(handle));
    ASSERT_OK(handle, sk_write_deleted(handle, 10));
    ASSERT_OK(handle, sk_complete_writing(handle));

    char* value = nullptr;
    EXPECT_NE(0, sk_get(handle, 10, &value));
    EXPECT_EQ(nullptr, value);

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));
    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, staged_write_rejects_invalid_call_order) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "dot"));

    EXPECT_NE(0, sk_write_vector(handle, 10, "10.1, 10.1, 10.1, 10.1"));
    EXPECT_NE(0, sk_write_deleted(handle, 10));
    EXPECT_NE(0, sk_abort_writing(handle));
    EXPECT_NE(0, sk_complete_writing(handle));

    ASSERT_OK(handle, sk_start_writing(handle));
    EXPECT_NE(0, sk_start_writing(handle));

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));
    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, staged_abort_removes_input_file_and_allows_restart) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dataset_dir = root / "ds";

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "dot"));

    ASSERT_OK(handle, sk_start_writing(handle));
    ASSERT_OK(handle, sk_write_vector(handle, 10, "10.1, 10.1, 10.1, 10.1"));
    const std::vector<std::filesystem::path> staged_inputs = find_staged_input_files(dataset_dir);
    ASSERT_EQ(1u, staged_inputs.size());
    const std::filesystem::path input_path = staged_inputs.front();
    ASSERT_TRUE(std::filesystem::exists(input_path));
    ASSERT_OK(handle, sk_abort_writing(handle));
    EXPECT_TRUE(find_staged_input_files(dataset_dir).empty());

    ASSERT_OK(handle, sk_start_writing(handle));
    ASSERT_OK(handle, sk_write_vector(handle, 11, "11.1, 11.1, 11.1, 11.1"));
    ASSERT_OK(handle, sk_complete_writing(handle));

    char* value = nullptr;
    EXPECT_NE(0, sk_get(handle, 10, &value));
    EXPECT_EQ(nullptr, value);
    EXPECT_NE(api_get(handle, 11).find("[ 11.1"), std::string::npos);

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));
    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, staged_write_many_vectors_then_knn) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "dot"));

    ASSERT_OK(handle, sk_start_writing(handle));
    for (uint64_t id = 0; id < 100; ++id) {
        const std::string value = std::to_string(static_cast<float>(id)) + ", " +
            std::to_string(static_cast<float>(id)) + ", " +
            std::to_string(static_cast<float>(id)) + ", " +
            std::to_string(static_cast<float>(id));
        ASSERT_OK(handle, sk_write_vector(handle, id, value.c_str()));
    }
    ASSERT_OK(handle, sk_complete_writing(handle));

    const std::vector<uint64_t> ids = api_knn(handle, "0.0, 0.0, 0.0, 0.0", 3);
    ASSERT_EQ(3u, ids.size());
    EXPECT_EQ(0u, ids[0]);
    EXPECT_EQ(1u, ids[1]);
    EXPECT_EQ(2u, ids[2]);

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));
    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, staged_write_mixed_vectors_and_deletes) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "dot"));

    // First session: create all vectors
    ASSERT_OK(handle, sk_start_writing(handle));
    ASSERT_OK(handle, sk_write_vector(handle, 10, "0.0, 0.0, 0.0, 0.0"));
    ASSERT_OK(handle, sk_write_vector(handle, 20, "5.0, 5.0, 5.0, 5.0"));
    ASSERT_OK(handle, sk_write_vector(handle, 30, "10.0, 10.0, 10.0, 10.0"));
    ASSERT_OK(handle, sk_write_vector(handle, 40, "15.0, 15.0, 15.0, 15.0"));
    ASSERT_OK(handle, sk_complete_writing(handle));

    // Second session: delete one
    ASSERT_OK(handle, sk_start_writing(handle));
    ASSERT_OK(handle, sk_write_deleted(handle, 40));
    ASSERT_OK(handle, sk_complete_writing(handle));

    EXPECT_NE(api_get(handle, 10).find("[ 0"), std::string::npos);
    EXPECT_NE(api_get(handle, 20).find("[ 5"), std::string::npos);
    EXPECT_NE(api_get(handle, 30).find("[ 10"), std::string::npos);

    char* value = nullptr;
    EXPECT_NE(0, sk_get(handle, 40, &value));
    EXPECT_EQ(nullptr, value);

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));
    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, staged_write_multiple_sessions_accumulate) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "dot"));

    ASSERT_OK(handle, sk_start_writing(handle));
    ASSERT_OK(handle, sk_write_vector(handle, 10, "10.0, 10.0, 10.0, 10.0"));
    ASSERT_OK(handle, sk_complete_writing(handle));

    ASSERT_OK(handle, sk_start_writing(handle));
    ASSERT_OK(handle, sk_write_vector(handle, 20, "20.0, 20.0, 20.0, 20.0"));
    ASSERT_OK(handle, sk_complete_writing(handle));

    ASSERT_OK(handle, sk_start_writing(handle));
    ASSERT_OK(handle, sk_write_vector(handle, 30, "30.0, 30.0, 30.0, 30.0"));
    ASSERT_OK(handle, sk_complete_writing(handle));

    EXPECT_NE(api_get(handle, 10).find("[ 10"), std::string::npos);
    EXPECT_NE(api_get(handle, 20).find("[ 20"), std::string::npos);
    EXPECT_NE(api_get(handle, 30).find("[ 30"), std::string::npos);

    const std::vector<uint64_t> ids = api_knn(handle, "10.0, 10.0, 10.0, 10.0", 3);
    ASSERT_EQ(3u, ids.size());
    EXPECT_EQ(30u, ids[0]);

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));
    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, staged_write_rejects_null_and_empty_vector) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "dot"));

    ASSERT_OK(handle, sk_start_writing(handle));
    EXPECT_NE(0, sk_write_vector(handle, 1, nullptr));
    EXPECT_NE(0, sk_write_vector(handle, 1, ""));

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));
    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, staged_write_without_open_dataset_fails) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);

    EXPECT_NE(0, sk_start_writing(handle));
    EXPECT_NE(0, sk_write_vector(handle, 1, "1.0, 1.0, 1.0, 1.0"));
    EXPECT_NE(0, sk_write_deleted(handle, 1));
    EXPECT_NE(0, sk_complete_writing(handle));

    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}
