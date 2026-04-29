// Unit tests for sketch2api bitset functionality.

#include "sketch2.h"
#include "sketch2api_testing.h"

#include "core/bitset/bitset_file_cache.h"
#include "core/bitset/chunked_bits.h"
#include "core/utils/singleton.h"
#include "core/bitset/utest_chunked_bits_helpers.h"

#include <cstdlib>
#include <experimental/scope>
#include <filesystem>
#include <functional>
#include <string>
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

using sketch2::test::unique_filter_name;

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
        EXPECT_EQ(0, sk_bitset_add_id(&state, id, &out_of_memory, &error_message, nullptr))
            << (error_message != nullptr ? error_message : "");
        if (::testing::Test::HasFailure()) {
            delete static_cast<sketch2::ChunkedBits*>(state);
            return nullptr;
        }
    }

    void* out = nullptr;
    EXPECT_EQ(0, sk_bitset_finish(&state, &out, &out_of_memory, &error_message))
        << (error_message != nullptr ? error_message : "");
    EXPECT_EQ(nullptr, state);
    return out;
}

std::vector<uint64_t> make_spill_bitset_filter_ids();

void assert_knn_with_spilled_bitset_filter(
        const std::function<void(sk_handle_t*, void*, uint64_t**, double**, size_t*)>& run_query) {
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
    ASSERT_EQ(2, sk_bitset_storage_kind_for_testing(bitset_filter));

    uint64_t* ids = nullptr;
    double* scores = nullptr;
    size_t count = 0;
    run_query(handle, bitset_filter, &ids, &scores, &count);

    ASSERT_EQ(2u, count);
    EXPECT_EQ(40u, ids[0]);
    EXPECT_EQ(20u, ids[1]);

    sk_free(ids);
    sk_free(scores);
    sk_bitset_delete(bitset_filter);

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));
    sk_release_handle(handle);
    std::filesystem::remove_all(root);
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
    EXPECT_EQ(0, sk_bitset_storage_kind_for_testing(bitset_filter));
    sk_bitset_delete(bitset_filter);
}

TEST(sketch2api, bitset_filter_builder_named_empty_publishes_file) {
    const std::string name = unique_filter_name("api_empty_filter");

    void* state = nullptr;
    bool out_of_memory = false;
    const char* error_message = nullptr;
    ASSERT_EQ(0, sk_bitset_create_builder(
        &state, &out_of_memory, &error_message, name.c_str()))
        << (error_message != nullptr ? error_message : "");

    void* bitset_filter = nullptr;
    ASSERT_EQ(0, sk_bitset_finish(
        &state, &bitset_filter, &out_of_memory, &error_message))
        << (error_message != nullptr ? error_message : "");
    EXPECT_EQ(nullptr, state);
    ASSERT_NE(nullptr, bitset_filter);

    const std::filesystem::path expected_path = sketch2::named_bitset_filter_path(name);
    EXPECT_TRUE(std::filesystem::exists(expected_path));
    EXPECT_GT(std::filesystem::file_size(expected_path), 0u);

    sk_bitset_delete(bitset_filter);
    std::filesystem::remove(expected_path);
}

TEST(sketch2api, bitset_filter_drop_removes_named_file_and_is_idempotent) {
    const std::string name = unique_filter_name("api_drop_filter");

    void* state = nullptr;
    bool out_of_memory = false;
    const char* error_message = nullptr;
    ASSERT_EQ(0, sk_bitset_add_id(
        &state, 20, &out_of_memory, &error_message, name.c_str()))
        << (error_message != nullptr ? error_message : "");

    void* bitset_filter = nullptr;
    ASSERT_EQ(0, sk_bitset_finish(
        &state, &bitset_filter, &out_of_memory, &error_message))
        << (error_message != nullptr ? error_message : "");
    ASSERT_NE(nullptr, bitset_filter);
    sk_bitset_delete(bitset_filter);

    const std::filesystem::path expected_path = sketch2::named_bitset_filter_path(name);
    ASSERT_TRUE(std::filesystem::exists(expected_path));

    int removed = -1;
    EXPECT_EQ(0, sk_bitset_drop(name.c_str(), &removed, &out_of_memory, &error_message))
        << (error_message != nullptr ? error_message : "");
    EXPECT_EQ(1, removed);
    EXPECT_FALSE(std::filesystem::exists(expected_path));

    removed = -1;
    EXPECT_EQ(0, sk_bitset_drop(name.c_str(), &removed, &out_of_memory, &error_message))
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
    ASSERT_EQ(0, sk_bitset_add_id(
        &state, 20, &out_of_memory, &error_message, name.c_str()))
        << (error_message != nullptr ? error_message : "");
    ASSERT_EQ(0, sk_bitset_add_id(
        &state, 40, &out_of_memory, &error_message, name.c_str()))
        << (error_message != nullptr ? error_message : "");

    void* built_filter = nullptr;
    ASSERT_EQ(0, sk_bitset_finish(
        &state, &built_filter, &out_of_memory, &error_message))
        << (error_message != nullptr ? error_message : "");
    ASSERT_NE(nullptr, built_filter);
    sk_bitset_delete(built_filter);

    void* loaded_filter = nullptr;
    ASSERT_EQ(0, sk_bitset_load(
        name.c_str(), &loaded_filter, &out_of_memory, &error_message))
        << (error_message != nullptr ? error_message : "");
    ASSERT_NE(nullptr, loaded_filter);
    EXPECT_EQ(1, sk_bitset_storage_kind_for_testing(loaded_filter));
    sk_bitset_delete(loaded_filter);
}

TEST(sketch2api, bitset_filter_load_rejects_null_name) {
    void* bitset_filter = reinterpret_cast<void*>(0x1);
    bool out_of_memory = true;
    const char* error_message = nullptr;

    EXPECT_NE(0, sk_bitset_load(
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

    EXPECT_NE(0, sk_bitset_load(
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
    EXPECT_NE(0, sk_bitset_create_builder(
        &state, &out_of_memory, &error_message, ""));
    EXPECT_FALSE(out_of_memory);
    EXPECT_NE(nullptr, error_message);
    EXPECT_EQ(nullptr, state);
}

TEST(sketch2api, bitset_filter_drop_rejects_invalid_names) {
    int removed = -1;
    bool out_of_memory = false;
    const char* error_message = nullptr;

    EXPECT_NE(0, sk_bitset_drop(nullptr, &removed, &out_of_memory, &error_message));
    EXPECT_FALSE(out_of_memory);
    EXPECT_EQ(0, removed);
    ASSERT_NE(nullptr, error_message);
    EXPECT_NE(
        std::string(error_message).find("bitset_drop: name must not be null"),
        std::string::npos);

    removed = -1;
    error_message = nullptr;
    EXPECT_NE(0, sk_bitset_drop("", &removed, &out_of_memory, &error_message));
    EXPECT_FALSE(out_of_memory);
    EXPECT_EQ(0, removed);
    ASSERT_NE(nullptr, error_message);
    EXPECT_NE(
        std::string(error_message).find("bitset_drop: invalid bitset filter name"),
        std::string::npos);

    removed = -1;
    error_message = nullptr;
    EXPECT_NE(0, sk_bitset_drop("bad-name", &removed, &out_of_memory, &error_message));
    EXPECT_FALSE(out_of_memory);
    EXPECT_EQ(0, removed);
    ASSERT_NE(nullptr, error_message);
    EXPECT_NE(
        std::string(error_message).find("bitset_drop: invalid bitset filter name"),
        std::string::npos);
}

TEST(sketch2api, bitset_filter_cache_remove_evicts_entry_keeps_file) {
    const std::string name = unique_filter_name("api_cache_remove");
    const std::filesystem::path expected_path = sketch2::named_bitset_filter_path(name);
    std::experimental::scope_exit cleanup([&]() {
        sketch2::bitset_file_cache().remove(name);
        std::filesystem::remove(expected_path);
    });

    void* state = nullptr;
    bool out_of_memory = false;
    const char* error_message = nullptr;
    ASSERT_EQ(0, sk_bitset_add_id(
        &state, 20, &out_of_memory, &error_message, name.c_str()))
        << (error_message != nullptr ? error_message : "");

    void* built_filter = nullptr;
    ASSERT_EQ(0, sk_bitset_finish(
        &state, &built_filter, &out_of_memory, &error_message))
        << (error_message != nullptr ? error_message : "");
    ASSERT_NE(nullptr, built_filter);
    sk_bitset_delete(built_filter);

    ASSERT_TRUE(sketch2::bitset_file_cache().contains(name));
    ASSERT_TRUE(std::filesystem::exists(expected_path));

    int removed = -1;
    EXPECT_EQ(0, sk_bitset_cache_remove(
        name.c_str(), &removed, &out_of_memory, &error_message))
        << (error_message != nullptr ? error_message : "");
    EXPECT_EQ(1, removed);
    EXPECT_FALSE(sketch2::bitset_file_cache().contains(name));
    EXPECT_TRUE(std::filesystem::exists(expected_path));

    removed = -1;
    EXPECT_EQ(0, sk_bitset_cache_remove(
        name.c_str(), &removed, &out_of_memory, &error_message))
        << (error_message != nullptr ? error_message : "");
    EXPECT_EQ(0, removed);
}

TEST(sketch2api, bitset_filter_cache_clear_evicts_all_keeps_files) {
    const std::string name_a = unique_filter_name("api_cache_clear_a");
    const std::string name_b = unique_filter_name("api_cache_clear_b");
    const std::filesystem::path path_a = sketch2::named_bitset_filter_path(name_a);
    const std::filesystem::path path_b = sketch2::named_bitset_filter_path(name_b);
    std::experimental::scope_exit cleanup([&]() {
        sketch2::bitset_file_cache().remove(name_a);
        sketch2::bitset_file_cache().remove(name_b);
        std::filesystem::remove(path_a);
        std::filesystem::remove(path_b);
    });

    bool out_of_memory = false;
    const char* error_message = nullptr;

    for (const std::string& name : {name_a, name_b}) {
        void* state = nullptr;
        ASSERT_EQ(0, sk_bitset_add_id(
            &state, 20, &out_of_memory, &error_message, name.c_str()))
            << (error_message != nullptr ? error_message : "");
        void* built = nullptr;
        ASSERT_EQ(0, sk_bitset_finish(
            &state, &built, &out_of_memory, &error_message))
            << (error_message != nullptr ? error_message : "");
        sk_bitset_delete(built);
    }

    ASSERT_TRUE(sketch2::bitset_file_cache().contains(name_a));
    ASSERT_TRUE(sketch2::bitset_file_cache().contains(name_b));

    EXPECT_EQ(0, sk_bitset_cache_clear(&out_of_memory, &error_message))
        << (error_message != nullptr ? error_message : "");
    EXPECT_FALSE(sketch2::bitset_file_cache().contains(name_a));
    EXPECT_FALSE(sketch2::bitset_file_cache().contains(name_b));
    EXPECT_TRUE(std::filesystem::exists(path_a));
    EXPECT_TRUE(std::filesystem::exists(path_b));
}

TEST(sketch2api, bitset_filter_cache_remove_rejects_invalid_args) {
    int removed = -1;
    bool out_of_memory = false;
    const char* error_message = nullptr;

    EXPECT_NE(0, sk_bitset_cache_remove(
        nullptr, &removed, &out_of_memory, &error_message));
    EXPECT_FALSE(out_of_memory);
    EXPECT_EQ(0, removed);
    ASSERT_NE(nullptr, error_message);
    EXPECT_NE(
        std::string(error_message).find("bitset_cache_remove: name must not be null"),
        std::string::npos);

    removed = -1;
    error_message = nullptr;
    EXPECT_NE(0, sk_bitset_cache_remove(
        "", &removed, &out_of_memory, &error_message));
    EXPECT_FALSE(out_of_memory);
    EXPECT_EQ(0, removed);
    ASSERT_NE(nullptr, error_message);
    EXPECT_NE(
        std::string(error_message).find("bitset_cache_remove: invalid bitset filter name"),
        std::string::npos);

    removed = -1;
    error_message = nullptr;
    EXPECT_NE(0, sk_bitset_cache_remove(
        "bad-name", &removed, &out_of_memory, &error_message));
    EXPECT_FALSE(out_of_memory);
    EXPECT_EQ(0, removed);
    ASSERT_NE(nullptr, error_message);
    EXPECT_NE(
        std::string(error_message).find("bitset_cache_remove: invalid bitset filter name"),
        std::string::npos);

    error_message = nullptr;
    EXPECT_NE(0, sk_bitset_cache_remove(
        "k", nullptr, &out_of_memory, &error_message));
    EXPECT_FALSE(out_of_memory);
    ASSERT_NE(nullptr, error_message);
    EXPECT_NE(
        std::string(error_message).find("bitset_cache_remove: invalid result output"),
        std::string::npos);
}

TEST(sketch2api, bitset_filter_add_multiple_ids_name_appends_all) {
    const std::string name = unique_filter_name("api_add_multi_filter");
    const std::filesystem::path expected_path = sketch2::named_bitset_filter_path(name);
    std::experimental::scope_exit cleanup([&]() { std::filesystem::remove(expected_path); });

    void* state = nullptr;
    bool out_of_memory = false;
    const char* error_message = nullptr;
    ASSERT_EQ(0, sk_bitset_create_builder(
        &state, &out_of_memory, &error_message, name.c_str()))
        << (error_message != nullptr ? error_message : "");

    const std::vector<uint64_t> ids = {7, 42, 100, 1'000'001, 5'000'000};
    ASSERT_EQ(0, sk_bitset_add_multiple_ids_name(
        &state, ids.data(), ids.size(), &out_of_memory, &error_message))
        << (error_message != nullptr ? error_message : "");

    void* built_filter = nullptr;
    ASSERT_EQ(0, sk_bitset_finish(
        &state, &built_filter, &out_of_memory, &error_message))
        << (error_message != nullptr ? error_message : "");
    ASSERT_NE(nullptr, built_filter);
    sk_bitset_delete(built_filter);

    sketch2::test::expect_persisted_filter_contains(
        expected_path, {7, 42, 100, 1'000'001, 5'000'000});
}

TEST(sketch2api, bitset_filter_add_multiple_ids_name_rejects_null_state) {
    bool out_of_memory = true;
    const char* error_message = nullptr;
    const uint64_t ids[] = {1, 2, 3};

    EXPECT_NE(0, sk_bitset_add_multiple_ids_name(
        nullptr, ids, 3, &out_of_memory, &error_message));
    EXPECT_FALSE(out_of_memory);
    ASSERT_NE(nullptr, error_message);
    EXPECT_NE(
        std::string(error_message).find("bitset filter builder: invalid builder state"),
        std::string::npos);

    void* state = nullptr;
    out_of_memory = true;
    error_message = nullptr;
    EXPECT_NE(0, sk_bitset_add_multiple_ids_name(
        &state, ids, 3, &out_of_memory, &error_message));
    EXPECT_FALSE(out_of_memory);
    EXPECT_EQ(nullptr, state);
    ASSERT_NE(nullptr, error_message);
    EXPECT_NE(
        std::string(error_message).find("bitset filter builder: invalid builder state"),
        std::string::npos);
}

TEST(sketch2api, bitset_filter_add_multiple_ids_name_rejects_zero_size) {
    const std::string name = unique_filter_name("api_add_multi_zero_size");

    void* state = nullptr;
    bool out_of_memory = false;
    const char* error_message = nullptr;
    ASSERT_EQ(0, sk_bitset_create_builder(
        &state, &out_of_memory, &error_message, name.c_str()))
        << (error_message != nullptr ? error_message : "");

    const uint64_t ids[] = {1};
    out_of_memory = true;
    error_message = nullptr;
    EXPECT_NE(0, sk_bitset_add_multiple_ids_name(
        &state, ids, 0, &out_of_memory, &error_message));
    EXPECT_FALSE(out_of_memory);
    ASSERT_NE(nullptr, error_message);
    EXPECT_NE(
        std::string(error_message).find(
            "bitset filter builder: ids size must be greater than zero"),
        std::string::npos);

    delete static_cast<sketch2::ChunkedBits*>(state);
}

TEST(sketch2api, bitset_filter_add_multiple_ids_name_rejects_null_ids) {
    const std::string name = unique_filter_name("api_add_multi_null_ids");

    void* state = nullptr;
    bool out_of_memory = false;
    const char* error_message = nullptr;
    ASSERT_EQ(0, sk_bitset_create_builder(
        &state, &out_of_memory, &error_message, name.c_str()))
        << (error_message != nullptr ? error_message : "");

    out_of_memory = true;
    error_message = nullptr;
    EXPECT_NE(0, sk_bitset_add_multiple_ids_name(
        &state, nullptr, 3, &out_of_memory, &error_message));
    EXPECT_FALSE(out_of_memory);
    ASSERT_NE(nullptr, error_message);
    EXPECT_NE(
        std::string(error_message).find("bitset filter builder: ids pointer is null"),
        std::string::npos);

    delete static_cast<sketch2::ChunkedBits*>(state);
}

TEST(sketch2api, bitset_filter_add_multiple_ids_name_mixes_with_single_id) {
    const std::string name = unique_filter_name("api_add_multi_mixed");
    const std::filesystem::path expected_path = sketch2::named_bitset_filter_path(name);
    std::experimental::scope_exit cleanup([&]() { std::filesystem::remove(expected_path); });

    void* state = nullptr;
    bool out_of_memory = false;
    const char* error_message = nullptr;
    ASSERT_EQ(0, sk_bitset_create_builder(
        &state, &out_of_memory, &error_message, name.c_str()))
        << (error_message != nullptr ? error_message : "");

    ASSERT_EQ(0, sk_bitset_add_id_name(
        &state, 11, &out_of_memory, &error_message))
        << (error_message != nullptr ? error_message : "");

    const std::vector<uint64_t> batch = {22, 33, 44};
    ASSERT_EQ(0, sk_bitset_add_multiple_ids_name(
        &state, batch.data(), batch.size(), &out_of_memory, &error_message))
        << (error_message != nullptr ? error_message : "");

    ASSERT_EQ(0, sk_bitset_add_id_name(
        &state, 55, &out_of_memory, &error_message))
        << (error_message != nullptr ? error_message : "");

    void* built_filter = nullptr;
    ASSERT_EQ(0, sk_bitset_finish(
        &state, &built_filter, &out_of_memory, &error_message))
        << (error_message != nullptr ? error_message : "");
    ASSERT_NE(nullptr, built_filter);
    sk_bitset_delete(built_filter);

    sketch2::test::expect_persisted_filter_contains(
        expected_path, {11, 22, 33, 44, 55});
}

TEST(sketch2api, knn_items_bitset_filter_filters_with_spilled_bitset_filter) {
    assert_knn_with_spilled_bitset_filter([](
            sk_handle_t* handle, void* bitset_filter, uint64_t** ids, double** scores, size_t* count) {
        ASSERT_EQ(0, sk_knn_items_bitset_filter(
            handle, "25.0, 25.0, 25.0, 25.0", 4, bitset_filter, ids, scores, count))
            << sk_error_message(handle);
    });
}

TEST(sketch2api, knn_vector_items_bitset_filter_filters_with_spilled_bitset_filter) {
    assert_knn_with_spilled_bitset_filter([](
            sk_handle_t* handle, void* bitset_filter, uint64_t** ids, double** scores, size_t* count) {
        const std::vector<float> query = {25.0f, 25.0f, 25.0f, 25.0f};
        ASSERT_EQ(0, sk_knn_vector_items_bitset_filter(
            handle, query.data(), query.size(), 4, bitset_filter, ids, scores, count))
            << sk_error_message(handle);
    });
}
