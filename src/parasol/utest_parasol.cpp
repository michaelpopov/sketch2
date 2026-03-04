#include "parasol.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <unistd.h>

#include <gtest/gtest.h>

namespace {

std::filesystem::path make_temp_dir() {
    std::filesystem::path base = std::filesystem::temp_directory_path();
    std::filesystem::create_directories(base);
    std::filesystem::path dir =
        base / std::filesystem::path("sketch2_parasol_ut_" + std::to_string(::getpid()) + "_" +
                                     std::to_string(std::rand()));
    std::filesystem::create_directories(dir);
    return dir;
}

std::filesystem::path make_dataset_dir(const std::filesystem::path& root) {
    return root / "dataset";
}

sk_dataset_metadata_t make_metadata(const std::filesystem::path &dir) {
    sk_dataset_metadata_t md {};
    std::snprintf(md.dir, sizeof(md.dir), "%s", dir.string().c_str());
    std::snprintf(md.type, sizeof(md.type), "%s", "f32");
    md.dim = 4;
    md.range_size = 1000;
    md.data_merge_ratio = 2;
    return md;
}

std::string read_file(const std::filesystem::path &path) {
    std::ifstream in(path);
    return std::string((std::istreambuf_iterator<char>(in)),
        std::istreambuf_iterator<char>());
}

void write_metadata_file(const std::filesystem::path& dir, const char* type = "f32") {
    std::filesystem::create_directories(dir);
    const std::filesystem::path metadata = dir / "sketch2.metadata";
    std::ofstream out(metadata);
    out << "[dataset]\n";
    out << "dirs=" << dir.string() << "\n";
    out << "range_size=1000\n";
    out << "dim=4\n";
    out << "type=" << type << "\n";
}

void write_marker_file(const std::filesystem::path& dir) {
    std::filesystem::create_directories(dir);
    const std::filesystem::path marker = dir / ".sketch2.managed";
    std::ofstream out(marker);
    out << "managed=1\n";
}

} // namespace

TEST(parasol, create_writes_metadata_ini) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir = make_dataset_dir(root);
    const sk_dataset_metadata_t md = make_metadata(dir);

    const sk_ret_t create_ret = sk_create(md);
    ASSERT_EQ(create_ret.code, 0) << create_ret.message;

    const std::filesystem::path metadata_path = dir / "sketch2.metadata";
    ASSERT_TRUE(std::filesystem::exists(metadata_path));

    const std::string body = read_file(metadata_path);
    EXPECT_NE(body.find("[dataset]\n"), std::string::npos);
    EXPECT_NE(body.find("dirs=" + dir.string() + "\n"), std::string::npos);
    EXPECT_NE(body.find("range_size=1000\n"), std::string::npos);
    EXPECT_NE(body.find("dim=4\n"), std::string::npos);
    EXPECT_NE(body.find("type=f32\n"), std::string::npos);

    const sk_ret_t drop_ret = sk_drop(dir.string().c_str());
    ASSERT_EQ(drop_ret.code, 0) << drop_ret.message;
    std::filesystem::remove_all(root);
}

TEST(parasol, create_fails_on_duplicate_metadata) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir = make_dataset_dir(root);
    const sk_dataset_metadata_t md = make_metadata(dir);

    ASSERT_EQ(sk_create(md).code, 0);
    const sk_ret_t second = sk_create(md);
    EXPECT_NE(second.code, 0);

    ASSERT_EQ(sk_drop(dir.string().c_str()).code, 0);
    std::filesystem::remove_all(root);
}

TEST(parasol, open_close_success) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir = make_dataset_dir(root);
    const sk_dataset_metadata_t md = make_metadata(dir);
    ASSERT_EQ(sk_create(md).code, 0);

    const sk_ret_t open_ret = sk_open(dir.string().c_str());
    ASSERT_EQ(open_ret.code, 0) << open_ret.message;
    ASSERT_NE(open_ret.handle, nullptr);

    const sk_ret_t close_ret = sk_close(open_ret.handle);
    EXPECT_EQ(close_ret.code, 0) << close_ret.message;

    ASSERT_EQ(sk_drop(dir.string().c_str()).code, 0);
    std::filesystem::remove_all(root);
}

TEST(parasol, add_delete_write_to_input_file) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir = make_dataset_dir(root);
    const sk_dataset_metadata_t md = make_metadata(dir);
    ASSERT_EQ(sk_create(md).code, 0);

    const sk_ret_t open_ret = sk_open(dir.string().c_str());
    ASSERT_EQ(open_ret.code, 0) << open_ret.message;

    const sk_ret_t add_ret = sk_add(open_ret.handle, 42, "1 2 3 4");
    ASSERT_EQ(add_ret.code, 0) << add_ret.message;

    const sk_ret_t del_ret = sk_delete(open_ret.handle, 42);
    ASSERT_EQ(del_ret.code, 0) << del_ret.message;

    ASSERT_EQ(sk_close(open_ret.handle).code, 0);

    const std::filesystem::path input_path = dir / "data.input";
    ASSERT_TRUE(std::filesystem::exists(input_path));
    const std::string body = read_file(input_path);
    EXPECT_NE(body.find("42 : [ 1 2 3 4 ]\n"), std::string::npos);
    EXPECT_NE(body.find("42 : []\n"), std::string::npos);

    ASSERT_EQ(sk_drop(dir.string().c_str()).code, 0);
    std::filesystem::remove_all(root);
}

TEST(parasol, create_fails_on_invalid_dir) {
    sk_dataset_metadata_t md {};
    std::snprintf(md.type, sizeof(md.type), "%s", "f32");
    md.dim = 4;
    md.range_size = 1000;
    md.data_merge_ratio = 2;

    const sk_ret_t ret = sk_create(md);
    EXPECT_NE(ret.code, 0);
}

TEST(parasol, create_fails_on_invalid_type) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir = make_dataset_dir(root);
    sk_dataset_metadata_t md = make_metadata(dir);
    std::snprintf(md.type, sizeof(md.type), "%s", "bad_type");

    const sk_ret_t ret = sk_create(md);
    EXPECT_NE(ret.code, 0);

    std::filesystem::remove_all(root);
}

TEST(parasol, drop_fails_on_null_dir) {
    const sk_ret_t ret = sk_drop(nullptr);
    EXPECT_NE(ret.code, 0);
}

TEST(parasol, drop_fails_on_nonexistent_dir) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path missing = root / "missing";
    const sk_ret_t ret = sk_drop(missing.string().c_str());
    EXPECT_NE(ret.code, 0);
    std::filesystem::remove_all(root);
}

TEST(parasol, drop_fails_without_metadata) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir = make_dataset_dir(root);
    std::filesystem::create_directories(dir);
    write_marker_file(dir);

    const sk_ret_t ret = sk_drop(dir.string().c_str());
    EXPECT_NE(ret.code, 0);

    std::filesystem::remove_all(root);
}

TEST(parasol, drop_fails_without_marker) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir = make_dataset_dir(root);
    write_metadata_file(dir);

    const sk_ret_t ret = sk_drop(dir.string().c_str());
    EXPECT_NE(ret.code, 0);

    std::filesystem::remove_all(root);
}

TEST(parasol, open_fails_on_null_path) {
    const sk_ret_t ret = sk_open(nullptr);
    EXPECT_NE(ret.code, 0);
}

TEST(parasol, open_fails_without_metadata_file) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir = make_dataset_dir(root);
    std::filesystem::create_directories(dir);

    const sk_ret_t ret = sk_open(dir.string().c_str());
    EXPECT_NE(ret.code, 0);

    std::filesystem::remove_all(root);
}

TEST(parasol, open_fails_on_invalid_metadata_content) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir = make_dataset_dir(root);
    write_metadata_file(dir, "bad_type");
    write_marker_file(dir);

    const sk_ret_t ret = sk_open(dir.string().c_str());
    EXPECT_NE(ret.code, 0);

    std::filesystem::remove_all(root);
}

TEST(parasol, close_fails_on_null_handle) {
    const sk_ret_t ret = sk_close(nullptr);
    EXPECT_NE(ret.code, 0);
}

TEST(parasol, repeated_close_contract_requires_nulling_handle) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir = make_dataset_dir(root);
    const sk_dataset_metadata_t md = make_metadata(dir);
    ASSERT_EQ(sk_create(md).code, 0);

    const sk_ret_t open_ret = sk_open(dir.string().c_str());
    ASSERT_EQ(open_ret.code, 0) << open_ret.message;

    sk_handle_t* handle = open_ret.handle;
    ASSERT_EQ(sk_close(handle).code, 0);
    handle = nullptr;

    const sk_ret_t second_close = sk_close(handle);
    EXPECT_NE(second_close.code, 0);

    ASSERT_EQ(sk_drop(dir.string().c_str()).code, 0);
    std::filesystem::remove_all(root);
}

TEST(parasol, add_fails_on_null_handle) {
    const sk_ret_t ret = sk_add(nullptr, 1, "1 2 3 4");
    EXPECT_NE(ret.code, 0);
}

TEST(parasol, add_fails_on_null_value) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir = make_dataset_dir(root);
    const sk_dataset_metadata_t md = make_metadata(dir);
    ASSERT_EQ(sk_create(md).code, 0);

    const sk_ret_t open_ret = sk_open(dir.string().c_str());
    ASSERT_EQ(open_ret.code, 0) << open_ret.message;

    const sk_ret_t ret = sk_add(open_ret.handle, 7, nullptr);
    EXPECT_NE(ret.code, 0);

    ASSERT_EQ(sk_close(open_ret.handle).code, 0);
    ASSERT_EQ(sk_drop(dir.string().c_str()).code, 0);
    std::filesystem::remove_all(root);
}

TEST(parasol, delete_fails_on_null_handle) {
    const sk_ret_t ret = sk_delete(nullptr, 1);
    EXPECT_NE(ret.code, 0);
}

// ---------------------------------------------------------------------------
// sk_load tests
// ---------------------------------------------------------------------------

// Helper: count regular files with a given extension inside a directory.
namespace {
size_t count_files_with_ext(const std::filesystem::path& dir, const std::string& ext) {
    size_t n = 0;
    for (const auto& entry : std::filesystem::directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ext) {
            ++n;
        }
    }
    return n;
}
} // namespace

// 1. Null handle is rejected immediately.
TEST(parasol, load_fails_on_null_handle) {
    const sk_ret_t ret = sk_load(nullptr);
    EXPECT_NE(ret.code, 0);
}

// 2. No input file was ever created (sk_add/sk_delete were never called).
TEST(parasol, load_fails_without_input_file) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir  = make_dataset_dir(root);
    ASSERT_EQ(sk_create(make_metadata(dir)).code, 0);

    const sk_ret_t open_ret = sk_open(dir.string().c_str());
    ASSERT_EQ(open_ret.code, 0) << open_ret.message;

    const sk_ret_t load_ret = sk_load(open_ret.handle);
    EXPECT_NE(load_ret.code, 0);

    ASSERT_EQ(sk_close(open_ret.handle).code, 0);
    ASSERT_EQ(sk_drop(dir.string().c_str()).code, 0);
    std::filesystem::remove_all(root);
}

// 3. After sk_add + sk_load the input file is consumed (deleted).
TEST(parasol, load_removes_input_file_on_success) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir  = make_dataset_dir(root);
    ASSERT_EQ(sk_create(make_metadata(dir)).code, 0);

    const sk_ret_t open_ret = sk_open(dir.string().c_str());
    ASSERT_EQ(open_ret.code, 0) << open_ret.message;

    ASSERT_EQ(sk_add(open_ret.handle, 5, "1.0, 2.0, 3.0, 4.0").code, 0);

    const sk_ret_t load_ret = sk_load(open_ret.handle);
    ASSERT_EQ(load_ret.code, 0) << load_ret.message;

    EXPECT_FALSE(std::filesystem::exists(dir / "data.input"));

    ASSERT_EQ(sk_close(open_ret.handle).code, 0);
    ASSERT_EQ(sk_drop(dir.string().c_str()).code, 0);
    std::filesystem::remove_all(root);
}

// 4. sk_add leaves handle->input open for writing; sk_load must close/flush
//    it before handing the file to InputReader, otherwise data may be missing.
TEST(parasol, load_flushes_open_write_handle_before_reading) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir  = make_dataset_dir(root);
    ASSERT_EQ(sk_create(make_metadata(dir)).code, 0);

    const sk_ret_t open_ret = sk_open(dir.string().c_str());
    ASSERT_EQ(open_ret.code, 0) << open_ret.message;

    // sk_add opens handle->input lazily; the file handle stays open after this.
    ASSERT_EQ(sk_add(open_ret.handle, 7, "1.0, 2.0, 3.0, 4.0").code, 0);

    // sk_load must close/flush the write handle, then load successfully.
    const sk_ret_t load_ret = sk_load(open_ret.handle);
    EXPECT_EQ(load_ret.code, 0) << load_ret.message;

    ASSERT_EQ(sk_close(open_ret.handle).code, 0);
    ASSERT_EQ(sk_drop(dir.string().c_str()).code, 0);
    std::filesystem::remove_all(root);
}

// 5. A successful load produces a binary .data file in the dataset directory.
TEST(parasol, load_creates_data_file) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir  = make_dataset_dir(root);
    ASSERT_EQ(sk_create(make_metadata(dir)).code, 0);

    const sk_ret_t open_ret = sk_open(dir.string().c_str());
    ASSERT_EQ(open_ret.code, 0) << open_ret.message;

    // id=5, range_size=1000 -> file_id=0 -> "0.data"
    ASSERT_EQ(sk_add(open_ret.handle, 5, "1.0, 2.0, 3.0, 4.0").code, 0);
    ASSERT_EQ(sk_load(open_ret.handle).code, 0);

    EXPECT_TRUE(std::filesystem::exists(dir / "0.data"));

    ASSERT_EQ(sk_close(open_ret.handle).code, 0);
    ASSERT_EQ(sk_drop(dir.string().c_str()).code, 0);
    std::filesystem::remove_all(root);
}

// 6. A delete-only input succeeds once the dataset already has a base data file.
//    (A delete-only first load is rejected because there is nothing to delete yet.)
TEST(parasol, load_with_delete_only_input_succeeds) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir  = make_dataset_dir(root);
    ASSERT_EQ(sk_create(make_metadata(dir)).code, 0);

    const sk_ret_t open_ret = sk_open(dir.string().c_str());
    ASSERT_EQ(open_ret.code, 0) << open_ret.message;
    sk_handle_t* handle = open_ret.handle;

    // Round 1: establish base data so that a .data file exists.
    ASSERT_EQ(sk_add(handle, 42, "1.0, 2.0, 3.0, 4.0").code, 0);
    ASSERT_EQ(sk_load(handle).code, 0);

    // Round 2: delete-only input — this is accepted on an established dataset.
    ASSERT_EQ(sk_delete(handle, 42).code, 0);
    const sk_ret_t load_ret = sk_load(handle);
    EXPECT_EQ(load_ret.code, 0) << load_ret.message;

    ASSERT_EQ(sk_close(handle).code, 0);
    ASSERT_EQ(sk_drop(dir.string().c_str()).code, 0);
    std::filesystem::remove_all(root);
}

// 7. Calling sk_load a second time without any intervening sk_add/sk_delete
//    fails because the previous load consumed (deleted) the input file.
TEST(parasol, load_twice_second_call_fails) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir  = make_dataset_dir(root);
    ASSERT_EQ(sk_create(make_metadata(dir)).code, 0);

    const sk_ret_t open_ret = sk_open(dir.string().c_str());
    ASSERT_EQ(open_ret.code, 0) << open_ret.message;

    ASSERT_EQ(sk_add(open_ret.handle, 3, "1.0, 2.0, 3.0, 4.0").code, 0);
    ASSERT_EQ(sk_load(open_ret.handle).code, 0);

    // No new sk_add/sk_delete, so data.input does not exist.
    const sk_ret_t second_load = sk_load(open_ret.handle);
    EXPECT_NE(second_load.code, 0);

    ASSERT_EQ(sk_close(open_ret.handle).code, 0);
    ASSERT_EQ(sk_drop(dir.string().c_str()).code, 0);
    std::filesystem::remove_all(root);
}

// 8. Ids must be in strictly ascending order in the input file.
//    sk_add does not enforce ordering; sk_load must reject the malformed file.
TEST(parasol, load_fails_with_unsorted_ids) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir  = make_dataset_dir(root);
    ASSERT_EQ(sk_create(make_metadata(dir)).code, 0);

    const sk_ret_t open_ret = sk_open(dir.string().c_str());
    ASSERT_EQ(open_ret.code, 0) << open_ret.message;

    // Write id=10 before id=5 — InputReader will reject out-of-order ids.
    ASSERT_EQ(sk_add(open_ret.handle, 10, "1.0, 2.0, 3.0, 4.0").code, 0);
    ASSERT_EQ(sk_add(open_ret.handle, 5,  "1.0, 2.0, 3.0, 4.0").code, 0);

    const sk_ret_t load_ret = sk_load(open_ret.handle);
    EXPECT_NE(load_ret.code, 0);

    ASSERT_EQ(sk_close(open_ret.handle).code, 0);
    ASSERT_EQ(sk_drop(dir.string().c_str()).code, 0);
    std::filesystem::remove_all(root);
}

// 9. Vectors spanning multiple id ranges produce one .data file per range.
TEST(parasol, load_vectors_across_multiple_ranges) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir  = make_dataset_dir(root);
    ASSERT_EQ(sk_create(make_metadata(dir)).code, 0);

    const sk_ret_t open_ret = sk_open(dir.string().c_str());
    ASSERT_EQ(open_ret.code, 0) << open_ret.message;

    // range_size=1000: id=5 -> file_id=0, id=1005 -> file_id=1, id=2005 -> file_id=2
    ASSERT_EQ(sk_add(open_ret.handle,    5, "1.0, 2.0, 3.0, 4.0").code, 0);
    ASSERT_EQ(sk_add(open_ret.handle, 1005, "2.0, 3.0, 4.0, 5.0").code, 0);
    ASSERT_EQ(sk_add(open_ret.handle, 2005, "3.0, 4.0, 5.0, 6.0").code, 0);

    ASSERT_EQ(sk_load(open_ret.handle).code, 0);

    EXPECT_TRUE(std::filesystem::exists(dir / "0.data"));
    EXPECT_TRUE(std::filesystem::exists(dir / "1.data"));
    EXPECT_TRUE(std::filesystem::exists(dir / "2.data"));
    EXPECT_EQ(3u, count_files_with_ext(dir, ".data"));

    ASSERT_EQ(sk_close(open_ret.handle).code, 0);
    ASSERT_EQ(sk_drop(dir.string().c_str()).code, 0);
    std::filesystem::remove_all(root);
}

// 10. The handle remains fully usable across two independent add->load cycles.
TEST(parasol, load_handle_usable_across_two_cycles) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir  = make_dataset_dir(root);
    ASSERT_EQ(sk_create(make_metadata(dir)).code, 0);

    const sk_ret_t open_ret = sk_open(dir.string().c_str());
    ASSERT_EQ(open_ret.code, 0) << open_ret.message;
    sk_handle_t* handle = open_ret.handle;

    // Cycle 1: id=5 -> 0.data
    ASSERT_EQ(sk_add(handle, 5, "1.0, 2.0, 3.0, 4.0").code, 0);
    ASSERT_EQ(sk_load(handle).code, 0);
    EXPECT_TRUE(std::filesystem::exists(dir / "0.data"));
    EXPECT_FALSE(std::filesystem::exists(dir / "data.input"));

    // Cycle 2: id=1005 -> 1.data
    ASSERT_EQ(sk_add(handle, 1005, "5.0, 6.0, 7.0, 8.0").code, 0);
    ASSERT_EQ(sk_load(handle).code, 0);
    EXPECT_TRUE(std::filesystem::exists(dir / "1.data"));
    EXPECT_FALSE(std::filesystem::exists(dir / "data.input"));

    ASSERT_EQ(sk_close(handle).code, 0);
    ASSERT_EQ(sk_drop(dir.string().c_str()).code, 0);
    std::filesystem::remove_all(root);
}

// ---------------------------------------------------------------------------
// sk_knn tests
// ---------------------------------------------------------------------------

TEST(parasol, knn_finds_expected_neighbors) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir  = make_dataset_dir(root);
    ASSERT_EQ(sk_create(make_metadata(dir)).code, 0);

    const sk_ret_t open_ret = sk_open(dir.string().c_str());
    ASSERT_EQ(open_ret.code, 0) << open_ret.message;
    sk_handle_t* handle = open_ret.handle;

    ASSERT_EQ(sk_add(handle, 1, "0.0, 0.0, 0.0, 0.0").code, 0);
    ASSERT_EQ(sk_add(handle, 2, "10.0, 10.0, 10.0, 10.0").code, 0);
    ASSERT_EQ(sk_add(handle, 3, "1.0, 1.0, 1.0, 1.0").code, 0);
    ASSERT_EQ(sk_load(handle).code, 0);

    uint64_t ids[2] = {0, 0};
    const sk_ret_t knn_ret = sk_knn(handle, "0.0, 0.0, 0.0, 0.0", ids, 2);
    ASSERT_EQ(knn_ret.code, 0) << knn_ret.message;
    EXPECT_EQ(ids[0], 1u);
    EXPECT_EQ(ids[1], 3u);

    ASSERT_EQ(sk_close(handle).code, 0);
    ASSERT_EQ(sk_drop(dir.string().c_str()).code, 0);
    std::filesystem::remove_all(root);
}

TEST(parasol, knn_fails_on_invalid_arguments) {
    uint64_t ids[2] = {0, 0};

    EXPECT_NE(sk_knn(nullptr, "0.0,0.0,0.0,0.0", ids, 2).code, 0);

    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir  = make_dataset_dir(root);
    ASSERT_EQ(sk_create(make_metadata(dir)).code, 0);

    const sk_ret_t open_ret = sk_open(dir.string().c_str());
    ASSERT_EQ(open_ret.code, 0) << open_ret.message;
    sk_handle_t* handle = open_ret.handle;

    EXPECT_NE(sk_knn(handle, nullptr, ids, 2).code, 0);
    EXPECT_NE(sk_knn(handle, "0.0,0.0,0.0,0.0", nullptr, 2).code, 0);
    EXPECT_NE(sk_knn(handle, "0.0,0.0,0.0,0.0", ids, 0).code, 0);

    ASSERT_EQ(sk_close(handle).code, 0);
    ASSERT_EQ(sk_drop(dir.string().c_str()).code, 0);
    std::filesystem::remove_all(root);
}

TEST(parasol, knn_fails_on_invalid_query_vector) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir  = make_dataset_dir(root);
    ASSERT_EQ(sk_create(make_metadata(dir)).code, 0);

    const sk_ret_t open_ret = sk_open(dir.string().c_str());
    ASSERT_EQ(open_ret.code, 0) << open_ret.message;
    sk_handle_t* handle = open_ret.handle;

    ASSERT_EQ(sk_add(handle, 1, "0.0, 0.0, 0.0, 0.0").code, 0);
    ASSERT_EQ(sk_load(handle).code, 0);

    uint64_t ids[1] = {0};
    const sk_ret_t ret = sk_knn(handle, "bad,bad,bad,bad", ids, 1);
    EXPECT_NE(ret.code, 0);

    ASSERT_EQ(sk_close(handle).code, 0);
    ASSERT_EQ(sk_drop(dir.string().c_str()).code, 0);
    std::filesystem::remove_all(root);
}
