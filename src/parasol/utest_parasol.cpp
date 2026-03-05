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

#define ASSERT_OK(handle, call_expr) ASSERT_EQ(0, (call_expr)) << sk_error_message(handle)
#define EXPECT_OK(handle, call_expr) EXPECT_EQ(0, (call_expr)) << sk_error_message(handle)

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

TEST(parasol, create_writes_metadata_ini) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir = make_dataset_dir(root);
    const sk_dataset_metadata_t md = make_metadata(dir);

    sk_handle_t* handle = connect();
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, md));

    const std::filesystem::path metadata_path = dir / "sketch2.metadata";
    ASSERT_TRUE(std::filesystem::exists(metadata_path));

    const std::string body = read_file(metadata_path);
    EXPECT_NE(body.find("[dataset]\n"), std::string::npos);
    EXPECT_NE(body.find("dirs=" + dir.string() + "\n"), std::string::npos);
    EXPECT_NE(body.find("range_size=1000\n"), std::string::npos);
    EXPECT_NE(body.find("dim=4\n"), std::string::npos);
    EXPECT_NE(body.find("type=f32\n"), std::string::npos);

    ASSERT_OK(handle, sk_drop(handle));
    disconnect(handle);
    std::filesystem::remove_all(root);
}

TEST(parasol, create_fails_on_duplicate_metadata) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir = make_dataset_dir(root);
    const sk_dataset_metadata_t md = make_metadata(dir);

    sk_handle_t* handle = connect();
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, md));
    EXPECT_NE(0, sk_create(handle, md));

    ASSERT_OK(handle, sk_drop(handle));
    disconnect(handle);
    std::filesystem::remove_all(root);
}

TEST(parasol, open_close_success) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir = make_dataset_dir(root);

    sk_handle_t* handle = connect();
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, make_metadata(dir)));
    ASSERT_OK(handle, sk_open(handle, dir.string().c_str()));
    EXPECT_OK(handle, sk_close(handle));

    disconnect(handle);
    std::filesystem::remove_all(root);
}

TEST(parasol, add_delete_write_to_input_file) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir = make_dataset_dir(root);

    sk_handle_t* handle = connect();
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, make_metadata(dir)));
    ASSERT_OK(handle, sk_open(handle, dir.string().c_str()));

    ASSERT_OK(handle, sk_add(handle, 42, "1 2 3 4"));
    ASSERT_OK(handle, sk_delete(handle, 42));
    ASSERT_OK(handle, sk_close(handle));

    const std::filesystem::path input_path = dir / "data.input";
    ASSERT_TRUE(std::filesystem::exists(input_path));
    const std::string body = read_file(input_path);
    EXPECT_NE(body.find("42 : [ 1 2 3 4 ]\n"), std::string::npos);
    EXPECT_NE(body.find("42 : []\n"), std::string::npos);

    disconnect(handle);
    std::filesystem::remove_all(root);
}

TEST(parasol, create_fails_on_invalid_dir) {
    sk_dataset_metadata_t md {};
    std::snprintf(md.type, sizeof(md.type), "%s", "f32");
    md.dim = 4;
    md.range_size = 1000;
    md.data_merge_ratio = 2;

    sk_handle_t* handle = connect();
    ASSERT_NE(handle, nullptr);
    EXPECT_NE(0, sk_create(handle, md));
    disconnect(handle);
}

TEST(parasol, create_fails_on_invalid_type) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir = make_dataset_dir(root);
    sk_dataset_metadata_t md = make_metadata(dir);
    std::snprintf(md.type, sizeof(md.type), "%s", "bad_type");

    sk_handle_t* handle = connect();
    ASSERT_NE(handle, nullptr);
    EXPECT_NE(0, sk_create(handle, md));

    disconnect(handle);
    std::filesystem::remove_all(root);
}

TEST(parasol, drop_fails_on_null_handle) {
    EXPECT_NE(0, sk_drop(nullptr));
}

TEST(parasol, drop_fails_without_initialized_dir) {
    sk_handle_t* handle = connect();
    ASSERT_NE(handle, nullptr);
    EXPECT_NE(0, sk_drop(handle));
    disconnect(handle);
}

TEST(parasol, drop_fails_without_metadata) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir = make_dataset_dir(root);

    sk_handle_t* handle = connect();
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, make_metadata(dir)));
    std::filesystem::remove(dir / "sketch2.metadata");

    EXPECT_NE(0, sk_drop(handle));

    disconnect(handle);
    std::filesystem::remove_all(root);
}

TEST(parasol, drop_fails_without_marker) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir = make_dataset_dir(root);

    sk_handle_t* handle = connect();
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, make_metadata(dir)));
    std::filesystem::remove(dir / ".sketch2.managed");

    EXPECT_NE(0, sk_drop(handle));

    disconnect(handle);
    std::filesystem::remove_all(root);
}

TEST(parasol, open_fails_on_null_path) {
    sk_handle_t* handle = connect();
    ASSERT_NE(handle, nullptr);
    EXPECT_NE(0, sk_open(handle, nullptr));
    disconnect(handle);
}

TEST(parasol, open_fails_without_metadata_file) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir = make_dataset_dir(root);
    std::filesystem::create_directories(dir);

    sk_handle_t* handle = connect();
    ASSERT_NE(handle, nullptr);
    EXPECT_NE(0, sk_open(handle, dir.string().c_str()));

    disconnect(handle);
    std::filesystem::remove_all(root);
}

TEST(parasol, open_fails_on_invalid_metadata_content) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir = make_dataset_dir(root);
    write_metadata_file(dir, "bad_type");
    write_marker_file(dir);

    sk_handle_t* handle = connect();
    ASSERT_NE(handle, nullptr);
    EXPECT_NE(0, sk_open(handle, dir.string().c_str()));

    disconnect(handle);
    std::filesystem::remove_all(root);
}

TEST(parasol, close_fails_on_null_handle) {
    EXPECT_NE(0, sk_close(nullptr));
}

TEST(parasol, repeated_close_is_idempotent) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir = make_dataset_dir(root);

    sk_handle_t* handle = connect();
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, make_metadata(dir)));
    ASSERT_OK(handle, sk_open(handle, dir.string().c_str()));
    ASSERT_OK(handle, sk_close(handle));

    EXPECT_OK(handle, sk_close(handle));

    disconnect(handle);
    std::filesystem::remove_all(root);
}

TEST(parasol, add_fails_on_null_handle) {
    EXPECT_NE(0, sk_add(nullptr, 1, "1 2 3 4"));
}

TEST(parasol, add_fails_on_null_value) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir = make_dataset_dir(root);

    sk_handle_t* handle = connect();
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, make_metadata(dir)));
    ASSERT_OK(handle, sk_open(handle, dir.string().c_str()));

    EXPECT_NE(0, sk_add(handle, 7, nullptr));

    ASSERT_OK(handle, sk_drop(handle));
    disconnect(handle);
    std::filesystem::remove_all(root);
}

TEST(parasol, delete_fails_on_null_handle) {
    EXPECT_NE(0, sk_delete(nullptr, 1));
}

TEST(parasol, load_fails_on_null_handle) {
    EXPECT_NE(0, sk_load(nullptr));
}

TEST(parasol, load_fails_without_input_file) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir  = make_dataset_dir(root);

    sk_handle_t* handle = connect();
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, make_metadata(dir)));
    ASSERT_OK(handle, sk_open(handle, dir.string().c_str()));

    EXPECT_NE(0, sk_load(handle));

    ASSERT_OK(handle, sk_drop(handle));
    disconnect(handle);
    std::filesystem::remove_all(root);
}

TEST(parasol, load_removes_input_file_on_success) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir  = make_dataset_dir(root);

    sk_handle_t* handle = connect();
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, make_metadata(dir)));
    ASSERT_OK(handle, sk_open(handle, dir.string().c_str()));

    ASSERT_OK(handle, sk_add(handle, 5, "1.0, 2.0, 3.0, 4.0"));
    ASSERT_OK(handle, sk_load(handle));

    EXPECT_FALSE(std::filesystem::exists(dir / "data.input"));

    ASSERT_OK(handle, sk_drop(handle));
    disconnect(handle);
    std::filesystem::remove_all(root);
}

TEST(parasol, load_flushes_open_write_handle_before_reading) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir  = make_dataset_dir(root);

    sk_handle_t* handle = connect();
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, make_metadata(dir)));
    ASSERT_OK(handle, sk_open(handle, dir.string().c_str()));

    ASSERT_OK(handle, sk_add(handle, 7, "1.0, 2.0, 3.0, 4.0"));
    EXPECT_OK(handle, sk_load(handle));

    ASSERT_OK(handle, sk_drop(handle));
    disconnect(handle);
    std::filesystem::remove_all(root);
}

TEST(parasol, load_creates_data_file) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir  = make_dataset_dir(root);

    sk_handle_t* handle = connect();
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, make_metadata(dir)));
    ASSERT_OK(handle, sk_open(handle, dir.string().c_str()));

    ASSERT_OK(handle, sk_add(handle, 5, "1.0, 2.0, 3.0, 4.0"));
    ASSERT_OK(handle, sk_load(handle));

    EXPECT_TRUE(std::filesystem::exists(dir / "0.data"));

    ASSERT_OK(handle, sk_drop(handle));
    disconnect(handle);
    std::filesystem::remove_all(root);
}

TEST(parasol, load_with_delete_only_input_succeeds) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir  = make_dataset_dir(root);

    sk_handle_t* handle = connect();
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, make_metadata(dir)));
    ASSERT_OK(handle, sk_open(handle, dir.string().c_str()));

    ASSERT_OK(handle, sk_add(handle, 42, "1.0, 2.0, 3.0, 4.0"));
    ASSERT_OK(handle, sk_load(handle));

    ASSERT_OK(handle, sk_delete(handle, 42));
    EXPECT_OK(handle, sk_load(handle));

    ASSERT_OK(handle, sk_drop(handle));
    disconnect(handle);
    std::filesystem::remove_all(root);
}

TEST(parasol, load_twice_second_call_fails) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir  = make_dataset_dir(root);

    sk_handle_t* handle = connect();
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, make_metadata(dir)));
    ASSERT_OK(handle, sk_open(handle, dir.string().c_str()));

    ASSERT_OK(handle, sk_add(handle, 3, "1.0, 2.0, 3.0, 4.0"));
    ASSERT_OK(handle, sk_load(handle));

    EXPECT_NE(0, sk_load(handle));

    ASSERT_OK(handle, sk_drop(handle));
    disconnect(handle);
    std::filesystem::remove_all(root);
}

TEST(parasol, load_accepts_unsorted_ids) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir  = make_dataset_dir(root);

    sk_handle_t* handle = connect();
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, make_metadata(dir)));
    ASSERT_OK(handle, sk_open(handle, dir.string().c_str()));

    ASSERT_OK(handle, sk_add(handle, 10, "1.0, 2.0, 3.0, 4.0"));
    ASSERT_OK(handle, sk_add(handle, 5,  "1.0, 2.0, 3.0, 4.0"));

    EXPECT_OK(handle, sk_load(handle));
    EXPECT_TRUE(std::filesystem::exists(dir / "0.data"));

    ASSERT_OK(handle, sk_drop(handle));
    disconnect(handle);
    std::filesystem::remove_all(root);
}

TEST(parasol, load_vectors_across_multiple_ranges) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir  = make_dataset_dir(root);

    sk_handle_t* handle = connect();
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, make_metadata(dir)));
    ASSERT_OK(handle, sk_open(handle, dir.string().c_str()));

    ASSERT_OK(handle, sk_add(handle,    5, "1.0, 2.0, 3.0, 4.0"));
    ASSERT_OK(handle, sk_add(handle, 1005, "2.0, 3.0, 4.0, 5.0"));
    ASSERT_OK(handle, sk_add(handle, 2005, "3.0, 4.0, 5.0, 6.0"));

    ASSERT_OK(handle, sk_load(handle));

    EXPECT_TRUE(std::filesystem::exists(dir / "0.data"));
    EXPECT_TRUE(std::filesystem::exists(dir / "1.data"));
    EXPECT_TRUE(std::filesystem::exists(dir / "2.data"));
    EXPECT_EQ(3u, count_files_with_ext(dir, ".data"));

    ASSERT_OK(handle, sk_drop(handle));
    disconnect(handle);
    std::filesystem::remove_all(root);
}

TEST(parasol, load_handle_usable_across_two_cycles) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir  = make_dataset_dir(root);

    sk_handle_t* handle = connect();
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, make_metadata(dir)));
    ASSERT_OK(handle, sk_open(handle, dir.string().c_str()));

    ASSERT_OK(handle, sk_add(handle, 5, "1.0, 2.0, 3.0, 4.0"));
    ASSERT_OK(handle, sk_load(handle));
    EXPECT_TRUE(std::filesystem::exists(dir / "0.data"));
    EXPECT_FALSE(std::filesystem::exists(dir / "data.input"));

    ASSERT_OK(handle, sk_add(handle, 1005, "5.0, 6.0, 7.0, 8.0"));
    ASSERT_OK(handle, sk_load(handle));
    EXPECT_TRUE(std::filesystem::exists(dir / "1.data"));
    EXPECT_FALSE(std::filesystem::exists(dir / "data.input"));

    ASSERT_OK(handle, sk_drop(handle));
    disconnect(handle);
    std::filesystem::remove_all(root);
}

TEST(parasol, knn_finds_expected_neighbors) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir  = make_dataset_dir(root);

    sk_handle_t* handle = connect();
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, make_metadata(dir)));
    ASSERT_OK(handle, sk_open(handle, dir.string().c_str()));

    ASSERT_OK(handle, sk_add(handle, 1, "0.0, 0.0, 0.0, 0.0"));
    ASSERT_OK(handle, sk_add(handle, 2, "10.0, 10.0, 10.0, 10.0"));
    ASSERT_OK(handle, sk_add(handle, 3, "1.0, 1.0, 1.0, 1.0"));
    ASSERT_OK(handle, sk_load(handle));

    uint64_t ids[2] = {0, 0};
    uint64_t ids_count = 2;
    ASSERT_OK(handle, sk_knn(handle, "0.0, 0.0, 0.0, 0.0", ids, &ids_count));
    ASSERT_EQ(2u, ids_count);
    EXPECT_EQ(ids[0], 1u);
    EXPECT_EQ(ids[1], 3u);

    ASSERT_OK(handle, sk_drop(handle));
    disconnect(handle);
    std::filesystem::remove_all(root);
}

TEST(parasol, knn_fails_on_invalid_arguments) {
    uint64_t ids[2] = {0, 0};
    uint64_t ids_count = 2;

    EXPECT_NE(0, sk_knn(nullptr, "0.0,0.0,0.0,0.0", ids, &ids_count));

    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir  = make_dataset_dir(root);

    sk_handle_t* handle = connect();
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, make_metadata(dir)));
    ASSERT_OK(handle, sk_open(handle, dir.string().c_str()));

    EXPECT_NE(0, sk_knn(handle, nullptr, ids, &ids_count));
    EXPECT_NE(0, sk_knn(handle, "0.0,0.0,0.0,0.0", nullptr, &ids_count));
    EXPECT_NE(0, sk_knn(handle, "0.0,0.0,0.0,0.0", ids, nullptr));
    ids_count = 0;
    EXPECT_NE(0, sk_knn(handle, "0.0,0.0,0.0,0.0", ids, &ids_count));

    ASSERT_OK(handle, sk_drop(handle));
    disconnect(handle);
    std::filesystem::remove_all(root);
}

TEST(parasol, knn_fails_on_invalid_query_vector) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path dir  = make_dataset_dir(root);

    sk_handle_t* handle = connect();
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, make_metadata(dir)));
    ASSERT_OK(handle, sk_open(handle, dir.string().c_str()));

    ASSERT_OK(handle, sk_add(handle, 1, "0.0, 0.0, 0.0, 0.0"));
    ASSERT_OK(handle, sk_load(handle));

    uint64_t ids[1] = {0};
    uint64_t ids_count = 1;
    EXPECT_NE(0, sk_knn(handle, "bad,bad,bad,bad", ids, &ids_count));

    ASSERT_OK(handle, sk_drop(handle));
    disconnect(handle);
    std::filesystem::remove_all(root);
}
