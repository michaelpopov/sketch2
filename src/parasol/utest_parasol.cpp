#include "parasol.h"

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
    const std::filesystem::path base = std::filesystem::temp_directory_path();
    std::filesystem::create_directories(base);
    const std::filesystem::path dir =
        base / std::filesystem::path("sketch2_parasol_ut_" + std::to_string(::getpid()) + "_" +
                                     std::to_string(std::rand()));
    std::filesystem::create_directories(dir);
    return dir;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    return std::string((std::istreambuf_iterator<char>(in)),
        std::istreambuf_iterator<char>());
}

} // namespace

TEST(parasol, create_open_close_drop_lifecycle) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_connect(root.string().c_str());
    ASSERT_NE(handle, nullptr);

    ASSERT_OK(handle, sk_create(handle, "dataset", 4, "f32", 1000));
    EXPECT_TRUE(std::filesystem::exists(root / "dataset"));
    EXPECT_TRUE(std::filesystem::exists(root / "dataset.ini"));
    EXPECT_TRUE(std::filesystem::exists(root / "dataset.lock"));

    const std::string ini = read_file(root / "dataset.ini");
    EXPECT_NE(ini.find("[dataset]\n"), std::string::npos);
    EXPECT_NE(ini.find("dirs=" + (root / "dataset").string() + "\n"), std::string::npos);
    EXPECT_NE(ini.find("range_size=1000\n"), std::string::npos);
    EXPECT_NE(ini.find("dim=4\n"), std::string::npos);
    EXPECT_NE(ini.find("type=f32\n"), std::string::npos);

    EXPECT_OK(handle, sk_close(handle, "dataset"));
    EXPECT_OK(handle, sk_drop(handle, "dataset"));

    sk_disconnect(handle);
    std::filesystem::remove_all(root);
}

TEST(parasol, upsert_get_gid_and_delete_follow_design_results) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_connect(root.string().c_str());
    ASSERT_NE(handle, nullptr);

    ASSERT_OK(handle, sk_create(handle, "ds", 4, "f32", 1000));
    ASSERT_OK(handle, sk_upsert(handle, 42, "1.0, 2.0, 3.0, 4.0"));
    ASSERT_OK(handle, sk_macc(handle));

    ASSERT_OK(handle, sk_get(handle, 42));
    EXPECT_STREQ("[ 1, 2, 3, 4 ]", sk_gres(handle));

    ASSERT_OK(handle, sk_gid(handle, "1.0, 2.0, 3.0, 4.0"));
    uint64_t id = 0;
    ASSERT_OK(handle, sk_ires(handle, &id));
    EXPECT_EQ(42u, id);

    ASSERT_OK(handle, sk_del(handle, 42));
    ASSERT_OK(handle, sk_macc(handle));
    EXPECT_NE(0, sk_get(handle, 42));

    EXPECT_OK(handle, sk_close(handle, "ds"));
    EXPECT_OK(handle, sk_drop(handle, "ds"));

    sk_disconnect(handle);
    std::filesystem::remove_all(root);
}

TEST(parasol, ups2_knn_and_kres_cache_ids_on_handle) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_connect(root.string().c_str());
    ASSERT_NE(handle, nullptr);

    ASSERT_OK(handle, sk_create(handle, "ds", 4, "f32", 1000));
    ASSERT_OK(handle, sk_ups2(handle, 1, 0.0));
    ASSERT_OK(handle, sk_ups2(handle, 2, 10.0));
    ASSERT_OK(handle, sk_ups2(handle, 3, 1.0));
    ASSERT_OK(handle, sk_macc(handle));

    ASSERT_OK(handle, sk_knn(handle, "0.0, 0.0, 0.0, 0.0", 2));
    EXPECT_EQ(2u, sk_kres(handle, -1));
    EXPECT_EQ(1u, sk_kres(handle, 0));
    EXPECT_EQ(3u, sk_kres(handle, 1));

    EXPECT_OK(handle, sk_close(handle, "ds"));
    EXPECT_OK(handle, sk_drop(handle, "ds"));

    sk_disconnect(handle);
    std::filesystem::remove_all(root);
}

TEST(parasol, generate_stats_and_print_smoke) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_connect(root.string().c_str());
    ASSERT_NE(handle, nullptr);

    ASSERT_OK(handle, sk_create(handle, "ds", 4, "f32", 1000));
    ASSERT_OK(handle, sk_generate(handle, 8, 10, 0));

    testing::internal::CaptureStdout();
    ASSERT_OK(handle, sk_stats(handle));
    const std::string stats_out = testing::internal::GetCapturedStdout();
    EXPECT_NE(stats_out.find("dataset:"), std::string::npos);
    EXPECT_NE(stats_out.find("Name: ds"), std::string::npos);
    EXPECT_NE(stats_out.find("Type: f32"), std::string::npos);
    EXPECT_NE(stats_out.find("Dim: 4"), std::string::npos);
    EXPECT_NE(stats_out.find("Range: 1000"), std::string::npos);
    EXPECT_NE(stats_out.find("accumulator:"), std::string::npos);
    EXPECT_NE(stats_out.find(".data:"), std::string::npos);

    testing::internal::CaptureStdout();
    ASSERT_OK(handle, sk_print(handle));
    const std::string print_out = testing::internal::GetCapturedStdout();
    EXPECT_NE(print_out.find("10 : ["), std::string::npos);

    EXPECT_OK(handle, sk_close(handle, "ds"));
    EXPECT_OK(handle, sk_drop(handle, "ds"));

    sk_disconnect(handle);
    std::filesystem::remove_all(root);
}

TEST(parasol, close_requires_matching_name) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_connect(root.string().c_str());
    ASSERT_NE(handle, nullptr);

    ASSERT_OK(handle, sk_create(handle, "ds", 4, "f32", 1000));
    EXPECT_NE(0, sk_close(handle, "other"));
    EXPECT_OK(handle, sk_close(handle, "ds"));
    EXPECT_OK(handle, sk_drop(handle, "ds"));

    sk_disconnect(handle);
    std::filesystem::remove_all(root);
}

TEST(parasol, gres_returns_empty_string_without_cached_vector) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_connect(root.string().c_str());
    ASSERT_NE(handle, nullptr);
    EXPECT_STREQ("", sk_gres(handle));

    sk_disconnect(handle);
    std::filesystem::remove_all(root);
}

TEST(parasol, kres_returns_zero_without_cached_result) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_connect(root.string().c_str());
    ASSERT_NE(handle, nullptr);
    EXPECT_EQ(0u, sk_kres(handle, -1));
    EXPECT_EQ(0u, sk_kres(handle, 0));

    sk_disconnect(handle);
    std::filesystem::remove_all(root);
}
