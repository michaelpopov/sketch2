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
    std::filesystem::path dir =
        base / std::filesystem::path("sketch2_parasol_ut_" + std::to_string(::getpid()) + "_" +
                                     std::to_string(std::rand()));
    std::filesystem::create_directories(dir);
    return dir;
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

} // namespace

TEST(parasol, create_writes_metadata_ini) {
    const std::filesystem::path dir = make_temp_dir();
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
}

TEST(parasol, create_fails_on_duplicate_metadata) {
    const std::filesystem::path dir = make_temp_dir();
    const sk_dataset_metadata_t md = make_metadata(dir);

    ASSERT_EQ(sk_create(md).code, 0);
    const sk_ret_t second = sk_create(md);
    EXPECT_NE(second.code, 0);

    ASSERT_EQ(sk_drop(dir.string().c_str()).code, 0);
}

TEST(parasol, open_close_success) {
    const std::filesystem::path dir = make_temp_dir();
    const sk_dataset_metadata_t md = make_metadata(dir);
    ASSERT_EQ(sk_create(md).code, 0);

    const sk_ret_t open_ret = sk_open(dir.string().c_str());
    ASSERT_EQ(open_ret.code, 0) << open_ret.message;
    ASSERT_NE(open_ret.handle, nullptr);

    const sk_ret_t close_ret = sk_close(open_ret.handle);
    EXPECT_EQ(close_ret.code, 0) << close_ret.message;

    ASSERT_EQ(sk_drop(dir.string().c_str()).code, 0);
}

TEST(parasol, add_delete_write_to_input_file) {
    const std::filesystem::path dir = make_temp_dir();
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
}
