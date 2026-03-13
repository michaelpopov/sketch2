// Unit tests for the INI reader.

#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include "utils/ini_reader.h"

using namespace sketch2;
namespace fs = std::filesystem;

class IniReaderTest : public ::testing::Test {
protected:
    std::string base_dir_;
    std::string ini_path_;

    void SetUp() override {
        base_dir_ = "/tmp/sketch2_utest_ini_reader_" + std::to_string(getpid());
        ini_path_ = base_dir_ + "/test.ini";
        fs::create_directories(base_dir_);
    }

    void TearDown() override {
        fs::remove_all(base_dir_);
    }

    void write_ini(const std::string& content) {
        std::ofstream f(ini_path_);
        f << content;
    }
};

TEST_F(IniReaderTest, InitFailsOnMissingFile) {
    IniReader reader;
    const Ret ret = reader.init(base_dir_ + "/missing.ini");
    EXPECT_NE(0, ret.code());
}

TEST_F(IniReaderTest, InitParsesGlobalAndSectionKeys) {
    write_ini(
        "port = 8080\n"
        "\n"
        "[db]\n"
        "host = localhost\n"
        "pool = 16\n");

    IniReader reader;
    ASSERT_EQ(0, reader.init(ini_path_).code());
    EXPECT_EQ(8080, reader.get_int("port", -1));
    EXPECT_EQ("localhost", reader.get_str("db.host", "missing"));
    EXPECT_EQ(16, reader.get_int("db.pool", -1));
}

TEST_F(IniReaderTest, GetReturnsDefaultsForMissingKeys) {
    write_ini("[main]\nname = app\n");

    IniReader reader;
    ASSERT_EQ(0, reader.init(ini_path_).code());
    EXPECT_EQ(7, reader.get_int("main.threads", 7));
    EXPECT_EQ("fallback", reader.get_str("main.path", "fallback"));
    EXPECT_TRUE(reader.get_str_list("main.nodes").empty());
}

TEST_F(IniReaderTest, GetStrListSplitsAndTrimsCsv) {
    write_ini(
        "[cluster]\n"
        "nodes = node1, node2 ,node3,, \n");

    IniReader reader;
    ASSERT_EQ(0, reader.init(ini_path_).code());
    const auto nodes = reader.get_str_list("cluster.nodes");
    ASSERT_EQ(3u, nodes.size());
    EXPECT_EQ("node1", nodes[0]);
    EXPECT_EQ("node2", nodes[1]);
    EXPECT_EQ("node3", nodes[2]);
}

TEST_F(IniReaderTest, IgnoresCommentsAndMalformedLines) {
    write_ini(
        "; comment\n"
        "# another comment\n"
        "[s]\n"
        "ok = 42\n"
        "bad line\n"
        "= no_key\n");

    IniReader reader;
    ASSERT_EQ(0, reader.init(ini_path_).code());
    EXPECT_EQ(42, reader.get_int("s.ok", -1));
    EXPECT_EQ("x", reader.get_str("s.bad", "x"));
}

TEST_F(IniReaderTest, InvalidIntValueThrows) {
    write_ini(
        "[main]\n"
        "count = not_an_int\n");

    IniReader reader;
    ASSERT_EQ(0, reader.init(ini_path_).code());
    EXPECT_THROW(reader.get_int("main.count", 11), std::runtime_error);
}
