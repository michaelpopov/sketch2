#include <gtest/gtest.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include "core/storage/input_generator.h"
#include "core/storage/storage_controller.h"
#include "core/storage/data_reader.h"

using namespace sketch2;
namespace fs = std::filesystem;

class StorageControllerTest : public ::testing::Test {
protected:
    std::string base_dir_;
    std::string input_path_;

    void SetUp() override {
        base_dir_   = "/tmp/sketch2_utest_sc_" + std::to_string(getpid());
        input_path_ = base_dir_ + "/input.txt";
        fs::create_directories(base_dir_);
    }

    void TearDown() override {
        fs::remove_all(base_dir_);
    }

    // Create a named subdirectory under base_dir_ and return its path.
    std::string make_dir(const std::string& name) {
        std::string path = base_dir_ + "/" + name;
        fs::create_directories(path);
        return path;
    }

    GeneratorConfig cfg(size_t count, size_t min_id, DataType type, size_t dim) {
        return {PatternType::Sequential, count, min_id, type, dim, 1000};
    }
};

// --- init error cases ---

TEST_F(StorageControllerTest, InitFailsOnEmptyDirs) {
    StorageController sc;
    EXPECT_NE(0, sc.init({}, 100).code());
}

TEST_F(StorageControllerTest, InitFailsOnZeroRangeSize) {
    StorageController sc;
    EXPECT_NE(0, sc.init({make_dir("d")}, 0).code());
}

TEST_F(StorageControllerTest, InitFailsOnDoubleInit) {
    StorageController sc;
    ASSERT_EQ(0, sc.init({make_dir("d")}, 100).code());
    EXPECT_NE(0, sc.init({make_dir("d2")}, 100).code());
}

// --- load error cases ---

TEST_F(StorageControllerTest, LoadFailsWhenNotInitialized) {
    StorageController sc;
    EXPECT_NE(0, sc.load(input_path_).code());
}

TEST_F(StorageControllerTest, LoadFailsOnBadInputPath) {
    StorageController sc;
    ASSERT_EQ(0, sc.init({make_dir("d")}, 100).code());
    EXPECT_NE(0, sc.load("/nonexistent/dir/input.txt").code());
}

TEST_F(StorageControllerTest, LoadFailsOnBadOutputDir) {
    generate_input_file(input_path_, cfg(3, 0, DataType::f32, 4));
    StorageController sc;
    ASSERT_EQ(0, sc.init({"/nonexistent/output/dir"}, 100).code());
    EXPECT_NE(0, sc.load(input_path_).code());
}

// --- load success: file placement ---

TEST_F(StorageControllerTest, LoadCreatesOutputFile) {
    auto dir = make_dir("d");
    generate_input_file(input_path_, cfg(5, 0, DataType::f32, 4));
    StorageController sc;
    ASSERT_EQ(0, sc.init({dir}, 1000).code());
    ASSERT_EQ(0, sc.load(input_path_).code());
    EXPECT_TRUE(fs::exists(dir + "/0.data"));
}

TEST_F(StorageControllerTest, LoadFileIdFromMinId) {
    // ids [2005..2009], range_size=1000 -> file_id=2 -> "2.data"
    auto dir = make_dir("d");
    generate_input_file(input_path_, cfg(5, 2005, DataType::f32, 4));
    StorageController sc;
    ASSERT_EQ(0, sc.init({dir}, 1000).code());
    ASSERT_EQ(0, sc.load(input_path_).code());
    EXPECT_TRUE(fs::exists(dir + "/2.data"));
    EXPECT_FALSE(fs::exists(dir + "/0.data"));
}

TEST_F(StorageControllerTest, LoadMultipleRangesCreateMultipleFiles) {
    // ids [5..19], range_size=10 -> file 0 (ids 5-9), file 1 (ids 10-19)
    auto dir = make_dir("d");
    generate_input_file(input_path_, cfg(15, 5, DataType::f32, 4));
    StorageController sc;
    ASSERT_EQ(0, sc.init({dir}, 10).code());
    ASSERT_EQ(0, sc.load(input_path_).code());
    EXPECT_TRUE(fs::exists(dir + "/0.data"));
    EXPECT_TRUE(fs::exists(dir + "/1.data"));
}

TEST_F(StorageControllerTest, LoadMultipleDirsRoutesByFileId) {
    // ids [0..29], range_size=10 -> file_id 0,1,2
    // 2 dirs: file_id%2 -> dir0 gets 0.data, 2.data; dir1 gets 1.data
    auto dir0 = make_dir("d0");
    auto dir1 = make_dir("d1");
    generate_input_file(input_path_, cfg(30, 0, DataType::f32, 4));
    StorageController sc;
    ASSERT_EQ(0, sc.init({dir0, dir1}, 10).code());
    ASSERT_EQ(0, sc.load(input_path_).code());
    EXPECT_TRUE(fs::exists(dir0 + "/0.data"));
    EXPECT_TRUE(fs::exists(dir1 + "/1.data"));
    EXPECT_TRUE(fs::exists(dir0 + "/2.data"));
}

// --- load success: output file content ---

TEST_F(StorageControllerTest, LoadFileCount) {
    auto dir = make_dir("d");
    generate_input_file(input_path_, cfg(7, 0, DataType::f32, 4));
    StorageController sc;
    ASSERT_EQ(0, sc.init({dir}, 1000).code());
    ASSERT_EQ(0, sc.load(input_path_).code());
    DataReader dr;
    ASSERT_EQ(0, dr.init(dir + "/0.data").code());
    EXPECT_EQ(7u, dr.count());
}

TEST_F(StorageControllerTest, LoadFileType) {
    auto dir = make_dir("d");
    generate_input_file(input_path_, cfg(3, 0, DataType::f32, 4));
    StorageController sc;
    ASSERT_EQ(0, sc.init({dir}, 1000).code());
    ASSERT_EQ(0, sc.load(input_path_).code());
    DataReader dr;
    ASSERT_EQ(0, dr.init(dir + "/0.data").code());
    EXPECT_EQ(DataType::f32, dr.type());
}

TEST_F(StorageControllerTest, LoadFileDim) {
    auto dir = make_dir("d");
    generate_input_file(input_path_, cfg(3, 0, DataType::f32, 64));
    StorageController sc;
    ASSERT_EQ(0, sc.init({dir}, 1000).code());
    ASSERT_EQ(0, sc.load(input_path_).code());
    DataReader dr;
    ASSERT_EQ(0, dr.init(dir + "/0.data").code());
    EXPECT_EQ(64u, dr.dim());
}

TEST_F(StorageControllerTest, LoadFileDataValues) {
    // generator writes id+0.1 for each dimension
    auto dir = make_dir("d");
    generate_input_file(input_path_, cfg(5, 10, DataType::f32, 4));
    StorageController sc;
    ASSERT_EQ(0, sc.init({dir}, 1000).code());
    ASSERT_EQ(0, sc.load(input_path_).code());
    DataReader dr;
    ASSERT_EQ(0, dr.init(dir + "/0.data").code());
    for (uint64_t id = 10; id < 15; ++id) {
        const float* v = reinterpret_cast<const float*>(dr.get(id));
        ASSERT_NE(nullptr, v) << "id " << id << " not found";
        float expected = static_cast<float>(id) + 0.1f;
        for (size_t d = 0; d < 4; ++d) {
            EXPECT_NEAR(expected, v[d], 1e-4f) << "id=" << id << " dim=" << d;
        }
    }
}

TEST_F(StorageControllerTest, LoadMultipleRangesCorrectCounts) {
    // ids [5..19], range_size=10
    // file 0: ids 5-9  -> 5 vectors
    // file 1: ids 10-19 -> 10 vectors
    auto dir = make_dir("d");
    generate_input_file(input_path_, cfg(15, 5, DataType::f32, 4));
    StorageController sc;
    ASSERT_EQ(0, sc.init({dir}, 10).code());
    ASSERT_EQ(0, sc.load(input_path_).code());

    DataReader dr0, dr1;
    ASSERT_EQ(0, dr0.init(dir + "/0.data").code());
    ASSERT_EQ(0, dr1.init(dir + "/1.data").code());
    EXPECT_EQ(5u,  dr0.count());
    EXPECT_EQ(10u, dr1.count());
}

TEST_F(StorageControllerTest, LoadSkipsMissingMiddleRanges) {
    auto dir = make_dir("d");
    {
        std::ofstream f(input_path_);
        f << "f32,4\n";
        f << "0 : [ 0.10, 0.10, 0.10, 0.10 ]\n";
        f << "20 : [ 20.10, 20.10, 20.10, 20.10 ]\n";
    }

    StorageController sc;
    ASSERT_EQ(0, sc.init({dir}, 10).code());
    ASSERT_EQ(0, sc.load(input_path_).code());

    EXPECT_TRUE(fs::exists(dir + "/0.data"));
    EXPECT_FALSE(fs::exists(dir + "/1.data"));
    EXPECT_TRUE(fs::exists(dir + "/2.data"));
}
