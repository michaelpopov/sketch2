// Unit tests for accumulator WAL persistence and recovery.

#include "core/storage/accumulator.h"
#include "core/storage/data_file.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <unistd.h>

#include <gtest/gtest.h>

namespace sketch2 {

namespace fs = std::filesystem;

namespace {

template <typename T, size_t N>
const uint8_t* as_bytes(const std::array<T, N>& data) {
    return reinterpret_cast<const uint8_t*>(data.data());
}

} // namespace

class AccumulatorWalTest : public ::testing::Test {
protected:
    std::string dir_;
    std::string wal_path_;

    void SetUp() override {
        dir_ = "/tmp/sketch2_utest_acc_wal_" + std::to_string(getpid());
        wal_path_ = dir_ + "/accumulator.wal";
        fs::create_directories(dir_);
    }

    void TearDown() override {
        fs::remove_all(dir_);
    }
};

TEST_F(AccumulatorWalTest, ReplayRestoresStateWithoutTruncatingValidWalRecords) {
    const std::array<float, 4> vec0 {1.0f, 2.0f, 3.0f, 4.0f};
    const std::array<float, 4> vec1 {5.0f, 6.0f, 7.0f, 8.0f};

    {
        Accumulator accumulator;
        ASSERT_EQ(0, accumulator.init(256, DataType::f32, 4).code());
        ASSERT_EQ(0, accumulator.attach_wal(wal_path_).code());
        ASSERT_EQ(0, accumulator.add_vector(1, as_bytes(vec0)).code());
        ASSERT_EQ(0, accumulator.delete_vector(2).code());
        ASSERT_EQ(0, accumulator.add_vector(2, as_bytes(vec1)).code());
        ASSERT_EQ(0, accumulator.delete_vector(1).code());
    }
    const auto wal_size_before_replay = fs::file_size(wal_path_);

    Accumulator restored;
    ASSERT_EQ(0, restored.init(256, DataType::f32, 4).code());
    ASSERT_EQ(0, restored.attach_wal(wal_path_).code());

    EXPECT_EQ((std::vector<uint64_t> {2}), restored.get_vector_ids());
    EXPECT_EQ((std::vector<uint64_t> {1}), restored.get_deleted_ids());
    const float* values = reinterpret_cast<const float*>(restored.get_vector(2));
    ASSERT_NE(nullptr, values);
    EXPECT_FLOAT_EQ(5.0f, values[0]);
    EXPECT_EQ(wal_size_before_replay, fs::file_size(wal_path_));
}

TEST_F(AccumulatorWalTest, ReplayIgnoresAndTruncatesPartialTailRecord) {
    const std::array<float, 4> vec {9.0f, 8.0f, 7.0f, 6.0f};

    {
        Accumulator accumulator;
        ASSERT_EQ(0, accumulator.init(256, DataType::f32, 4).code());
        ASSERT_EQ(0, accumulator.attach_wal(wal_path_).code());
        ASSERT_EQ(0, accumulator.add_vector(7, as_bytes(vec)).code());
    }
    const auto wal_size_before_garbage = fs::file_size(wal_path_);

    {
        std::ofstream out(wal_path_, std::ios::binary | std::ios::app);
        const std::array<uint8_t, 5> garbage {1, 2, 3, 4, 5};
        out.write(reinterpret_cast<const char*>(garbage.data()), garbage.size());
    }

    Accumulator restored;
    ASSERT_EQ(0, restored.init(256, DataType::f32, 4).code());
    ASSERT_EQ(0, restored.attach_wal(wal_path_).code());
    const float* values = reinterpret_cast<const float*>(restored.get_vector(7));
    ASSERT_NE(nullptr, values);
    EXPECT_FLOAT_EQ(9.0f, values[0]);
    EXPECT_EQ(wal_size_before_garbage, fs::file_size(wal_path_));
}

TEST_F(AccumulatorWalTest, ReplayFailsOnChecksumMismatch) {
    const std::array<float, 4> vec {1.0f, 1.0f, 1.0f, 1.0f};

    {
        Accumulator accumulator;
        ASSERT_EQ(0, accumulator.init(256, DataType::f32, 4).code());
        ASSERT_EQ(0, accumulator.attach_wal(wal_path_).code());
        ASSERT_EQ(0, accumulator.add_vector(9, as_bytes(vec)).code());
    }

    {
        std::fstream file(wal_path_, std::ios::binary | std::ios::in | std::ios::out);
        ASSERT_TRUE(file.is_open());
        file.seekp(-1, std::ios::end);
        const char bad = '\xFF';
        file.write(&bad, 1);
    }

    Accumulator restored;
    ASSERT_EQ(0, restored.init(256, DataType::f32, 4).code());
    const Ret ret = restored.attach_wal(wal_path_);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("AccumulatorWal::replay: checksum mismatch", ret.message());
}

TEST_F(AccumulatorWalTest, ReplayFailsOnTypeMismatch) {
    const std::array<float, 4> vec {2.0f, 2.0f, 2.0f, 2.0f};

    {
        Accumulator accumulator;
        ASSERT_EQ(0, accumulator.init(256, DataType::f32, 4).code());
        ASSERT_EQ(0, accumulator.attach_wal(wal_path_).code());
        ASSERT_EQ(0, accumulator.add_vector(11, as_bytes(vec)).code());
    }

    Accumulator restored;
    ASSERT_EQ(0, restored.init(256, DataType::i16, 4).code());
    const Ret ret = restored.attach_wal(wal_path_);
    EXPECT_NE(0, ret.code());
    EXPECT_EQ("AccumulatorWal: wal type mismatch", ret.message());
}

} // namespace sketch2
