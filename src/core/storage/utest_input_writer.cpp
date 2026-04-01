// Unit tests for indexed-binary input writing and reading.

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "core/storage/input_reader.h"
#include "core/storage/input_writer.h"
#include "core/utest_tmp_dir.h"

using namespace sketch2;

namespace {

class InputWriterTest : public ::testing::Test {
protected:
    std::string path_;

    void SetUp() override {
        path_ = tmp_dir() + "/sketch2_utest_iw_" + std::to_string(getpid()) + ".bin";
    }

    void TearDown() override {
        std::remove(path_.c_str());
    }

    static std::vector<float> read_f32_vector(const InputReader& reader, size_t index) {
        std::vector<uint8_t> buf(reader.size());
        EXPECT_EQ(0, reader.data(index, buf.data(), buf.size()).code());
        const float* values = reinterpret_cast<const float*>(buf.data());
        return std::vector<float>(values, values + reader.dim());
    }

    static void corrupt_first_payload_byte_after_header(const std::string& path) {
        std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
        ASSERT_TRUE(file.is_open());

        std::string header;
        std::getline(file, header);
        const std::streamoff payload_offset =
            static_cast<std::streamoff>(header.size()) + 1 + 8 + 8;

        file.seekg(payload_offset);
        char value = 0;
        file.read(&value, 1);
        ASSERT_TRUE(file.good());

        value ^= 0x01;
        file.seekp(payload_offset);
        file.write(&value, 1);
        ASSERT_TRUE(file.good());
    }
};

TEST_F(InputWriterTest, PartialBlockRoundTripsMixedVectorsAndDeletes) {
    InputWriter writer;
    ASSERT_EQ(0, writer.init(DataType::f32, 4, path_).code());
    ASSERT_EQ(0, writer.write_vector(10, "10.1, 10.1, 10.1, 10.1").code());
    ASSERT_EQ(0, writer.write_deleted(11).code());
    ASSERT_EQ(0, writer.write_vector(12, "12.1 12.1 12.1 12.1").code());
    ASSERT_EQ(0, writer.close_file().code());

    InputReader reader;
    ASSERT_EQ(0, reader.init(path_).code());
    ASSERT_TRUE(reader.is_binary());
    ASSERT_EQ(DataType::f32, reader.type());
    ASSERT_EQ(4u, reader.dim());
    ASSERT_EQ(3u, reader.count());

    EXPECT_EQ(10u, reader.id(0));
    EXPECT_EQ(11u, reader.id(1));
    EXPECT_EQ(12u, reader.id(2));

    EXPECT_FALSE(reader.is_no_data(0));
    EXPECT_TRUE(reader.is_no_data(1));
    EXPECT_FALSE(reader.is_no_data(2));

    const std::vector<float> v0 = read_f32_vector(reader, 0);
    for (float value : v0) {
        EXPECT_NEAR(10.1f, value, 1e-4f);
    }

    std::vector<uint8_t> deleted_buf(reader.size());
    EXPECT_NE(0, reader.data(1, deleted_buf.data(), deleted_buf.size()).code());
    const uint8_t* raw = nullptr;
    EXPECT_NE(0, reader.raw_data(1, &raw).code());

    const std::vector<float> v2 = read_f32_vector(reader, 2);
    for (float value : v2) {
        EXPECT_NEAR(12.1f, value, 1e-4f);
    }
}

TEST_F(InputWriterTest, FullBlockRoundTripsAndFooterIsAccepted) {
    InputWriter writer;
    ASSERT_EQ(0, writer.init(DataType::i16, 4, path_).code());
    for (uint64_t id = 100; id < 164; ++id) {
        if ((id % 7) == 0) {
            ASSERT_EQ(0, writer.write_deleted(id).code());
        } else {
            const std::string value = std::to_string(id) + ", " + std::to_string(id) + ", " +
                std::to_string(id) + ", " + std::to_string(id);
            ASSERT_EQ(0, writer.write_vector(id, value.c_str()).code());
        }
    }
    ASSERT_EQ(0, writer.close_file().code());

    InputReader reader;
    const Ret init_ret = reader.init(path_);
    ASSERT_EQ(0, init_ret.code()) << init_ret.message();
    ASSERT_TRUE(reader.is_binary());
    ASSERT_EQ(DataType::i16, reader.type());
    ASSERT_EQ(64u, reader.count());
    EXPECT_EQ(100u, reader.id(0));
    EXPECT_EQ(163u, reader.id(63));
    EXPECT_TRUE(reader.is_no_data(5));   // id 105
    EXPECT_FALSE(reader.is_no_data(0));  // id 100
}

TEST_F(InputWriterTest, ReaderRejectsCorruptedFullBlockChecksum) {
    InputWriter writer;
    ASSERT_EQ(0, writer.init(DataType::f32, 4, path_).code());
    for (uint64_t id = 0; id < 64; ++id) {
        ASSERT_EQ(0, writer.write_vector(id, "1.5, 1.5, 1.5, 1.5").code());
    }
    ASSERT_EQ(0, writer.close_file().code());

    corrupt_first_payload_byte_after_header(path_);

    InputReader reader;
    const Ret ret = reader.init(path_);
    EXPECT_NE(0, ret.code());
    EXPECT_NE(ret.message().find("checksum"), std::string::npos);
}

} // namespace
