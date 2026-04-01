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

TEST_F(InputWriterTest, FullBlockIsNoDataWorksForAllIndices) {
    InputWriter writer;
    ASSERT_EQ(0, writer.init(DataType::f32, 4, path_).code());
    for (uint64_t id = 0; id < 64; ++id) {
        if (id % 10 == 0) {
            ASSERT_EQ(0, writer.write_deleted(id).code());
        } else {
            ASSERT_EQ(0, writer.write_vector(id, "1.0, 1.0, 1.0, 1.0").code());
        }
    }
    ASSERT_EQ(0, writer.close_file().code());

    InputReader reader;
    ASSERT_EQ(0, reader.init(path_).code());
    ASSERT_EQ(64u, reader.count());

    for (size_t i = 0; i < reader.count(); ++i) {
        const uint64_t id = reader.id(i);
        if (id % 10 == 0) {
            EXPECT_TRUE(reader.is_no_data(i)) << "id=" << id;
        } else {
            EXPECT_FALSE(reader.is_no_data(i)) << "id=" << id;
        }
    }
}

TEST_F(InputWriterTest, FullBlockDataAndRawDataWorkForAllIndices) {
    InputWriter writer;
    ASSERT_EQ(0, writer.init(DataType::f32, 4, path_).code());
    for (uint64_t id = 0; id < 64; ++id) {
        if (id == 30) {
            ASSERT_EQ(0, writer.write_deleted(id).code());
        } else {
            const std::string value = std::to_string(static_cast<float>(id) + 0.5f) + ", " +
                std::to_string(static_cast<float>(id) + 0.5f) + ", " +
                std::to_string(static_cast<float>(id) + 0.5f) + ", " +
                std::to_string(static_cast<float>(id) + 0.5f);
            ASSERT_EQ(0, writer.write_vector(id, value.c_str()).code());
        }
    }
    ASSERT_EQ(0, writer.close_file().code());

    InputReader reader;
    ASSERT_EQ(0, reader.init(path_).code());
    ASSERT_EQ(64u, reader.count());

    for (size_t i = 0; i < reader.count(); ++i) {
        const uint64_t id = reader.id(i);
        if (id == 30) {
            std::vector<uint8_t> buf(reader.size());
            EXPECT_NE(0, reader.data(i, buf.data(), buf.size()).code());
            const uint8_t* raw = nullptr;
            EXPECT_NE(0, reader.raw_data(i, &raw).code());
        } else {
            const std::vector<float> v = read_f32_vector(reader, i);
            const float expected = static_cast<float>(id) + 0.5f;
            for (float val : v) {
                EXPECT_NEAR(expected, val, 1e-2f) << "id=" << id;
            }

            const uint8_t* raw = nullptr;
            ASSERT_EQ(0, reader.raw_data(i, &raw).code());
            ASSERT_NE(nullptr, raw);
        }
    }
}

TEST_F(InputWriterTest, TwoFullBlocksRoundTrip) {
    InputWriter writer;
    ASSERT_EQ(0, writer.init(DataType::f32, 4, path_).code());
    for (uint64_t id = 0; id < 128; ++id) {
        if (id == 10 || id == 70) {
            ASSERT_EQ(0, writer.write_deleted(id).code());
        } else {
            const std::string value = std::to_string(id) + ", " + std::to_string(id) + ", " +
                std::to_string(id) + ", " + std::to_string(id);
            ASSERT_EQ(0, writer.write_vector(id, value.c_str()).code());
        }
    }
    ASSERT_EQ(0, writer.close_file().code());

    InputReader reader;
    const Ret ret = reader.init(path_);
    ASSERT_EQ(0, ret.code()) << ret.message();
    ASSERT_TRUE(reader.is_binary());
    ASSERT_EQ(128u, reader.count());
    EXPECT_EQ(0u, reader.id(0));
    EXPECT_EQ(127u, reader.id(127));

    EXPECT_TRUE(reader.is_no_data(10));
    EXPECT_TRUE(reader.is_no_data(70));
    EXPECT_FALSE(reader.is_no_data(0));
    EXPECT_FALSE(reader.is_no_data(127));

    const std::vector<float> v0 = read_f32_vector(reader, 0);
    EXPECT_NEAR(0.0f, v0[0], 1e-4f);
    const std::vector<float> v127 = read_f32_vector(reader, 127);
    EXPECT_NEAR(127.0f, v127[0], 1e-4f);
}

TEST_F(InputWriterTest, FullBlockPlusPartialBlockRoundTrip) {
    const uint64_t total = 70;
    InputWriter writer;
    ASSERT_EQ(0, writer.init(DataType::f32, 4, path_).code());
    for (uint64_t id = 0; id < total; ++id) {
        if (id == 5 || id == 65) {
            ASSERT_EQ(0, writer.write_deleted(id).code());
        } else {
            const std::string value = std::to_string(id) + ".5, " + std::to_string(id) + ".5, " +
                std::to_string(id) + ".5, " + std::to_string(id) + ".5";
            ASSERT_EQ(0, writer.write_vector(id, value.c_str()).code());
        }
    }
    ASSERT_EQ(0, writer.close_file().code());

    InputReader reader;
    const Ret ret = reader.init(path_);
    ASSERT_EQ(0, ret.code()) << ret.message();
    ASSERT_EQ(total, reader.count());

    EXPECT_TRUE(reader.is_no_data(5));
    EXPECT_TRUE(reader.is_no_data(65));
    EXPECT_FALSE(reader.is_no_data(0));
    EXPECT_FALSE(reader.is_no_data(69));

    const std::vector<float> v69 = read_f32_vector(reader, 69);
    EXPECT_NEAR(69.5f, v69[0], 1e-4f);
}

TEST_F(InputWriterTest, UnsortedIdsRoundTripCorrectly) {
    InputWriter writer;
    ASSERT_EQ(0, writer.init(DataType::f32, 4, path_).code());
    ASSERT_EQ(0, writer.write_vector(50, "50.0, 50.0, 50.0, 50.0").code());
    ASSERT_EQ(0, writer.write_deleted(30).code());
    ASSERT_EQ(0, writer.write_vector(10, "10.0, 10.0, 10.0, 10.0").code());
    ASSERT_EQ(0, writer.write_vector(40, "40.0, 40.0, 40.0, 40.0").code());
    ASSERT_EQ(0, writer.write_deleted(20).code());
    ASSERT_EQ(0, writer.close_file().code());

    InputReader reader;
    ASSERT_EQ(0, reader.init(path_).code());
    ASSERT_EQ(5u, reader.count());

    // After sorting by id: 10, 20, 30, 40, 50
    EXPECT_EQ(10u, reader.id(0));
    EXPECT_EQ(20u, reader.id(1));
    EXPECT_EQ(30u, reader.id(2));
    EXPECT_EQ(40u, reader.id(3));
    EXPECT_EQ(50u, reader.id(4));

    // Deleted items: ids 20 and 30
    EXPECT_FALSE(reader.is_no_data(0)); // id 10
    EXPECT_TRUE(reader.is_no_data(1));  // id 20
    EXPECT_TRUE(reader.is_no_data(2));  // id 30
    EXPECT_FALSE(reader.is_no_data(3)); // id 40
    EXPECT_FALSE(reader.is_no_data(4)); // id 50

    const std::vector<float> v10 = read_f32_vector(reader, 0);
    EXPECT_NEAR(10.0f, v10[0], 1e-4f);
    const std::vector<float> v40 = read_f32_vector(reader, 3);
    EXPECT_NEAR(40.0f, v40[0], 1e-4f);
    const std::vector<float> v50 = read_f32_vector(reader, 4);
    EXPECT_NEAR(50.0f, v50[0], 1e-4f);
}

TEST_F(InputWriterTest, AllDeletedBlockRoundTrips) {
    InputWriter writer;
    ASSERT_EQ(0, writer.init(DataType::f32, 4, path_).code());
    for (uint64_t id = 0; id < 5; ++id) {
        ASSERT_EQ(0, writer.write_deleted(id).code());
    }
    ASSERT_EQ(0, writer.close_file().code());

    InputReader reader;
    ASSERT_EQ(0, reader.init(path_).code());
    ASSERT_EQ(5u, reader.count());

    for (size_t i = 0; i < reader.count(); ++i) {
        EXPECT_TRUE(reader.is_no_data(i)) << "index=" << i;
    }
}

TEST_F(InputWriterTest, F16TypeRoundTrips) {
    InputWriter writer;
    ASSERT_EQ(0, writer.init(DataType::f16, 4, path_).code());
    ASSERT_EQ(0, writer.write_vector(1, "1.0, 2.0, 3.0, 4.0").code());
    ASSERT_EQ(0, writer.write_deleted(2).code());
    ASSERT_EQ(0, writer.write_vector(3, "5.0, 6.0, 7.0, 8.0").code());
    ASSERT_EQ(0, writer.close_file().code());

    InputReader reader;
    ASSERT_EQ(0, reader.init(path_).code());
    ASSERT_EQ(DataType::f16, reader.type());
    ASSERT_EQ(4u, reader.dim());
    ASSERT_EQ(3u, reader.count());
    EXPECT_FALSE(reader.is_no_data(0));
    EXPECT_TRUE(reader.is_no_data(1));
    EXPECT_FALSE(reader.is_no_data(2));
}

TEST_F(InputWriterTest, InitRejectsDoubleInit) {
    InputWriter writer;
    ASSERT_EQ(0, writer.init(DataType::f32, 4, path_).code());
    EXPECT_NE(0, writer.init(DataType::f32, 4, path_).code());
}

TEST_F(InputWriterTest, InitRejectsInvalidDimension) {
    InputWriter writer;
    EXPECT_NE(0, writer.init(DataType::f32, 3, path_).code());
    EXPECT_NE(0, writer.init(DataType::f32, 5000, path_).code());
}

TEST_F(InputWriterTest, WriteVectorRejectsNullVector) {
    InputWriter writer;
    ASSERT_EQ(0, writer.init(DataType::f32, 4, path_).code());
    EXPECT_NE(0, writer.write_vector(1, nullptr).code());
}

TEST_F(InputWriterTest, WriteBeforeInitReturnsError) {
    InputWriter writer;
    EXPECT_NE(0, writer.write_vector(1, "1.0, 1.0, 1.0, 1.0").code());
    EXPECT_NE(0, writer.write_deleted(1).code());
}

TEST_F(InputWriterTest, CloseWithoutInitSucceeds) {
    InputWriter writer;
    EXPECT_EQ(0, writer.close_file().code());
}

TEST_F(InputWriterTest, EmptyFileRoundTrips) {
    InputWriter writer;
    ASSERT_EQ(0, writer.init(DataType::f32, 4, path_).code());
    ASSERT_EQ(0, writer.close_file().code());

    InputReader reader;
    ASSERT_EQ(0, reader.init(path_).code());
    EXPECT_TRUE(reader.is_binary());
    EXPECT_EQ(0u, reader.count());
}

TEST_F(InputWriterTest, ThreeFullBlocksRoundTrip) {
    const uint64_t total = 192;
    InputWriter writer;
    ASSERT_EQ(0, writer.init(DataType::i16, 4, path_).code());
    for (uint64_t id = 0; id < total; ++id) {
        if (id == 63 || id == 127 || id == 191) {
            ASSERT_EQ(0, writer.write_deleted(id).code());
        } else {
            const std::string value = std::to_string(id) + ", " + std::to_string(id) + ", " +
                std::to_string(id) + ", " + std::to_string(id);
            ASSERT_EQ(0, writer.write_vector(id, value.c_str()).code());
        }
    }
    ASSERT_EQ(0, writer.close_file().code());

    InputReader reader;
    const Ret ret = reader.init(path_);
    ASSERT_EQ(0, ret.code()) << ret.message();
    ASSERT_EQ(total, reader.count());

    EXPECT_TRUE(reader.is_no_data(63));
    EXPECT_TRUE(reader.is_no_data(127));
    EXPECT_TRUE(reader.is_no_data(191));
    EXPECT_FALSE(reader.is_no_data(0));
    EXPECT_FALSE(reader.is_no_data(100));
    EXPECT_FALSE(reader.is_no_data(190));
}

} // namespace
