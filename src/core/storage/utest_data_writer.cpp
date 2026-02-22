#include <gtest/gtest.h>
#include <cstdio>
#include <cstdint>
#include <unistd.h>
#include <vector>
#include "core/storage/data_file.h"
#include "core/storage/input_generator.h"
#include "core/storage/data_writer.h"

using namespace sketch2;

static constexpr uint32_t kExpectedMagic   = 0x534B5632;
static constexpr uint16_t kExpectedKind    = 0; // FileType::Data
static constexpr uint16_t kExpectedVersion = 1;

class DataWriterTest : public ::testing::Test {
protected:
    std::string input_path_;
    std::string output_path_;

    void SetUp() override {
        std::string base = "/tmp/sketch2_utest_dw_" + std::to_string(getpid());
        input_path_  = base + ".txt";
        output_path_ = base + ".bin";
    }

    void TearDown() override {
        std::remove(input_path_.c_str());
        std::remove(output_path_.c_str());
    }

    // Generate input and run DataWriter::exec(), return the Ret from exec()
    Ret run(size_t count, size_t min_id, DataType type, size_t dim) {
        GeneratorConfig cfg{PatternType::Sequential, count, min_id, type, dim};
        generate_input_file(input_path_, cfg);
        DataWriter w;
        w.init(input_path_, output_path_);
        return w.exec();
    }

    DataFileHeader read_header() {
        DataFileHeader hdr{};
        FILE* f = fopen(output_path_.c_str(), "rb");
        if (f) {
            fread(&hdr, sizeof(hdr), 1, f);
            fclose(f);
        }
        return hdr;
    }

    std::vector<uint64_t> read_ids(size_t count, size_t vec_size) {
        std::vector<uint64_t> ids(count);
        FILE* f = fopen(output_path_.c_str(), "rb");
        if (f) {
            fseek(f, static_cast<long>(sizeof(DataFileHeader) + count * vec_size), SEEK_SET);
            fread(ids.data(), sizeof(uint64_t), count, f);
            fclose(f);
        }
        return ids;
    }

    std::vector<float> read_f32_vectors(size_t count, size_t dim) {
        std::vector<float> data(count * dim);
        FILE* f = fopen(output_path_.c_str(), "rb");
        if (f) {
            fseek(f, static_cast<long>(sizeof(DataFileHeader)), SEEK_SET);
            fread(data.data(), sizeof(float), count * dim, f);
            fclose(f);
        }
        return data;
    }
};

// --- error cases ---

TEST_F(DataWriterTest, FailsOnBadInputPath) {
    DataWriter w;
    w.init("/nonexistent/dir/input.txt", output_path_);
    EXPECT_NE(0, w.exec().code());
}

TEST_F(DataWriterTest, FailsOnBadOutputPath) {
    GeneratorConfig cfg{PatternType::Sequential, 3, 0, DataType::f32, 4};
    generate_input_file(input_path_, cfg);
    DataWriter w;
    w.init(input_path_, "/nonexistent/dir/output.bin");
    EXPECT_NE(0, w.exec().code());
}

// --- success ---

TEST_F(DataWriterTest, SuccessReturnCode) {
    EXPECT_EQ(0, run(3, 0, DataType::f32, 4).code());
}

// --- output file size ---

TEST_F(DataWriterTest, OutputFileSize) {
    const size_t count = 5, dim = 4;
    run(count, 0, DataType::f32, dim);
    size_t expected = sizeof(DataFileHeader)
                    + count * dim * sizeof(float)  // vectors
                    + count * sizeof(uint64_t);    // ids
    FILE* f = fopen(output_path_.c_str(), "rb");
    ASSERT_NE(nullptr, f);
    fseek(f, 0, SEEK_END);
    size_t actual = static_cast<size_t>(ftell(f));
    fclose(f);
    EXPECT_EQ(expected, actual);
}

// --- header fields ---

TEST_F(DataWriterTest, HeaderMagic) {
    run(3, 0, DataType::f32, 4);
    EXPECT_EQ(kExpectedMagic, read_header().magic);
}

TEST_F(DataWriterTest, HeaderKind) {
    run(3, 0, DataType::f32, 4);
    EXPECT_EQ(kExpectedKind, read_header().kind);
}

TEST_F(DataWriterTest, HeaderVersion) {
    run(3, 0, DataType::f32, 4);
    EXPECT_EQ(kExpectedVersion, read_header().version);
}

TEST_F(DataWriterTest, HeaderCount) {
    run(7, 0, DataType::f32, 4);
    EXPECT_EQ(7u, read_header().count);
}

TEST_F(DataWriterTest, HeaderMinMaxId) {
    run(5, 10, DataType::f32, 4);
    auto hdr = read_header();
    EXPECT_EQ(10u, hdr.min_id);
    EXPECT_EQ(14u, hdr.max_id);
}

TEST_F(DataWriterTest, HeaderDim) {
    run(1, 0, DataType::f32, 64);
    EXPECT_EQ(64u, read_header().dim);
}

TEST_F(DataWriterTest, HeaderTypeF32) {
    run(1, 0, DataType::f32, 4);
    EXPECT_EQ(data_type_to_int(DataType::f32), read_header().type);
}

TEST_F(DataWriterTest, HeaderTypeF16) {
    EXPECT_THROW((void)run(1, 0, DataType::f16, 4), std::runtime_error);
}

TEST_F(DataWriterTest, HeaderTypeI32) {
    run(1, 0, DataType::i32, 4);
    EXPECT_EQ(data_type_to_int(DataType::i32), read_header().type);
}

// --- ids section ---

TEST_F(DataWriterTest, IdsAreCorrect) {
    const size_t count = 4, min_id = 20, dim = 4;
    run(count, min_id, DataType::f32, dim);
    auto ids = read_ids(count, dim * sizeof(float));
    for (size_t i = 0; i < count; ++i) {
        EXPECT_EQ(min_id + i, ids[i]) << "id at index " << i;
    }
}

// --- vector data section ---

TEST_F(DataWriterTest, F32VectorDataIsCorrect) {
    const size_t count = 3, min_id = 5, dim = 4;
    run(count, min_id, DataType::f32, dim);
    auto data = read_f32_vectors(count, dim);
    for (size_t i = 0; i < count; ++i) {
        float expected = static_cast<float>(min_id + i) + 0.1f;
        for (size_t d = 0; d < dim; ++d) {
            EXPECT_NEAR(expected, data[i * dim + d], 1e-4f)
                << "vector " << i << " dim " << d;
        }
    }
}
