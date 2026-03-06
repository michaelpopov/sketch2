#include <gtest/gtest.h>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <unistd.h>
#include <vector>
#include "core/storage/data_file.h"
#include "core/storage/input_generator.h"
#include "core/storage/data_writer.h"

using namespace sketch2;

static constexpr uint32_t kExpectedMagic   = 0x534B5632;
static constexpr uint16_t kExpectedKind    = 0; // FileType::Data
static constexpr uint16_t kExpectedVersion = kVersion;

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
    Ret run(size_t count, size_t min_id, DataType type, size_t dim, size_t every_n_deleted = 0) {
        GeneratorConfig cfg{PatternType::Sequential, count, min_id, type, dim, 1000, every_n_deleted};
        generate_input_file(input_path_, cfg);
        DataWriter w;
        w.init(input_path_, output_path_);
        return w.exec();
    }

    Ret run_raw_input(const std::string& content) {
        std::ofstream f(input_path_);
        f << content;
        f.close();
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

    size_t ids_offset(size_t count, size_t vec_size, const DataFileHeader& hdr) {
        return align_up<size_t>(static_cast<size_t>(hdr.data_offset) + count * vec_size, kIdsAlignment);
    }

    std::vector<uint64_t> read_ids(size_t count, size_t vec_size) {
        std::vector<uint64_t> ids(count);
        FILE* f = fopen(output_path_.c_str(), "rb");
        if (f) {
            DataFileHeader hdr{};
            fread(&hdr, sizeof(hdr), 1, f);
            fseek(f, static_cast<long>(ids_offset(count, vec_size, hdr)), SEEK_SET);
            fread(ids.data(), sizeof(uint64_t), count, f);
            fclose(f);
        }
        return ids;
    }

    std::vector<uint64_t> read_deleted_ids(size_t count, size_t deleted_count, size_t vec_size) {
        std::vector<uint64_t> ids(deleted_count);
        FILE* f = fopen(output_path_.c_str(), "rb");
        if (f) {
            DataFileHeader hdr{};
            fread(&hdr, sizeof(hdr), 1, f);
            const size_t offset = ids_offset(count, vec_size, hdr) + count * sizeof(uint64_t);
            fseek(f, static_cast<long>(offset), SEEK_SET);
            fread(ids.data(), sizeof(uint64_t), deleted_count, f);
            fclose(f);
        }
        return ids;
    }

    std::vector<float> read_f32_vectors(size_t count, size_t dim) {
        std::vector<float> data(count * dim);
        FILE* f = fopen(output_path_.c_str(), "rb");
        if (f) {
            DataFileHeader hdr{};
            fread(&hdr, sizeof(hdr), 1, f);
            fseek(f, static_cast<long>(hdr.data_offset), SEEK_SET);
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
    GeneratorConfig cfg{PatternType::Sequential, 3, 0, DataType::f32, 4, 1000};
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
    const DataFileHeader hdr = read_header();
    const size_t ids_off = ids_offset(count, dim * sizeof(float), hdr);
    size_t expected = sizeof(DataFileHeader)
                    + (hdr.data_offset - sizeof(DataFileHeader))
                    + count * dim * sizeof(float)  // vectors
                    + (ids_off - (hdr.data_offset + count * dim * sizeof(float))) // ids alignment padding
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
    EXPECT_EQ(kExpectedMagic, read_header().base.magic);
}

TEST_F(DataWriterTest, HeaderKind) {
    run(3, 0, DataType::f32, 4);
    EXPECT_EQ(kExpectedKind, read_header().base.kind);
}

TEST_F(DataWriterTest, HeaderVersion) {
    run(3, 0, DataType::f32, 4);
    EXPECT_EQ(kExpectedVersion, read_header().base.version);
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
    if (supports_f16()) {
        run(1, 0, DataType::f16, 4);
        EXPECT_EQ(data_type_to_int(DataType::f16), read_header().type);
    }
}

TEST_F(DataWriterTest, HeaderTypeI16) {
    run(1, 0, DataType::i16, 4);
    EXPECT_EQ(data_type_to_int(DataType::i16), read_header().type);
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

TEST_F(DataWriterTest, UnsortedInputIsWrittenInSortedIdOrder) {
    const std::string content =
        "f32,4\n"
        "20 : [ 20.1, 20.1, 20.1, 20.1 ]\n"
        "10 : [ 10.1, 10.1, 10.1, 10.1 ]\n"
        "15 : [ 15.1, 15.1, 15.1, 15.1 ]\n";
    ASSERT_EQ(0, run_raw_input(content).code());

    const auto hdr = read_header();
    ASSERT_EQ(3u, hdr.count);
    EXPECT_EQ(10u, hdr.min_id);
    EXPECT_EQ(20u, hdr.max_id);

    const size_t vec_size = 4 * sizeof(float);
    const auto ids = read_ids(hdr.count, vec_size);
    ASSERT_EQ((std::vector<uint64_t>{10, 15, 20}), ids);

    const auto data = read_f32_vectors(hdr.count, 4);
    ASSERT_EQ(12u, data.size());
    EXPECT_NEAR(10.1f, data[0], 1e-4f);
    EXPECT_NEAR(15.1f, data[4], 1e-4f);
    EXPECT_NEAR(20.1f, data[8], 1e-4f);
}

TEST_F(DataWriterTest, UnsortedInputWithDeletesWritesSortedActiveAndDeletedIds) {
    const std::string content =
        "f32,4\n"
        "12 : []\n"
        "11 : [ 11.1, 11.1, 11.1, 11.1 ]\n"
        "8 : []\n"
        "10 : [ 10.1, 10.1, 10.1, 10.1 ]\n";
    ASSERT_EQ(0, run_raw_input(content).code());

    const auto hdr = read_header();
    ASSERT_EQ(2u, hdr.count);
    ASSERT_EQ(2u, hdr.deleted_count);
    EXPECT_EQ(10u, hdr.min_id);
    EXPECT_EQ(11u, hdr.max_id);

    const size_t vec_size = 4 * sizeof(float);
    const auto ids = read_ids(hdr.count, vec_size);
    const auto deleted = read_deleted_ids(hdr.count, hdr.deleted_count, vec_size);
    ASSERT_EQ((std::vector<uint64_t>{10, 11}), ids);
    ASSERT_EQ((std::vector<uint64_t>{8, 12}), deleted);
}

TEST_F(DataWriterTest, HeaderDeletedCountAndActiveCountAreCorrect) {
    ASSERT_EQ(0, run(6, 0, DataType::f32, 4, 2).code()); // deleted ids: 2,4
    const auto hdr = read_header();
    EXPECT_EQ(4u, hdr.count);
    EXPECT_EQ(2u, hdr.deleted_count);
    EXPECT_EQ(0u, hdr.min_id);
    EXPECT_EQ(5u, hdr.max_id);
}

TEST_F(DataWriterTest, DeletedIdsSectionIsWritten) {
    ASSERT_EQ(0, run(6, 0, DataType::f32, 4, 2).code());
    const auto hdr = read_header();
    ASSERT_EQ(4u, hdr.count);
    ASSERT_EQ(2u, hdr.deleted_count);

    const size_t vec_size = 4 * sizeof(float);
    auto ids = read_ids(hdr.count, vec_size);
    auto deleted_ids = read_deleted_ids(hdr.count, hdr.deleted_count, vec_size);

    ASSERT_EQ(4u, ids.size());
    ASSERT_EQ(2u, deleted_ids.size());
    EXPECT_EQ(0u, ids[0]);
    EXPECT_EQ(1u, ids[1]);
    EXPECT_EQ(3u, ids[2]);
    EXPECT_EQ(5u, ids[3]);
    EXPECT_EQ(2u, deleted_ids[0]);
    EXPECT_EQ(4u, deleted_ids[1]);
}

TEST_F(DataWriterTest, AllDeletedInputProducesZeroActiveRangeAndCount) {
    const std::string content =
        "f32,4\n"
        "100 : []\n"
        "101 : []\n"
        "102 : []\n";
    ASSERT_EQ(0, run_raw_input(content).code());
    const auto hdr = read_header();
    EXPECT_EQ(0u, hdr.count);
    EXPECT_EQ(3u, hdr.deleted_count);
    EXPECT_EQ(0u, hdr.min_id);
    EXPECT_EQ(0u, hdr.max_id);
}
