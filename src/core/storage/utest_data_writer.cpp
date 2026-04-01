// Unit tests for writing binary data files from textual input.

#include <gtest/gtest.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <unistd.h>
#include <vector>
#include "core/storage/input_generator.h"
#include "core/storage/data_file.h"
#include "core/storage/data_file_layout.h"
#include "core/storage/data_writer.h"
#include "core/storage/data_reader.h"
#include "core/storage/dataset_writer.h"
#include "utest_tmp_dir.h"
#include <filesystem>

using namespace sketch2;
namespace fs = std::filesystem;

class DataWriterTest : public ::testing::Test {
protected:
    std::string input_path_;
    std::string output_path_;
    std::string dataset_dir_;
    std::string dataset_ini_path_;

    void SetUp() override {
        std::string base = tmp_dir() + "/sketch2_utest_dw_" + std::to_string(getpid());
        input_path_  = base + ".txt";
        output_path_ = base + ".bin";
        dataset_dir_ = base + "_dataset";
        dataset_ini_path_ = base + "_dataset.ini";
        fs::create_directories(dataset_dir_);
    }

    void TearDown() override {
        std::remove(input_path_.c_str());
        std::remove(output_path_.c_str());
        std::remove(dataset_ini_path_.c_str());
        fs::remove_all(dataset_dir_);
    }

    // Generate input and run DataWriter::exec(), return the Ret from exec()
    Ret run(size_t count, size_t min_id, DataType type, size_t dim,
            size_t every_n_deleted = 0, bool write_cosine_inv_norms = false) {
        GeneratorConfig cfg{PatternType::Sequential, count, min_id, type, dim, 1000, every_n_deleted};
        generate_input_file(input_path_, cfg);
        DataWriter w;
        w.init(input_path_, output_path_, 0, 0, write_cosine_inv_norms);
        return w.exec();
    }

    Ret run_binary(size_t count, size_t min_id, DataType type, size_t dim,
                   bool write_cosine_inv_norms = false) {
        GeneratorConfig cfg{PatternType::Sequential, count, min_id, type, dim, 1000, 0, true};
        generate_input_file(input_path_, cfg);
        DataWriter w;
        w.init(input_path_, output_path_, 0, 0, write_cosine_inv_norms);
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
            const auto sz = fread(&hdr, sizeof(hdr), 1, f);
            (void)sz;
            fclose(f);
        }
        return hdr;
    }

    size_t ids_offset(size_t count, size_t vec_size, const DataFileHeader& hdr) {
        (void)vec_size;
        return align_up<size_t>(static_cast<size_t>(hdr.data_offset)
            + count * static_cast<size_t>(hdr.vector_stride), kIdsAlignment);
    }

    std::vector<uint64_t> read_ids(size_t count, size_t vec_size) {
        std::vector<uint64_t> ids(count);
        FILE* f = fopen(output_path_.c_str(), "rb");
        if (f) {
            DataFileHeader hdr{};
            auto sz = fread(&hdr, sizeof(hdr), 1, f);
            (void)sz;
            fseek(f, static_cast<long>(ids_offset(count, vec_size, hdr)), SEEK_SET);
            sz = fread(ids.data(), sizeof(uint64_t), count, f);
            (void)sz;
            fclose(f);
        }
        return ids;
    }

    std::vector<uint64_t> read_deleted_ids(size_t count, size_t deleted_count, size_t vec_size) {
        std::vector<uint64_t> ids(deleted_count);
        FILE* f = fopen(output_path_.c_str(), "rb");
        if (f) {
            DataFileHeader hdr{};
            auto sz = fread(&hdr, sizeof(hdr), 1, f);
            (void)sz;
            const size_t offset = ids_offset(count, vec_size, hdr) + count * sizeof(uint64_t);
            fseek(f, static_cast<long>(offset), SEEK_SET);
            sz = fread(ids.data(), sizeof(uint64_t), deleted_count, f);
            (void)sz;
            fclose(f);
        }
        return ids;
    }

    std::vector<float> read_f32_vectors(size_t count, size_t dim) {
        std::vector<float> data(count * dim);
        FILE* f = fopen(output_path_.c_str(), "rb");
        if (f) {
            DataFileHeader hdr{};
            auto sz = fread(&hdr, sizeof(hdr), 1, f);
            (void)sz;
            for (size_t i = 0; i < count; ++i) {
                fseek(f, static_cast<long>(hdr.data_offset + i * static_cast<size_t>(hdr.vector_stride)), SEEK_SET);
                sz = fread(data.data() + i * dim, sizeof(float), dim, f);
                (void)sz;
            }
            fclose(f);
        }
        return data;
    }

    std::vector<float> read_cosine_inv_norms(size_t count) {
        std::vector<float> values(count);
        FILE* f = fopen(output_path_.c_str(), "rb");
        if (f) {
            DataFileHeader hdr{};
            auto sz = fread(&hdr, sizeof(hdr), 1, f);
            (void)sz;
            const IdsLayout layout = compute_ids_layout(hdr, count);
            fseek(f, static_cast<long>(layout.cosine_inv_norms_offset), SEEK_SET);
            sz = fread(values.data(), sizeof(float), count, f);
            (void)sz;
            fclose(f);
        }
        return values;
    }

    Ret init_dataset_writer(DatasetWriter* writer, DataType type = DataType::f32, uint64_t dim = 4,
            uint64_t range_size = 1000, DistFunc dist_func = DistFunc::L1) {
        DatasetMetadata metadata;
        metadata.dirs = {dataset_dir_};
        metadata.type = type;
        metadata.dim = dim;
        metadata.range_size = range_size;
        metadata.dist_func = dist_func;
        CHECK(write_dataset_ini(metadata, dataset_ini_path_));
        return writer->init(dataset_ini_path_);
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
                    + count * static_cast<size_t>(hdr.vector_stride)  // padded vector records
                    + (ids_off - (hdr.data_offset + count * static_cast<size_t>(hdr.vector_stride))) // ids alignment padding
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
    EXPECT_EQ(kMagic, read_header().base.magic);
}

TEST_F(DataWriterTest, HeaderKind) {
    run(3, 0, DataType::f32, 4);
    EXPECT_EQ(static_cast<uint16_t>(FileType::Data), read_header().base.kind);
}

TEST_F(DataWriterTest, HeaderVersion) {
    run(3, 0, DataType::f32, 4);
    EXPECT_EQ(kVersion, read_header().base.version);
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

TEST_F(DataWriterTest, HeaderVectorStrideIsAligned) {
    run(1, 0, DataType::f32, 5);
    const auto hdr = read_header();
    EXPECT_EQ(compute_vector_stride(5u * sizeof(float)), hdr.vector_stride);
    EXPECT_EQ(0u, hdr.vector_stride % kDataAlignment);
}

TEST_F(DataWriterTest, HeaderTypeF32) {
    run(1, 0, DataType::f32, 4);
    EXPECT_EQ(data_type_to_int(DataType::f32), read_header().type);
}

TEST_F(DataWriterTest, HeaderTypeF16) {
    run(1, 0, DataType::f16, 4);
    EXPECT_EQ(data_type_to_int(DataType::f16), read_header().type);
}

TEST_F(DataWriterTest, HeaderTypeI16) {
    run(1, 0, DataType::i16, 4);
    EXPECT_EQ(data_type_to_int(DataType::i16), read_header().type);
}

TEST_F(DataWriterTest, HeaderCosineFlagIsSetWhenRequested) {
    ASSERT_EQ(0, run(3, 0, DataType::f32, 4, 0, true).code());
    EXPECT_TRUE(data_file_has_cosine_inv_norms(read_header()));
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

TEST_F(DataWriterTest, BinaryInputVectorDataIsCorrect) {
    const size_t count = 3, min_id = 5, dim = 4;
    ASSERT_EQ(0, run_binary(count, min_id, DataType::f32, dim).code());
    auto data = read_f32_vectors(count, dim);
    for (size_t i = 0; i < count; ++i) {
        float expected = static_cast<float>(min_id + i) + 0.1f;
        for (size_t d = 0; d < dim; ++d) {
            EXPECT_NEAR(expected, data[i * dim + d], 1e-4f)
                << "vector " << i << " dim " << d;
        }
    }
}

TEST_F(DataWriterTest, CosineValuesSectionIsWritten) {
    ASSERT_EQ(0, run(2, 0, DataType::f32, 4, 0, true).code());
    const auto hdr = read_header();
    ASSERT_TRUE(data_file_has_cosine_inv_norms(hdr));

    const auto cosine_inv_norms = read_cosine_inv_norms(hdr.count);
    ASSERT_EQ(2u, cosine_inv_norms.size());
    const float inv_norm0 = 1.0f / std::sqrt(4.0f * 0.1f * 0.1f);
    const float inv_norm1 = 1.0f / std::sqrt(4.0f * 1.1f * 1.1f);
    EXPECT_NEAR(inv_norm0, cosine_inv_norms[0], 1e-5f);
    EXPECT_NEAR(inv_norm1, cosine_inv_norms[1], 1e-5f);
}

TEST_F(DataWriterTest, CosineValuesSectionIsWrittenForI16) {
    ASSERT_EQ(0, run(2, 0, DataType::i16, 4, 0, true).code());
    const auto hdr = read_header();
    ASSERT_TRUE(data_file_has_cosine_inv_norms(hdr));

    const auto cosine_inv_norms = read_cosine_inv_norms(hdr.count);
    ASSERT_EQ(2u, cosine_inv_norms.size());
    EXPECT_FLOAT_EQ(0.0f, cosine_inv_norms[0]);
    EXPECT_NEAR(1.0f / std::sqrt(4.0f), cosine_inv_norms[1], 1e-6f);
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

// --- DatasetWriter staged input path ---

TEST_F(DataWriterTest, DatasetWriterStagedWriteCreatesDataFileAndRemovesInputFile) {
    DatasetWriter writer;
    ASSERT_EQ(0, init_dataset_writer(&writer).code());

    const fs::path input_path = fs::path(dataset_dir_) / "sketch2.owner.input";
    const fs::path data_path = fs::path(dataset_dir_) / "0.data";

    ASSERT_EQ(0, writer.start_writing().code());
    ASSERT_TRUE(fs::exists(input_path));
    ASSERT_EQ(0, writer.write_vector(10, "10.1, 10.1, 10.1, 10.1").code());
    ASSERT_EQ(0, writer.write_vector(11, "11.1 11.1 11.1 11.1").code());
    ASSERT_EQ(0, writer.complete_writing().code());

    EXPECT_FALSE(fs::exists(input_path));
    ASSERT_TRUE(fs::exists(data_path));

    DataReader reader;
    ASSERT_EQ(0, reader.init(data_path.string()).code());
    ASSERT_EQ(2u, reader.count());
    EXPECT_EQ(10u, reader.id(0));
    EXPECT_EQ(11u, reader.id(1));
}

TEST_F(DataWriterTest, DatasetWriterStagedWriteRejectsInvalidCallOrder) {
    DatasetWriter writer;
    ASSERT_EQ(0, init_dataset_writer(&writer).code());

    EXPECT_NE(0, writer.write_vector(10, "10.1, 10.1, 10.1, 10.1").code());
    EXPECT_NE(0, writer.write_deleted(10).code());
    EXPECT_NE(0, writer.complete_writing().code());

    ASSERT_EQ(0, writer.start_writing().code());
    EXPECT_NE(0, writer.start_writing().code());
}

TEST_F(DataWriterTest, DatasetWriterStagedDeleteOnExistingDataRemovesVisibleItem) {
    DatasetWriter writer;
    ASSERT_EQ(0, init_dataset_writer(&writer).code());

    ASSERT_EQ(0, writer.start_writing().code());
    ASSERT_EQ(0, writer.write_vector(10, "10.1, 10.1, 10.1, 10.1").code());
    ASSERT_EQ(0, writer.complete_writing().code());

    ASSERT_EQ(0, writer.start_writing().code());
    ASSERT_EQ(0, writer.write_deleted(10).code());
    ASSERT_EQ(0, writer.complete_writing().code());

    const fs::path data_path = fs::path(dataset_dir_) / "0.data";
    const fs::path delta_path = fs::path(dataset_dir_) / "0.delta";
    ASSERT_TRUE(fs::exists(data_path));
    EXPECT_FALSE(fs::exists(delta_path));

    DataReader data_reader;
    ASSERT_EQ(0, data_reader.init(data_path.string()).code());
    EXPECT_EQ(0u, data_reader.count());
    EXPECT_EQ(0u, data_reader.deleted_count());
    EXPECT_EQ(nullptr, data_reader.get(10));
}

TEST_F(DataWriterTest, DatasetWriterStagedWriteManyVectorsCreatesCorrectDataFile) {
    DatasetWriter writer;
    ASSERT_EQ(0, init_dataset_writer(&writer, DataType::f32, 4, 1000).code());

    ASSERT_EQ(0, writer.start_writing().code());
    for (uint64_t id = 0; id < 100; ++id) {
        const std::string value = std::to_string(static_cast<float>(id) + 0.1f) + ", " +
            std::to_string(static_cast<float>(id) + 0.1f) + ", " +
            std::to_string(static_cast<float>(id) + 0.1f) + ", " +
            std::to_string(static_cast<float>(id) + 0.1f);
        ASSERT_EQ(0, writer.write_vector(id, value.c_str()).code());
    }
    ASSERT_EQ(0, writer.complete_writing().code());

    const fs::path data_path = fs::path(dataset_dir_) / "0.data";
    ASSERT_TRUE(fs::exists(data_path));

    DataReader data_reader;
    ASSERT_EQ(0, data_reader.init(data_path.string()).code());
    ASSERT_EQ(100u, data_reader.count());
    EXPECT_EQ(0u, data_reader.id(0));
    EXPECT_EQ(99u, data_reader.id(99));
}

TEST_F(DataWriterTest, DatasetWriterStagedWriteWithDeletesProducesCorrectCounts) {
    DatasetWriter writer;
    ASSERT_EQ(0, init_dataset_writer(&writer).code());

    // First session: create the vectors
    ASSERT_EQ(0, writer.start_writing().code());
    ASSERT_EQ(0, writer.write_vector(10, "10.0, 10.0, 10.0, 10.0").code());
    ASSERT_EQ(0, writer.write_vector(11, "11.0, 11.0, 11.0, 11.0").code());
    ASSERT_EQ(0, writer.write_vector(12, "12.0, 12.0, 12.0, 12.0").code());
    ASSERT_EQ(0, writer.write_vector(13, "13.0, 13.0, 13.0, 13.0").code());
    ASSERT_EQ(0, writer.write_vector(14, "14.0, 14.0, 14.0, 14.0").code());
    ASSERT_EQ(0, writer.complete_writing().code());

    // Second session: delete two of them
    ASSERT_EQ(0, writer.start_writing().code());
    ASSERT_EQ(0, writer.write_deleted(13).code());
    ASSERT_EQ(0, writer.write_deleted(14).code());
    ASSERT_EQ(0, writer.complete_writing().code());

    // Force merge so deletes are folded into the data file
    ASSERT_EQ(0, writer.merge().code());

    const fs::path data_path = fs::path(dataset_dir_) / "0.data";
    ASSERT_TRUE(fs::exists(data_path));

    DataReader data_reader;
    ASSERT_EQ(0, data_reader.init(data_path.string()).code());
    EXPECT_EQ(3u, data_reader.count());
}

TEST_F(DataWriterTest, DatasetWriterStagedWriteUnsortedIdsProducesSortedDataFile) {
    DatasetWriter writer;
    ASSERT_EQ(0, init_dataset_writer(&writer).code());

    ASSERT_EQ(0, writer.start_writing().code());
    ASSERT_EQ(0, writer.write_vector(30, "30.0, 30.0, 30.0, 30.0").code());
    ASSERT_EQ(0, writer.write_vector(10, "10.0, 10.0, 10.0, 10.0").code());
    ASSERT_EQ(0, writer.write_vector(20, "20.0, 20.0, 20.0, 20.0").code());
    ASSERT_EQ(0, writer.complete_writing().code());

    const fs::path data_path = fs::path(dataset_dir_) / "0.data";
    ASSERT_TRUE(fs::exists(data_path));

    DataReader data_reader;
    ASSERT_EQ(0, data_reader.init(data_path.string()).code());
    EXPECT_EQ(3u, data_reader.count());
    EXPECT_EQ(10u, data_reader.id(0));
    EXPECT_EQ(20u, data_reader.id(1));
    EXPECT_EQ(30u, data_reader.id(2));
}

TEST_F(DataWriterTest, DatasetWriterStagedWriteSpansMultipleRanges) {
    DatasetWriter writer;
    ASSERT_EQ(0, init_dataset_writer(&writer, DataType::f32, 4, 100).code());

    ASSERT_EQ(0, writer.start_writing().code());
    ASSERT_EQ(0, writer.write_vector(50, "50.0, 50.0, 50.0, 50.0").code());
    ASSERT_EQ(0, writer.write_vector(150, "150.0, 150.0, 150.0, 150.0").code());
    ASSERT_EQ(0, writer.write_vector(250, "250.0, 250.0, 250.0, 250.0").code());
    ASSERT_EQ(0, writer.complete_writing().code());

    EXPECT_TRUE(fs::exists(fs::path(dataset_dir_) / "0.data"));
    EXPECT_TRUE(fs::exists(fs::path(dataset_dir_) / "1.data"));
    EXPECT_TRUE(fs::exists(fs::path(dataset_dir_) / "2.data"));

    DataReader r0, r1, r2;
    ASSERT_EQ(0, r0.init((fs::path(dataset_dir_) / "0.data").string()).code());
    ASSERT_EQ(0, r1.init((fs::path(dataset_dir_) / "1.data").string()).code());
    ASSERT_EQ(0, r2.init((fs::path(dataset_dir_) / "2.data").string()).code());
    EXPECT_EQ(1u, r0.count());
    EXPECT_EQ(1u, r1.count());
    EXPECT_EQ(1u, r2.count());
    EXPECT_EQ(50u, r0.id(0));
    EXPECT_EQ(150u, r1.id(0));
    EXPECT_EQ(250u, r2.id(0));
}

TEST_F(DataWriterTest, DatasetWriterMultipleStagedWriteSessions) {
    DatasetWriter writer;
    ASSERT_EQ(0, init_dataset_writer(&writer).code());

    ASSERT_EQ(0, writer.start_writing().code());
    ASSERT_EQ(0, writer.write_vector(10, "10.0, 10.0, 10.0, 10.0").code());
    ASSERT_EQ(0, writer.complete_writing().code());

    ASSERT_EQ(0, writer.start_writing().code());
    ASSERT_EQ(0, writer.write_vector(11, "11.0, 11.0, 11.0, 11.0").code());
    ASSERT_EQ(0, writer.complete_writing().code());

    const fs::path data_path = fs::path(dataset_dir_) / "0.data";
    ASSERT_TRUE(fs::exists(data_path));

    DataReader data_reader;
    ASSERT_EQ(0, data_reader.init(data_path.string()).code());
    EXPECT_EQ(2u, data_reader.count());
    EXPECT_EQ(10u, data_reader.id(0));
    EXPECT_EQ(11u, data_reader.id(1));
}

TEST_F(DataWriterTest, DatasetWriterStagedWriteWithCosineDistFunc) {
    DatasetWriter writer;
    ASSERT_EQ(0, init_dataset_writer(&writer, DataType::f32, 4, 1000, DistFunc::COS).code());

    ASSERT_EQ(0, writer.start_writing().code());
    ASSERT_EQ(0, writer.write_vector(10, "1.0, 0.0, 0.0, 0.0").code());
    ASSERT_EQ(0, writer.write_vector(11, "0.0, 1.0, 0.0, 0.0").code());
    ASSERT_EQ(0, writer.complete_writing().code());

    const fs::path data_path = fs::path(dataset_dir_) / "0.data";
    ASSERT_TRUE(fs::exists(data_path));

    DataReader data_reader;
    ASSERT_EQ(0, data_reader.init(data_path.string()).code());
    ASSERT_EQ(2u, data_reader.count());
    EXPECT_NEAR(1.0f, data_reader.cosine_inv_norm(0), 1e-5f);
    EXPECT_NEAR(1.0f, data_reader.cosine_inv_norm(1), 1e-5f);
}
