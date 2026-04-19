// Unit tests for binary data-file reading.

#include <gtest/gtest.h>
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <unistd.h>
#include <memory>
#include <stdexcept>
#include <vector>
#include <fstream>
#include "core/compute/norm_utils.h"
#include "core/storage/data_file.h"
#include "core/storage/data_file_layout.h"
#include "core/storage/input_generator.h"
#include "core/storage/data_writer.h"
#include "core/storage/data_reader.h"
#include "core/storage/compact_ids.h"
#include "utest_tmp_dir.h"

using namespace sketch2;

namespace {

std::vector<uint8_t> read_file_bytes(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        return {};
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return {};
    }
    const long size = ftell(f);
    if (size < 0) {
        fclose(f);
        return {};
    }
    rewind(f);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (!bytes.empty()) {
        const size_t read = fread(bytes.data(), 1, bytes.size(), f);
        if (read != bytes.size()) {
            fclose(f);
            return {};
        }
    }
    fclose(f);
    return bytes;
}

} // namespace

class DataReaderTest : public ::testing::Test {
protected:
    std::string input_path_;
    std::string data_path_;
    std::string delta_input_path_;
    std::string delta_path_;

    void SetUp() override {
        std::string base = tmp_dir() + "/sketch2_utest_dr_" + std::to_string(getpid());
        input_path_ = base + ".txt";
        data_path_  = base + ".bin";
        delta_input_path_ = base + ".delta.txt";
        delta_path_ = base + ".delta.bin";
    }

    void TearDown() override {
        std::remove(input_path_.c_str());
        std::remove(data_path_.c_str());
        std::remove(delta_input_path_.c_str());
        std::remove(delta_path_.c_str());
    }

    void generate_file(const std::string& in_path, const std::string& out_path, const GeneratorConfig& cfg) {
        Ret ret = generate_input_file(in_path, cfg);
        ASSERT_EQ(0, ret.code()) << "generate_input_file failed: " << ret.message();
        DataWriter w;
        ret = w.exec_for_testing(in_path, out_path);
        ASSERT_EQ(0, ret.code()) << "DataWriter::exec_for_testing failed: " << ret.message();
    }

    void generate(size_t count, size_t min_id, DataType type, size_t dim, size_t every_n_deleted = 0) {
        GeneratorConfig cfg{PatternType::Sequential, count, min_id, type, dim, 1000, every_n_deleted};
        generate_file(input_path_, data_path_, cfg);
    }

    void generate_delta(size_t count, size_t min_id, DataType type, size_t dim, size_t every_n_deleted = 0) {
        GeneratorConfig cfg{PatternType::Sequential, count, min_id, type, dim, 1000, every_n_deleted};
        generate_file(delta_input_path_, delta_path_, cfg);
    }

    void generate_delta_detailed(size_t count, size_t min_id, DataType type, size_t dim, size_t max_val = 1000, size_t every_n_deleted = 0) {
        GeneratorConfig cfg{PatternType::Detailed, count, min_id, type, dim, max_val, every_n_deleted};
        generate_file(delta_input_path_, delta_path_, cfg);
    }

    void write_raw_to_data_file(const std::string& in_path, const std::string& out_path, const std::string& content) {
        std::ofstream f(in_path);
        f << content;
        f.close();
        DataWriter w;
        ASSERT_EQ(0, w.exec_for_testing(in_path, out_path).code());
    }

    std::unique_ptr<DataReader> make_delta_reader() {
        auto delta = std::make_unique<DataReader>();
        EXPECT_EQ(0, delta->init(delta_path_).code());
        return delta;
    }

    // Write a minimal valid binary data file with vector bytes and compact id trailer.
    // type_field: 0=f16, 1=f32, 2=i16  (matches DataWriter encoding)
    void write_raw(DataType type_field, uint16_t dim, uint64_t min_id,
                   const std::vector<std::vector<uint8_t>>& vecs,
                   uint32_t norm_flags = 0) {
        CompactIds compact_ids;
        std::vector<uint64_t> ids;
        ids.reserve(vecs.size());
        for (size_t i = 0; i < vecs.size(); ++i) {
            ids.push_back(min_id + static_cast<uint64_t>(i));
        }
        ASSERT_EQ(0, compact_ids.init(ids).code());
        CompactIds compact_deleted_ids;
        std::vector<uint64_t> deleted_ids;
        ASSERT_EQ(0, compact_deleted_ids.init(deleted_ids).code());

        DataFileHeader hdr = make_data_header(
            min_id,
            min_id + static_cast<uint64_t>(vecs.size()) - 1,
            static_cast<uint32_t>(vecs.size()),
            0,
            type_field,
            dim,
            norm_flags);
        ASSERT_EQ(0, set_data_header_layout(
            &hdr, compact_ids.serialized_size_bytes(), compact_deleted_ids.serialized_size_bytes()).code());
        FILE* f = fopen(data_path_.c_str(), "wb");
        fwrite(&hdr, sizeof(hdr), 1, f);
        const size_t pad_size = static_cast<size_t>(hdr.data_offset) - sizeof(DataFileHeader);
        if (pad_size > 0) {
            std::vector<uint8_t> pad(pad_size, 0);
            fwrite(pad.data(), 1, pad.size(), f);
        }
        const DataRecordLayout record_layout = compute_data_record_layout(type_field, dim, norm_flags != 0);
        for (const auto& v : vecs) {
            float norm = 0.0f;
            float* norm_ptr = nullptr;
            if (norm_flags != 0) {
                if (norm_flags == kDataFileHasCosineInvNorms) {
                    norm = compute_cosine_inverse_norm(v.data(), type_field, dim);
                } else if (norm_flags == kDataFileHasSquaredNorms) {
                    norm = compute_squared_norm(v.data(), type_field, dim);
                } else {
                    FAIL() << "write_raw: invalid stored norm flags";
                }
                norm_ptr = &norm;
            }
            ASSERT_EQ(0, write_data_record(
                f, v.data(), record_layout, norm_ptr, "DataReaderTest::write_raw").code());
        }
        const DataMetadataLayout metadata_layout = compute_data_metadata_layout(hdr, vecs.size());
        const size_t ids_pad_size = metadata_layout.vectors_padding;
        if (ids_pad_size > 0) {
            std::vector<uint8_t> pad(ids_pad_size, 0);
            fwrite(pad.data(), 1, pad.size(), f);
        }
        ASSERT_EQ(0, compact_ids.write(f, "DataReaderTest::write_raw ids write failed").code());
        const size_t deleted_ids_padding =
            compute_deleted_ids_padding(metadata_layout.ids_trailer_offset, compact_ids.serialized_size_bytes());
        if (deleted_ids_padding > 0) {
            std::vector<uint8_t> pad(deleted_ids_padding, 0);
            fwrite(pad.data(), 1, pad.size(), f);
        }
        ASSERT_EQ(0, compact_deleted_ids.write(f, "DataReaderTest::write_raw deleted ids write failed").code());
        fclose(f);
    }

    void write_raw_with_legacy_id_arrays(DataType type_field, uint16_t dim, uint64_t min_id,
                                         const std::vector<std::vector<uint8_t>>& vecs) {
        DataFileHeader hdr = make_data_header(
            min_id,
            min_id + static_cast<uint64_t>(vecs.size()) - 1,
            static_cast<uint32_t>(vecs.size()),
            0,
            type_field,
            dim);
        ASSERT_EQ(0, set_data_header_layout(
            &hdr, vecs.size() * sizeof(uint64_t), 0).code());
        FILE* f = fopen(data_path_.c_str(), "wb");
        ASSERT_NE(nullptr, f);
        ASSERT_EQ(1u, fwrite(&hdr, sizeof(hdr), 1, f));
        const size_t pad_size = static_cast<size_t>(hdr.data_offset) - sizeof(DataFileHeader);
        if (pad_size > 0) {
            std::vector<uint8_t> pad(pad_size, 0);
            ASSERT_EQ(pad.size(), fwrite(pad.data(), 1, pad.size(), f));
        }
        for (const auto& v : vecs) {
            ASSERT_EQ(0, write_vector_record(f, v.data(), v.size(), hdr.vector_stride,
                      "DataReaderTest::write_raw_with_legacy_id_arrays").code());
        }
        const DataMetadataLayout metadata_layout = compute_data_metadata_layout(hdr, vecs.size());
        if (metadata_layout.vectors_padding > 0) {
            std::vector<uint8_t> pad(metadata_layout.vectors_padding, 0);
            ASSERT_EQ(pad.size(), fwrite(pad.data(), 1, pad.size(), f));
        }
        // Intentionally write legacy raw uint64 arrays to ensure hard cutover rejects them.
        for (size_t i = 0; i < vecs.size(); ++i) {
            const uint64_t id = min_id + static_cast<uint64_t>(i);
            ASSERT_EQ(1u, fwrite(&id, sizeof(id), 1, f));
        }
        fclose(f);
    }
};

// --- init error cases ---

TEST_F(DataReaderTest, FailsOnBadPath) {
    DataReader r;
    EXPECT_NE(0, r.init("/nonexistent/dir/file.bin").code());
}

TEST_F(DataReaderTest, FailsOnFileTooSmall) {
    { std::ofstream f(data_path_); f << "tiny"; }
    DataReader r;
    EXPECT_NE(0, r.init(data_path_).code());
}

TEST_F(DataReaderTest, FailsOnInvalidMagic) {
    generate(3, 0, DataType::f32, 4);
    FILE* f = fopen(data_path_.c_str(), "r+b");
    uint32_t bad = 0xDEADBEEF;
    fwrite(&bad, sizeof(bad), 1, f);
    fclose(f);
    DataReader r;
    EXPECT_NE(0, r.init(data_path_).code());
}

TEST_F(DataReaderTest, FailsOnWrongKind) {
    generate(1, 0, DataType::f32, 4);
    // kind is at offset 4 (after the 4-byte magic)
    FILE* f = fopen(data_path_.c_str(), "r+b");
    fseek(f, static_cast<long>(sizeof(uint32_t)), SEEK_SET);
    uint16_t bad_kind = 0xFFFF;
    fwrite(&bad_kind, sizeof(bad_kind), 1, f);
    fclose(f);
    DataReader r;
    EXPECT_NE(0, r.init(data_path_).code());
}

TEST_F(DataReaderTest, FailsOnWrongVersion) {
    generate(1, 0, DataType::f32, 4);
    // version is at offset 6 (after magic and kind)
    FILE* f = fopen(data_path_.c_str(), "r+b");
    ASSERT_NE(nullptr, f);
    fseek(f, static_cast<long>(sizeof(uint32_t) + sizeof(uint16_t)), SEEK_SET);
    uint16_t bad_version = static_cast<uint16_t>(kVersion + 1);
    fwrite(&bad_version, sizeof(bad_version), 1, f);
    fclose(f);
    DataReader r;
    EXPECT_NE(0, r.init(data_path_).code());
}

TEST_F(DataReaderTest, FailsOnPreviousVersionHardCutover) {
    generate(1, 0, DataType::f32, 4);
    FILE* f = fopen(data_path_.c_str(), "r+b");
    ASSERT_NE(nullptr, f);
    ASSERT_EQ(0, fseek(f, static_cast<long>(sizeof(uint32_t) + sizeof(uint16_t)), SEEK_SET));
    const uint16_t prev_version = static_cast<uint16_t>(kVersion - 1);
    ASSERT_EQ(1u, fwrite(&prev_version, sizeof(prev_version), 1, f));
    fclose(f);

    DataReader r;
    const Ret ret = r.init(data_path_);
    EXPECT_NE(0, ret.code());
}

TEST_F(DataReaderTest, FailsOnInvalidCombinedNormFlags) {
    generate(1, 0, DataType::f32, 4);
    FILE* f = fopen(data_path_.c_str(), "r+b");
    ASSERT_NE(nullptr, f);
    DataFileHeader hdr{};
    ASSERT_EQ(1u, fread(&hdr, sizeof(hdr), 1, f));
    hdr.flags = kDataFileHasCosineInvNorms | kDataFileHasSquaredNorms;
    rewind(f);
    ASSERT_EQ(1u, fwrite(&hdr, sizeof(hdr), 1, f));
    fclose(f);

    DataReader r;
    const Ret ret = r.init(data_path_);
    EXPECT_NE(0, ret.code());
    EXPECT_NE(std::string::npos, ret.message().find("invalid stored norm flags"));
}

TEST_F(DataReaderTest, FailsWhenInlineNormDoesNotFitStride) {
    generate(2, 0, DataType::f32, 8);

    FILE* f = fopen(data_path_.c_str(), "r+b");
    ASSERT_NE(nullptr, f);
    DataFileHeader hdr{};
    ASSERT_EQ(1u, fread(&hdr, sizeof(hdr), 1, f));
    hdr.flags = kDataFileHasCosineInvNorms;
    hdr.vector_stride = static_cast<uint32_t>(compute_vector_size(DataType::f32, hdr.dim));
    rewind(f);
    ASSERT_EQ(1u, fwrite(&hdr, sizeof(hdr), 1, f));
    fclose(f);

    DataReader r;
    const Ret ret = r.init(data_path_);
    EXPECT_NE(0, ret.code());
    EXPECT_NE(std::string::npos, ret.message().find("invalid inline norm layout"));
}

TEST_F(DataReaderTest, FailsWhenInlineNormUsesNonCanonicalStride) {
    const uint16_t dim = 5;
    const DataType type = DataType::f32;
    const DataRecordLayout canonical_layout = compute_data_record_layout(type, dim, true);
    const size_t shifted_norm_offset = canonical_layout.norm_offset + kDataAlignment;
    const size_t noncanonical_stride = canonical_layout.stride + kDataAlignment;

    CompactIds compact_ids;
    ASSERT_EQ(0, compact_ids.init(std::vector<uint64_t>{7}).code());
    CompactIds compact_deleted_ids;
    ASSERT_EQ(0, compact_deleted_ids.init(std::vector<uint64_t>{}).code());

    DataFileHeader hdr = make_data_header(7, 7, 1, 0, type, dim, kDataFileHasCosineInvNorms);
    hdr.vector_stride = static_cast<uint32_t>(noncanonical_stride);
    ASSERT_EQ(0, set_data_header_layout(
        &hdr, compact_ids.serialized_size_bytes(), compact_deleted_ids.serialized_size_bytes()).code());

    FILE* f = fopen(data_path_.c_str(), "wb");
    ASSERT_NE(nullptr, f);
    ASSERT_EQ(1u, fwrite(&hdr, sizeof(hdr), 1, f));
    const size_t header_padding = static_cast<size_t>(hdr.data_offset) - sizeof(DataFileHeader);
    if (header_padding > 0) {
        std::vector<uint8_t> pad(header_padding, 0);
        ASSERT_EQ(pad.size(), fwrite(pad.data(), 1, pad.size(), f));
    }

    const std::vector<float> vector_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    ASSERT_EQ(canonical_layout.vector_size, vector_values.size() * sizeof(float));
    ASSERT_EQ(vector_values.size(), fwrite(vector_values.data(), sizeof(float), vector_values.size(), f));
    const size_t bytes_before_shifted_norm = shifted_norm_offset - canonical_layout.vector_size;
    if (bytes_before_shifted_norm > 0) {
        std::vector<uint8_t> pad(bytes_before_shifted_norm, 0);
        ASSERT_EQ(pad.size(), fwrite(pad.data(), 1, pad.size(), f));
    }
    const float norm = static_cast<float>(compute_cosine_inverse_norm(
        reinterpret_cast<const uint8_t*>(vector_values.data()), type, dim));
    ASSERT_EQ(1u, fwrite(&norm, sizeof(norm), 1, f));
    const size_t bytes_after_norm = noncanonical_stride - (shifted_norm_offset + sizeof(norm));
    if (bytes_after_norm > 0) {
        std::vector<uint8_t> pad(bytes_after_norm, 0);
        ASSERT_EQ(pad.size(), fwrite(pad.data(), 1, pad.size(), f));
    }

    const DataMetadataLayout metadata_layout = compute_data_metadata_layout(hdr, 1);
    if (metadata_layout.vectors_padding > 0) {
        std::vector<uint8_t> pad(metadata_layout.vectors_padding, 0);
        ASSERT_EQ(pad.size(), fwrite(pad.data(), 1, pad.size(), f));
    }
    ASSERT_EQ(0, compact_ids.write(f, "DataReaderTest::FailsWhenInlineNormUsesNonCanonicalStride ids").code());
    const size_t deleted_ids_padding =
        compute_deleted_ids_padding(metadata_layout.ids_trailer_offset, compact_ids.serialized_size_bytes());
    if (deleted_ids_padding > 0) {
        std::vector<uint8_t> pad(deleted_ids_padding, 0);
        ASSERT_EQ(pad.size(), fwrite(pad.data(), 1, pad.size(), f));
    }
    ASSERT_EQ(0, compact_deleted_ids.write(
        f, "DataReaderTest::FailsWhenInlineNormUsesNonCanonicalStride deleted ids").code());
    fclose(f);

    DataReader r;
    const Ret ret = r.init(data_path_);
    EXPECT_NE(0, ret.code());
    EXPECT_NE(std::string::npos, ret.message().find("invalid inline norm layout"));
}

TEST_F(DataReaderTest, FailsOnLegacyRawIdsTrailer) {
    const std::vector<std::vector<uint8_t>> vecs = {
        std::vector<uint8_t>(4 * sizeof(float), 0),
        std::vector<uint8_t>(4 * sizeof(float), 1),
    };
    write_raw_with_legacy_id_arrays(DataType::f32, 4, 10, vecs);

    DataReader r;
    EXPECT_NE(0, r.init(data_path_).code());
}

TEST_F(DataReaderTest, FailsOnTruncatedPayload) {
    generate(3, 0, DataType::f32, 4);
    FILE* f = fopen(data_path_.c_str(), "r+b");
    ASSERT_NE(nullptr, f);
    ASSERT_EQ(0, fseek(f, 0, SEEK_END));
    long size = ftell(f);
    ASSERT_GT(size, 0);
    ASSERT_EQ(0, ftruncate(fileno(f), size - 1));
    fclose(f);

    DataReader r;
    EXPECT_NE(0, r.init(data_path_).code());
}

TEST_F(DataReaderTest, CanRetryInitAfterInvalidTypeField) {
    generate(1, 0, DataType::f32, 4);

    FILE* f = fopen(data_path_.c_str(), "r+b");
    ASSERT_NE(nullptr, f);
    DataFileHeader hdr{};
    ASSERT_EQ(1u, fread(&hdr, sizeof(hdr), 1, f));
    hdr.type = 99;
    rewind(f);
    ASSERT_EQ(1u, fwrite(&hdr, sizeof(hdr), 1, f));
    fclose(f);

    DataReader r;
    Ret ret = r.init(data_path_);
    ASSERT_NE(0, ret.code());

    generate(1, 0, DataType::f32, 4);
    ret = r.init(data_path_);
    EXPECT_EQ(0, ret.code()) << ret.message();
    EXPECT_EQ(DataType::f32, r.type());
    EXPECT_EQ(1u, r.count());
}

// --- metadata after init ---

TEST_F(DataReaderTest, SuccessReturnCode) {
    generate(3, 0, DataType::f32, 4);
    DataReader r;
    const auto ret = r.init(data_path_);
    EXPECT_EQ(0, ret.code()) << ret.message();
}

TEST_F(DataReaderTest, TypeF32) {
    generate(1, 0, DataType::f32, 4);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    EXPECT_EQ(DataType::f32, r.type());
}

TEST_F(DataReaderTest, TypeF16) {

    generate(1, 0, DataType::f16, 4);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    EXPECT_EQ(DataType::f16, r.type());
}

TEST_F(DataReaderTest, TypeI16) {
    generate(1, 0, DataType::i16, 4);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    EXPECT_EQ(DataType::i16, r.type());
}

TEST_F(DataReaderTest, DimIsCorrect) {
    generate(1, 0, DataType::f32, 64);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    EXPECT_EQ(64u, r.dim());
}

TEST_F(DataReaderTest, CountIsCorrect) {
    generate(7, 0, DataType::f32, 4);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    EXPECT_EQ(7u, r.count());
}

TEST_F(DataReaderTest, GetNormThrowsWhenInlineNormsAreAbsent) {
    generate(2, 0, DataType::f32, 4);
    DataReader r;
    ASSERT_EQ(0, r.init(data_path_).code());
    ASSERT_FALSE(r.has_norms());

    EXPECT_THROW(static_cast<void>(r.get_norm(0)), std::logic_error);

    auto it = r.base_begin();
    ASSERT_FALSE(it.eof());
    EXPECT_THROW(static_cast<void>(it.get_norm()), std::logic_error);
}

TEST_F(DataReaderTest, ReadsCosineValuesFromInlineRecords) {
    GeneratorConfig cfg{PatternType::Sequential, 2, 0, DataType::f32, 4, 1000};
    generate_input_file(input_path_, cfg);
    DataWriter w;
    ASSERT_EQ(0, w.exec_for_testing(input_path_, data_path_, 0, 0, DistFunc::COS).code());

    DataReader r;
    ASSERT_EQ(0, r.init(data_path_).code());
    ASSERT_TRUE(r.has_norms());
    EXPECT_EQ(kDataFileHasCosineInvNorms, r.norm_flags());
    EXPECT_TRUE(r.has_matching_stored_norms(DistFunc::COS));
    EXPECT_FALSE(r.has_matching_stored_norms(DistFunc::L2));
    EXPECT_NEAR(1.0 / std::sqrt(4.0 * 0.1 * 0.1), static_cast<double>(r.get_norm(0)), 1e-6);
    EXPECT_NEAR(1.0 / std::sqrt(4.0 * 1.1 * 1.1), static_cast<double>(r.get_norm(1)), 1e-6);

    auto it = r.base_begin();
    ASSERT_FALSE(it.eof());
    EXPECT_NEAR(static_cast<double>(r.get_norm(0)), static_cast<double>(it.get_norm()), 1e-6);
    EXPECT_NEAR(static_cast<double>(r.get_norm(0)),
        static_cast<double>(r.get_norm_from_record_unchecked(it.data())), 1e-6);
}

TEST_F(DataReaderTest, ReadsL2NormValuesFromInlineRecords) {
    GeneratorConfig cfg{PatternType::Sequential, 2, 0, DataType::f32, 4, 1000};
    generate_input_file(input_path_, cfg);
    DataWriter w;
    ASSERT_EQ(0, w.exec_for_testing(input_path_, data_path_, 0, 0, DistFunc::L2).code());

    DataReader r;
    ASSERT_EQ(0, r.init(data_path_).code());
    ASSERT_TRUE(r.has_norms());
    EXPECT_EQ(kDataFileHasSquaredNorms, r.norm_flags());
    EXPECT_TRUE(r.has_matching_stored_norms(DistFunc::L2));
    EXPECT_FALSE(r.has_matching_stored_norms(DistFunc::COS));
    EXPECT_NEAR(4.0 * 0.1 * 0.1, static_cast<double>(r.get_norm(0)), 1e-6);
    EXPECT_NEAR(4.0 * 1.1 * 1.1, static_cast<double>(r.get_norm(1)), 1e-6);

    auto it = r.base_begin();
    ASSERT_FALSE(it.eof());
    EXPECT_NEAR(static_cast<double>(r.get_norm(0)), static_cast<double>(it.get_norm()), 1e-6);
}

TEST_F(DataReaderTest, WriteRawSupportsInlineNormsForI16) {
    const std::vector<int16_t> vec0 = {3, 3, 3, 3};
    const std::vector<int16_t> vec1 = {4, 4, 4, 4};
    const std::vector<std::vector<uint8_t>> vecs = {
        std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(vec0.data()),
            reinterpret_cast<const uint8_t*>(vec0.data()) + vec0.size() * sizeof(int16_t)),
        std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(vec1.data()),
            reinterpret_cast<const uint8_t*>(vec1.data()) + vec1.size() * sizeof(int16_t))
    };

    write_raw(DataType::i16, 4, 7, vecs, kDataFileHasSquaredNorms);

    DataReader r;
    ASSERT_EQ(0, r.init(data_path_).code());
    ASSERT_TRUE(r.has_norms());
    EXPECT_TRUE(r.has_matching_stored_norms(DistFunc::L2));
    EXPECT_FALSE(r.has_matching_stored_norms(DistFunc::COS));
    EXPECT_GT(r.stride(), r.size());
    EXPECT_FLOAT_EQ(36.0f, r.get_norm(0));
    EXPECT_FLOAT_EQ(64.0f, r.get_norm(1));
}

TEST_F(DataReaderTest, EmptyDataFileInitSucceeds) {
    DataFileHeader hdr = make_data_header(0, 0, 0, 0, DataType::f32, 4);
    CompactIds compact_ids;
    std::vector<uint64_t> ids;
    ASSERT_EQ(0, compact_ids.init(ids).code());
    CompactIds compact_deleted_ids;
    std::vector<uint64_t> deleted_ids;
    ASSERT_EQ(0, compact_deleted_ids.init(deleted_ids).code());
    ASSERT_EQ(0, set_data_header_layout(
        &hdr, compact_ids.serialized_size_bytes(), compact_deleted_ids.serialized_size_bytes()).code());
    FILE* f = fopen(data_path_.c_str(), "wb");
    ASSERT_NE(nullptr, f);
    ASSERT_EQ(1u, fwrite(&hdr, sizeof(hdr), 1, f));
    const size_t padding = static_cast<size_t>(hdr.data_offset) - sizeof(DataFileHeader);
    if (padding > 0) {
        std::vector<uint8_t> zeros(padding, 0);
        ASSERT_EQ(padding, fwrite(zeros.data(), 1, padding, f));
    }
    ASSERT_EQ(0, compact_ids.write(f, "DataReaderTest::EmptyDataFileInitSucceeds ids write failed").code());
    const DataMetadataLayout metadata_layout = compute_data_metadata_layout(hdr, 0);
    const size_t deleted_ids_padding =
        compute_deleted_ids_padding(metadata_layout.ids_trailer_offset, compact_ids.serialized_size_bytes());
    if (deleted_ids_padding > 0) {
        std::vector<uint8_t> zeros(deleted_ids_padding, 0);
        ASSERT_EQ(deleted_ids_padding, fwrite(zeros.data(), 1, deleted_ids_padding, f));
    }
    ASSERT_EQ(0, compact_deleted_ids.write(f, "DataReaderTest::EmptyDataFileInitSucceeds deleted ids write failed").code());
    fclose(f);

    DataReader r;
    ASSERT_EQ(0, r.init(data_path_).code());
    EXPECT_EQ(0u, r.count());
    EXPECT_TRUE(r.begin().eof());
    EXPECT_EQ(nullptr, r.get(123));
}

TEST_F(DataReaderTest, SizeF32IsCorrect) {
    generate(1, 0, DataType::f32, 4);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    EXPECT_EQ(4u * 4u, r.size()); // 4 dims * 4 bytes
}

TEST_F(DataReaderTest, SizeF16IsCorrect) {

    generate(1, 0, DataType::f16, 16);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    EXPECT_EQ(16u * 2u, r.size()); // 16 dims * 2 bytes
}

TEST_F(DataReaderTest, SizeI16IsCorrect) {
    generate(1, 0, DataType::i16, 8);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    EXPECT_EQ(8u * 2u, r.size()); // 8 dims * 2 bytes
}

// --- at() ---

TEST_F(DataReaderTest, AtReturnsNonNull) {
    generate(3, 0, DataType::f32, 4);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    EXPECT_NE(nullptr, r.at(0));
}

TEST_F(DataReaderTest, AtOutOfBoundsThrows) {
    generate(3, 0, DataType::f32, 4);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    EXPECT_THROW(r.at(3),   std::out_of_range);
    EXPECT_THROW(r.at(100), std::out_of_range);
}

TEST_F(DataReaderTest, AtReturnsNullForHiddenVector) {
    generate(3, 0, DataType::f32, 4);
    write_raw_to_data_file(
        delta_input_path_, delta_path_,
        "f32,4\n"
        "1 : []\n");

    DataReader r;
    ASSERT_EQ(0, r.init(data_path_, make_delta_reader()).code());
    EXPECT_NE(nullptr, r.at(0));
    EXPECT_EQ(nullptr, r.at(1));
    EXPECT_NE(nullptr, r.at(2));
}

TEST_F(DataReaderTest, GetNormThrowsForHiddenVector) {
    GeneratorConfig cfg{PatternType::Sequential, 3, 0, DataType::f32, 4, 1000};
    generate_input_file(input_path_, cfg);
    std::ofstream delta(delta_input_path_);
    ASSERT_TRUE(delta.is_open());
    delta << "f32,4\n"
          << "1 : [ 11.0, 11.0, 11.0, 11.0 ]\n";
    delta.close();

    DataWriter w;
    ASSERT_EQ(0, w.exec_for_testing(input_path_, data_path_, 0, 0, DistFunc::COS).code());
    ASSERT_EQ(0, w.exec_for_testing(delta_input_path_, delta_path_, 0, 0, DistFunc::COS).code());

    DataReader r;
    ASSERT_EQ(0, r.init(data_path_, make_delta_reader()).code());
    ASSERT_TRUE(r.has_norms());

    EXPECT_NEAR(1.0 / std::sqrt(4.0 * 0.1 * 0.1), static_cast<double>(r.get_norm(0)), 1e-6);
    EXPECT_THROW(static_cast<void>(r.get_norm(1)), std::logic_error);
    EXPECT_NEAR(1.0 / std::sqrt(4.0 * 2.1 * 2.1), static_cast<double>(r.get_norm(2)), 1e-6);
}

TEST_F(DataReaderTest, AtF32VectorDataIsCorrect) {
    const size_t count = 4, min_id = 10, dim = 4;
    generate(count, min_id, DataType::f32, dim);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    for (size_t i = 0; i < count; ++i) {
        const float* v = reinterpret_cast<const float*>(r.at(i));
        float expected = static_cast<float>(min_id + i) + 0.1f;
        for (size_t d = 0; d < dim; ++d)
            EXPECT_NEAR(expected, v[d], 1e-4f) << "vector " << i << " dim " << d;
    }
}

TEST_F(DataReaderTest, AtI16VectorDataIsCorrect) {
    const size_t count = 3, min_id = 5, dim = 4;
    generate(count, min_id, DataType::i16, dim);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    for (size_t i = 0; i < count; ++i) {
        const int16_t* v = reinterpret_cast<const int16_t*>(r.at(i));
        int16_t expected = static_cast<int16_t>(min_id + i);
        for (size_t d = 0; d < dim; ++d)
            EXPECT_EQ(expected, v[d]) << "vector " << i << " dim " << d;
    }
}

TEST_F(DataReaderTest, AtConsecutivePointersSpacedByStride) {
    generate(4, 0, DataType::f32, 8);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    for (size_t i = 0; i < 3; ++i) {
        ptrdiff_t gap = r.at(i + 1) - r.at(i);
        EXPECT_EQ(static_cast<ptrdiff_t>(r.stride()), gap) << "at index " << i;
    }
}

TEST_F(DataReaderTest, AtConsecutivePointersSpacedByStrideWithInlineNorms) {
    GeneratorConfig cfg{PatternType::Sequential, 4, 0, DataType::f32, 8, 1000};
    generate_input_file(input_path_, cfg);
    DataWriter w;
    ASSERT_EQ(0, w.exec_for_testing(input_path_, data_path_, 0, 0, DistFunc::COS).code());

    DataReader r;
    ASSERT_EQ(0, r.init(data_path_).code());
    ASSERT_GT(r.stride(), r.size());
    for (size_t i = 0; i < 3; ++i) {
        ptrdiff_t gap = r.at(i + 1) - r.at(i);
        EXPECT_EQ(static_cast<ptrdiff_t>(r.stride()), gap) << "at index " << i;
    }
}

// --- get() ---

TEST_F(DataReaderTest, GetReturnsNullForMissingId) {
    generate(3, 0, DataType::f32, 4);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    EXPECT_EQ(nullptr, r.get(999));
}

TEST_F(DataReaderTest, GetReturnsCorrectVector) {
    const size_t min_id = 5;
    generate(3, min_id, DataType::f32, 4);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    for (size_t i = 0; i < 3; ++i) {
        const uint8_t* via_get = r.get(min_id + i);
        ASSERT_NE(nullptr, via_get) << "id " << min_id + i << " not found";
        EXPECT_EQ(r.at(i), via_get) << "id " << min_id + i;
    }
}

TEST_F(DataReaderTest, GetFirstId) {
    const size_t min_id = 7;
    generate(3, min_id, DataType::f32, 4);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    EXPECT_EQ(r.at(0), r.get(min_id));
}

TEST_F(DataReaderTest, GetLastId) {
    const size_t count = 3, min_id = 7;
    generate(count, min_id, DataType::f32, 4);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    EXPECT_EQ(r.at(count - 1), r.get(min_id + count - 1));
}

// --- iterator ---

TEST_F(DataReaderTest, IteratorTraversesAllVectors) {
    const size_t count = 5;
    generate(count, 0, DataType::f32, 4);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    size_t seen = 0;
    for (auto it = r.begin(); !it.eof(); it.next())
        ++seen;
    EXPECT_EQ(count, seen);
}

TEST_F(DataReaderTest, IteratorIdsAreCorrect) {
    const size_t count = 4, min_id = 20;
    generate(count, min_id, DataType::f32, 4);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    size_t i = 0;
    for (auto it = r.begin(); !it.eof(); it.next(), ++i)
        EXPECT_EQ(min_id + i, it.id()) << "index " << i;
}

TEST_F(DataReaderTest, IteratorDataMatchesAt) {
    generate(3, 0, DataType::f32, 4);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    size_t i = 0;
    for (auto it = r.begin(); !it.eof(); it.next(), ++i)
        EXPECT_EQ(r.at(i), it.data()) << "index " << i;
}

TEST_F(DataReaderTest, IteratorSingleVector) {
    generate(1, 42, DataType::f32, 4);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    auto it = r.begin();
    ASSERT_FALSE(it.eof());
    EXPECT_EQ(42u, it.id());
    EXPECT_EQ(r.at(0), it.data());
    it.next();
    EXPECT_TRUE(it.eof());
}

TEST_F(DataReaderTest, IteratorF32ValuesCorrect) {
    const size_t count = 3, min_id = 2, dim = 4;
    generate(count, min_id, DataType::f32, dim);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    size_t i = 0;
    for (auto it = r.begin(); !it.eof(); it.next(), ++i) {
        const float* v = reinterpret_cast<const float*>(it.data());
        float expected = static_cast<float>(min_id + i) + 0.1f;
        for (size_t d = 0; d < dim; ++d)
            EXPECT_NEAR(expected, v[d], 1e-4f) << "vector " << i << " dim " << d;
    }
}

// --- delta-based modifications ---

TEST_F(DataReaderTest, DeltaSkipsDeletedInIterator) {
    generate(6, 0, DataType::f32, 4);
    generate_delta(6, 0, DataType::f32, 4, 2); // deleted ids: 2,4

    DataReader r;
    ASSERT_EQ(0, r.init(data_path_, make_delta_reader()).code());
    std::vector<uint64_t> seen_ids;
    for (auto it = r.begin(); !it.eof(); it.next())
        seen_ids.push_back(it.id());
    ASSERT_EQ(4u, seen_ids.size());
    EXPECT_EQ(0u, seen_ids[0]);
    EXPECT_EQ(1u, seen_ids[1]);
    EXPECT_EQ(3u, seen_ids[2]);
    EXPECT_EQ(5u, seen_ids[3]);
}

TEST_F(DataReaderTest, DeltaGetReturnsNullForDeleted) {
    generate(6, 0, DataType::f32, 4);
    generate_delta(6, 0, DataType::f32, 4, 2); // deleted ids: 2,4

    DataReader r;
    ASSERT_EQ(0, r.init(data_path_, make_delta_reader()).code());
    EXPECT_NE(nullptr, r.get(0));
    EXPECT_EQ(nullptr, r.get(2));
    EXPECT_NE(nullptr, r.get(3));
    EXPECT_EQ(nullptr, r.get(4));
}

TEST_F(DataReaderTest, NoDeltaIteratorSeesAll) {
    const size_t count = 4;
    generate(count, 0, DataType::f32, 4);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    size_t seen = 0;
    for (auto it = r.begin(); !it.eof(); it.next())
        ++seen;
    EXPECT_EQ(count, seen);
}

TEST_F(DataReaderTest, AllDeletedDeltaIteratorIsImmediatelyEof) {
    generate(3, 0, DataType::f32, 4);
    write_raw_to_data_file(
        delta_input_path_, delta_path_,
        "f32,4\n"
        "0 : []\n"
        "1 : []\n"
        "2 : []\n");

    DataReader r;
    ASSERT_EQ(0, r.init(data_path_, make_delta_reader()).code());
    EXPECT_TRUE(r.begin().eof());
}

TEST_F(DataReaderTest, DeltaFirstVectorDeletedIteratorStartsAtSecond) {
    generate(3, 0, DataType::f32, 4);
    write_raw_to_data_file(
        delta_input_path_, delta_path_,
        "f32,4\n"
        "0 : []\n");

    DataReader r;
    ASSERT_EQ(0, r.init(data_path_, make_delta_reader()).code());
    auto it = r.begin();
    ASSERT_FALSE(it.eof());
    EXPECT_EQ(1u, it.id());
}

TEST_F(DataReaderTest, DeltaLastVectorDeletedNotVisited) {
    generate(3, 0, DataType::f32, 4);
    write_raw_to_data_file(
        delta_input_path_, delta_path_,
        "f32,4\n"
        "2 : []\n");

    DataReader r;
    ASSERT_EQ(0, r.init(data_path_, make_delta_reader()).code());
    std::vector<uint64_t> seen_ids;
    for (auto it = r.begin(); !it.eof(); it.next())
        seen_ids.push_back(it.id());
    ASSERT_EQ(2u, seen_ids.size());
    EXPECT_EQ(0u, seen_ids[0]);
    EXPECT_EQ(1u, seen_ids[1]);
}

TEST_F(DataReaderTest, CountIncludesDeletedWhenDeltaApplied) {
    const size_t count = 4;
    generate(count, 0, DataType::f32, 4);
    write_raw_to_data_file(
        delta_input_path_, delta_path_,
        "f32,4\n"
        "1 : []\n"
        "2 : []\n");

    DataReader r;
    ASSERT_EQ(0, r.init(data_path_, make_delta_reader()).code());
    EXPECT_EQ(count, r.count());
}

TEST_F(DataReaderTest, DeltaIdsOutsideDataRangeAreIgnored) {
    const size_t count = 3;
    generate(count, 10, DataType::f32, 4); // ids: 10,11,12
    write_raw_to_data_file(
        delta_input_path_, delta_path_,
        "f32,4\n"
        "1 : []\n"
        "200 : []\n");

    DataReader r;
    ASSERT_EQ(0, r.init(data_path_, make_delta_reader()).code());
    size_t seen = 0;
    for (auto it = r.begin(); !it.eof(); it.next())
        ++seen;
    EXPECT_EQ(count, seen);
}

TEST_F(DataReaderTest, DeletedFromDeltaReturnsNull) {
    generate(3, 10, DataType::f32, 4);
    write_raw_to_data_file(
        delta_input_path_, delta_path_,
        "f32,4\n"
        "11 : []\n");

    DataReader r;
    ASSERT_EQ(0, r.init(data_path_, make_delta_reader()).code());
    EXPECT_NE(nullptr, r.get(10));
    EXPECT_EQ(nullptr, r.get(11));
    EXPECT_NE(nullptr, r.get(12));
}

TEST_F(DataReaderTest, UpdatedValueComesFromDelta) {
    generate(3, 10, DataType::i16, 4);
    generate_delta_detailed(1, 11, DataType::i16, 4);

    DataReader r;
    ASSERT_EQ(0, r.init(data_path_, make_delta_reader()).code());

    const int16_t* v10 = reinterpret_cast<const int16_t*>(r.get(10));
    const int16_t* v11 = reinterpret_cast<const int16_t*>(r.get(11));
    const int16_t* v12 = reinterpret_cast<const int16_t*>(r.get(12));
    ASSERT_NE(nullptr, v10);
    ASSERT_NE(nullptr, v11);
    ASSERT_NE(nullptr, v12);

    EXPECT_EQ(10, v10[0]); // untouched
    EXPECT_EQ(0, v11[0]);  // updated from detailed delta first vector
    EXPECT_EQ(12, v12[0]); // untouched
}

TEST_F(DataReaderTest, DeltaIteratorSkipsDeletedAndAppendsUpdatedFromDelta) {
    generate(4, 10, DataType::f32, 4);
    write_raw_to_data_file(
        delta_input_path_, delta_path_,
        "f32,4\n"
        "11 : []\n"
        "12 : [ 99.0, 99.0, 99.0, 99.0 ]\n");

    DataReader r;
    ASSERT_EQ(0, r.init(data_path_, make_delta_reader()).code());

    std::vector<uint64_t> seen_ids;
    for (auto it = r.begin(); !it.eof(); it.next()) {
        seen_ids.push_back(it.id());
    }
    ASSERT_EQ(3u, seen_ids.size());
    EXPECT_EQ(10u, seen_ids[0]);
    // New iterator contract: visible base rows first, then rows from attached delta.
    EXPECT_EQ(13u, seen_ids[1]);
    EXPECT_EQ(12u, seen_ids[2]);

    const float* v12 = reinterpret_cast<const float*>(r.get(12));
    ASSERT_NE(nullptr, v12);
    EXPECT_NEAR(99.0f, v12[0], 1e-5f);
}

TEST_F(DataReaderTest, BaseBeginReturnsVisibleBaseIdsInSortedOrder) {
    generate(4, 10, DataType::f32, 4);
    write_raw_to_data_file(
        delta_input_path_, delta_path_,
        "f32,4\n"
        "11 : []\n"
        "12 : [ 99.0, 99.0, 99.0, 99.0 ]\n");

    DataReader r;
    ASSERT_EQ(0, r.init(data_path_, make_delta_reader()).code());

    std::vector<uint64_t> seen_ids;
    for (auto it = r.base_begin(); !it.eof(); it.next()) {
        seen_ids.push_back(it.id());
    }

    EXPECT_EQ((std::vector<uint64_t> {10u, 13u}), seen_ids);
}

TEST_F(DataReaderTest, BaseBeginDataReturnsVisibleBaseRows) {
    generate(4, 10, DataType::f32, 4);
    write_raw_to_data_file(
        delta_input_path_, delta_path_,
        "f32,4\n"
        "11 : []\n"
        "12 : [ 99.0, 99.0, 99.0, 99.0 ]\n");

    DataReader r;
    ASSERT_EQ(0, r.init(data_path_, make_delta_reader()).code());

    auto it = r.base_begin();
    ASSERT_FALSE(it.eof());
    EXPECT_EQ(r.at(0), it.data());
    it.next();
    ASSERT_FALSE(it.eof());
    EXPECT_EQ(r.at(3), it.data());
    it.next();
    EXPECT_TRUE(it.eof());
}

TEST_F(DataReaderTest, DeltaBeginReturnsDeltaIdsInSortedOrder) {
    generate(4, 10, DataType::f32, 4);
    write_raw_to_data_file(
        delta_input_path_, delta_path_,
        "f32,4\n"
        "13 : [ 77.0, 77.0, 77.0, 77.0 ]\n"
        "11 : []\n"
        "12 : [ 99.0, 99.0, 99.0, 99.0 ]\n");

    DataReader r;
    ASSERT_EQ(0, r.init(data_path_, make_delta_reader()).code());

    std::vector<uint64_t> seen_ids;
    for (auto it = r.delta_begin(); !it.eof(); it.next()) {
        seen_ids.push_back(it.id());
    }

    EXPECT_EQ((std::vector<uint64_t> {12u, 13u}), seen_ids);
}

TEST_F(DataReaderTest, DeltaBeginIsEmptyWithoutAttachedDelta) {
    generate(3, 10, DataType::f32, 4);

    DataReader r;
    ASSERT_EQ(0, r.init(data_path_).code());
    EXPECT_TRUE(r.delta_begin().eof());
}

TEST_F(DataReaderTest, InitFailsWhenDeltaTypeMismatch) {
    generate(3, 0, DataType::f32, 4);
    generate_delta(3, 0, DataType::i16, 4);
    DataReader r;
    EXPECT_NE(0, r.init(data_path_, make_delta_reader()).code());
}

TEST_F(DataReaderTest, InitFailsWhenDeltaDimMismatch) {
    generate(3, 0, DataType::f32, 4);
    generate_delta(3, 0, DataType::f32, 8);
    DataReader r;
    EXPECT_NE(0, r.init(data_path_, make_delta_reader()).code());
}

// --- deleted-id section ---

TEST_F(DataReaderTest, DeletedCountAndDeletedIdsAreReadable) {
    generate(6, 0, DataType::f32, 4, 2); // deleted ids: 2,4
    DataReader r;
    ASSERT_EQ(0, r.init(data_path_).code());
    EXPECT_EQ(4u, r.count());
    EXPECT_EQ(2u, r.deleted_count());
    EXPECT_EQ(2u, r.deleted_id(0));
    EXPECT_EQ(4u, r.deleted_id(1));
}

TEST_F(DataReaderTest, DeletedIdOutOfBoundsThrows) {
    generate(6, 0, DataType::f32, 4, 2);
    DataReader r;
    ASSERT_EQ(0, r.init(data_path_).code());
    EXPECT_THROW(r.deleted_id(2), std::out_of_range);
    EXPECT_THROW(r.deleted_id(100), std::out_of_range);
}

TEST_F(DataReaderTest, GetReturnsNullForIdsInDeletedSection) {
    generate(6, 0, DataType::f32, 4, 2);
    DataReader r;
    ASSERT_EQ(0, r.init(data_path_).code());
    EXPECT_NE(nullptr, r.get(0));
    EXPECT_EQ(nullptr, r.get(2));
    EXPECT_NE(nullptr, r.get(3));
    EXPECT_EQ(nullptr, r.get(4));
    EXPECT_NE(nullptr, r.get(5));
}

TEST_F(DataReaderTest, CheckConsistencyReturnsTrueForValidFile) {
    generate(6, 0, DataType::f32, 4, 2);
    DataReader r;
    ASSERT_EQ(0, r.init(data_path_).code());
    EXPECT_TRUE(r.check_consistency());
}

TEST_F(DataReaderTest, CheckConsistencyReturnsFalseWhenIdsOverlapDeletedIds) {
    generate(6, 0, DataType::f32, 4, 2);

    FILE* f = fopen(data_path_.c_str(), "r+b");
    ASSERT_NE(nullptr, f);

    DataFileHeader hdr{};
    ASSERT_EQ(1u, fread(&hdr, sizeof(hdr), 1, f));
    const DataType type = data_type_from_int(hdr.type);
    const size_t vec_size = compute_vector_size(type, hdr.dim);
    EXPECT_LE(vec_size, static_cast<size_t>(hdr.vector_stride));
    const size_t ids_offset = compute_data_metadata_layout(hdr, hdr.count).ids_trailer_offset;
    fclose(f);

    const std::vector<uint8_t> bytes = read_file_bytes(data_path_);
    ASSERT_FALSE(bytes.empty());
    CompactIds active_ids;
    size_t active_ids_size = 0;
    ASSERT_EQ(0, active_ids.map(bytes.data() + ids_offset, bytes.size() - ids_offset, &active_ids_size).code());
    const size_t deleted_ids_offset = compute_deleted_ids_offset(ids_offset, active_ids_size);

    CompactIds deleted_ids;
    ASSERT_EQ(0, deleted_ids.map(bytes.data() + deleted_ids_offset, bytes.size() - deleted_ids_offset, nullptr).code());
    const size_t original_deleted_serialized_size = deleted_ids.serialized_size_bytes();
    std::vector<uint64_t> overlapping_deleted;
    overlapping_deleted.reserve(deleted_ids.count());
    for (size_t i = 0; i < deleted_ids.count(); ++i) {
        overlapping_deleted.push_back(deleted_ids.id(i) + 1u);
    }
    CompactIds overlapping_deleted_ids;
    ASSERT_EQ(0, overlapping_deleted_ids.init(overlapping_deleted).code());
    ASSERT_EQ(original_deleted_serialized_size, overlapping_deleted_ids.serialized_size_bytes());

    f = fopen(data_path_.c_str(), "r+b");
    ASSERT_NE(nullptr, f);
    ASSERT_EQ(0, fseek(f, deleted_ids_offset, SEEK_SET));
    ASSERT_EQ(0, overlapping_deleted_ids.write(f,
        "DataReaderTest::CheckConsistencyReturnsFalseWhenIdsOverlapDeletedIds write failed").code());
    fclose(f);

    DataReader r;
    ASSERT_EQ(0, r.init(data_path_).code());
    EXPECT_FALSE(r.check_consistency());
}

// --- double init ---

TEST_F(DataReaderTest, DoubleInitFails) {
    generate(3, 0, DataType::f32, 4);
    DataReader r;
    ASSERT_EQ(0, r.init(data_path_).code());
    EXPECT_NE(0, r.init(data_path_).code());
}
