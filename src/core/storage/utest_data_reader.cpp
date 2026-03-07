#include <gtest/gtest.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <unistd.h>
#include <memory>
#include <vector>
#include <fstream>
#include "core/storage/data_file.h"
#include "core/storage/input_generator.h"
#include "core/storage/data_writer.h"
#include "core/storage/data_reader.h"

using namespace sketch2;

class DataReaderTest : public ::testing::Test {
protected:
    std::string input_path_;
    std::string data_path_;
    std::string delta_input_path_;
    std::string delta_path_;

    void SetUp() override {
        std::string base = "/tmp/sketch2_utest_dr_" + std::to_string(getpid());
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
        ret = w.init(in_path, out_path);
        ASSERT_EQ(0, ret.code()) << "DataWriter::init failed: " << ret.message();
        ret = w.exec();
        ASSERT_EQ(0, ret.code()) << "DataWriter::exec failed: " << ret.message();
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
        w.init(in_path, out_path);
        ASSERT_EQ(0, w.exec().code());
    }

    std::unique_ptr<DataReader> make_delta_reader() {
        auto delta = std::make_unique<DataReader>();
        EXPECT_EQ(0, delta->init(delta_path_).code());
        return delta;
    }

    // Write a minimal valid binary data file with raw vector bytes.
    // type_field: 0=f16, 1=f32, 2=i16  (matches DataWriter encoding)
    void write_raw(DataType type_field, uint16_t dim, uint64_t min_id,
                   const std::vector<std::vector<uint8_t>>& vecs) {
        DataFileHeader hdr{};
        hdr.base.magic   = kMagic;
        hdr.base.kind    = static_cast<uint16_t>(FileType::Data);
        hdr.base.version = kVersion;
        hdr.min_id  = min_id;
        hdr.max_id  = min_id + static_cast<uint64_t>(vecs.size()) - 1;
        hdr.count   = static_cast<uint32_t>(vecs.size());
        hdr.type    = data_type_to_int(type_field);
        hdr.dim     = dim;
        hdr.data_offset = static_cast<uint32_t>(
            ((sizeof(DataFileHeader) + kDataAlignment - 1) / kDataAlignment) * kDataAlignment);
        FILE* f = fopen(data_path_.c_str(), "wb");
        fwrite(&hdr, sizeof(hdr), 1, f);
        const size_t pad_size = static_cast<size_t>(hdr.data_offset) - sizeof(DataFileHeader);
        if (pad_size > 0) {
            std::vector<uint8_t> pad(pad_size, 0);
            fwrite(pad.data(), 1, pad.size(), f);
        }
        for (const auto& v : vecs)
            fwrite(v.data(), v.size(), 1, f);
        const size_t vectors_bytes = vecs.size() * static_cast<size_t>(dim) * data_type_size(type_field);
        const size_t ids_offset = align_up<size_t>(static_cast<size_t>(hdr.data_offset) + vectors_bytes, kIdsAlignment);
        const size_t ids_pad_size = ids_offset - (static_cast<size_t>(hdr.data_offset) + vectors_bytes);
        if (ids_pad_size > 0) {
            std::vector<uint8_t> pad(ids_pad_size, 0);
            fwrite(pad.data(), 1, pad.size(), f);
        }
        for (size_t i = 0; i < vecs.size(); ++i) {
            uint64_t id = min_id + static_cast<uint64_t>(i);
            fwrite(&id, sizeof(id), 1, f);
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
    if (!supports_f16()) {
        return;
    }
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

TEST_F(DataReaderTest, SizeF32IsCorrect) {
    generate(1, 0, DataType::f32, 4);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    EXPECT_EQ(4u * 4u, r.size()); // 4 dims * 4 bytes
}

TEST_F(DataReaderTest, SizeF16IsCorrect) {
    if (!supports_f16()) {
        return;
    }
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

TEST_F(DataReaderTest, AtConsecutivePointersSpacedBySize) {
    generate(4, 0, DataType::f32, 8);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    for (size_t i = 0; i < 3; ++i) {
        ptrdiff_t gap = r.at(i + 1) - r.at(i);
        EXPECT_EQ(static_cast<ptrdiff_t>(r.size()), gap) << "at index " << i;
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
    const size_t vec_size = static_cast<size_t>(hdr.dim) * data_type_size(type);
    const size_t vectors_bytes = static_cast<size_t>(hdr.count) * vec_size;
    const size_t ids_offset = align_up<size_t>(static_cast<size_t>(hdr.data_offset) + vectors_bytes, kIdsAlignment);
    const size_t deleted_ids_offset = ids_offset + static_cast<size_t>(hdr.count) * sizeof(uint64_t);

    ASSERT_EQ(0, fseek(f, static_cast<long>(deleted_ids_offset), SEEK_SET));
    const uint64_t overlapped_id = 3; // 3 is in ids_ for this generated input
    ASSERT_EQ(1u, fwrite(&overlapped_id, sizeof(overlapped_id), 1, f));
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
