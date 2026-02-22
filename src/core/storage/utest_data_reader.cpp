#include <gtest/gtest.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <unistd.h>
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

    void SetUp() override {
        std::string base = "/tmp/sketch2_utest_dr_" + std::to_string(getpid());
        input_path_ = base + ".txt";
        data_path_  = base + ".bin";
    }

    void TearDown() override {
        std::remove(input_path_.c_str());
        std::remove(data_path_.c_str());
    }

    void generate(size_t count, size_t min_id, DataType type, size_t dim) {
        GeneratorConfig cfg{PatternType::Sequential, count, min_id, type, dim};
        generate_input_file(input_path_, cfg);
        DataWriter w;
        w.init(input_path_, data_path_);
        w.exec();
    }

    // Write a minimal valid binary data file with raw vector bytes.
    // type_field: 0=f32, 1=f16, 2=i32  (matches DataWriter encoding)
    void write_raw(DataType type_field, uint16_t dim, uint64_t min_id,
                   const std::vector<std::vector<uint8_t>>& vecs) {
        DataFileHeader hdr{};
        hdr.magic   = kMagic;
        hdr.kind    = static_cast<uint16_t>(FileType::Data);
        hdr.version = kVersion;
        hdr.min_id  = min_id;
        hdr.max_id  = min_id + static_cast<uint64_t>(vecs.size()) - 1;
        hdr.count   = static_cast<uint32_t>(vecs.size());
        hdr.type    = data_type_to_int(type_field);
        hdr.dim     = dim;
        FILE* f = fopen(data_path_.c_str(), "wb");
        fwrite(&hdr, sizeof(hdr), 1, f);
        for (const auto& v : vecs)
            fwrite(v.data(), v.size(), 1, f);
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
    uint16_t delta = static_cast<uint16_t>(FileType::Delta);
    fwrite(&delta, sizeof(delta), 1, f);
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

// --- metadata after init ---

TEST_F(DataReaderTest, SuccessReturnCode) {
    generate(3, 0, DataType::f32, 4);
    DataReader r;
    const auto ret = r.init(data_path_);
    EXPECT_EQ(0, ret) << ret.message();
}

TEST_F(DataReaderTest, TypeF32) {
    generate(1, 0, DataType::f32, 4);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    EXPECT_EQ(DataType::f32, r.type());
}

TEST_F(DataReaderTest, TypeF16) {
    EXPECT_THROW(generate(1, 0, DataType::f16, 4), std::runtime_error);
}

TEST_F(DataReaderTest, TypeI32) {
    generate(1, 0, DataType::i32, 4);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    EXPECT_EQ(DataType::i32, r.type());
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
    EXPECT_THROW(generate(1, 0, DataType::f16, 8), std::runtime_error);
}

TEST_F(DataReaderTest, SizeI32IsCorrect) {
    generate(1, 0, DataType::i32, 8);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    EXPECT_EQ(8u * 4u, r.size()); // 8 dims * 4 bytes
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

TEST_F(DataReaderTest, AtI32VectorDataIsCorrect) {
    const size_t count = 3, min_id = 5, dim = 4;
    generate(count, min_id, DataType::i32, dim);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_).code());
    for (size_t i = 0; i < count; ++i) {
        const int32_t* v = reinterpret_cast<const int32_t*>(r.at(i));
        int32_t expected = static_cast<int32_t>(min_id + i);
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

// --- bitset / deletion (InPlace) ---

TEST_F(DataReaderTest, InPlaceBitsetSkipsDeletedInIterator) {
    const size_t count = 4;
    generate(count, 0, DataType::f32, 4);
    std::vector<bool> bitset(count, false);
    bitset[1] = true;
    bitset[3] = true;
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_, ReaderMode::InPlace, &bitset).code());
    std::vector<uint64_t> seen_ids;
    for (auto it = r.begin(); !it.eof(); it.next())
        seen_ids.push_back(it.id());
    ASSERT_EQ(2u, seen_ids.size());
    EXPECT_EQ(0u, seen_ids[0]);
    EXPECT_EQ(2u, seen_ids[1]);
}

TEST_F(DataReaderTest, InPlaceBitsetGetReturnsNullForDeleted) {
    const size_t count = 3;
    generate(count, 0, DataType::f32, 4);
    std::vector<bool> bitset(count, false);
    bitset[1] = true;
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_, ReaderMode::InPlace, &bitset).code());
    EXPECT_NE(nullptr, r.get(0));
    EXPECT_EQ(nullptr, r.get(1));
    EXPECT_NE(nullptr, r.get(2));
}

TEST_F(DataReaderTest, NoBitsetIteratorSeesAll) {
    const size_t count = 4;
    generate(count, 0, DataType::f32, 4);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_, ReaderMode::InPlace, nullptr).code());
    size_t seen = 0;
    for (auto it = r.begin(); !it.eof(); it.next())
        ++seen;
    EXPECT_EQ(count, seen);
}

TEST_F(DataReaderTest, AllDeletedIteratorIsImmediatelyEof) {
    const size_t count = 3;
    generate(count, 0, DataType::f32, 4);
    std::vector<bool> bitset(count, true);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_, ReaderMode::InPlace, &bitset).code());
    EXPECT_TRUE(r.begin().eof());
}

TEST_F(DataReaderTest, FirstVectorDeletedIteratorStartsAtSecond) {
    const size_t count = 3;
    generate(count, 0, DataType::f32, 4);
    std::vector<bool> bitset(count, false);
    bitset[0] = true;
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_, ReaderMode::InPlace, &bitset).code());
    auto it = r.begin();
    ASSERT_FALSE(it.eof());
    EXPECT_EQ(1u, it.id());
}

TEST_F(DataReaderTest, LastVectorDeletedNotVisited) {
    const size_t count = 3;
    generate(count, 0, DataType::f32, 4);
    std::vector<bool> bitset(count, false);
    bitset[count - 1] = true;
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_, ReaderMode::InPlace, &bitset).code());
    std::vector<uint64_t> seen_ids;
    for (auto it = r.begin(); !it.eof(); it.next())
        seen_ids.push_back(it.id());
    ASSERT_EQ(count - 1, seen_ids.size());
    EXPECT_EQ(count - 2, seen_ids.back());
}

TEST_F(DataReaderTest, CountIncludesDeleted) {
    const size_t count = 4;
    generate(count, 0, DataType::f32, 4);
    std::vector<bool> bitset(count, false);
    bitset[1] = true;
    bitset[2] = true;
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_, ReaderMode::InPlace, &bitset).code());
    EXPECT_EQ(count, r.count()); // total, including deleted
}

TEST_F(DataReaderTest, BitsetShorterThanCountNoCrash) {
    // Bitset covers only the first 2 vectors; remaining 3 have no bit → not deleted.
    const size_t count = 5;
    generate(count, 0, DataType::f32, 4);
    std::vector<bool> bitset(2, true);
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_, ReaderMode::InPlace, &bitset).code());
    size_t seen = 0;
    for (auto it = r.begin(); !it.eof(); it.next())
        ++seen;
    EXPECT_EQ(count - 2, seen);
}

// --- bitset / deletion (Reference) ---

TEST_F(DataReaderTest, ReferenceModeBitSetNullPointerIsDeleted) {
    // vec0: first 8 bytes = null → deleted when bitset bit is set
    // vec1: first 8 bytes = non-null → alive even when bitset bit is set
    const uint16_t dim = 4;
    const size_t vec_size = dim * 4; // f32: 4 bytes per element
    std::vector<uint8_t> vec0(vec_size, 0);
    std::vector<uint8_t> vec1(vec_size, 0);
    uint64_t nonzero = 1ULL;
    memcpy(vec1.data(), &nonzero, sizeof(nonzero));
    write_raw(DataType::f32, dim, 0, {vec0, vec1});

    std::vector<bool> bitset = {true, true};
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_, ReaderMode::Reference, &bitset).code());
    EXPECT_EQ(nullptr, r.get(0)); // null pointer → deleted
    EXPECT_NE(nullptr, r.get(1)); // non-null pointer → alive
}

TEST_F(DataReaderTest, ReferenceModeNoBitsetNotDeleted) {
    // Null pointer bytes in the vector, but bitset bit is not set → not deleted.
    const uint16_t dim = 4;
    const size_t vec_size = dim * 4;
    std::vector<uint8_t> vec0(vec_size, 0);
    write_raw(DataType::f32, dim, 0, {vec0});

    std::vector<bool> bitset = {false};
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_, ReaderMode::Reference, &bitset).code());
    EXPECT_NE(nullptr, r.get(0));
}

TEST_F(DataReaderTest, ReferenceModeIteratorSkipsNullPointerVectors) {
    const uint16_t dim = 4;
    const size_t vec_size = dim * 4;
    std::vector<uint8_t> vec0(vec_size, 0); // null → deleted
    std::vector<uint8_t> vec1(vec_size, 0);
    uint64_t nonzero = 1ULL;
    memcpy(vec1.data(), &nonzero, sizeof(nonzero));
    write_raw(DataType::f32, dim, 10, {vec0, vec1});

    std::vector<bool> bitset = {true, true};
    DataReader r;
    EXPECT_EQ(0, r.init(data_path_, ReaderMode::Reference, &bitset).code());

    std::vector<uint64_t> seen_ids;
    for (auto it = r.begin(); !it.eof(); it.next())
        seen_ids.push_back(it.id());
    ASSERT_EQ(1u, seen_ids.size());
    EXPECT_EQ(11u, seen_ids[0]);
}
