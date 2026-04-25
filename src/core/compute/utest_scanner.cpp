// Unit tests for Scanner nearest-neighbor scanning.

#include <gtest/gtest.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <fstream>
#include <memory>
#include <filesystem>
#include <experimental/scope>
#include "core/compute/scanner.h"
#include "core/compute/scanner_heap_utils.h"
#include "core/compute/scanner_query_context.h"
#include "core/compute/scanner_scan_loops.h"
#include "core/utils/singleton.h"
#include "core/utils/thread_pool.h"
#include "core/utils/chunked_bits.h"
#include "core/storage/input_generator.h"
#include "core/storage/data_writer.h"
#include "core/storage/data_reader.h"
#include "core/storage/dataset_node.h"
#include "utest_tmp_dir.h"

using namespace sketch2;
namespace fs = std::filesystem;

namespace {

size_t g_counting_dot_calls = 0;

Scanner make_compiled_scanner() {
    return Scanner();
}

Ret find_ids(const Scanner& scanner, const DatasetReader& dataset, size_t count, const uint8_t* vec,
             std::vector<uint64_t>& result, const BitsetFilter* bitset = nullptr) {
    result.clear();
    std::vector<DistItem> items;
    Ret ret = scanner.find_items(dataset, count, vec, items, bitset);
    if (ret.code() != 0) {
        return ret;
    }
    extract_ids_from_items(items, &result);
    return ret;
}

struct ScannerBitsetFilter {
    ScannerBitsetFilter() {
        filter.view = &view;
    }
    ScannerBitsetFilter(const ScannerBitsetFilter&) = delete;
    ScannerBitsetFilter& operator=(const ScannerBitsetFilter&) = delete;
    ~ScannerBitsetFilter() {
        std::free(data);
    }

    void* data = nullptr;
    size_t size = 0;
    size_t allocation_size = 0;
    ChunkedBitsView view;
    BitsetFilter filter;
};

size_t align_up_for_chunked_bits(size_t value) {
    const size_t mask = kChunkedBitsBlobAlignment - 1u;
    return (value + mask) & ~mask;
}

void build_allowed_ids_filter(const std::vector<uint64_t>& ids, ScannerBitsetFilter* out) {
    ASSERT_NE(nullptr, out);
    ASSERT_EQ(nullptr, out->data);

    ChunkedBits bits;
    for (uint64_t id : ids) {
        ASSERT_EQ(0, bits.add(id).code());
    }
    ASSERT_EQ(0, bits.finish().code());

    out->size = bits.serialized_size_bytes();
    out->allocation_size = align_up_for_chunked_bits(out->size);
    out->data = std::aligned_alloc(kChunkedBitsBlobAlignment, out->allocation_size);
    ASSERT_NE(nullptr, out->data);
    ASSERT_EQ(0, bits.serialize(out->data, out->size).code());
    ASSERT_EQ(0, out->view.init_blob(out->data, out->size).code());
    out->filter.view = &out->view;
}

void extract_absolute_reader_heap_items(const DataReader& reader, LocalDistHeap* heap,
        std::vector<DistItem>* result) {
    std::vector<LocalDistItem> local_items;
    extract_local_items(heap, &local_items);
    result->clear();
    result->reserve(local_items.size());
    const uint64_t heap_base_id = reader_heap_base_id(reader);
    for (const LocalDistItem& item : local_items) {
        result->push_back(DistItem{item.id + heap_base_id, item.score});
    }
}

double counting_f32_dot(const uint8_t* a, const uint8_t* b, size_t dim) {
    ++g_counting_dot_calls;
    const auto* aa = reinterpret_cast<const float*>(a);
    const auto* bb = reinterpret_cast<const float*>(b);
    double dot = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        dot += static_cast<double>(aa[i]) * static_cast<double>(bb[i]);
    }
    return dot;
}

}

class ScannerTest : public ::testing::Test {
protected:
    std::string input_path_;
    std::string data_path_;
    std::string delta_input_path_;
    std::string delta_path_;
    std::vector<std::string> cleanup_dirs_;
    std::vector<std::string> cleanup_files_;

    void SetUp() override {
        std::string base = tmp_dir() + "/sketch2_utest_scanner_ex_" + std::to_string(getpid());
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
        for (const std::string& path : cleanup_files_) {
            std::remove(path.c_str());
        }
        for (const std::string& path : cleanup_dirs_) {
            fs::remove_all(path);
        }
    }

    void generate_file(const std::string& in_path, const std::string& out_path, const GeneratorConfig& cfg) {
        generate_input_file(in_path, cfg);
        DataWriter w;
        ASSERT_EQ(0, w.exec_for_testing(in_path, out_path, 0).code());
    }

    void generate(size_t count, size_t min_id, DataType type, size_t dim) {
        GeneratorConfig cfg{PatternType::Sequential, count, min_id, type, dim, 1000};
        generate_file(input_path_, data_path_, cfg);
    }

    void generate_delta(size_t count, size_t min_id, DataType type, size_t dim, size_t every_n_deleted = 0) {
        GeneratorConfig cfg{PatternType::Sequential, count, min_id, type, dim, 1000, every_n_deleted};
        generate_file(delta_input_path_, delta_path_, cfg);
    }

    void write_delta_raw(const std::string& content) {
        std::ofstream f(delta_input_path_);
        f << content;
        f.close();
        DataWriter w;
        ASSERT_EQ(0, w.exec_for_testing(delta_input_path_, delta_path_, 0).code());
    }

    void write_input_raw(const std::string& path, const std::string& content) {
        std::ofstream f(path);
        f << content;
        f.close();
    }

    std::unique_ptr<DatasetReader> make_dataset_reader(
            DataType type,
            uint64_t dim,
            DistFunc func,
            const std::vector<std::string>& store_inputs,
            uint64_t range_size = 1000) {
        const std::string dataset_dir = data_path_ + ".dataset_" + std::to_string(cleanup_dirs_.size());
        const std::string config_path = data_path_ + ".dataset_" + std::to_string(cleanup_dirs_.size()) + ".ini";
        cleanup_dirs_.push_back(dataset_dir);
        cleanup_files_.push_back(config_path);
        fs::create_directories(dataset_dir);

        DatasetNode ds;
        EXPECT_EQ(0, ds.init_for_test({dataset_dir}, range_size, type, dim, func).code());
        for (const std::string& input : store_inputs) {
            EXPECT_EQ(0, ds.store(input).code());
        }

        write_input_raw(
            config_path,
            std::string("[dataset]\n") +
            "dirs = " + dataset_dir + "\n"
            "range_size = " + std::to_string(range_size) + "\n"
            "type = " + data_type_to_string(type) + "\n"
            "dist_func = " + dist_func_to_string(func) + "\n"
            "dim = " + std::to_string(dim) + "\n");

        auto reader = std::make_unique<DatasetReader>();
        EXPECT_EQ(0, reader->init(config_path).code());
        return reader;
    }

    std::vector<uint8_t> f32_vec(float val, size_t dim) {
        std::vector<uint8_t> buf(dim * sizeof(float));
        auto* p = reinterpret_cast<float*>(buf.data());
        for (size_t i = 0; i < dim; ++i) p[i] = val;
        return buf;
    }

    std::vector<uint8_t> f32_values(std::initializer_list<float> values) {
        std::vector<uint8_t> buf(values.size() * sizeof(float));
        auto* p = reinterpret_cast<float*>(buf.data());
        size_t i = 0;
        for (float v : values) {
            p[i++] = v;
        }
        return buf;
    }

    std::vector<uint8_t> i16_vec(int16_t val, size_t dim) {
        std::vector<uint8_t> buf(dim * sizeof(int16_t));
        auto* p = reinterpret_cast<int16_t*>(buf.data());
        for (size_t i = 0; i < dim; ++i) p[i] = val;
        return buf;
    }

    std::vector<uint8_t> f16_vec(float val, size_t dim) {
        std::vector<uint8_t> buf(dim * sizeof(uint16_t));
        auto* p = reinterpret_cast<uint16_t*>(buf.data());
        for (size_t i = 0; i < dim; ++i) p[i] = float_to_f16(val);
        return buf;
    }

    static uint16_t float_to_f16(float f) {
        uint32_t x;
        memcpy(&x, &f, sizeof(x));
        uint16_t sign     = static_cast<uint16_t>((x >> 16) & 0x8000);
        int      exp      = static_cast<int>((x >> 23) & 0xFF) - 127 + 15;
        uint32_t mantissa = x & 0x7FFFFFu;
        if (exp <= 0)  return sign;
        if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
        return static_cast<uint16_t>(sign | (exp << 10) | (mantissa >> 13));
    }

    void overwrite_stored_norm(const std::string& path, size_t index, float value) {
        const int fd = open(path.c_str(), O_RDWR);
        ASSERT_GE(fd, 0);

        DataFileHeader hdr{};
        ASSERT_EQ(static_cast<ssize_t>(sizeof(hdr)), pread(fd, &hdr, sizeof(hdr), 0));
        ASSERT_TRUE(data_file_has_norms(hdr));
        ASSERT_LT(index, static_cast<size_t>(hdr.count));
        const DataRecordLayout record_layout = compute_data_record_layout(
            data_type_from_int(static_cast<int>(hdr.type)), hdr.dim, true);
        ASSERT_GE(static_cast<size_t>(hdr.vector_stride), record_layout.norm_offset + sizeof(value));

        const size_t record_offset =
            static_cast<size_t>(hdr.data_offset) + index * static_cast<size_t>(hdr.vector_stride);
        const off_t norm_offset =
            static_cast<off_t>(record_offset + record_layout.norm_offset);
        ASSERT_EQ(static_cast<ssize_t>(sizeof(value)),
                  pwrite(fd, &value, sizeof(value), norm_offset));
        ASSERT_EQ(0, close(fd));
    }
};

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

TEST_F(ScannerTest, FindFailsOnCountZero) {
    generate(3, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_});
    Scanner s;
    auto q = f32_vec(0.0f, 4);
    std::vector<uint64_t> result;
    EXPECT_NE(0, find_ids(s, *reader, 0, q.data(), result).code());
}

TEST_F(ScannerTest, FindFailsOnNullQueryPointer) {
    generate(3, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_});
    Scanner s;
    std::vector<uint64_t> result;
    EXPECT_NE(0, find_ids(s, *reader, 1, nullptr, result).code());
}

TEST_F(ScannerTest, FindFailsOnUnknownFunction) {
    DatasetReader reader;
    Scanner s = make_compiled_scanner();
    auto q = f32_vec(0.0f, 4);
    std::vector<uint64_t> result;
    EXPECT_NE(0, find_ids(s, reader, 1, q.data(), result).code());
}

TEST_F(ScannerTest, FindClearsReusedResultBufferOnFailure) {
    generate(3, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_});
    Scanner s;
    auto q = f32_vec(0.0f, 4);
    std::vector<uint64_t> result;

    ASSERT_EQ(0, find_ids(s, *reader, 2, q.data(), result).code());
    ASSERT_FALSE(result.empty());

    EXPECT_NE(0, find_ids(s, *reader, 0, q.data(), result).code());
    EXPECT_TRUE(result.empty());
}

TEST_F(ScannerTest, FindItemsClearsReusedResultBufferOnFailure) {
    generate(3, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_});
    Scanner s;
    auto q = f32_vec(0.0f, 4);
    std::vector<DistItem> result;

    ASSERT_EQ(0, s.find_items(*reader, 2, q.data(), result).code());
    ASSERT_FALSE(result.empty());

    EXPECT_NE(0, s.find_items(*reader, 1, nullptr, result).code());
    EXPECT_TRUE(result.empty());
}

// ---------------------------------------------------------------------------
// DOT metric
// ---------------------------------------------------------------------------

TEST_F(ScannerTest, FindF32DOTK3ReturnsInOrder) {
    generate(5, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_});
    Scanner s;
    auto q = f32_vec(3.2f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, *reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(4u, result[0]);
    EXPECT_EQ(3u, result[1]);
    EXPECT_EQ(2u, result[2]);
}

TEST_F(ScannerTest, FindF32DOTK3ReturnsInOrderWithHighway) {
    generate(5, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_});
    Scanner s = make_compiled_scanner();
    auto q = f32_vec(3.2f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, *reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(4u, result[0]);
    EXPECT_EQ(3u, result[1]);
    EXPECT_EQ(2u, result[2]);
}

TEST_F(ScannerTest, FindCountExceedsTotalReturnsCapped) {
    const size_t total = 3;
    generate(total, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_});
    Scanner s;
    auto q = f32_vec(0.0f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, *reader, 100, q.data(), result).code());
    EXPECT_EQ(total, result.size());
}

TEST_F(ScannerTest, FindResultSizeMatchesRequest) {
    generate(5, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_});
    Scanner s;
    auto q = f32_vec(3.2f, 4);
    std::vector<uint64_t> result;

    ASSERT_EQ(0, find_ids(s, *reader, 1, q.data(), result).code());
    EXPECT_EQ(1u, result.size());

    ASSERT_EQ(0, find_ids(s, *reader, 3, q.data(), result).code());
    EXPECT_EQ(3u, result.size());

    ASSERT_EQ(0, find_ids(s, *reader, 5, q.data(), result).code());
    EXPECT_EQ(5u, result.size());
}

TEST_F(ScannerTest, FindItemsF32DOTReturnsIdsAndDistances) {
    generate(5, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_});
    Scanner s;
    auto q = f32_vec(3.2f, 4);
    std::vector<DistItem> result;
    ASSERT_EQ(0, s.find_items(*reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(4u, result[0].id);
    EXPECT_EQ(3u, result[1].id);
    EXPECT_EQ(2u, result[2].id);
    EXPECT_NEAR(52.48, result[0].score, 1e-4);
    EXPECT_NEAR(39.68, result[1].score, 1e-4);
    EXPECT_NEAR(26.88, result[2].score, 1e-4);
}

// ---------------------------------------------------------------------------
// L2 metric
// ---------------------------------------------------------------------------

TEST_F(ScannerTest, FindF32L2K3ReturnsInOrder) {
    generate(5, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::L2, {input_path_});
    Scanner s;
    auto q = f32_vec(3.2f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, *reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(3u, result[0]);
    EXPECT_EQ(4u, result[1]);
    EXPECT_EQ(2u, result[2]);
}

TEST_F(ScannerTest, FindF32L2K3ReturnsInOrderWithHighway) {
    generate(5, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::L2, {input_path_});
    Scanner s = make_compiled_scanner();
    auto q = f32_vec(3.2f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, *reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(3u, result[0]);
    EXPECT_EQ(4u, result[1]);
    EXPECT_EQ(2u, result[2]);
}

// ---------------------------------------------------------------------------
// Cosine metric
// ---------------------------------------------------------------------------

TEST_F(ScannerTest, FindF32CosK3ReturnsInOrder) {
    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 100.0, 1.0, 0.0, 0.0 ]\n"
        "20 : [ 1.0, 1.0, 0.0, 0.0 ]\n"
        "30 : [ -1.0, 0.0, 0.0, 0.0 ]\n");
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::COS, {input_path_});
    Scanner s;
    auto q = f32_values({1.0f, 0.0f, 0.0f, 0.0f});
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, *reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(10u, result[0]);
    EXPECT_EQ(20u, result[1]);
    EXPECT_EQ(30u, result[2]);
}

TEST_F(ScannerTest, FindF32CosK3ReturnsInOrderWithHighway) {
    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 100.0, 1.0, 0.0, 0.0 ]\n"
        "20 : [ 1.0, 1.0, 0.0, 0.0 ]\n"
        "30 : [ -1.0, 0.0, 0.0, 0.0 ]\n");
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::COS, {input_path_});
    Scanner s = make_compiled_scanner();
    auto q = f32_values({1.0f, 0.0f, 0.0f, 0.0f});
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, *reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(10u, result[0]);
    EXPECT_EQ(20u, result[1]);
    EXPECT_EQ(30u, result[2]);
}

TEST_F(ScannerTest, FindF32CosStoredCosineValuesHandleZeroVectors) {
    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 0.0, 0.0, 0.0, 0.0 ]\n"
        "20 : [ 1.0, 0.0, 0.0, 0.0 ]\n");
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::COS, {input_path_});

    Scanner s = make_compiled_scanner();
    auto q = f32_values({0.0f, 0.0f, 0.0f, 0.0f});
    std::vector<DistItem> result;
    ASSERT_EQ(0, s.find_items(*reader, 2, q.data(), result).code());
    ASSERT_EQ(2u, result.size());
    EXPECT_EQ(10u, result[0].id);
    EXPECT_DOUBLE_EQ(0.0, result[0].score);
    EXPECT_EQ(20u, result[1].id);
    EXPECT_DOUBLE_EQ(1.0, result[1].score);
}

TEST_F(ScannerTest, FindF32CosStoredPathsMatchRanking) {
    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 10.0, 0.0, 0.0, 0.0 ]\n"
        "20 : [ 2.0, 1.0, 0.0, 0.0 ]\n"
        "30 : [ 1.0, 2.0, 0.0, 0.0 ]\n"
        "40 : [ 0.0, 1.0, 0.0, 0.0 ]\n"
        "50 : [ -1.0, 0.0, 0.0, 0.0 ]\n");

    auto reader_a = make_dataset_reader(DataType::f32, 4, DistFunc::COS, {input_path_});
    auto reader_b = make_dataset_reader(DataType::f32, 4, DistFunc::COS, {input_path_});

    Scanner s = make_compiled_scanner();
    auto q = f32_values({1.0f, 0.0f, 0.0f, 0.0f});
    std::vector<uint64_t> result_a;
    std::vector<uint64_t> result_b;
    ASSERT_EQ(0, find_ids(s, *reader_a, 5, q.data(), result_a).code());
    ASSERT_EQ(0, find_ids(s, *reader_b, 5, q.data(), result_b).code());

    ASSERT_EQ((std::vector<uint64_t> {10u, 20u, 30u, 40u, 50u}), result_a);
    EXPECT_EQ(result_a, result_b);
}

// ---------------------------------------------------------------------------
// Other data types
// ---------------------------------------------------------------------------

TEST_F(ScannerTest, FindI16AllSortedByDistance) {
    generate(3, 0, DataType::i16, 4);
    auto reader = make_dataset_reader(DataType::i16, 4, DistFunc::DOT, {input_path_});
    Scanner s;
    auto q = i16_vec(0, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, *reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(0u, result[0]);
    EXPECT_EQ(1u, result[1]);
    EXPECT_EQ(2u, result[2]);
}

TEST_F(ScannerTest, FindF16Works) {
    generate(3, 0, DataType::f16, 4);
    auto reader = make_dataset_reader(DataType::f16, 4, DistFunc::DOT, {input_path_});
    Scanner s;
    auto q = f16_vec(1.1f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, *reader, 1, q.data(), result).code());
    ASSERT_EQ(1u, result.size());
    EXPECT_EQ(2u, result[0]);
}

TEST_F(ScannerTest, FindF16WorksWithHighway) {
    generate(3, 0, DataType::f16, 4);
    auto reader = make_dataset_reader(DataType::f16, 4, DistFunc::DOT, {input_path_});
    Scanner s = make_compiled_scanner();
    auto q = f16_vec(1.1f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, *reader, 1, q.data(), result).code());
    ASSERT_EQ(1u, result.size());
    EXPECT_EQ(2u, result[0]);
}

TEST_F(ScannerTest, FindF16CosWorksWithHighway) {
    write_input_raw(
        input_path_,
        "f16,4\n"
        "10 : [ 100.0, 1.0, 0.0, 0.0 ]\n"
        "20 : [ 1.0, 1.0, 0.0, 0.0 ]\n"
        "30 : [ -1.0, 0.0, 0.0, 0.0 ]\n");
    auto reader = make_dataset_reader(DataType::f16, 4, DistFunc::COS, {input_path_});
    Scanner s = make_compiled_scanner();
    auto q = f16_vec(0.0f, 4);
    reinterpret_cast<uint16_t*>(q.data())[0] = float_to_f16(1.0f);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, *reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(10u, result[0]);
    EXPECT_EQ(20u, result[1]);
    EXPECT_EQ(30u, result[2]);
}

// ---------------------------------------------------------------------------
// Bitset filter tests
// ---------------------------------------------------------------------------

TEST_F(ScannerTest, BitsetFilterWalksSparseAllowedIds) {
    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 10.0, 10.0, 10.0, 10.0 ]\n"
        "20 : [ 20.0, 20.0, 20.0, 20.0 ]\n"
        "30 : [ 30.0, 30.0, 30.0, 30.0 ]\n"
        "40 : [ 40.0, 40.0, 40.0, 40.0 ]\n");
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_}, 100);

    ScannerBitsetFilter allowed;
    ASSERT_NO_FATAL_FAILURE(build_allowed_ids_filter({5, 30, 999}, &allowed));

    Scanner s = make_compiled_scanner();
    auto q = f32_vec(1.0f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, *reader, 3, q.data(), result, &allowed.filter).code());
    ASSERT_EQ(1u, result.size());
    EXPECT_EQ(30u, result[0]);
}

TEST_F(ScannerTest, BitsetFilterWalksAllowedIdsAcrossChunks) {
    const uint64_t id0 = 7;
    const uint64_t id1 = kChunkSize + 11;
    const uint64_t id2 = 2 * kChunkSize + 13;
    write_input_raw(
        input_path_,
        std::string("f32,4\n") +
        std::to_string(id0) + " : [ 1.0, 1.0, 1.0, 1.0 ]\n" +
        std::to_string(kChunkSize / 2) + " : [ 50.0, 50.0, 50.0, 50.0 ]\n" +
        std::to_string(id1) + " : [ 2.0, 2.0, 2.0, 2.0 ]\n" +
        std::to_string(id2) + " : [ 3.0, 3.0, 3.0, 3.0 ]\n");
    auto reader = make_dataset_reader(
        DataType::f32, 4, DistFunc::DOT, {input_path_}, 3 * kChunkSize);

    ScannerBitsetFilter allowed;
    ASSERT_NO_FATAL_FAILURE(build_allowed_ids_filter({id0, id1, id2}, &allowed));

    Scanner s = make_compiled_scanner();
    auto q = f32_vec(1.0f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, *reader, 3, q.data(), result, &allowed.filter).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(id2, result[0]);
    EXPECT_EQ(id1, result[1]);
    EXPECT_EQ(id0, result[2]);
}

TEST_F(ScannerTest, BitsetFilterAllowsDenseReaderWithoutDuplicates) {
    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 1.0, 1.0, 1.0, 1.0 ]\n"
        "20 : [ 2.0, 2.0, 2.0, 2.0 ]\n"
        "30 : [ 3.0, 3.0, 3.0, 3.0 ]\n"
        "40 : [ 4.0, 4.0, 4.0, 4.0 ]\n");
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_}, 100);

    ScannerBitsetFilter allowed;
    ASSERT_NO_FATAL_FAILURE(build_allowed_ids_filter({10, 20, 30, 40}, &allowed));

    Scanner s = make_compiled_scanner();
    auto q = f32_vec(1.0f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, *reader, 4, q.data(), result, &allowed.filter).code());
    ASSERT_EQ(4u, result.size());
    EXPECT_EQ(40u, result[0]);
    EXPECT_EQ(30u, result[1]);
    EXPECT_EQ(20u, result[2]);
    EXPECT_EQ(10u, result[3]);
}

// ---------------------------------------------------------------------------
// Delta tests
// ---------------------------------------------------------------------------

TEST_F(ScannerTest, DeltaSkipsDeletedIds) {
    generate(6, 0, DataType::f32, 4);
    generate_delta(6, 0, DataType::f32, 4, 2);

    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_, delta_input_path_});

    Scanner s;
    auto q = f32_vec(3.2f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, *reader, 6, q.data(), result).code());

    for (uint64_t id : result) {
        EXPECT_NE(2u, id);
        EXPECT_NE(4u, id);
    }
}

TEST_F(ScannerTest, DeltaUsesUpdatedVectors) {
    generate(4, 10, DataType::f32, 4);
    write_delta_raw(
        "f32,4\n"
        "11 : [ 20.0, 20.0, 20.0, 20.0 ]\n");

    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_, delta_input_path_});

    Scanner s;
    auto q = f32_vec(20.0f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, *reader, 1, q.data(), result).code());
    ASSERT_EQ(1u, result.size());
    EXPECT_EQ(11u, result[0]);
}

TEST_F(ScannerTest, DeltaDeletingAllVectorsReturnsEmptyResult) {
    generate(3, 0, DataType::f32, 4);
    write_delta_raw(
        "f32,4\n"
        "0 : []\n"
        "1 : []\n"
        "2 : []\n");

    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_, delta_input_path_});

    Scanner s = make_compiled_scanner();
    auto q = f32_vec(1.1f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, *reader, 3, q.data(), result).code());
    EXPECT_TRUE(result.empty());
}

TEST_F(ScannerTest, BitsetFilterRestartsForDeltaRows) {
    write_input_raw(
        input_path_,
        "f32,4\n"
        "0 : [ 0.0, 0.0, 0.0, 0.0 ]\n"
        "1 : [ 1.0, 1.0, 1.0, 1.0 ]\n"
        "2 : [ 2.0, 2.0, 2.0, 2.0 ]\n"
        "3 : [ 3.0, 3.0, 3.0, 3.0 ]\n"
        "4 : [ 4.0, 4.0, 4.0, 4.0 ]\n"
        "5 : [ 5.0, 5.0, 5.0, 5.0 ]\n");
    write_delta_raw(
        "f32,4\n"
        "2 : []\n"
        "4 : [ 100.0, 100.0, 100.0, 100.0 ]\n");
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT,
        {input_path_, delta_input_path_});

    ScannerBitsetFilter allowed;
    ASSERT_NO_FATAL_FAILURE(build_allowed_ids_filter({2, 4, 5}, &allowed));

    Scanner s = make_compiled_scanner();
    auto q = f32_vec(1.0f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, *reader, 3, q.data(), result, &allowed.filter).code());
    ASSERT_EQ(2u, result.size());
    EXPECT_EQ(4u, result[0]);
    EXPECT_EQ(5u, result[1]);
}

TEST_F(ScannerTest, BitsetFilterSkipsHiddenBaseRowTarget) {
    write_input_raw(
        input_path_,
        "f32,4\n"
        "0 : [ 0.0, 0.0, 0.0, 0.0 ]\n"
        "1 : [ 1.0, 1.0, 1.0, 1.0 ]\n"
        "2 : [ 100.0, 100.0, 100.0, 100.0 ]\n"
        "3 : [ 3.0, 3.0, 3.0, 3.0 ]\n"
        "4 : [ 4.0, 4.0, 4.0, 4.0 ]\n");
    write_delta_raw(
        "f32,4\n"
        "2 : []\n");
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT,
        {input_path_, delta_input_path_});

    ScannerBitsetFilter allowed;
    ASSERT_NO_FATAL_FAILURE(build_allowed_ids_filter({2, 3}, &allowed));

    Scanner s = make_compiled_scanner();
    auto q = f32_vec(1.0f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, *reader, 2, q.data(), result, &allowed.filter).code());
    ASSERT_EQ(1u, result.size());
    EXPECT_EQ(3u, result[0]);
}

// ---------------------------------------------------------------------------
// Multi-file dataset
// ---------------------------------------------------------------------------

TEST_F(ScannerTest, FindDatasetWorks) {
    generate_input_file(input_path_, GeneratorConfig{PatternType::Sequential, 30, 0, DataType::f32, 4, 1000});
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_}, 10);

    Scanner s;
    auto q = f32_vec(15.2f, 4);
    std::vector<uint64_t> result;
    const auto ret = find_ids(s, *reader, 3, q.data(), result);
    ASSERT_EQ(0, ret.code()) << "\n\nfind failed: " << ret.message() << "\n\n";
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(29u, result[0]);
    EXPECT_EQ(28u, result[1]);
    EXPECT_EQ(27u, result[2]);
}

TEST_F(ScannerTest, FindDatasetItemsReturnsIdsAndDistancesInOrder) {
    std::string d = tmp_dir() + "/sketch2_utest_scanner_ex_dsitems_" + std::to_string(getpid());
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d); });

    auto input = d + "/input.txt";
    generate_input_file(input, GeneratorConfig{PatternType::Sequential, 30, 0, DataType::f32, 4, 1000});
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input}, 100);

    Scanner s = make_compiled_scanner();
    auto q = f32_vec(15.2f, 4);
    std::vector<DistItem> result;
    ASSERT_EQ(0, s.find_items(*reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(29u, result[0].id);
    EXPECT_EQ(28u, result[1].id);
    EXPECT_EQ(27u, result[2].id);
    EXPECT_NEAR(1769.28, result[0].score, 1e-2);
    EXPECT_NEAR(1708.48, result[1].score, 1e-2);
    EXPECT_NEAR(1647.68, result[2].score, 1e-2);
}

TEST_F(ScannerTest, FindDatasetL2Works) {
    generate_input_file(input_path_, GeneratorConfig{PatternType::Sequential, 30, 0, DataType::f32, 4, 1000});
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::L2, {input_path_}, 10);

    Scanner s = make_compiled_scanner();
    auto q = f32_vec(15.2f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, *reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(15u, result[0]);
    EXPECT_EQ(16u, result[1]);
    EXPECT_EQ(14u, result[2]);
}

TEST_F(ScannerTest, FindDatasetL2UsesStoredSquaredNormsForCompiledEngine) {
    const std::string dataset_dir =
        tmp_dir() + "/sketch2_utest_scanner_ex_l2_stored_norms_" + std::to_string(getpid());
    const std::string config_path = dataset_dir + ".ini";
    fs::create_directories(dataset_dir);
    std::experimental::scope_exit cleanup([&]() {
        fs::remove_all(dataset_dir);
        std::remove(config_path.c_str());
    });

    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 1.0, 0.0, 0.0, 0.0 ]\n"
        "20 : [ 0.0, 1.0, 0.0, 0.0 ]\n");

    DatasetNode ds;
    ASSERT_EQ(0, ds.init_for_test({dataset_dir}, 100, DataType::f32, 4, DistFunc::L2).code());
    ASSERT_EQ(0, ds.store(input_path_).code());

    overwrite_stored_norm(dataset_dir + "/0.data", 0, 100.0f);

    write_input_raw(
        config_path,
        std::string("[dataset]\n") +
        "dirs = " + dataset_dir + "\n"
        "range_size = 100\n"
        "type = f32\n"
        "dist_func = l2\n"
        "dim = 4\n");

    DatasetReader reader;
    ASSERT_EQ(0, reader.init(config_path).code());

    const auto q = f32_values({1.0f, 0.0f, 0.0f, 0.0f});
    Scanner s = make_compiled_scanner();
    std::vector<DistItem> result;
    ASSERT_EQ(0, s.find_items(reader, 2, q.data(), result).code());
    ASSERT_EQ(2u, result.size());
    EXPECT_EQ(20u, result[0].id);
    EXPECT_NEAR(2.0, result[0].score, 1e-6);
    EXPECT_EQ(10u, result[1].id);
    EXPECT_NEAR(99.0, result[1].score, 1e-6);
}

TEST_F(ScannerTest, L2StoredNormScanSkipsDotWhenNormLowerBoundCannotBeatHeap) {
    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 1.0, 0.0, 0.0, 0.0 ]\n"
        "20 : [ 10.0, 0.0, 0.0, 0.0 ]\n");

    const std::string dataset_dir = data_path_ + ".dataset_" + std::to_string(cleanup_dirs_.size());
    cleanup_dirs_.push_back(dataset_dir);
    fs::create_directories(dataset_dir);

    DatasetNode ds;
    ASSERT_EQ(0, ds.init_for_test({dataset_dir}, 100, DataType::f32, 4, DistFunc::L2).code());
    ASSERT_EQ(0, ds.store(input_path_).code());

    DataReader reader;
    ASSERT_EQ(0, reader.init(dataset_dir + "/0.data").code());

    const auto q = f32_values({1.0f, 0.0f, 0.0f, 0.0f});
    const QueryL2Context query{q.data(), 4, 1.0};
    LocalDistHeap heap(LocalDistItemCompare{DistFunc::L2});
    heap.reserve(1);

    g_counting_dot_calls = 0;
    scan_l2_stored_norms<counting_f32_dot>(reader, 1, &heap, query);

    std::vector<DistItem> result;
    extract_absolute_reader_heap_items(reader, &heap, &result);

    ASSERT_EQ(1u, result.size());
    EXPECT_EQ(10u, result[0].id);
    EXPECT_DOUBLE_EQ(0.0, result[0].score);
    EXPECT_EQ(1u, g_counting_dot_calls);
}

TEST_F(ScannerTest, L2StoredNormScanRefreshesCachedBoundsWhenHeapThresholdTightens) {
    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 10.0, 0.0, 0.0, 0.0 ]\n"
        "20 : [ 1.0, 0.0, 0.0, 0.0 ]\n"
        "30 : [ 2.0, 0.0, 0.0, 0.0 ]\n"
        "40 : [ 3.0, 0.0, 0.0, 0.0 ]\n");

    const std::string dataset_dir = data_path_ + ".dataset_" + std::to_string(cleanup_dirs_.size());
    cleanup_dirs_.push_back(dataset_dir);
    fs::create_directories(dataset_dir);

    DatasetNode ds;
    ASSERT_EQ(0, ds.init_for_test({dataset_dir}, 100, DataType::f32, 4, DistFunc::L2).code());
    ASSERT_EQ(0, ds.store(input_path_).code());

    DataReader reader;
    ASSERT_EQ(0, reader.init(dataset_dir + "/0.data").code());

    const auto q = f32_values({0.0f, 0.0f, 0.0f, 0.0f});
    const QueryL2Context query{q.data(), 4, 0.0};
    LocalDistHeap heap(LocalDistItemCompare{DistFunc::L2});
    heap.reserve(2);

    g_counting_dot_calls = 0;
    scan_l2_stored_norms<counting_f32_dot>(reader, 2, &heap, query);

    std::vector<DistItem> result;
    extract_absolute_reader_heap_items(reader, &heap, &result);

    ASSERT_EQ(2u, result.size());
    EXPECT_EQ(20u, result[0].id);
    EXPECT_DOUBLE_EQ(1.0, result[0].score);
    EXPECT_EQ(30u, result[1].id);
    EXPECT_DOUBLE_EQ(4.0, result[1].score);
    EXPECT_EQ(3u, g_counting_dot_calls);
}

TEST_F(ScannerTest, CosStoredNormScanSkipsDotForZeroStoredVector) {
    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 0.0, 0.0, 0.0, 0.0 ]\n"
        "20 : [ 1.0, 0.0, 0.0, 0.0 ]\n");

    const std::string dataset_dir = data_path_ + ".dataset_" + std::to_string(cleanup_dirs_.size());
    cleanup_dirs_.push_back(dataset_dir);
    fs::create_directories(dataset_dir);

    DatasetNode ds;
    ASSERT_EQ(0, ds.init_for_test({dataset_dir}, 100, DataType::f32, 4, DistFunc::COS).code());
    ASSERT_EQ(0, ds.store(input_path_).code());

    DataReader reader;
    ASSERT_EQ(0, reader.init(dataset_dir + "/0.data").code());

    const auto q = f32_values({1.0f, 0.0f, 0.0f, 0.0f});
    const double query_norm_sq = 1.0;
    const QueryCosContext query{q.data(), 4, query_norm_sq, query_inverse_norm(query_norm_sq)};
    LocalDistHeap heap(LocalDistItemCompare{DistFunc::COS});
    heap.reserve(2);

    g_counting_dot_calls = 0;
    scan_cos_stored_norms<counting_f32_dot>(reader, 2, &heap, query);

    std::vector<DistItem> result;
    extract_absolute_reader_heap_items(reader, &heap, &result);

    ASSERT_EQ(2u, result.size());
    EXPECT_EQ(20u, result[0].id);
    EXPECT_DOUBLE_EQ(0.0, result[0].score);
    EXPECT_EQ(10u, result[1].id);
    EXPECT_DOUBLE_EQ(1.0, result[1].score);
    EXPECT_EQ(1u, g_counting_dot_calls);
}

TEST_F(ScannerTest, CosStoredNormScanTreatsBothZeroVectorsAsExactMatchWithoutDot) {
    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 0.0, 0.0, 0.0, 0.0 ]\n"
        "20 : [ 1.0, 0.0, 0.0, 0.0 ]\n");

    const std::string dataset_dir = data_path_ + ".dataset_" + std::to_string(cleanup_dirs_.size());
    cleanup_dirs_.push_back(dataset_dir);
    fs::create_directories(dataset_dir);

    DatasetNode ds;
    ASSERT_EQ(0, ds.init_for_test({dataset_dir}, 100, DataType::f32, 4, DistFunc::COS).code());
    ASSERT_EQ(0, ds.store(input_path_).code());

    DataReader reader;
    ASSERT_EQ(0, reader.init(dataset_dir + "/0.data").code());

    const auto q = f32_values({0.0f, 0.0f, 0.0f, 0.0f});
    const double query_norm_sq = 0.0;
    const QueryCosContext query{q.data(), 4, query_norm_sq, query_inverse_norm(query_norm_sq)};
    LocalDistHeap heap(LocalDistItemCompare{DistFunc::COS});
    heap.reserve(2);

    g_counting_dot_calls = 0;
    scan_cos_stored_norms<counting_f32_dot>(reader, 2, &heap, query);

    std::vector<DistItem> result;
    extract_absolute_reader_heap_items(reader, &heap, &result);

    ASSERT_EQ(2u, result.size());
    EXPECT_EQ(10u, result[0].id);
    EXPECT_DOUBLE_EQ(0.0, result[0].score);
    EXPECT_EQ(20u, result[1].id);
    EXPECT_DOUBLE_EQ(1.0, result[1].score);
    EXPECT_EQ(1u, g_counting_dot_calls);
}

TEST_F(ScannerTest, FindDatasetCosWorks) {
    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 100.0, 1.0, 0.0, 0.0 ]\n"
        "20 : [ 1.0, 1.0, 0.0, 0.0 ]\n"
        "30 : [ -1.0, 0.0, 0.0, 0.0 ]\n");
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::COS, {input_path_}, 100);

    Scanner s = make_compiled_scanner();
    auto q = f32_values({1.0f, 0.0f, 0.0f, 0.0f});
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, *reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(10u, result[0]);
    EXPECT_EQ(20u, result[1]);
    EXPECT_EQ(30u, result[2]);
}

TEST_F(ScannerTest, FindDatasetCosRejectsFilesMissingStoredInverseNorms) {
    std::string d = tmp_dir() + "/sketch2_utest_scanner_ex_cos_missing_inv_" + std::to_string(getpid());
    std::string cfg = d + ".ini";
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() {
        fs::remove_all(d);
        std::remove(cfg.c_str());
    });

    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 100.0, 1.0, 0.0, 0.0 ]\n"
        "20 : [ 1.0, 1.0, 0.0, 0.0 ]\n"
        "30 : [ -1.0, 0.0, 0.0, 0.0 ]\n");
    DataWriter writer;
    ASSERT_EQ(0, writer.exec_for_testing(input_path_, d + "/0.data", 0).code());

    DatasetNode ds;
    ASSERT_EQ(0, ds.init_for_test({d}, 100, DataType::f32, 4, DistFunc::COS).code());

    write_input_raw(
        cfg,
        std::string("[dataset]\n") +
        "dirs = " + d + "\n"
        "range_size = 100\n"
        "type = f32\n"
        "dist_func = cos\n"
        "dim = 4\n");

    DatasetReader reader;
    ASSERT_EQ(0, reader.init(cfg).code());

    Scanner s = make_compiled_scanner();
    auto q = f32_values({1.0f, 0.0f, 0.0f, 0.0f});
    std::vector<uint64_t> result;
    const Ret ret = find_ids(s, reader, 3, q.data(), result);
    EXPECT_NE(0, ret.code());
    EXPECT_TRUE(result.empty());
    EXPECT_NE(std::string(ret.message()).find("missing stored norms"), std::string::npos);
}

TEST_F(ScannerTest, FindDatasetL2RejectsFilesMissingStoredNorms) {
    std::string d = tmp_dir() + "/sketch2_utest_scanner_ex_l2_missing_norms_" + std::to_string(getpid());
    std::string cfg = d + ".ini";
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() {
        fs::remove_all(d);
        std::remove(cfg.c_str());
    });

    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 100.0, 1.0, 0.0, 0.0 ]\n"
        "20 : [ 1.0, 1.0, 0.0, 0.0 ]\n"
        "30 : [ -1.0, 0.0, 0.0, 0.0 ]\n");
    DataWriter writer;
    ASSERT_EQ(0, writer.exec_for_testing(input_path_, d + "/0.data", 0).code());

    DatasetNode ds;
    ASSERT_EQ(0, ds.init_for_test({d}, 100, DataType::f32, 4, DistFunc::L2).code());

    write_input_raw(
        cfg,
        std::string("[dataset]\n") +
        "dirs = " + d + "\n"
        "range_size = 100\n"
        "type = f32\n"
        "dist_func = l2\n"
        "dim = 4\n");

    DatasetReader reader;
    ASSERT_EQ(0, reader.init(cfg).code());

    Scanner s = make_compiled_scanner();
    auto q = f32_values({1.0f, 0.0f, 0.0f, 0.0f});
    std::vector<uint64_t> result;
    const Ret ret = find_ids(s, reader, 3, q.data(), result);
    EXPECT_NE(0, ret.code());
    EXPECT_TRUE(result.empty());
    EXPECT_NE(std::string(ret.message()).find("missing stored norms"), std::string::npos);
}

TEST_F(ScannerTest, FindDatasetFailsOnNullQueryPointer) {
    generate(3, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_}, 100);
    Scanner s = make_compiled_scanner();
    std::vector<uint64_t> result;
    EXPECT_NE(0, find_ids(s, *reader, 1, nullptr, result).code());
}

TEST_F(ScannerTest, FindDatasetFailsOnZeroCount) {
    generate(3, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_}, 100);
    Scanner s = make_compiled_scanner();
    auto q = f32_vec(1.0f, 4);
    std::vector<uint64_t> result;
    EXPECT_NE(0, find_ids(s, *reader, 0, q.data(), result).code());
}

TEST_F(ScannerTest, FindDatasetSkipsDeletedVectorsFromDelta) {
    std::string d = tmp_dir() + "/sketch2_utest_scanner_ex_dsdel_" + std::to_string(getpid());
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d); });

    DatasetNode ds;
    ASSERT_EQ(0, ds.init_for_test({d}, 100, DataType::f32, 4).code());

    generate_input_file(input_path_,
        GeneratorConfig{PatternType::Sequential, 5, 0, DataType::f32, 4, 1000});
    ASSERT_EQ(0, ds.store(input_path_).code());

    {
        std::ofstream f(input_path_);
        f << "f32,4\n2 : []\n";
    }
    ASSERT_EQ(0, ds.store(input_path_).code());
    ASSERT_TRUE(fs::exists(d + "/0.delta")) << "expected a delta file to exist";

    Scanner s = make_compiled_scanner();
    auto q = f32_vec(2.1f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, ds, 5, q.data(), result).code());

    EXPECT_EQ(4u, result.size());
    for (uint64_t id : result) {
        EXPECT_NE(2u, id) << "deleted id=2 must not appear in results";
    }
}

TEST_F(ScannerTest, FindDatasetUsesUpdatedVectorFromDelta) {
    std::string d = tmp_dir() + "/sketch2_utest_scanner_ex_dsupd_" + std::to_string(getpid());
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d); });

    DatasetNode ds;
    ASSERT_EQ(0, ds.init_for_test({d}, 100, DataType::f32, 4).code());

    generate_input_file(input_path_,
        GeneratorConfig{PatternType::Sequential, 5, 0, DataType::f32, 4, 1000});
    ASSERT_EQ(0, ds.store(input_path_).code());

    {
        std::ofstream f(input_path_);
        f << "f32,4\n1 : [ 500.0, 500.0, 500.0, 500.0 ]\n";
    }
    ASSERT_EQ(0, ds.store(input_path_).code());
    ASSERT_TRUE(fs::exists(d + "/0.delta")) << "expected a delta file to exist";

    Scanner s = make_compiled_scanner();
    auto q = f32_vec(0.0f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, ds, 1, q.data(), result).code());
    ASSERT_EQ(1u, result.size());
    EXPECT_EQ(0u, result[0]);
}

TEST_F(ScannerTest, FindDatasetHandlesDeltaIdBelowBaseMinId) {
    std::string d = tmp_dir() + "/sketch2_utest_scanner_ex_delta_low_id_" + std::to_string(getpid());
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d); });

    DatasetNode ds;
    ASSERT_EQ(0, ds.init_for_test({d}, 100, DataType::f32, 4, DistFunc::L2).code());

    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 10.0, 10.0, 10.0, 10.0 ]\n"
        "11 : [ 11.0, 11.0, 11.0, 11.0 ]\n"
        "12 : [ 12.0, 12.0, 12.0, 12.0 ]\n");
    ASSERT_EQ(0, ds.store(input_path_).code());

    write_input_raw(
        input_path_,
        "f32,4\n"
        "1 : [ 0.0, 0.0, 0.0, 0.0 ]\n");
    ASSERT_EQ(0, ds.store(input_path_).code());
    ASSERT_TRUE(fs::exists(d + "/0.delta")) << "expected a delta file to exist";

    Scanner s = make_compiled_scanner();
    auto q = f32_vec(0.0f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, ds, 1, q.data(), result).code());
    ASSERT_EQ(1u, result.size());
    EXPECT_EQ(1u, result[0]);
}

TEST_F(ScannerTest, BitsetFilterHandlesDeltaIdBelowBaseMinId) {
    std::string d = tmp_dir() + "/sketch2_utest_scanner_ex_bitset_delta_low_id_" +
        std::to_string(getpid());
    fs::create_directories(d);
    std::experimental::scope_exit cleanup([&]() { fs::remove_all(d); });

    DatasetNode ds;
    ASSERT_EQ(0, ds.init_for_test({d}, 100, DataType::f32, 4, DistFunc::L2).code());

    write_input_raw(
        input_path_,
        "f32,4\n"
        "10 : [ 10.0, 10.0, 10.0, 10.0 ]\n"
        "11 : [ 11.0, 11.0, 11.0, 11.0 ]\n"
        "12 : [ 12.0, 12.0, 12.0, 12.0 ]\n");
    ASSERT_EQ(0, ds.store(input_path_).code());

    write_input_raw(
        input_path_,
        "f32,4\n"
        "1 : [ 0.0, 0.0, 0.0, 0.0 ]\n");
    ASSERT_EQ(0, ds.store(input_path_).code());
    ASSERT_TRUE(fs::exists(d + "/0.delta")) << "expected a delta file to exist";

    ScannerBitsetFilter allowed;
    ASSERT_NO_FATAL_FAILURE(build_allowed_ids_filter({1}, &allowed));

    Scanner s = make_compiled_scanner();
    auto q = f32_vec(0.0f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, ds, 1, q.data(), result, &allowed.filter).code());
    ASSERT_EQ(1u, result.size());
    EXPECT_EQ(1u, result[0]);
}

// ---------------------------------------------------------------------------
// Concurrent scan tests
// ---------------------------------------------------------------------------

class ScannerConcurrentTest : public ScannerTest {
protected:
    void SetUp() override {
        ScannerTest::SetUp();
        prior_pool_ = get_singleton().thread_pool();
        Singleton::force_thread_pool_for_testing(4);
    }

    void TearDown() override {
        Singleton::force_thread_pool_for_testing(prior_pool_);
        ScannerTest::TearDown();
    }

private:
    std::shared_ptr<ThreadPool> prior_pool_;
};

TEST_F(ScannerConcurrentTest, DOTTopKSpansMultipleReaders) {
    generate(30, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_}, 10);

    Scanner s = make_compiled_scanner();
    auto q = f32_vec(9.5f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, *reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(29u, result[0]);
    EXPECT_EQ(28u, result[1]);
    EXPECT_EQ(27u, result[2]);
}

TEST_F(ScannerConcurrentTest, L2TopKSpansMultipleReaders) {
    generate(30, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::L2, {input_path_}, 10);

    Scanner s = make_compiled_scanner();
    auto q = f32_vec(9.5f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, *reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(9u, result[0]);
    EXPECT_EQ(10u, result[1]);
    EXPECT_EQ(8u, result[2]);
}

TEST_F(ScannerConcurrentTest, FindItemsSpansMultipleReaders) {
    generate(30, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_}, 10);

    Scanner s = make_compiled_scanner();
    auto q = f32_vec(15.2f, 4);
    std::vector<DistItem> result;
    ASSERT_EQ(0, s.find_items(*reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(29u, result[0].id);
    EXPECT_EQ(28u, result[1].id);
    EXPECT_EQ(27u, result[2].id);
    EXPECT_NEAR(1769.28, result[0].score, 1e-2);
    EXPECT_NEAR(1708.48, result[1].score, 1e-2);
    EXPECT_NEAR(1647.68, result[2].score, 1e-2);
}

TEST_F(ScannerConcurrentTest, SingleReaderWithPoolFallsBackToSequential) {
    generate(5, 0, DataType::f32, 4);
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::DOT, {input_path_}, 1000);

    Scanner s = make_compiled_scanner();
    auto q = f32_vec(2.2f, 4);
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, *reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(4u, result[0]);
    EXPECT_EQ(3u, result[1]);
    EXPECT_EQ(2u, result[2]);
}

TEST_F(ScannerConcurrentTest, CosineTopKSpansMultipleReaders) {
    write_input_raw(
        input_path_,
        "f32,4\n"
        "5  : [ 100.0, 1.0, 0.0, 0.0 ]\n"
        "15 : [ 1.0, 1.0, 0.0, 0.0 ]\n"
        "25 : [ -1.0, 0.0, 0.0, 0.0 ]\n");
    auto reader = make_dataset_reader(DataType::f32, 4, DistFunc::COS, {input_path_}, 10);

    Scanner s = make_compiled_scanner();
    auto q = f32_values({1.0f, 0.0f, 0.0f, 0.0f});
    std::vector<uint64_t> result;
    ASSERT_EQ(0, find_ids(s, *reader, 3, q.data(), result).code());
    ASSERT_EQ(3u, result.size());
    EXPECT_EQ(5u, result[0]);
    EXPECT_EQ(15u, result[1]);
    EXPECT_EQ(25u, result[2]);
}
