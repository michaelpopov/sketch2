// End-to-end tests covering the full storage write/read/merge flow.

#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <unistd.h>
#include <vector>
#include "core/storage/input_generator.h"
#include "core/storage/data_file.h"
#include "core/storage/data_file_layout.h"
#include "core/storage/dataset_node.h"
#include "core/storage/data_reader.h"
#include "core/utils/utest_float8_helpers.h"
#include "utest_tmp_dir.h"

using namespace sketch2;
using sketch2::test::f8_ordinal_bytes;
using sketch2::test::reference_f8_squared_norm;
namespace fs = std::filesystem;

class DatasetFullCycleTest : public ::testing::Test {
protected:
    std::string base_dir_;
    std::string input_path_;

    void SetUp() override {
        base_dir_ = tmp_dir() + "/sketch2_utest_full_cycle_" + std::to_string(getpid());
        input_path_ = base_dir_ + "/input.txt";
        fs::create_directories(base_dir_);
    }

    void TearDown() override {
        fs::remove_all(base_dir_);
    }

    std::string make_dir(const std::string& name) {
        const std::string path = base_dir_ + "/" + name;
        fs::create_directories(path);
        return path;
    }

    GeneratorConfig seq_cfg(size_t count, size_t min_id, DataType type, size_t dim, size_t every_n_deleted = 0) {
        return {PatternType::Sequential, count, min_id, type, dim, 1000, every_n_deleted};
    }

    GeneratorConfig detailed_cfg(size_t count, size_t min_id, DataType type, size_t dim, size_t max_val = 1000, size_t every_n_deleted = 0) {
        return {PatternType::Detailed, count, min_id, type, dim, max_val, every_n_deleted};
    }

    void write_manual(const ManualInputGenerator& gen) {
        const Ret ret = generate_input_file(input_path_, gen);
        ASSERT_EQ(0, ret.code()) << ret.message();
    }

    void write_generated(const GeneratorConfig& cfg) {
        const Ret ret = generate_input_file(input_path_, cfg);
        ASSERT_EQ(0, ret.code()) << ret.message();
    }

    size_t visible_count(const DataReader& reader) {
        size_t n = 0;
        for (auto it = reader.begin(); !it.eof(); it.next()) {
            ++n;
        }
        return n;
    }

    DataFileHeader read_header(const std::string& data_path) {
        FILE* f = fopen(data_path.c_str(), "rb");
        EXPECT_NE(nullptr, f);
        DataFileHeader hdr{};
        if (f == nullptr) {
            return hdr;
        }
        EXPECT_EQ(1u, fread(&hdr, sizeof(hdr), 1, f));
        fclose(f);
        return hdr;
    }
};

TEST_F(DatasetFullCycleTest, SequentialSingleRangeRoundTripThroughDatasetRangeReader) {
    const std::string dir = make_dir("d");
    DatasetNode ds;
    ASSERT_EQ(0, ds.init_for_test({dir}, 100, DataType::f32, 4).code());

    write_generated(seq_cfg(10, 100, DataType::f32, 4));
    ASSERT_EQ(0, ds.store(input_path_).code());

    DatasetRangeReaderPtr drs = ds.reader();
    ASSERT_NE(nullptr, drs);

    auto [r, ret] = drs->next();
    ASSERT_EQ(0, ret.code());
    ASSERT_NE(nullptr, r);
    EXPECT_EQ(10u, r->count());
    EXPECT_EQ(100u, r->id(0));
    EXPECT_EQ(109u, r->id(9));

    const float* v = reinterpret_cast<const float*>(r->get(103));
    ASSERT_NE(nullptr, v);
    EXPECT_NEAR(103.1f, v[0], 1e-4f);
    EXPECT_NEAR(103.1f, v[3], 1e-4f);

    EXPECT_EQ(nullptr, drs->next().first);
}

TEST_F(DatasetFullCycleTest, MultiDirMultiRangeReaderOrderAndCoverage) {
    const std::string d0 = make_dir("d0");
    const std::string d1 = make_dir("d1");
    DatasetNode ds;
    ASSERT_EQ(0, ds.init_for_test({d0, d1}, 10, DataType::f32, 4).code());

    write_generated(seq_cfg(30, 0, DataType::f32, 4));
    ASSERT_EQ(0, ds.store(input_path_).code());

    DatasetRangeReaderPtr drs = ds.reader();
    ASSERT_NE(nullptr, drs);

    auto [r0, ret0] = drs->next();
    auto [r1, ret1] = drs->next();
    auto [r2, ret2] = drs->next();
    ASSERT_EQ(0, ret0.code());
    ASSERT_EQ(0, ret1.code());
    ASSERT_EQ(0, ret2.code());
    ASSERT_NE(nullptr, r0);
    ASSERT_NE(nullptr, r1);
    ASSERT_NE(nullptr, r2);
    EXPECT_EQ(nullptr, drs->next().first);

    EXPECT_EQ(0u, r0->id(0));
    EXPECT_EQ(10u, r1->id(0));
    EXPECT_EQ(20u, r2->id(0));
    EXPECT_EQ(10u, r0->count());
    EXPECT_EQ(10u, r1->count());
    EXPECT_EQ(10u, r2->count());
}

TEST_F(DatasetFullCycleTest, OverrideAndDeleteAreAppliedByDatasetRangeReaderWithDelta) {
    const std::string dir = make_dir("d");
    DatasetNode ds;
    ASSERT_EQ(0, ds.init_for_test({dir}, 100, DataType::f32, 4).code());

    // Base data in 0.data: ids 0..19 with values id+0.1
    write_generated(seq_cfg(20, 0, DataType::f32, 4));
    ASSERT_EQ(0, ds.store(input_path_).code());

    // Small update in same range => stored in 0.delta.
    // Detailed pattern makes values independent from id, so we can detect override.
    write_generated(detailed_cfg(2, 5, DataType::f32, 4, 10));
    ASSERT_EQ(0, ds.store(input_path_).code());

    // Delete one existing id in delta.
    ManualInputGenerator gen;
    gen.type = DataType::f32;
    gen.dim = 4;
    gen.deleted(7);
    write_manual(gen);
    ASSERT_EQ(0, ds.store(input_path_).code());

    DatasetRangeReaderPtr drs = ds.reader();
    auto [r, ret] = drs->next();
    ASSERT_EQ(0, ret.code());
    ASSERT_NE(nullptr, r);
    EXPECT_EQ(nullptr, drs->next().first);

    const float* v5 = reinterpret_cast<const float*>(r->get(5));
    ASSERT_NE(nullptr, v5);
    EXPECT_NEAR(0.0f, v5[0], 1e-4f); // overridden from ~5.1 to detailed value
    EXPECT_NEAR(0.0f, v5[3], 1e-4f);

    const float* v6 = reinterpret_cast<const float*>(r->get(6));
    ASSERT_NE(nullptr, v6);
    EXPECT_NEAR(0.01f, v6[0], 1e-4f);
    EXPECT_NEAR(0.0f, v6[1], 1e-4f);

    EXPECT_EQ(nullptr, r->get(7)); // deleted in delta

    const float* v8 = reinterpret_cast<const float*>(r->get(8));
    ASSERT_NE(nullptr, v8);
    EXPECT_NEAR(8.1f, v8[0], 1e-4f); // untouched base value

    EXPECT_EQ(19u, visible_count(*r)); // one id deleted
}

TEST_F(DatasetFullCycleTest, DeltaMergeBackToDataKeepsReaderConsistent) {
    const std::string dir = make_dir("d");
    DatasetNode ds;
    ASSERT_EQ(0, ds.init_for_test({dir}, 100, DataType::f32, 4).code());

    write_generated(seq_cfg(10, 0, DataType::f32, 4));
    ASSERT_EQ(0, ds.store(input_path_).code());

    write_generated(seq_cfg(1, 80, DataType::f32, 4)); // create delta
    ASSERT_EQ(0, ds.store(input_path_).code());

    write_generated(seq_cfg(5, 50, DataType::f32, 4)); // force data+delta merge
    ASSERT_EQ(0, ds.store(input_path_).code());

    DatasetRangeReaderPtr drs = ds.reader();
    auto [r, ret] = drs->next();
    ASSERT_EQ(0, ret.code());
    ASSERT_NE(nullptr, r);
    EXPECT_EQ(nullptr, drs->next().first);

    EXPECT_EQ(16u, r->count());
    EXPECT_NE(nullptr, r->get(80));
    EXPECT_NE(nullptr, r->get(52));
}

TEST_F(DatasetFullCycleTest, ReaderAppliesDeltaOnlyToTouchedRange) {
    const std::string dir = make_dir("d");
    DatasetNode ds;
    ASSERT_EQ(0, ds.init_for_test({dir}, 10, DataType::f32, 4).code());

    write_generated(seq_cfg(20, 0, DataType::f32, 4)); // 0.data and 1.data
    ASSERT_EQ(0, ds.store(input_path_).code());

    write_generated(detailed_cfg(1, 2, DataType::f32, 4, 10)); // touches only file_id=0
    ASSERT_EQ(0, ds.store(input_path_).code());

    DatasetRangeReaderPtr drs = ds.reader();
    auto [r0, ret0] = drs->next();
    auto [r1, ret1] = drs->next();
    ASSERT_EQ(0, ret0.code());
    ASSERT_EQ(0, ret1.code());
    ASSERT_NE(nullptr, r0);
    ASSERT_NE(nullptr, r1);
    EXPECT_EQ(nullptr, drs->next().first);

    EXPECT_EQ(0u, r0->id(0));
    EXPECT_EQ(10u, r1->id(0));

    const float* changed = reinterpret_cast<const float*>(r0->get(2));
    ASSERT_NE(nullptr, changed);
    EXPECT_NEAR(0.0f, changed[0], 1e-4f); // detailed override applied via delta

    const float* untouched = reinterpret_cast<const float*>(r1->get(12));
    ASSERT_NE(nullptr, untouched);
    EXPECT_NEAR(12.1f, untouched[0], 1e-4f); // no delta in second range
}

TEST_F(DatasetFullCycleTest, FullCycleI16WithOverrideAndDelete) {
    const std::string dir = make_dir("d");
    DatasetNode ds;
    ASSERT_EQ(0, ds.init_for_test({dir}, 100, DataType::i16, 4).code());

    write_generated(seq_cfg(12, 0, DataType::i16, 4));
    ASSERT_EQ(0, ds.store(input_path_).code());

    write_generated(detailed_cfg(2, 3, DataType::i16, 4, 3));
    ASSERT_EQ(0, ds.store(input_path_).code());

    ManualInputGenerator gen;
    gen.type = DataType::i16;
    gen.dim = 4;
    gen.deleted(4);
    write_manual(gen);
    ASSERT_EQ(0, ds.store(input_path_).code());

    DatasetRangeReaderPtr drs = ds.reader();
    auto [r, ret] = drs->next();
    ASSERT_EQ(0, ret.code());
    ASSERT_NE(nullptr, r);
    EXPECT_EQ(nullptr, drs->next().first);

    const int16_t* v3 = reinterpret_cast<const int16_t*>(r->get(3));
    ASSERT_NE(nullptr, v3);
    EXPECT_EQ(0, v3[0]); // overridden from 3 to detailed value

    EXPECT_EQ(nullptr, r->get(4)); // deleted

    const int16_t* v5 = reinterpret_cast<const int16_t*>(r->get(5));
    ASSERT_NE(nullptr, v5);
    EXPECT_EQ(6, v5[0]); // untouched bounded sequential payload
}

TEST_F(DatasetFullCycleTest, FullCycleF8CreateUpdateDeleteMergeReopenAndLookup) {
    constexpr uint64_t kMinId = 1000;
    constexpr size_t kDim = 5;
    const std::string dir = make_dir("f8");
    DatasetNode ds;
    ASSERT_EQ(0, ds.init_for_test({dir}, 100, DataType::f8, kDim, DistFunc::L2).code());

    // Base ordinals are range-relative. The update starts a new f8 sequence,
    // so id 1002 changes from ordinal 2 to ordinal 0 without relying on id order.
    write_generated(seq_cfg(6, kMinId, DataType::f8, kDim));
    ASSERT_EQ(0, ds.store(input_path_).code());
    write_generated(seq_cfg(1, kMinId + 2, DataType::f8, kDim));
    ASSERT_EQ(0, ds.store(input_path_).code());

    ManualInputGenerator tombstone;
    tombstone.type = DataType::f8;
    tombstone.dim = kDim;
    tombstone.deleted(kMinId + 3);
    write_manual(tombstone);
    ASSERT_EQ(0, ds.store(input_path_).code());

    const std::string data_path = dir + "/10.data";
    const std::string delta_path = dir + "/10.delta";
    const std::vector<uint8_t> expected_update = f8_ordinal_bytes(0, kDim);
    ASSERT_TRUE(fs::exists(data_path));
    ASSERT_TRUE(fs::exists(delta_path));

    // Check the attached overlay before compaction: updates replace base bytes,
    // tombstones mask base rows, and the delta keeps its own L2 norm.
    {
        DataReader delta;
        ASSERT_EQ(0, delta.init(delta_path).code());
        ASSERT_TRUE(delta.has_matching_stored_norms(DistFunc::L2));
        ASSERT_EQ(kMinId + 2, delta.id(0));
        EXPECT_NEAR(static_cast<float>(reference_f8_squared_norm(expected_update)), delta.get_norm(0), 1e-5f);

        auto attached_delta = std::make_unique<DataReader>();
        ASSERT_EQ(0, attached_delta->init(delta_path).code());
        DataReader live;
        ASSERT_EQ(0, live.init(data_path, std::move(attached_delta)).code());
        ASSERT_TRUE(live.has_matching_stored_norms(DistFunc::L2));
        const uint8_t* live_updated = live.get(kMinId + 2);
        ASSERT_NE(nullptr, live_updated);
        EXPECT_TRUE(std::equal(expected_update.begin(), expected_update.end(), live_updated));
        EXPECT_EQ(nullptr, live.get(kMinId + 3));
    }

    ASSERT_EQ(0, ds.merge().code());
    EXPECT_FALSE(fs::exists(delta_path));
    const DataFileHeader hdr = read_header(data_path);
    EXPECT_EQ(3u, hdr.type);
    EXPECT_TRUE(data_file_has_squared_norms(hdr));
    EXPECT_EQ(compute_data_record_layout(DataType::f8, kDim, true).stride, hdr.vector_stride);

    DataReader reopened;
    ASSERT_EQ(0, reopened.init(data_path).code());
    ASSERT_TRUE(reopened.has_matching_stored_norms(DistFunc::L2));
    EXPECT_EQ(5u, reopened.count());
    EXPECT_EQ(nullptr, reopened.get(kMinId + 3));
    EXPECT_TRUE(reopened.check_consistency());

    ASSERT_EQ(kMinId + 2, reopened.id(2));
    const uint8_t* updated = reopened.get(kMinId + 2);
    ASSERT_NE(nullptr, updated);
    EXPECT_TRUE(std::equal(expected_update.begin(), expected_update.end(), updated));
    EXPECT_NEAR(static_cast<float>(reference_f8_squared_norm(expected_update)), reopened.get_norm(2), 1e-5f);

    // DatasetNode lookup exercises the post-merge reader cache/query surface.
    auto [looked_up, lookup_ret] = ds.get_vector(kMinId + 2);
    ASSERT_EQ(0, lookup_ret.code());
    ASSERT_NE(nullptr, looked_up);
    EXPECT_TRUE(std::equal(expected_update.begin(), expected_update.end(), looked_up));
    auto [deleted, deleted_ret] = ds.get_vector(kMinId + 3);
    EXPECT_EQ(0, deleted_ret.code());
    EXPECT_EQ(nullptr, deleted);
}

TEST_F(DatasetFullCycleTest, DenseRangeStoredWithRoaringIdsTrailer) {
    const std::string dir = make_dir("dense");
    DatasetNode ds;
    ASSERT_EQ(0, ds.init_for_test({dir}, 100000, DataType::f32, 4).code());

    ManualInputGenerator gen;
    gen.type = DataType::f32;
    gen.dim = 4;
    for (uint64_t i = 0; i < 9000; ++i) {
        gen.add(20000 + i * 2, 1);
    }
    write_manual(gen);
    ASSERT_EQ(0, ds.store(input_path_).code());

    const std::string data_path = dir + "/0.data";
    ASSERT_TRUE(fs::exists(data_path));
    const DataFileHeader hdr = read_header(data_path);
    EXPECT_EQ(0u, hdr.ids_offset % kDataRegionAlignment);
    EXPECT_GT(hdr.ids_bytes, 0u);
    DataReader reader;
    ASSERT_EQ(0, reader.init(data_path).code());
    EXPECT_EQ(20000u, reader.id(0));
    EXPECT_EQ(20000u + 8999u * 2u, reader.id(8999));
}
