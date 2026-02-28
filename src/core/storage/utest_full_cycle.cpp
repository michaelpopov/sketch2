#include <gtest/gtest.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include <vector>
#include "core/storage/input_generator.h"
#include "core/storage/dataset.h"
#include "core/storage/data_reader.h"

using namespace sketch2;
namespace fs = std::filesystem;

class DatasetFullCycleTest : public ::testing::Test {
protected:
    std::string base_dir_;
    std::string input_path_;

    void SetUp() override {
        base_dir_ = "/tmp/sketch2_utest_full_cycle_" + std::to_string(getpid());
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
};

TEST_F(DatasetFullCycleTest, SequentialSingleRangeRoundTripThroughDatasetReader) {
    const std::string dir = make_dir("d");
    Dataset ds;
    ASSERT_EQ(0, ds.init({dir}, 100, DataType::f32, 4).code());

    write_generated(seq_cfg(10, 100, DataType::f32, 4));
    ASSERT_EQ(0, ds.store(input_path_).code());

    DatasetReaderPtr drs = ds.reader();
    ASSERT_NE(nullptr, drs);

    DataReaderPtr r = drs->next();
    ASSERT_NE(nullptr, r);
    EXPECT_EQ(10u, r->count());
    EXPECT_EQ(100u, r->id(0));
    EXPECT_EQ(109u, r->id(9));

    const float* v = reinterpret_cast<const float*>(r->get(103));
    ASSERT_NE(nullptr, v);
    EXPECT_NEAR(103.1f, v[0], 1e-4f);
    EXPECT_NEAR(103.1f, v[3], 1e-4f);

    EXPECT_EQ(nullptr, drs->next());
}

TEST_F(DatasetFullCycleTest, MultiDirMultiRangeReaderOrderAndCoverage) {
    const std::string d0 = make_dir("d0");
    const std::string d1 = make_dir("d1");
    Dataset ds;
    ASSERT_EQ(0, ds.init({d0, d1}, 10, DataType::f32, 4).code());

    write_generated(seq_cfg(30, 0, DataType::f32, 4));
    ASSERT_EQ(0, ds.store(input_path_).code());

    DatasetReaderPtr drs = ds.reader();
    ASSERT_NE(nullptr, drs);

    DataReaderPtr r0 = drs->next();
    DataReaderPtr r1 = drs->next();
    DataReaderPtr r2 = drs->next();
    ASSERT_NE(nullptr, r0);
    ASSERT_NE(nullptr, r1);
    ASSERT_NE(nullptr, r2);
    EXPECT_EQ(nullptr, drs->next());

    EXPECT_EQ(0u, r0->id(0));
    EXPECT_EQ(10u, r1->id(0));
    EXPECT_EQ(20u, r2->id(0));
    EXPECT_EQ(10u, r0->count());
    EXPECT_EQ(10u, r1->count());
    EXPECT_EQ(10u, r2->count());
}

TEST_F(DatasetFullCycleTest, OverrideAndDeleteAreAppliedByDatasetReaderWithDelta) {
    const std::string dir = make_dir("d");
    Dataset ds;
    ASSERT_EQ(0, ds.init({dir}, 100, DataType::f32, 4).code());

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

    DatasetReaderPtr drs = ds.reader();
    DataReaderPtr r = drs->next();
    ASSERT_NE(nullptr, r);
    EXPECT_EQ(nullptr, drs->next());

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
    Dataset ds;
    ASSERT_EQ(0, ds.init({dir}, 100, DataType::f32, 4).code());

    write_generated(seq_cfg(10, 0, DataType::f32, 4));
    ASSERT_EQ(0, ds.store(input_path_).code());

    write_generated(seq_cfg(1, 80, DataType::f32, 4)); // create delta
    ASSERT_EQ(0, ds.store(input_path_).code());

    write_generated(seq_cfg(5, 50, DataType::f32, 4)); // force data+delta merge
    ASSERT_EQ(0, ds.store(input_path_).code());

    DatasetReaderPtr drs = ds.reader();
    DataReaderPtr r = drs->next();
    ASSERT_NE(nullptr, r);
    EXPECT_EQ(nullptr, drs->next());

    EXPECT_EQ(16u, r->count());
    EXPECT_NE(nullptr, r->get(80));
    EXPECT_NE(nullptr, r->get(52));
}

TEST_F(DatasetFullCycleTest, ReaderAppliesDeltaOnlyToTouchedRange) {
    const std::string dir = make_dir("d");
    Dataset ds;
    ASSERT_EQ(0, ds.init({dir}, 10, DataType::f32, 4).code());

    write_generated(seq_cfg(20, 0, DataType::f32, 4)); // 0.data and 1.data
    ASSERT_EQ(0, ds.store(input_path_).code());

    write_generated(detailed_cfg(1, 2, DataType::f32, 4, 10)); // touches only file_id=0
    ASSERT_EQ(0, ds.store(input_path_).code());

    DatasetReaderPtr drs = ds.reader();
    DataReaderPtr r0 = drs->next();
    DataReaderPtr r1 = drs->next();
    ASSERT_NE(nullptr, r0);
    ASSERT_NE(nullptr, r1);
    EXPECT_EQ(nullptr, drs->next());

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
    Dataset ds;
    ASSERT_EQ(0, ds.init({dir}, 100, DataType::i16, 4).code());

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

    DatasetReaderPtr drs = ds.reader();
    DataReaderPtr r = drs->next();
    ASSERT_NE(nullptr, r);
    EXPECT_EQ(nullptr, drs->next());

    const int16_t* v3 = reinterpret_cast<const int16_t*>(r->get(3));
    ASSERT_NE(nullptr, v3);
    EXPECT_EQ(0, v3[0]); // overridden from 3 to detailed value

    EXPECT_EQ(nullptr, r->get(4)); // deleted

    const int16_t* v5 = reinterpret_cast<const int16_t*>(r->get(5));
    ASSERT_NE(nullptr, v5);
    EXPECT_EQ(5, v5[0]); // untouched value
}

