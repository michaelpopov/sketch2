// Unit tests for the SQLite virtual table integration.

#include <gtest/gtest.h>

#include "sqlite3.h"

#include "core/storage/dataset.h"
#include "utils/shared_types.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <tuple>
#include <unistd.h>
#include <vector>

using namespace sketch2;
namespace fs = std::filesystem;

namespace {

struct SqliteDbCloser {
    void operator()(sqlite3* db) const {
        if (db != nullptr) {
            sqlite3_close(db);
        }
    }
};

using SqliteDbPtr = std::unique_ptr<sqlite3, SqliteDbCloser>;

class VliteTest : public ::testing::Test {
protected:
    fs::path root_;
    fs::path dataset_dir_;
    fs::path input_path_;
    fs::path ini_path_;
    std::string table_name_ = "nn";

    void SetUp() override {
        root_ = fs::temp_directory_path() / ("sketch2_utest_vlite_" + std::to_string(getpid()));
        dataset_dir_ = root_ / "dataset";
        input_path_ = root_ / "input.txt";
        ini_path_ = root_ / "dataset.ini";
        fs::create_directories(dataset_dir_);
    }

    void TearDown() override {
        std::error_code ec;
        fs::remove_all(root_, ec);
    }

    void write_input(const std::string& content) {
        std::ofstream out(input_path_);
        ASSERT_TRUE(out.is_open());
        out << content;
    }

    void write_ini(DataType type, uint64_t dim, uint64_t range_size, DistFunc dist_func) {
        std::ofstream out(ini_path_);
        ASSERT_TRUE(out.is_open());
        out << "[dataset]\n";
        out << "dirs=" << dataset_dir_.string() << "\n";
        out << "range_size=" << range_size << "\n";
        out << "dim=" << dim << "\n";
        out << "type=" << data_type_to_string(type) << "\n";
        out << "dist_func=" << dist_func_to_string(dist_func) << "\n";
    }

    void create_dataset(DataType type, uint64_t dim, uint64_t range_size, DistFunc dist_func) {
        Dataset dataset;
        ASSERT_EQ(0, dataset.init({dataset_dir_.string()}, range_size, type, dim,
            kAccumulatorBufferSize, dist_func).code());
        ASSERT_EQ(0, dataset.store(input_path_.string()).code());
        write_ini(type, dim, range_size, dist_func);
    }

    SqliteDbPtr open_db_with_extension() {
        sqlite3* raw_db = nullptr;
        if (sqlite3_open(":memory:", &raw_db) != SQLITE_OK) {
            ADD_FAILURE() << "sqlite3_open failed";
            if (raw_db != nullptr) {
                sqlite3_close(raw_db);
            }
            return {};
        }
        SqliteDbPtr db(raw_db);

        if (sqlite3_enable_load_extension(db.get(), 1) != SQLITE_OK) {
            ADD_FAILURE() << sqlite3_errmsg(db.get());
            return {};
        }
        char* err_msg = nullptr;
        const int rc = sqlite3_load_extension(db.get(), VLITE_EXTENSION_PATH, nullptr, &err_msg);
        const std::string error_text = err_msg ? err_msg : "";
        sqlite3_free(err_msg);
        if (rc != SQLITE_OK) {
            ADD_FAILURE() << error_text;
            return {};
        }
        return db;
    }

    void create_virtual_table(sqlite3* db) {
        char* sql = sqlite3_mprintf(
            "CREATE VIRTUAL TABLE %s USING vlite('%q')",
            table_name_.c_str(),
            ini_path_.string().c_str());
        ASSERT_NE(nullptr, sql);
        char* err_msg = nullptr;
        const int rc = sqlite3_exec(db, sql, nullptr, nullptr, &err_msg);
        const std::string error_text = err_msg ? err_msg : "";
        sqlite3_free(err_msg);
        sqlite3_free(sql);
        ASSERT_EQ(SQLITE_OK, rc) << error_text;
    }

    std::vector<std::pair<uint64_t, double>> query_results(sqlite3* db, const std::string& sql) {
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
            ADD_FAILURE() << sqlite3_errmsg(db);
            return {};
        }

        std::vector<std::pair<uint64_t, double>> rows;
        while (true) {
            const int rc = sqlite3_step(stmt);
            if (rc == SQLITE_DONE) {
                break;
            }
            if (rc != SQLITE_ROW) {
                ADD_FAILURE() << sqlite3_errmsg(db);
                sqlite3_finalize(stmt);
                return {};
            }
            rows.emplace_back(
                static_cast<uint64_t>(sqlite3_column_int64(stmt, 0)),
                sqlite3_column_double(stmt, 1));
        }

        if (sqlite3_finalize(stmt) != SQLITE_OK) {
            ADD_FAILURE() << sqlite3_errmsg(db);
            return {};
        }
        return rows;
    }

    void expect_query_error(sqlite3* db, const std::string& sql, const std::string& message_substr) {
        sqlite3_stmt* stmt = nullptr;
        ASSERT_EQ(SQLITE_OK, sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr))
            << sqlite3_errmsg(db);

        const int rc = sqlite3_step(stmt);
        EXPECT_EQ(SQLITE_ERROR, rc);
        EXPECT_NE(std::string(sqlite3_errmsg(db)).find(message_substr), std::string::npos);
        sqlite3_finalize(stmt);
    }
};

TEST_F(VliteTest, ReturnsKnnIdsAndDistancesForL1Dataset) {
    write_input("f32,4\n"
                "0 : [ 0.1, 0.1, 0.1, 0.1 ]\n"
                "14 : [ 14.1, 14.1, 14.1, 14.1 ]\n"
                "15 : [ 15.1, 15.1, 15.1, 15.1 ]\n"
                "16 : [ 16.1, 16.1, 16.1, 16.1 ]\n"
                "30 : [ 30.1, 30.1, 30.1, 30.1 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::L1);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto rows = query_results(db.get(),
        "SELECT id, distance FROM nn "
        "WHERE query = '15.2, 15.2, 15.2, 15.2' AND k = 3 "
        "ORDER BY distance");

    ASSERT_EQ(3u, rows.size());
    EXPECT_EQ(15u, rows[0].first);
    EXPECT_EQ(16u, rows[1].first);
    EXPECT_EQ(14u, rows[2].first);
    EXPECT_NEAR(0.4, rows[0].second, 1e-5);
    EXPECT_NEAR(3.6, rows[1].second, 1e-5);
    EXPECT_NEAR(4.4, rows[2].second, 1e-5);
}

TEST_F(VliteTest, UsesDatasetDistanceFunctionForCosineQueries) {
    write_input("f32,4\n"
                "10 : [ 100.0, 1.0, 0.0, 0.0 ]\n"
                "20 : [ 1.0, 1.0, 0.0, 0.0 ]\n"
                "30 : [ -1.0, 0.0, 0.0, 0.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::COS);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto rows = query_results(db.get(),
        "SELECT id, distance FROM nn "
        "WHERE query = '1.0, 0.0, 0.0, 0.0' AND k = 3 "
        "ORDER BY distance");

    ASSERT_EQ(3u, rows.size());
    EXPECT_EQ(10u, rows[0].first);
    EXPECT_EQ(20u, rows[1].first);
    EXPECT_EQ(30u, rows[2].first);
    EXPECT_LE(rows[0].second, rows[1].second);
    EXPECT_LE(rows[1].second, rows[2].second);
}

TEST_F(VliteTest, UsesDatasetDistanceFunctionForL2Queries) {
    write_input("f32,4\n"
                "10 : [ 3.0, 0.0, 0.0, 0.0 ]\n"
                "20 : [ 2.0, 2.0, 0.0, 0.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::L2);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto rows = query_results(db.get(),
        "SELECT id, distance FROM nn "
        "WHERE query = '0.0, 0.0, 0.0, 0.0' AND k = 2 "
        "ORDER BY distance");

    ASSERT_EQ(2u, rows.size());
    EXPECT_EQ(20u, rows[0].first);
    EXPECT_EQ(10u, rows[1].first);
    EXPECT_NEAR(8.0, rows[0].second, 1e-9);
    EXPECT_NEAR(9.0, rows[1].second, 1e-9);
}

TEST_F(VliteTest, SupportsI16Datasets) {
    write_input("i16,4\n"
                "10 : [ 10, 10, 10, 10 ]\n"
                "20 : [ 11, 11, 11, 11 ]\n");
    create_dataset(DataType::i16, 4, 100, DistFunc::L1);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto rows = query_results(db.get(),
        "SELECT id, distance FROM nn "
        "WHERE query = '10, 10, 10, 10' AND k = 2 "
        "ORDER BY distance");

    ASSERT_EQ(2u, rows.size());
    EXPECT_EQ(10u, rows[0].first);
    EXPECT_EQ(20u, rows[1].first);
    EXPECT_NEAR(0.0, rows[0].second, 1e-9);
    EXPECT_NEAR(4.0, rows[1].second, 1e-9);
}

TEST_F(VliteTest, SupportsF16Datasets) {
    if (!supports_f16()) {
        GTEST_SKIP() << "f16 is not supported on this platform";
    }

    write_input("f16,4\n"
                "10 : [ 10.0, 10.0, 10.0, 10.0 ]\n"
                "20 : [ 11.0, 11.0, 11.0, 11.0 ]\n");
    create_dataset(DataType::f16, 4, 100, DistFunc::L1);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto rows = query_results(db.get(),
        "SELECT id, distance FROM nn "
        "WHERE query = '10.0, 10.0, 10.0, 10.0' AND k = 2 "
        "ORDER BY distance");

    ASSERT_EQ(2u, rows.size());
    EXPECT_EQ(10u, rows[0].first);
    EXPECT_EQ(20u, rows[1].first);
    EXPECT_NEAR(0.0, rows[0].second, 1e-6);
    EXPECT_NEAR(4.0, rows[1].second, 1e-6);
}

TEST_F(VliteTest, SupportsMatchOperator) {
    write_input("f32,4\n"
                "10 : [ 100.0, 1.0, 0.0, 0.0 ]\n"
                "20 : [ 1.0, 1.0, 0.0, 0.0 ]\n"
                "30 : [ -1.0, 0.0, 0.0, 0.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::COS);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto rows = query_results(db.get(),
        "SELECT id, distance FROM nn "
        "WHERE query MATCH '1.0, 0.0, 0.0, 0.0' AND k = 2 "
        "ORDER BY distance");

    ASSERT_EQ(2u, rows.size());
    EXPECT_EQ(10u, rows[0].first);
    EXPECT_EQ(20u, rows[1].first);
}

TEST_F(VliteTest, PushesLimitIntoImplicitKWithoutChangingVisibleK) {
    write_input("f32,4\n"
                "0 : [ 0.1, 0.1, 0.1, 0.1 ]\n"
                "14 : [ 14.1, 14.1, 14.1, 14.1 ]\n"
                "15 : [ 15.1, 15.1, 15.1, 15.1 ]\n"
                "16 : [ 16.1, 16.1, 16.1, 16.1 ]\n"
                "30 : [ 30.1, 30.1, 30.1, 30.1 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::L1);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    sqlite3_stmt* stmt = nullptr;
    ASSERT_EQ(SQLITE_OK, sqlite3_prepare_v2(
        db.get(),
        "SELECT id, k FROM nn "
        "WHERE query MATCH '15.2, 15.2, 15.2, 15.2' "
        "ORDER BY distance LIMIT 2 OFFSET 1",
        -1, &stmt, nullptr));

    std::vector<std::pair<uint64_t, sqlite3_int64>> rows;
    while (true) {
        const int rc = sqlite3_step(stmt);
        if (rc == SQLITE_DONE) {
            break;
        }
        ASSERT_EQ(SQLITE_ROW, rc) << sqlite3_errmsg(db.get());
        rows.emplace_back(
            static_cast<uint64_t>(sqlite3_column_int64(stmt, 0)),
            sqlite3_column_int64(stmt, 1));
    }
    ASSERT_EQ(SQLITE_OK, sqlite3_finalize(stmt));

    ASSERT_EQ(2u, rows.size());
    EXPECT_EQ(16u, rows[0].first);
    EXPECT_EQ(14u, rows[1].first);
    EXPECT_EQ(10, rows[0].second);
    EXPECT_EQ(10, rows[1].second);
}

TEST_F(VliteTest, ExplicitKRemainsVisibleWhenLimitPushdownApplies) {
    write_input("f32,4\n"
                "0 : [ 0.1, 0.1, 0.1, 0.1 ]\n"
                "14 : [ 14.1, 14.1, 14.1, 14.1 ]\n"
                "15 : [ 15.1, 15.1, 15.1, 15.1 ]\n"
                "16 : [ 16.1, 16.1, 16.1, 16.1 ]\n"
                "30 : [ 30.1, 30.1, 30.1, 30.1 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::L1);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    sqlite3_stmt* stmt = nullptr;
    ASSERT_EQ(SQLITE_OK, sqlite3_prepare_v2(
        db.get(),
        "SELECT id, k FROM nn "
        "WHERE query = '15.2, 15.2, 15.2, 15.2' AND k = 100 "
        "ORDER BY distance LIMIT 1",
        -1, &stmt, nullptr));

    ASSERT_EQ(SQLITE_ROW, sqlite3_step(stmt));
    EXPECT_EQ(15, sqlite3_column_int64(stmt, 0));
    EXPECT_EQ(100, sqlite3_column_int64(stmt, 1));
    ASSERT_EQ(SQLITE_DONE, sqlite3_step(stmt));
    ASSERT_EQ(SQLITE_OK, sqlite3_finalize(stmt));
}

TEST_F(VliteTest, ReusesCachedDatasetAcrossQueries) {
    write_input("f32,4\n"
                "10 : [ 10.1, 10.1, 10.1, 10.1 ]\n"
                "11 : [ 11.1, 11.1, 11.1, 11.1 ]\n"
                "12 : [ 12.1, 12.1, 12.1, 12.1 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::L1);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto first_rows = query_results(db.get(),
        "SELECT id, distance FROM nn "
        "WHERE query = '11.2, 11.2, 11.2, 11.2' AND k = 2 "
        "ORDER BY distance");
    ASSERT_EQ(2u, first_rows.size());
    EXPECT_EQ(11u, first_rows[0].first);
    EXPECT_EQ(12u, first_rows[1].first);

    ASSERT_TRUE(fs::remove(ini_path_));

    const auto second_rows = query_results(db.get(),
        "SELECT id, distance FROM nn "
        "WHERE query = '10.2, 10.2, 10.2, 10.2' AND k = 2 "
        "ORDER BY distance");
    ASSERT_EQ(2u, second_rows.size());
    EXPECT_EQ(10u, second_rows[0].first);
    EXPECT_EQ(11u, second_rows[1].first);
}

TEST_F(VliteTest, LargeKReturnsAllAvailableRows) {
    write_input("f32,4\n"
                "10 : [ 10.0, 10.0, 10.0, 10.0 ]\n"
                "11 : [ 11.0, 11.0, 11.0, 11.0 ]\n"
                "12 : [ 12.0, 12.0, 12.0, 12.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::L1);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto rows = query_results(db.get(),
        "SELECT id, distance FROM nn "
        "WHERE query = '10.0, 10.0, 10.0, 10.0' AND k = 100 "
        "ORDER BY distance");

    ASSERT_EQ(3u, rows.size());
    EXPECT_EQ(10u, rows[0].first);
    EXPECT_EQ(11u, rows[1].first);
    EXPECT_EQ(12u, rows[2].first);
}

TEST_F(VliteTest, LimitZeroReturnsNoRows) {
    write_input("f32,4\n"
                "10 : [ 10.0, 10.0, 10.0, 10.0 ]\n"
                "11 : [ 11.0, 11.0, 11.0, 11.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::L1);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto rows = query_results(db.get(),
        "SELECT id, distance FROM nn "
        "WHERE query = '10.0, 10.0, 10.0, 10.0' "
        "ORDER BY distance LIMIT 0");

    EXPECT_TRUE(rows.empty());
}

TEST_F(VliteTest, DeletedVectorsStayHidden) {
    write_input("f32,4\n"
                "0 : [ 0.1, 0.1, 0.1, 0.1 ]\n"
                "1 : [ 1.1, 1.1, 1.1, 1.1 ]\n"
                "2 : [ 2.1, 2.1, 2.1, 2.1 ]\n"
                "3 : [ 3.1, 3.1, 3.1, 3.1 ]\n");

    Dataset dataset;
    ASSERT_EQ(0, dataset.init({dataset_dir_.string()}, 100, DataType::f32, 4,
        kAccumulatorBufferSize, DistFunc::L1).code());
    ASSERT_EQ(0, dataset.store(input_path_.string()).code());
    ASSERT_EQ(0, dataset.delete_vector(2).code());
    ASSERT_EQ(0, dataset.store_accumulator().code());
    write_ini(DataType::f32, 4, 100, DistFunc::L1);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto rows = query_results(db.get(),
        "SELECT id, distance FROM nn "
        "WHERE query = '2.1, 2.1, 2.1, 2.1' AND k = 4 "
        "ORDER BY distance");

    ASSERT_EQ(3u, rows.size());
    for (const auto& [id, dist] : rows) {
        (void)dist;
        EXPECT_NE(2u, id);
    }
}

TEST_F(VliteTest, FailsWithoutQueryConstraint) {
    write_input("f32,4\n"
                "1 : [ 1.0, 1.0, 1.0, 1.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::L1);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    sqlite3_stmt* stmt = nullptr;
    ASSERT_EQ(SQLITE_OK, sqlite3_prepare_v2(
        db.get(), "SELECT id FROM nn WHERE k = 1", -1, &stmt, nullptr));

    const int rc = sqlite3_step(stmt);
    EXPECT_EQ(SQLITE_ERROR, rc);
    EXPECT_NE(std::string(sqlite3_errmsg(db.get())).find("WHERE query"), std::string::npos);
    sqlite3_finalize(stmt);
}

TEST_F(VliteTest, RejectsZeroK) {
    write_input("f32,4\n"
                "1 : [ 1.0, 1.0, 1.0, 1.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::L1);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    expect_query_error(db.get(),
        "SELECT id FROM nn WHERE query = '1.0, 1.0, 1.0, 1.0' AND k = 0",
        "k must be > 0");
}

TEST_F(VliteTest, RejectsNegativeK) {
    write_input("f32,4\n"
                "1 : [ 1.0, 1.0, 1.0, 1.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::L1);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    expect_query_error(db.get(),
        "SELECT id FROM nn WHERE query = '1.0, 1.0, 1.0, 1.0' AND k = -1",
        "k must be > 0");
}

TEST_F(VliteTest, RejectsEmptyQueryString) {
    write_input("f32,4\n"
                "1 : [ 1.0, 1.0, 1.0, 1.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::L1);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    expect_query_error(db.get(),
        "SELECT id FROM nn WHERE query = '' AND k = 1",
        "non-empty string");
}

TEST_F(VliteTest, RejectsWrongDimensionQueryVector) {
    write_input("f32,4\n"
                "1 : [ 1.0, 1.0, 1.0, 1.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::L1);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    expect_query_error(db.get(),
        "SELECT id FROM nn WHERE query = '1.0, 1.0, 1.0' AND k = 1",
        "truncated vector payload");
}

TEST_F(VliteTest, RejectsMalformedQueryVectorText) {
    write_input("f32,4\n"
                "1 : [ 1.0, 1.0, 1.0, 1.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::L1);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    expect_query_error(db.get(),
        "SELECT id FROM nn WHERE query = '1.0, nope, 1.0, 1.0' AND k = 1",
        "invalid f32 token");
}

TEST_F(VliteTest, RejectsIdsOutsideSqliteIntegerRange) {
    const uint64_t large_id = static_cast<uint64_t>(std::numeric_limits<sqlite3_int64>::max()) + 1u;
    write_input("f32,4\n" + std::to_string(large_id) + " : [ 1.0, 1.0, 1.0, 1.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::L1);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    sqlite3_stmt* stmt = nullptr;
    ASSERT_EQ(SQLITE_OK, sqlite3_prepare_v2(
        db.get(),
        "SELECT id FROM nn "
        "WHERE query = '1.0, 1.0, 1.0, 1.0' AND k = 1 "
        "ORDER BY distance",
        -1, &stmt, nullptr));

    const int rc = sqlite3_step(stmt);
    EXPECT_EQ(SQLITE_ERROR, rc);
    EXPECT_NE(std::string(sqlite3_errmsg(db.get())).find("SQLite INTEGER range"), std::string::npos);
    sqlite3_finalize(stmt);
}

} // namespace
