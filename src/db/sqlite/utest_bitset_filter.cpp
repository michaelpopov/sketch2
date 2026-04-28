// Unit tests covering SQL pattern variations of bitset_agg() in SQLite.
// Complements utest_vlite.cpp, which exercises core bitset_agg semantics; the
// tests below focus on the breadth of SQL idioms that can feed bitset_agg and
// the breadth of source shapes that can produce its input rows.

#include <gtest/gtest.h>

#include "sqlite3.h"

#include "sketch2api/sketch2.h"
#include "core/bitset/bitset_filter_control.h"
#include "core/bitset/chunked_bits.h"
#include "utils/singleton.h"
#include "utils/shared_types.h"
#include "core/bitset/utest_chunked_bits_helpers.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace sketch2;
namespace fs = std::filesystem;

namespace {

using test::expect_persisted_filter_contains;
using test::ScopedPathCleanup;
using test::unique_filter_name;

struct SqliteDbCloser {
    void operator()(sqlite3* db) const {
        if (db != nullptr) {
            sqlite3_close(db);
        }
    }
};

using SqliteDbPtr = std::unique_ptr<sqlite3, SqliteDbCloser>;

class BitsetFilterSqlTest : public ::testing::Test {
protected:
    fs::path root_;
    fs::path db_root_;
    fs::path dataset_dir_;
    fs::path input_path_;
    std::string table_name_ = "nn";
    std::string dataset_name_ = "ds";

    void SetUp() override {
        static uint64_t counter = 0;
        root_ = fs::temp_directory_path() /
            ("sketch2_utest_bitset_filter_" + std::to_string(getpid()) + "_" + std::to_string(++counter));
        fs::create_directories(root_);
        db_root_ = root_ / "db";
        dataset_dir_ = db_root_ / dataset_name_;
        input_path_ = root_ / "input.txt";
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

    void create_dataset(DataType type, uint64_t dim, uint64_t range_size, DistFunc dist_func) {
        sk_handle_t* handle = sk_new_handle(db_root_.string().c_str());
        ASSERT_NE(nullptr, handle);
        ASSERT_EQ(0, sk_create(handle, dataset_name_.c_str(), nullptr, static_cast<unsigned int>(dim),
            data_type_to_string(type), static_cast<unsigned int>(range_size), dist_func_to_string(dist_func)))
            << sk_error_message(handle);
        ASSERT_EQ(0, sk_load_file(handle, input_path_.string().c_str())) << sk_error_message(handle);
        ASSERT_EQ(0, sk_close(handle)) << sk_error_message(handle);
        sk_release_handle(handle);
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
            "CREATE VIRTUAL TABLE %s USING vlite('%q', '%q')",
            table_name_.c_str(),
            db_root_.string().c_str(),
            dataset_name_.c_str());
        ASSERT_NE(nullptr, sql);
        char* err_msg = nullptr;
        const int rc = sqlite3_exec(db, sql, nullptr, nullptr, &err_msg);
        const std::string error_text = err_msg ? err_msg : "";
        sqlite3_free(err_msg);
        sqlite3_free(sql);
        ASSERT_EQ(SQLITE_OK, rc) << error_text;
    }

    void exec_sql(sqlite3* db, const std::string& sql) {
        char* err_msg = nullptr;
        const int rc = sqlite3_exec(db, sql.c_str(), nullptr, nullptr, &err_msg);
        const std::string error_text = err_msg ? err_msg : "";
        sqlite3_free(err_msg);
        ASSERT_EQ(SQLITE_OK, rc) << error_text << " : " << sql;
    }

    int query_single_int(sqlite3* db, const std::string& sql) {
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
            ADD_FAILURE() << sqlite3_errmsg(db) << " : " << sql;
            return -1;
        }
        int value = -1;
        int rc = sqlite3_step(stmt);
        if (rc != SQLITE_ROW) {
            ADD_FAILURE() << sqlite3_errmsg(db) << " : " << sql;
            sqlite3_finalize(stmt);
            return -1;
        }
        value = sqlite3_column_int(stmt, 0);
        rc = sqlite3_step(stmt);
        EXPECT_EQ(SQLITE_DONE, rc) << sqlite3_errmsg(db) << " : " << sql;
        if (sqlite3_finalize(stmt) != SQLITE_OK) {
            ADD_FAILURE() << sqlite3_errmsg(db);
            return -1;
        }
        return value;
    }

    std::vector<uint64_t> query_ids(sqlite3* db, const std::string& sql) {
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr) != SQLITE_OK) {
            ADD_FAILURE() << sqlite3_errmsg(db) << " : " << sql;
            return {};
        }
        std::vector<uint64_t> ids;
        while (true) {
            const int rc = sqlite3_step(stmt);
            if (rc == SQLITE_DONE) {
                break;
            }
            if (rc != SQLITE_ROW) {
                ADD_FAILURE() << sqlite3_errmsg(db) << " : " << sql;
                sqlite3_finalize(stmt);
                return {};
            }
            ids.push_back(static_cast<uint64_t>(sqlite3_column_int64(stmt, 0)));
        }
        if (sqlite3_finalize(stmt) != SQLITE_OK) {
            ADD_FAILURE() << sqlite3_errmsg(db);
            return {};
        }
        return ids;
    }

    std::vector<uint64_t> sorted_query_ids(sqlite3* db, const std::string& sql) {
        auto ids = query_ids(db, sql);
        std::sort(ids.begin(), ids.end());
        return ids;
    }

    void expect_query_error(sqlite3* db, const std::string& sql, const std::string& message_substr) {
        sqlite3_stmt* stmt = nullptr;
        ASSERT_EQ(SQLITE_OK, sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, nullptr))
            << sqlite3_errmsg(db) << " : " << sql;
        const int rc = sqlite3_step(stmt);
        EXPECT_EQ(SQLITE_ERROR, rc) << "expected SQLITE_ERROR for: " << sql;
        EXPECT_NE(std::string(sqlite3_errmsg(db)).find(message_substr), std::string::npos)
            << "actual message: " << sqlite3_errmsg(db);
        sqlite3_finalize(stmt);
    }

    // Standard 5-row dataset with ids 10, 20, 30, 40, 50, each whose vector
    // scales with the id. For DOT against query [25,25,25,25] the natural DESC
    // order is 50, 40, 30, 20, 10 — easy to verify bitset filtering.
    void setup_decimal_dataset() {
        write_input("f32,4\n"
                    "10 : [ 10.0, 10.0, 10.0, 10.0 ]\n"
                    "20 : [ 20.0, 20.0, 20.0, 20.0 ]\n"
                    "30 : [ 30.0, 30.0, 30.0, 30.0 ]\n"
                    "40 : [ 40.0, 40.0, 40.0, 40.0 ]\n"
                    "50 : [ 50.0, 50.0, 50.0, 50.0 ]\n");
        create_dataset(DataType::f32, 4, 100, DistFunc::DOT);
    }

    void populate_spill_sized_allowed_table(sqlite3* db) {
        exec_sql(db, "CREATE TABLE many(id INTEGER)");
        exec_sql(db, "BEGIN");
        sqlite3_stmt* stmt = nullptr;
        ASSERT_EQ(SQLITE_OK,
            sqlite3_prepare_v2(db, "INSERT INTO many(id) VALUES (?)", -1, &stmt, nullptr));
        sqlite3_bind_int64(stmt, 1, 20);
        ASSERT_EQ(SQLITE_DONE, sqlite3_step(stmt));
        sqlite3_reset(stmt);
        sqlite3_bind_int64(stmt, 1, 40);
        ASSERT_EQ(SQLITE_DONE, sqlite3_step(stmt));
        sqlite3_reset(stmt);
        for (uint64_t i = 0; i < 50000; ++i) {
            sqlite3_bind_int64(stmt, 1, static_cast<sqlite3_int64>((i + 1) << kChunkBits));
            ASSERT_EQ(SQLITE_DONE, sqlite3_step(stmt));
            sqlite3_reset(stmt);
        }
        sqlite3_finalize(stmt);
        exec_sql(db, "COMMIT");
    }
};

// ---------------------------------------------------------------------------
// Source variations: the rows feeding bitset_agg can come from many shapes.
// ---------------------------------------------------------------------------

TEST_F(BitsetFilterSqlTest, BitsetAggSourcedFromValuesClause) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto ids = sorted_query_ids(db.get(),
        "SELECT id FROM nn "
        "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        "AND allowed_ids = ("
        "  SELECT bitset_agg(column1) FROM (VALUES (20), (40))"
        ")");
    EXPECT_EQ((std::vector<uint64_t>{20, 40}), ids);
}

TEST_F(BitsetFilterSqlTest, BitsetAggSourcedFromCte) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto ids = sorted_query_ids(db.get(),
        "WITH chosen(id) AS (VALUES (10), (30), (50)) "
        "SELECT id FROM nn "
        "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        "AND allowed_ids = (SELECT bitset_agg(id) FROM chosen)");
    EXPECT_EQ((std::vector<uint64_t>{10, 30, 50}), ids);
}

TEST_F(BitsetFilterSqlTest, BitsetAggSourcedFromRecursiveCte) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    // Recursive CTE generates 10, 20, 30, 40, 50 — every dataset id.
    const auto ids = sorted_query_ids(db.get(),
        "WITH RECURSIVE seq(x) AS ("
        "  SELECT 10 UNION ALL SELECT x + 10 FROM seq WHERE x < 50"
        ") "
        "SELECT id FROM nn "
        "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        "AND allowed_ids = (SELECT bitset_agg(x) FROM seq)");
    EXPECT_EQ((std::vector<uint64_t>{10, 20, 30, 40, 50}), ids);
}

TEST_F(BitsetFilterSqlTest, BitsetAggSourcedFromRegularTable) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    exec_sql(db.get(), "CREATE TABLE allowed(id INTEGER)");
    exec_sql(db.get(), "INSERT INTO allowed(id) VALUES (20), (40), (50)");

    const auto ids = sorted_query_ids(db.get(),
        "SELECT id FROM nn "
        "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        "AND allowed_ids = (SELECT bitset_agg(id) FROM allowed)");
    EXPECT_EQ((std::vector<uint64_t>{20, 40, 50}), ids);
}

TEST_F(BitsetFilterSqlTest, BitsetAggSourcedFromTempTable) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    exec_sql(db.get(), "CREATE TEMP TABLE tmp_allowed(id INTEGER)");
    exec_sql(db.get(), "INSERT INTO temp.tmp_allowed(id) VALUES (10), (50)");

    const auto ids = sorted_query_ids(db.get(),
        "SELECT id FROM nn "
        "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        "AND allowed_ids = (SELECT bitset_agg(id) FROM temp.tmp_allowed)");
    EXPECT_EQ((std::vector<uint64_t>{10, 50}), ids);
}

// ---------------------------------------------------------------------------
// Filtering / shaping the bitset_agg input rows.
// ---------------------------------------------------------------------------

TEST_F(BitsetFilterSqlTest, BitsetAggSourceFilteredByWhereClause) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    exec_sql(db.get(), "CREATE TABLE candidates(id INTEGER, tag TEXT)");
    exec_sql(db.get(),
        "INSERT INTO candidates VALUES "
        "(10, 'cold'), (20, 'hot'), (30, 'cold'), (40, 'hot'), (50, 'cold')");

    const auto ids = sorted_query_ids(db.get(),
        "SELECT id FROM nn "
        "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        "AND allowed_ids = ("
        "  SELECT bitset_agg(id) FROM candidates WHERE tag = 'hot'"
        ")");
    EXPECT_EQ((std::vector<uint64_t>{20, 40}), ids);
}

TEST_F(BitsetFilterSqlTest, BitsetAggSourceShapedByLimit) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    exec_sql(db.get(), "CREATE TABLE c(id INTEGER)");
    exec_sql(db.get(), "INSERT INTO c VALUES (10), (20), (30), (40), (50)");

    const auto ids = sorted_query_ids(db.get(),
        "SELECT id FROM nn "
        "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        "AND allowed_ids = ("
        "  SELECT bitset_agg(id) FROM (SELECT id FROM c ORDER BY id DESC LIMIT 2)"
        ")");
    EXPECT_EQ((std::vector<uint64_t>{40, 50}), ids);
}

TEST_F(BitsetFilterSqlTest, BitsetAggInputOrderIndependence) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    // Whether the source is sorted, reverse-sorted, or random, the resulting
    // Bitset filters must be identical: the same ids end up in the result set.
    const auto sorted_ids = sorted_query_ids(db.get(),
        "SELECT id FROM nn "
        "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        "AND allowed_ids = ("
        "  SELECT bitset_agg(column1) FROM (VALUES (10), (20), (30))"
        ")");
    const auto reverse_ids = sorted_query_ids(db.get(),
        "SELECT id FROM nn "
        "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        "AND allowed_ids = ("
        "  SELECT bitset_agg(column1) FROM (VALUES (30), (20), (10))"
        ")");
    const auto random_ids = sorted_query_ids(db.get(),
        "SELECT id FROM nn "
        "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        "AND allowed_ids = ("
        "  SELECT bitset_agg(column1) FROM (VALUES (20), (10), (30))"
        ")");
    EXPECT_EQ((std::vector<uint64_t>{10, 20, 30}), sorted_ids);
    EXPECT_EQ(sorted_ids, reverse_ids);
    EXPECT_EQ(sorted_ids, random_ids);
}

TEST_F(BitsetFilterSqlTest, BitsetAggWithDistinctOnSource) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    // bitset_agg dedups internally; SELECT DISTINCT is just a different SQL
    // path to the same result.
    const auto ids = sorted_query_ids(db.get(),
        "SELECT id FROM nn "
        "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        "AND allowed_ids = ("
        "  SELECT bitset_agg(id) FROM ("
        "    SELECT DISTINCT column1 AS id FROM (VALUES (10), (10), (30), (30))"
        "  )"
        ")");
    EXPECT_EQ((std::vector<uint64_t>{10, 30}), ids);
}

TEST_F(BitsetFilterSqlTest, BitsetAggInputFromGroupByHaving) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    exec_sql(db.get(), "CREATE TABLE votes(id INTEGER, voter TEXT)");
    exec_sql(db.get(),
        "INSERT INTO votes VALUES "
        "(10,'a'), (20,'a'), (20,'b'), (30,'a'), (40,'a'), (40,'b'), (40,'c'), (50,'a')");

    // Allow only ids that received >= 2 votes.
    const auto ids = sorted_query_ids(db.get(),
        "SELECT id FROM nn "
        "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        "AND allowed_ids = ("
        "  SELECT bitset_agg(id) FROM ("
        "    SELECT id FROM votes GROUP BY id HAVING COUNT(*) >= 2"
        "  )"
        ")");
    EXPECT_EQ((std::vector<uint64_t>{20, 40}), ids);
}

// ---------------------------------------------------------------------------
// Set-operation combinations to compose the input set.
// ---------------------------------------------------------------------------

TEST_F(BitsetFilterSqlTest, BitsetAggFromUnion) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto ids = sorted_query_ids(db.get(),
        "SELECT id FROM nn "
        "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        "AND allowed_ids = ("
        "  SELECT bitset_agg(id) FROM ("
        "    SELECT 10 AS id UNION SELECT 20 UNION SELECT 30 UNION SELECT 20"
        "  )"
        ")");
    EXPECT_EQ((std::vector<uint64_t>{10, 20, 30}), ids);
}

TEST_F(BitsetFilterSqlTest, BitsetAggFromIntersect) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto ids = sorted_query_ids(db.get(),
        "WITH a(id) AS (VALUES (10),(20),(30),(40)), "
        "     b(id) AS (VALUES (20),(40),(50)) "
        "SELECT id FROM nn "
        "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        "AND allowed_ids = ("
        "  SELECT bitset_agg(id) FROM ("
        "    SELECT id FROM a INTERSECT SELECT id FROM b"
        "  )"
        ")");
    EXPECT_EQ((std::vector<uint64_t>{20, 40}), ids);
}

TEST_F(BitsetFilterSqlTest, BitsetAggFromExcept) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto ids = sorted_query_ids(db.get(),
        "WITH a(id) AS (VALUES (10),(20),(30),(40),(50)), "
        "     blocked(id) AS (VALUES (20),(40)) "
        "SELECT id FROM nn "
        "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        "AND allowed_ids = ("
        "  SELECT bitset_agg(id) FROM ("
        "    SELECT id FROM a EXCEPT SELECT id FROM blocked"
        "  )"
        ")");
    EXPECT_EQ((std::vector<uint64_t>{10, 30, 50}), ids);
}

TEST_F(BitsetFilterSqlTest, BitsetAggFromInnerJoin) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    exec_sql(db.get(), "CREATE TABLE owners(id INTEGER, owner TEXT)");
    exec_sql(db.get(),
        "INSERT INTO owners VALUES (10,'alice'),(20,'bob'),(30,'alice'),(40,'carol'),(50,'bob')");
    exec_sql(db.get(), "CREATE TABLE active(owner TEXT)");
    exec_sql(db.get(), "INSERT INTO active VALUES ('alice'),('bob')");

    const auto ids = sorted_query_ids(db.get(),
        "SELECT id FROM nn "
        "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        "AND allowed_ids = ("
        "  SELECT bitset_agg(o.id) "
        "  FROM owners AS o INNER JOIN active AS a ON o.owner = a.owner"
        ")");
    EXPECT_EQ((std::vector<uint64_t>{10, 20, 30, 50}), ids);
}

TEST_F(BitsetFilterSqlTest, BitsetAggFromLeftJoinTreatsUnmatchedNullsAsNoop) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    exec_sql(db.get(), "CREATE TABLE candidates(id INTEGER)");
    exec_sql(db.get(), "INSERT INTO candidates VALUES (10),(20),(30),(40),(50)");
    exec_sql(db.get(), "CREATE TABLE allowed(id INTEGER)");
    exec_sql(db.get(), "INSERT INTO allowed VALUES (10),(30)");

    // LEFT JOIN produces NULLs for unmatched rows; bitset_agg ignores those.
    const auto ids = sorted_query_ids(db.get(),
        "SELECT id FROM nn "
        "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        "AND allowed_ids = ("
        "  SELECT bitset_agg(a.id) "
        "  FROM candidates AS c LEFT JOIN allowed AS a ON c.id = a.id"
        ")");
    EXPECT_EQ((std::vector<uint64_t>{10, 30}), ids);
}

// ---------------------------------------------------------------------------
// Expression variations on the aggregated value.
// ---------------------------------------------------------------------------

TEST_F(BitsetFilterSqlTest, BitsetAggOnArithmeticExpression) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    // base values 1..5 multiplied by 10 produce dataset ids.
    const auto ids = sorted_query_ids(db.get(),
        "SELECT id FROM nn "
        "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        "AND allowed_ids = ("
        "  SELECT bitset_agg(column1 * 10) FROM (VALUES (1), (3), (5))"
        ")");
    EXPECT_EQ((std::vector<uint64_t>{10, 30, 50}), ids);
}

TEST_F(BitsetFilterSqlTest, BitsetAggOnCaseExpression) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    exec_sql(db.get(), "CREATE TABLE labels(label TEXT)");
    exec_sql(db.get(), "INSERT INTO labels VALUES ('a'),('b'),('c'),('d')");

    // CASE maps labels to ids; 'd' yields NULL which bitset_agg drops.
    const auto ids = sorted_query_ids(db.get(),
        "SELECT id FROM nn "
        "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        "AND allowed_ids = ("
        "  SELECT bitset_agg(CASE label "
        "    WHEN 'a' THEN 10 WHEN 'b' THEN 30 WHEN 'c' THEN 50 ELSE NULL "
        "  END) FROM labels"
        ")");
    EXPECT_EQ((std::vector<uint64_t>{10, 30, 50}), ids);
}

TEST_F(BitsetFilterSqlTest, BitsetAggOnExplicitCastFromText) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    // Source rows are TEXT; CAST(... AS INTEGER) feeds a valid integer to
    // bitset_agg. Without the CAST this would fail (see RejectsTextValue).
    const auto ids = sorted_query_ids(db.get(),
        "SELECT id FROM nn "
        "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        "AND allowed_ids = ("
        "  SELECT bitset_agg(CAST(column1 AS INTEGER)) "
        "  FROM (VALUES ('20'), ('40'))"
        ")");
    EXPECT_EQ((std::vector<uint64_t>{20, 40}), ids);
}

// ---------------------------------------------------------------------------
// Edge values: id = 0, very large ids, empty input, all-NULL input.
// ---------------------------------------------------------------------------

TEST_F(BitsetFilterSqlTest, BitsetAggIncludesZeroId) {
    write_input("f32,4\n"
                "0  : [ 0.5, 0.5, 0.5, 0.5 ]\n"
                "10 : [ 10.0, 10.0, 10.0, 10.0 ]\n"
                "20 : [ 20.0, 20.0, 20.0, 20.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto ids = sorted_query_ids(db.get(),
        "SELECT id FROM nn "
        "WHERE query = '0.0, 0.0, 0.0, 0.0' AND k = 3 "
        "AND allowed_ids = ("
        "  SELECT bitset_agg(column1) FROM (VALUES (0), (20))"
        ")");
    EXPECT_EQ((std::vector<uint64_t>{0, 20}), ids);
}

TEST_F(BitsetFilterSqlTest, BitsetAggHandlesIdsAcrossManyChunks) {
    constexpr uint64_t kFirstChunkId = 5;
    constexpr uint64_t kSecondChunkId = (1ull << 16u) + 7u;
    constexpr uint64_t kFarChunkId = (1ull << 24u) + 11u;
    write_input(
        std::string("f32,4\n")
        + std::to_string(kFirstChunkId)  + " : [ 1.0, 1.0, 1.0, 1.0 ]\n"
        + std::to_string(kSecondChunkId) + " : [ 1.0, 1.0, 1.0, 1.0 ]\n"
        + std::to_string(kFarChunkId)    + " : [ 1.0, 1.0, 1.0, 1.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const std::string sql =
        "SELECT id FROM nn "
        "WHERE query = '1.0, 1.0, 1.0, 1.0' AND k = 3 "
        "AND allowed_ids = ("
        "  SELECT bitset_agg(column1) FROM (VALUES ("
        + std::to_string(kFirstChunkId) + "), ("
        + std::to_string(kFarChunkId) + ")))";
    const auto ids = sorted_query_ids(db.get(), sql);
    EXPECT_EQ((std::vector<uint64_t>{kFirstChunkId, kFarChunkId}), ids);
}

TEST_F(BitsetFilterSqlTest, BitsetAggOnAllNullInputMatchesNothing) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto ids = query_ids(db.get(),
        "SELECT id FROM nn "
        "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        "AND allowed_ids = ("
        "  SELECT bitset_agg(column1) FROM (VALUES (NULL), (NULL), (NULL))"
        ")");
    EXPECT_TRUE(ids.empty());
}

TEST_F(BitsetFilterSqlTest, NamedBitsetAggOnAllNullInputPersistsEmptyFile) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const std::string name = unique_filter_name("sql_named_empty_all_null");
    const fs::path expected_path = named_bitset_filter_path(name);
    fs::remove(expected_path);

    const auto ids = query_ids(db.get(),
        std::string("SELECT id FROM nn ")
        + "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        + "AND allowed_ids = ("
        + "  SELECT bitset_agg(column1, '" + name + "') FROM (VALUES (NULL), (NULL), (NULL))"
        + ")");
    EXPECT_TRUE(ids.empty());
    ASSERT_TRUE(fs::exists(expected_path));
    EXPECT_GT(fs::file_size(expected_path), 0u);
    expect_persisted_filter_contains(expected_path, {});

    fs::remove(expected_path);
}

TEST_F(BitsetFilterSqlTest, BitsetAggOnFilteredOutInputMatchesNothing) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto ids = query_ids(db.get(),
        "SELECT id FROM nn "
        "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        "AND allowed_ids = ("
        "  SELECT bitset_agg(column1) FROM (VALUES (10),(20)) WHERE 1 = 0"
        ")");
    EXPECT_TRUE(ids.empty());
}

TEST_F(BitsetFilterSqlTest, NamedBitsetAggOnZeroRowsMatchesNothingWithoutFile) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const std::string name = unique_filter_name("sql_named_empty_zero_rows");
    const fs::path expected_path = named_bitset_filter_path(name);
    fs::remove(expected_path);

    const auto ids = query_ids(db.get(),
        std::string("SELECT id FROM nn ")
        + "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        + "AND allowed_ids = ("
        + "  SELECT bitset_agg(column1, '" + name + "') FROM (VALUES (10),(20)) WHERE 1 = 0"
        + ")");
    EXPECT_TRUE(ids.empty());
    EXPECT_FALSE(fs::exists(expected_path));
}

TEST_F(BitsetFilterSqlTest, BitsetAggOnIdsNotInDatasetReturnsNoRows) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto ids = query_ids(db.get(),
        "SELECT id FROM nn "
        "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        "AND allowed_ids = ("
        "  SELECT bitset_agg(column1) FROM (VALUES (1), (2), (3))"
        ")");
    EXPECT_TRUE(ids.empty());
}

TEST_F(BitsetFilterSqlTest, BitsetAggMixedExistingAndMissingIdsKeepsExisting) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto ids = sorted_query_ids(db.get(),
        "SELECT id FROM nn "
        "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        "AND allowed_ids = ("
        "  SELECT bitset_agg(column1) FROM (VALUES (10), (999), (40), (12345))"
        ")");
    EXPECT_EQ((std::vector<uint64_t>{10, 40}), ids);
}

// ---------------------------------------------------------------------------
// Composition with the host query.
// ---------------------------------------------------------------------------

TEST_F(BitsetFilterSqlTest, BitsetAggLargeAggregateAcrossManyRows) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    exec_sql(db.get(), "CREATE TABLE many(id INTEGER)");
    // Build a big input set with ids 0..999, only 30 of which are real
    // dataset ids (10, 20, 30, 40, 50).
    {
        sqlite3_exec(db.get(), "BEGIN", nullptr, nullptr, nullptr);
        sqlite3_stmt* stmt = nullptr;
        ASSERT_EQ(SQLITE_OK,
            sqlite3_prepare_v2(db.get(), "INSERT INTO many(id) VALUES (?)", -1, &stmt, nullptr));
        for (int i = 0; i < 1000; ++i) {
            sqlite3_bind_int64(stmt, 1, i);
            ASSERT_EQ(SQLITE_DONE, sqlite3_step(stmt));
            sqlite3_reset(stmt);
        }
        sqlite3_finalize(stmt);
        sqlite3_exec(db.get(), "COMMIT", nullptr, nullptr, nullptr);
    }

    const auto ids = sorted_query_ids(db.get(),
        "SELECT id FROM nn "
        "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 10 "
        "AND allowed_ids = (SELECT bitset_agg(id) FROM many)");
    EXPECT_EQ((std::vector<uint64_t>{10, 20, 30, 40, 50}), ids);
}

TEST_F(BitsetFilterSqlTest, BitsetAggSpillSizedTypedPointerFiltersQuery) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    populate_spill_sized_allowed_table(db.get());

    const auto ids = sorted_query_ids(db.get(),
        "SELECT id FROM nn "
        "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        "AND allowed_ids = (SELECT bitset_agg(id) FROM many)");
    EXPECT_EQ((std::vector<uint64_t>{20, 40}), ids);
}

TEST_F(BitsetFilterSqlTest, NamedSpillSizedBitsetAggPersistsFile) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());
    populate_spill_sized_allowed_table(db.get());

    const std::string name = unique_filter_name("sql_named_spill");
    const fs::path expected_path = named_bitset_filter_path(name);
    fs::remove(expected_path);

    const auto ids = sorted_query_ids(db.get(),
        std::string("SELECT id FROM nn ")
        + "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        + "AND allowed_ids = (SELECT bitset_agg(id, '" + name + "') FROM many)");
    EXPECT_EQ((std::vector<uint64_t>{20, 40}), ids);

    ASSERT_TRUE(fs::exists(expected_path));
    EXPECT_GT(fs::file_size(expected_path), 0u);
    expect_persisted_filter_contains(expected_path, {20, 40});

    fs::remove(expected_path);
}

TEST_F(BitsetFilterSqlTest, BitsetLoadUsesPersistedNamedFilter) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const std::string name = unique_filter_name("sql_load_filter");
    const fs::path expected_path = named_bitset_filter_path(name);
    fs::remove(expected_path);
    ScopedPathCleanup cleanup{expected_path};

    exec_sql(db.get(),
        std::string("SELECT bitset_agg(column1, '") + name
        + "') FROM (VALUES (20), (40))");
    ASSERT_TRUE(fs::exists(expected_path));

    const auto ids = sorted_query_ids(db.get(),
        std::string("SELECT id FROM nn ")
        + "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        + "AND allowed_ids = bitset_load('" + name + "')");
    EXPECT_EQ((std::vector<uint64_t>{20, 40}), ids);
}

TEST_F(BitsetFilterSqlTest, BitsetLoadRejectsInvalidAndMissingNames) {
    SqliteDbPtr db = open_db_with_extension();

    expect_query_error(db.get(), "SELECT bitset_load(NULL)", "name must be a string");
    expect_query_error(db.get(), "SELECT bitset_load('')", "bitset_load: invalid bitset filter name");
    expect_query_error(
        db.get(), "SELECT bitset_load('bad-name')", "bitset_load: invalid bitset filter name");

    const std::string name = unique_filter_name("sql_missing_filter");
    expect_query_error(db.get(), "SELECT bitset_load('" + name + "')", "failed to open");
}

TEST_F(BitsetFilterSqlTest, BitsetLoadCachesWithinPreparedStatement) {
    SqliteDbPtr db = open_db_with_extension();

    const std::string name = unique_filter_name("sql_load_cache_filter");
    const fs::path expected_path = named_bitset_filter_path(name);
    fs::remove(expected_path);
    ScopedPathCleanup cleanup{expected_path};

    exec_sql(db.get(),
        std::string("SELECT bitset_agg(column1, '") + name
        + "') FROM (VALUES (20), (40))");
    ASSERT_TRUE(fs::exists(expected_path));

    sqlite3_stmt* stmt = nullptr;
    const char* sql = "SELECT bitset_load(?) FROM (VALUES (1), (2), (3))";
    ASSERT_EQ(SQLITE_OK, sqlite3_prepare_v2(db.get(), sql, -1, &stmt, nullptr))
        << sqlite3_errmsg(db.get());
    ASSERT_EQ(SQLITE_OK, sqlite3_bind_text(stmt, 1, name.c_str(), -1, SQLITE_TRANSIENT))
        << sqlite3_errmsg(db.get());

    ASSERT_EQ(SQLITE_ROW, sqlite3_step(stmt)) << sqlite3_errmsg(db.get());
    ASSERT_TRUE(fs::remove(expected_path));

    EXPECT_EQ(SQLITE_ROW, sqlite3_step(stmt)) << sqlite3_errmsg(db.get());
    EXPECT_EQ(SQLITE_ROW, sqlite3_step(stmt)) << sqlite3_errmsg(db.get());
    EXPECT_EQ(SQLITE_DONE, sqlite3_step(stmt)) << sqlite3_errmsg(db.get());
    EXPECT_EQ(SQLITE_OK, sqlite3_finalize(stmt)) << sqlite3_errmsg(db.get());
}

TEST_F(BitsetFilterSqlTest, BitsetLoadCacheReplacesWhenArgumentChanges) {
    SqliteDbPtr db = open_db_with_extension();

    const std::string first_name = unique_filter_name("sql_load_cache_first");
    const std::string second_name = unique_filter_name("sql_load_cache_second");
    const fs::path first_path = named_bitset_filter_path(first_name);
    const fs::path second_path = named_bitset_filter_path(second_name);
    fs::remove(first_path);
    fs::remove(second_path);
    ScopedPathCleanup first_cleanup{first_path};
    ScopedPathCleanup second_cleanup{second_path};

    exec_sql(db.get(),
        std::string("SELECT bitset_agg(column1, '") + first_name
        + "') FROM (VALUES (20), (40))");
    exec_sql(db.get(),
        std::string("SELECT bitset_agg(column1, '") + second_name
        + "') FROM (VALUES (10), (50))");
    ASSERT_TRUE(fs::exists(first_path));
    ASSERT_TRUE(fs::exists(second_path));

    sqlite3_stmt* stmt = nullptr;
    const char* sql = "SELECT bitset_load(?) FROM (VALUES (1), (2))";
    ASSERT_EQ(SQLITE_OK, sqlite3_prepare_v2(db.get(), sql, -1, &stmt, nullptr))
        << sqlite3_errmsg(db.get());

    ASSERT_EQ(SQLITE_OK, sqlite3_bind_text(stmt, 1, first_name.c_str(), -1, SQLITE_TRANSIENT))
        << sqlite3_errmsg(db.get());
    ASSERT_EQ(SQLITE_ROW, sqlite3_step(stmt)) << sqlite3_errmsg(db.get());
    ASSERT_TRUE(fs::remove(first_path));
    EXPECT_EQ(SQLITE_ROW, sqlite3_step(stmt)) << sqlite3_errmsg(db.get());
    EXPECT_EQ(SQLITE_DONE, sqlite3_step(stmt)) << sqlite3_errmsg(db.get());

    ASSERT_EQ(SQLITE_OK, sqlite3_reset(stmt)) << sqlite3_errmsg(db.get());
    ASSERT_EQ(SQLITE_OK, sqlite3_bind_text(stmt, 1, second_name.c_str(), -1, SQLITE_TRANSIENT))
        << sqlite3_errmsg(db.get());
    ASSERT_EQ(SQLITE_ROW, sqlite3_step(stmt)) << sqlite3_errmsg(db.get());
    ASSERT_TRUE(fs::remove(second_path));
    EXPECT_EQ(SQLITE_ROW, sqlite3_step(stmt)) << sqlite3_errmsg(db.get());
    EXPECT_EQ(SQLITE_DONE, sqlite3_step(stmt)) << sqlite3_errmsg(db.get());
    EXPECT_EQ(SQLITE_OK, sqlite3_finalize(stmt)) << sqlite3_errmsg(db.get());
}

TEST_F(BitsetFilterSqlTest, BitsetDropRemovesNamedFileAndReturnsStatus) {
    SqliteDbPtr db = open_db_with_extension();

    const std::string name = unique_filter_name("sql_drop_filter");
    const fs::path expected_path = named_bitset_filter_path(name);
    fs::remove(expected_path);

    exec_sql(db.get(),
        std::string("SELECT bitset_agg(column1, '") + name + "') FROM (VALUES (20), (40))");
    ASSERT_TRUE(fs::exists(expected_path));

    EXPECT_EQ(1, query_single_int(db.get(), "SELECT bitset_drop('" + name + "')"));
    EXPECT_FALSE(fs::exists(expected_path));
    EXPECT_EQ(0, query_single_int(db.get(), "SELECT bitset_drop('" + name + "')"));
}

TEST_F(BitsetFilterSqlTest, BitsetDropRejectsInvalidNameArguments) {
    SqliteDbPtr db = open_db_with_extension();

    expect_query_error(db.get(), "SELECT bitset_drop(NULL)", "name must be a string");
    expect_query_error(db.get(), "SELECT bitset_drop('')", "bitset_drop: invalid bitset filter name");
    expect_query_error(
        db.get(), "SELECT bitset_drop('bad-name')", "bitset_drop: invalid bitset filter name");
}

TEST_F(BitsetFilterSqlTest, BitsetAggRejectsEmptyName) {
    SqliteDbPtr db = open_db_with_extension();

    expect_query_error(db.get(),
        "SELECT bitset_agg(column1, '') FROM (VALUES (1))",
        "invalid bitset filter name");
}

TEST_F(BitsetFilterSqlTest, BitsetAggGroupByCollapsedToSingleBitsetFilter) {
    setup_decimal_dataset();
    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    exec_sql(db.get(), "CREATE TABLE buckets(bucket INTEGER, id INTEGER)");
    exec_sql(db.get(),
        "INSERT INTO buckets VALUES (1,10),(1,30),(2,20),(2,40),(2,50)");

    // GROUP BY produces one bitset filter per group; pick bucket=2 with LIMIT 1.
    const auto ids = sorted_query_ids(db.get(),
        "SELECT id FROM nn "
        "WHERE query = '25.0, 25.0, 25.0, 25.0' AND k = 5 "
        "AND allowed_ids = ("
        "  SELECT bitset_agg(id) FROM buckets WHERE bucket = 2 GROUP BY bucket"
        ")");
    EXPECT_EQ((std::vector<uint64_t>{20, 40, 50}), ids);
}

// ---------------------------------------------------------------------------
// Type-validation paths not covered by utest_vlite.cpp.
// ---------------------------------------------------------------------------

TEST_F(BitsetFilterSqlTest, BitsetAggRejectsRealValue) {
    SqliteDbPtr db = open_db_with_extension();
    expect_query_error(db.get(),
        "SELECT bitset_agg(column1) FROM (VALUES (1.5))",
        "must be an integer");
}

TEST_F(BitsetFilterSqlTest, BitsetAggRejectsBlobValue) {
    SqliteDbPtr db = open_db_with_extension();
    expect_query_error(db.get(),
        "SELECT bitset_agg(column1) FROM (VALUES (X'deadbeef'))",
        "must be an integer");
}

TEST_F(BitsetFilterSqlTest, BitsetAggRejectsNegativeFromExpression) {
    SqliteDbPtr db = open_db_with_extension();
    expect_query_error(db.get(),
        "SELECT bitset_agg(column1 - 100) FROM (VALUES (10))",
        "non-negative");
}

TEST_F(BitsetFilterSqlTest, BitsetAggSurfacesErrorMidStream) {
    SqliteDbPtr db = open_db_with_extension();
    // First two ids are valid; the third is negative and must abort the query.
    expect_query_error(db.get(),
        "SELECT bitset_agg(column1) FROM (VALUES (1), (2), (-3), (4))",
        "non-negative");
}

} // namespace
