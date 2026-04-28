// Unit tests for the SQLite virtual table integration.

#include <gtest/gtest.h>

#include "dlfcn.h"
#include "sqlite3.h"

#include "sketch2api/sketch2.h"
#include "core/bitset/bitset_filter_control.h"
#include "utils/singleton.h"
#include "utils/shared_types.h"
#include "core/bitset/utest_chunked_bits_helpers.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <tuple>
#include <sys/wait.h>
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

constexpr const char* kScenarioEnv = "SKETCH2API_VLITE_COMPUTE_SCENARIO";

using test::ScopedPathCleanup;
using test::unique_filter_name;

class EnvVarGuard {
public:
    explicit EnvVarGuard(const char* name) : name_(name) {
        const char* current = std::getenv(name_);
        if (current != nullptr) {
            had_original_ = true;
            original_ = current;
        }
    }

    ~EnvVarGuard() {
        if (had_original_) {
            (void)setenv(name_, original_.c_str(), 1);
        } else {
            (void)unsetenv(name_);
        }
    }

private:
    const char* name_;
    bool had_original_ = false;
    std::string original_;
};

class VliteTest : public ::testing::Test {
protected:
    fs::path root_;
    fs::path db_root_;
    fs::path dataset_dir_;
    fs::path input_path_;
    fs::path ini_path_;
    std::string table_name_ = "nn";
    std::string dataset_name_ = "dataset";

    void SetUp() override {
        static uint64_t counter = 0;
        root_ = fs::temp_directory_path() /
            ("sketch2_utest_vlite_" + std::to_string(getpid()) + "_" + std::to_string(++counter));
        fs::create_directories(root_);
        db_root_ = root_ / "db";
        dataset_dir_ = db_root_ / dataset_name_;
        input_path_ = root_ / "input.txt";
        ini_path_ = dataset_dir_ / (dataset_name_ + ".ini");
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

    void write_query_file(const std::string& filename, const std::string& content) {
        std::ofstream out(root_ / filename);
        ASSERT_TRUE(out.is_open());
        out << content;
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

    std::string loaded_knn_engine_name() {
        void* handle = dlopen(VLITE_EXTENSION_PATH, RTLD_NOW | RTLD_LOCAL);
        EXPECT_NE(nullptr, handle);
        if (handle == nullptr) {
            return "";
        }

        using EngineNameFn = const char* (*)();
        auto* fn = reinterpret_cast<EngineNameFn>(dlsym(handle, "sqlite3_sketch2_knn_engine_name_for_testing"));
        EXPECT_NE(nullptr, fn);
        const char* name = fn == nullptr ? nullptr : fn();
        EXPECT_NE(nullptr, name);
        dlclose(handle);
        return name == nullptr ? "" : std::string(name);
    }
};

TEST_F(VliteTest, ReturnsKnnIdsAndScoresForDOTDataset) {
    write_input("f32,4\n"
                "0 : [ 0.1, 0.1, 0.1, 0.1 ]\n"
                "14 : [ 14.1, 14.1, 14.1, 14.1 ]\n"
                "15 : [ 15.1, 15.1, 15.1, 15.1 ]\n"
                "16 : [ 16.1, 16.1, 16.1, 16.1 ]\n"
                "30 : [ 30.1, 30.1, 30.1, 30.1 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto rows = query_results(db.get(),
        "SELECT id, score FROM nn "
        "WHERE query = '15.2, 15.2, 15.2, 15.2' AND k = 3 "
        "ORDER BY score DESC");

    ASSERT_EQ(3u, rows.size());
    EXPECT_EQ(30u, rows[0].first);
    EXPECT_EQ(16u, rows[1].first);
    EXPECT_EQ(15u, rows[2].first);
    EXPECT_NEAR(1830.08, rows[0].second, 1e-4);
    EXPECT_NEAR(978.88, rows[1].second, 1e-4);
    EXPECT_NEAR(918.08, rows[2].second, 1e-4);
}

TEST_F(VliteTest, UsesDatasetScoreFunctionForCosineQueries) {
    write_input("f32,4\n"
                "10 : [ 100.0, 1.0, 0.0, 0.0 ]\n"
                "20 : [ 1.0, 1.0, 0.0, 0.0 ]\n"
                "30 : [ -1.0, 0.0, 0.0, 0.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::COS);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto rows = query_results(db.get(),
        "SELECT id, score FROM nn "
        "WHERE query = '1.0, 0.0, 0.0, 0.0' AND k = 3 "
        "ORDER BY score");

    ASSERT_EQ(3u, rows.size());
    EXPECT_EQ(10u, rows[0].first);
    EXPECT_EQ(20u, rows[1].first);
    EXPECT_EQ(30u, rows[2].first);
    EXPECT_LE(rows[0].second, rows[1].second);
    EXPECT_LE(rows[1].second, rows[2].second);
}

TEST_F(VliteTest, ChildScenario) {
    const char* scenario = std::getenv(kScenarioEnv);
    if (scenario == nullptr || scenario[0] == '\0') {
        GTEST_SKIP() << "Scenario is only executed through the parent launcher.";
    }

    EnvVarGuard scenario_guard(kScenarioEnv);
    write_input("f32,4\n"
                "10 : [ 100.0, 1.0, 0.0, 0.0 ]\n"
                "20 : [ 1.0, 1.0, 0.0, 0.0 ]\n"
                "30 : [ -1.0, 0.0, 0.0, 0.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::COS);

    if (std::string_view(scenario) != "compiled_engine") {
        FAIL() << "Unknown scenario: " << scenario;
    }

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    EXPECT_EQ("highway", loaded_knn_engine_name());

    const auto rows = query_results(db.get(),
        "SELECT id, score FROM nn "
        "WHERE query = '1.0, 0.0, 0.0, 0.0' AND k = 3 "
        "ORDER BY score");

    ASSERT_EQ(3u, rows.size());
    EXPECT_EQ(10u, rows[0].first);
    EXPECT_EQ(20u, rows[1].first);
    EXPECT_EQ(30u, rows[2].first);
}

void run_child_scenario(const char* scenario) {
    EnvVarGuard scenario_guard(kScenarioEnv);

    ASSERT_EQ(0, setenv(kScenarioEnv, scenario, 1));

    pid_t pid = fork();
    ASSERT_NE(-1, pid);
    if (pid == 0) {
        execl("/proc/self/exe", "/proc/self/exe",
            "--gtest_filter=VliteTest.ChildScenario", nullptr);
        _exit(127);
    }

    int status = 0;
    ASSERT_NE(-1, waitpid(pid, &status, 0));
    ASSERT_TRUE(WIFEXITED(status));
    ASSERT_EQ(0, WEXITSTATUS(status));
}

TEST_F(VliteTest, UsesCompiledEngine) {
    run_child_scenario("compiled_engine");
}

TEST_F(VliteTest, UsesDatasetScoreFunctionForL2Queries) {
    write_input("f32,4\n"
                "10 : [ 3.0, 0.0, 0.0, 0.0 ]\n"
                "20 : [ 2.0, 2.0, 0.0, 0.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::L2);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto rows = query_results(db.get(),
        "SELECT id, score FROM nn "
        "WHERE query = '0.0, 0.0, 0.0, 0.0' AND k = 2 "
        "ORDER BY score");

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
    create_dataset(DataType::i16, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto rows = query_results(db.get(),
        "SELECT id, score FROM nn "
        "WHERE query = '10, 10, 10, 10' AND k = 2 "
        "ORDER BY score DESC");

    ASSERT_EQ(2u, rows.size());
    EXPECT_EQ(20u, rows[0].first);
    EXPECT_EQ(10u, rows[1].first);
    EXPECT_NEAR(440.0, rows[0].second, 1e-9);
    EXPECT_NEAR(400.0, rows[1].second, 1e-9);
}

TEST_F(VliteTest, SupportsF16Datasets) {
    write_input("f16,4\n"
                "10 : [ 10.0, 10.0, 10.0, 10.0 ]\n"
                "20 : [ 11.0, 11.0, 11.0, 11.0 ]\n");
    create_dataset(DataType::f16, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto rows = query_results(db.get(),
        "SELECT id, score FROM nn "
        "WHERE query = '10.0, 10.0, 10.0, 10.0' AND k = 2 "
        "ORDER BY score DESC");

    ASSERT_EQ(2u, rows.size());
    EXPECT_EQ(20u, rows[0].first);
    EXPECT_EQ(10u, rows[1].first);
    EXPECT_NEAR(440.0, rows[0].second, 1e-6);
    EXPECT_NEAR(400.0, rows[1].second, 1e-6);
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
        "SELECT id, score FROM nn "
        "WHERE query MATCH '1.0, 0.0, 0.0, 0.0' AND k = 2 "
        "ORDER BY score");

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
    create_dataset(DataType::f32, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    sqlite3_stmt* stmt = nullptr;
    ASSERT_EQ(SQLITE_OK, sqlite3_prepare_v2(
        db.get(),
        "SELECT id, k FROM nn "
        "WHERE query MATCH '15.2, 15.2, 15.2, 15.2' "
        "ORDER BY score DESC LIMIT 2 OFFSET 1",
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
    EXPECT_EQ(15u, rows[1].first);
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
    create_dataset(DataType::f32, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    sqlite3_stmt* stmt = nullptr;
    ASSERT_EQ(SQLITE_OK, sqlite3_prepare_v2(
        db.get(),
        "SELECT id, k FROM nn "
        "WHERE query = '15.2, 15.2, 15.2, 15.2' AND k = 100 "
        "ORDER BY score DESC LIMIT 1",
        -1, &stmt, nullptr));

    ASSERT_EQ(SQLITE_ROW, sqlite3_step(stmt));
    EXPECT_EQ(30, sqlite3_column_int64(stmt, 0));
    EXPECT_EQ(100, sqlite3_column_int64(stmt, 1));
    ASSERT_EQ(SQLITE_DONE, sqlite3_step(stmt));
    ASSERT_EQ(SQLITE_OK, sqlite3_finalize(stmt));
}

TEST_F(VliteTest, ReusesCachedDatasetAcrossQueries) {
    write_input("f32,4\n"
                "10 : [ 10.1, 10.1, 10.1, 10.1 ]\n"
                "11 : [ 11.1, 11.1, 11.1, 11.1 ]\n"
                "12 : [ 12.1, 12.1, 12.1, 12.1 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto first_rows = query_results(db.get(),
        "SELECT id, score FROM nn "
        "WHERE query = '11.2, 11.2, 11.2, 11.2' AND k = 2 "
        "ORDER BY score DESC");
    ASSERT_EQ(2u, first_rows.size());
    EXPECT_EQ(12u, first_rows[0].first);
    EXPECT_EQ(11u, first_rows[1].first);

    ASSERT_TRUE(fs::remove(ini_path_));

    const auto second_rows = query_results(db.get(),
        "SELECT id, score FROM nn "
        "WHERE query = '10.2, 10.2, 10.2, 10.2' AND k = 2 "
        "ORDER BY score DESC");
    ASSERT_EQ(2u, second_rows.size());
    EXPECT_EQ(12u, second_rows[0].first);
    EXPECT_EQ(11u, second_rows[1].first);
}

TEST_F(VliteTest, LargeKReturnsAllAvailableRows) {
    write_input("f32,4\n"
                "10 : [ 10.0, 10.0, 10.0, 10.0 ]\n"
                "11 : [ 11.0, 11.0, 11.0, 11.0 ]\n"
                "12 : [ 12.0, 12.0, 12.0, 12.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto rows = query_results(db.get(),
        "SELECT id, score FROM nn "
        "WHERE query = '10.0, 10.0, 10.0, 10.0' AND k = 100 "
        "ORDER BY score");

    ASSERT_EQ(3u, rows.size());
    EXPECT_EQ(10u, rows[0].first);
    EXPECT_EQ(11u, rows[1].first);
    EXPECT_EQ(12u, rows[2].first);
}

TEST_F(VliteTest, LimitZeroReturnsNoRows) {
    write_input("f32,4\n"
                "10 : [ 10.0, 10.0, 10.0, 10.0 ]\n"
                "11 : [ 11.0, 11.0, 11.0, 11.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto rows = query_results(db.get(),
        "SELECT id, score FROM nn "
        "WHERE query = '10.0, 10.0, 10.0, 10.0' "
        "ORDER BY score LIMIT 0");

    EXPECT_TRUE(rows.empty());
}

TEST_F(VliteTest, FailsWithoutQueryConstraint) {
    write_input("f32,4\n"
                "1 : [ 1.0, 1.0, 1.0, 1.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::DOT);

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
    create_dataset(DataType::f32, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    expect_query_error(db.get(),
        "SELECT id FROM nn WHERE query = '1.0, 1.0, 1.0, 1.0' AND k = 0",
        "k must be > 0");
}

TEST_F(VliteTest, RejectsNegativeK) {
    write_input("f32,4\n"
                "1 : [ 1.0, 1.0, 1.0, 1.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    expect_query_error(db.get(),
        "SELECT id FROM nn WHERE query = '1.0, 1.0, 1.0, 1.0' AND k = -1",
        "k must be > 0");
}

TEST_F(VliteTest, RejectsEmptyQueryString) {
    write_input("f32,4\n"
                "1 : [ 1.0, 1.0, 1.0, 1.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    expect_query_error(db.get(),
        "SELECT id FROM nn WHERE query = '' AND k = 1",
        "non-empty string");
}

TEST_F(VliteTest, RejectsWrongDimensionQueryVector) {
    write_input("f32,4\n"
                "1 : [ 1.0, 1.0, 1.0, 1.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    expect_query_error(db.get(),
        "SELECT id FROM nn WHERE query = '1.0, 1.0, 1.0' AND k = 1",
        "truncated vector payload");
}

TEST_F(VliteTest, RejectsMalformedQueryVectorText) {
    write_input("f32,4\n"
                "1 : [ 1.0, 1.0, 1.0, 1.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    expect_query_error(db.get(),
        "SELECT id FROM nn WHERE query = '1.0, nope, 1.0, 1.0' AND k = 1",
        "invalid f32 token");
}

TEST_F(VliteTest, RejectsIdsOutsideSqliteIntegerRange) {
    const uint64_t large_id = static_cast<uint64_t>(std::numeric_limits<sqlite3_int64>::max()) + 1u;
    write_input("f32,4\n" + std::to_string(large_id) + " : [ 1.0, 1.0, 1.0, 1.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    sqlite3_stmt* stmt = nullptr;
    ASSERT_EQ(SQLITE_OK, sqlite3_prepare_v2(
        db.get(),
        "SELECT id FROM nn "
        "WHERE query = '1.0, 1.0, 1.0, 1.0' AND k = 1 "
        "ORDER BY score",
        -1, &stmt, nullptr));

    const int rc = sqlite3_step(stmt);
    EXPECT_EQ(SQLITE_ERROR, rc);
    EXPECT_NE(std::string(sqlite3_errmsg(db.get())).find("SQLite INTEGER range"), std::string::npos);
    sqlite3_finalize(stmt);
}

TEST_F(VliteTest, SpaceDelimitedQueryWorksForF32L2Dataset) {
    write_input("f32,4\n"
                "10 : [ 3.0, 0.0, 0.0, 0.0 ]\n"
                "20 : [ 2.0, 2.0, 0.0, 0.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::L2);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto rows = query_results(db.get(),
        "SELECT id, score FROM nn "
        "WHERE query = '0.0 0.0 0.0 0.0' AND k = 2 "
        "ORDER BY score");

    ASSERT_EQ(2u, rows.size());
    EXPECT_EQ(20u, rows[0].first);
    EXPECT_EQ(10u, rows[1].first);
    EXPECT_NEAR(8.0, rows[0].second, 1e-9);
    EXPECT_NEAR(9.0, rows[1].second, 1e-9);
}

TEST_F(VliteTest, SpaceDelimitedQueryWorksForI16Dataset) {
    write_input("i16,4\n"
                "10 : [ 10, 10, 10, 10 ]\n"
                "20 : [ 11, 11, 11, 11 ]\n");
    create_dataset(DataType::i16, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto rows = query_results(db.get(),
        "SELECT id, score FROM nn "
        "WHERE query = '10 10 10 10' AND k = 2 "
        "ORDER BY score DESC");

    ASSERT_EQ(2u, rows.size());
    EXPECT_EQ(20u, rows[0].first);
    EXPECT_EQ(10u, rows[1].first);
    EXPECT_NEAR(440.0, rows[0].second, 1e-9);
    EXPECT_NEAR(400.0, rows[1].second, 1e-9);
}

TEST_F(VliteTest, SpaceDelimitedQueryWorksForF16Dataset) {
    write_input("f16,4\n"
                "10 : [ 10.0, 10.0, 10.0, 10.0 ]\n"
                "20 : [ 11.0, 11.0, 11.0, 11.0 ]\n");
    create_dataset(DataType::f16, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto rows = query_results(db.get(),
        "SELECT id, score FROM nn "
        "WHERE query = '10.0 10.0 10.0 10.0' AND k = 2 "
        "ORDER BY score DESC");

    ASSERT_EQ(2u, rows.size());
    EXPECT_EQ(20u, rows[0].first);
    EXPECT_EQ(10u, rows[1].first);
    EXPECT_NEAR(440.0, rows[0].second, 1e-6);
    EXPECT_NEAR(400.0, rows[1].second, 1e-6);
}

TEST_F(VliteTest, SpaceDelimitedQueryWorksWithMatchOperator) {
    write_input("f32,4\n"
                "10 : [ 100.0, 1.0, 0.0, 0.0 ]\n"
                "20 : [ 1.0, 1.0, 0.0, 0.0 ]\n"
                "30 : [ -1.0, 0.0, 0.0, 0.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::COS);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto rows = query_results(db.get(),
        "SELECT id, score FROM nn "
        "WHERE query MATCH '1.0 0.0 0.0 0.0' AND k = 2 "
        "ORDER BY score");

    ASSERT_EQ(2u, rows.size());
    EXPECT_EQ(10u, rows[0].first);
    EXPECT_EQ(20u, rows[1].first);
}

TEST_F(VliteTest, AtPrefixLoadsVectorFromCommaDelimitedFile) {
    write_input("f32,4\n"
                "10 : [ 3.0, 0.0, 0.0, 0.0 ]\n"
                "20 : [ 2.0, 2.0, 0.0, 0.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::L2);
    write_query_file("query.txt", "0.0, 0.0, 0.0, 0.0\n");

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const std::string query_path = "@" + (root_ / "query.txt").string();
    const auto rows = query_results(db.get(),
        "SELECT id, score FROM nn "
        "WHERE query = '" + query_path + "' AND k = 2 "
        "ORDER BY score");

    ASSERT_EQ(2u, rows.size());
    EXPECT_EQ(20u, rows[0].first);
    EXPECT_EQ(10u, rows[1].first);
    EXPECT_NEAR(8.0, rows[0].second, 1e-9);
    EXPECT_NEAR(9.0, rows[1].second, 1e-9);
}

TEST_F(VliteTest, AtPrefixLoadsVectorFromSpaceDelimitedFile) {
    write_input("f32,4\n"
                "10 : [ 3.0, 0.0, 0.0, 0.0 ]\n"
                "20 : [ 2.0, 2.0, 0.0, 0.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::L2);
    write_query_file("query.txt", "0.0 0.0 0.0 0.0\n");

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const std::string query_path = "@" + (root_ / "query.txt").string();
    const auto rows = query_results(db.get(),
        "SELECT id, score FROM nn "
        "WHERE query = '" + query_path + "' AND k = 2 "
        "ORDER BY score");

    ASSERT_EQ(2u, rows.size());
    EXPECT_EQ(20u, rows[0].first);
    EXPECT_EQ(10u, rows[1].first);
    EXPECT_NEAR(8.0, rows[0].second, 1e-9);
    EXPECT_NEAR(9.0, rows[1].second, 1e-9);
}

TEST_F(VliteTest, AtPrefixWorksWithI16Dataset) {
    write_input("i16,4\n"
                "10 : [ 10, 10, 10, 10 ]\n"
                "20 : [ 11, 11, 11, 11 ]\n");
    create_dataset(DataType::i16, 4, 100, DistFunc::DOT);
    write_query_file("query.txt", "10, 10, 10, 10\n");

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const std::string query_path = "@" + (root_ / "query.txt").string();
    const auto rows = query_results(db.get(),
        "SELECT id, score FROM nn "
        "WHERE query = '" + query_path + "' AND k = 2 "
        "ORDER BY score DESC");

    ASSERT_EQ(2u, rows.size());
    EXPECT_EQ(20u, rows[0].first);
    EXPECT_EQ(10u, rows[1].first);
    EXPECT_NEAR(440.0, rows[0].second, 1e-9);
    EXPECT_NEAR(400.0, rows[1].second, 1e-9);
}

TEST_F(VliteTest, AtPrefixWorksWithF16Dataset) {
    write_input("f16,4\n"
                "10 : [ 10.0, 10.0, 10.0, 10.0 ]\n"
                "20 : [ 11.0, 11.0, 11.0, 11.0 ]\n");
    create_dataset(DataType::f16, 4, 100, DistFunc::DOT);
    write_query_file("query.txt", "10.0, 10.0, 10.0, 10.0\n");

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const std::string query_path = "@" + (root_ / "query.txt").string();
    const auto rows = query_results(db.get(),
        "SELECT id, score FROM nn "
        "WHERE query = '" + query_path + "' AND k = 2 "
        "ORDER BY score DESC");

    ASSERT_EQ(2u, rows.size());
    EXPECT_EQ(20u, rows[0].first);
    EXPECT_EQ(10u, rows[1].first);
    EXPECT_NEAR(440.0, rows[0].second, 1e-6);
    EXPECT_NEAR(400.0, rows[1].second, 1e-6);
}

TEST_F(VliteTest, AtPrefixWorksWithMatchOperator) {
    write_input("f32,4\n"
                "10 : [ 100.0, 1.0, 0.0, 0.0 ]\n"
                "20 : [ 1.0, 1.0, 0.0, 0.0 ]\n"
                "30 : [ -1.0, 0.0, 0.0, 0.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::COS);
    write_query_file("query.txt", "1.0, 0.0, 0.0, 0.0\n");

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const std::string query_path = "@" + (root_ / "query.txt").string();
    const auto rows = query_results(db.get(),
        "SELECT id, score FROM nn "
        "WHERE query MATCH '" + query_path + "' AND k = 2 "
        "ORDER BY score");

    ASSERT_EQ(2u, rows.size());
    EXPECT_EQ(10u, rows[0].first);
    EXPECT_EQ(20u, rows[1].first);
}

TEST_F(VliteTest, AtPrefixNonExistentFileReturnsError) {
    write_input("f32,4\n"
                "1 : [ 1.0, 1.0, 1.0, 1.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const std::string query_path = "@" + (root_ / "no_such_file.txt").string();
    expect_query_error(db.get(),
        "SELECT id FROM nn WHERE query = '" + query_path + "' AND k = 1",
        "failed to open file");
}

TEST_F(VliteTest, AtPrefixWrongDimensionFileReturnsError) {
    write_input("f32,4\n"
                "1 : [ 1.0, 1.0, 1.0, 1.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::DOT);
    write_query_file("query.txt", "1.0, 1.0, 1.0\n");

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const std::string query_path = "@" + (root_ / "query.txt").string();
    expect_query_error(db.get(),
        "SELECT id FROM nn WHERE query = '" + query_path + "' AND k = 1",
        "truncated vector payload");
}

TEST_F(VliteTest, AtPrefixMalformedVectorFileReturnsError) {
    write_input("f32,4\n"
                "1 : [ 1.0, 1.0, 1.0, 1.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::DOT);
    write_query_file("query.txt", "1.0, nope, 1.0, 1.0\n");

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const std::string query_path = "@" + (root_ / "query.txt").string();
    expect_query_error(db.get(),
        "SELECT id FROM nn WHERE query = '" + query_path + "' AND k = 1",
        "invalid f32 token");
}

TEST_F(VliteTest, BitsetAggBuildsBitsetFilterFromUnsortedIds) {
    write_input("f32,4\n"
                "10 : [ 10.0, 10.0, 10.0, 10.0 ]\n"
                "11 : [ 11.0, 11.0, 11.0, 11.0 ]\n"
                "18 : [ 18.0, 18.0, 18.0, 18.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    ASSERT_NE(nullptr, db);
    create_virtual_table(db.get());

    const auto rows = query_results(db.get(),
        "SELECT id, score FROM nn "
        "WHERE query = '18.0, 18.0, 18.0, 18.0' AND k = 3 "
        "AND allowed_ids = ("
        "  SELECT bitset_agg(id) "
        "  FROM (SELECT 18 AS id UNION ALL SELECT 10 UNION ALL SELECT NULL "
        "        UNION ALL SELECT 18)"
        ")");

    ASSERT_EQ(2u, rows.size());
    std::vector<uint64_t> ids{rows[0].first, rows[1].first};
    std::sort(ids.begin(), ids.end());
    EXPECT_EQ((std::vector<uint64_t>{10, 18}), ids);
}

TEST_F(VliteTest, BitsetAggRejectsInvalidInputValues) {
    SqliteDbPtr db = open_db_with_extension();
    ASSERT_NE(nullptr, db);

    expect_query_error(db.get(),
        "SELECT bitset_agg(id) FROM (SELECT -1 AS id)",
        "non-negative");
    expect_query_error(db.get(),
        "SELECT bitset_agg(id) FROM (SELECT 'oops' AS id)",
        "must be an integer");
    expect_query_error(db.get(),
        "SELECT bitset_agg(id, 123) FROM (SELECT 1 AS id)",
        "parameter must be a string");
}

TEST_F(VliteTest, BitsetAggNamedParameterPersistsFile) {
    write_input("f32,4\n"
                "10 : [ 10.0, 10.0, 10.0, 10.0 ]\n"
                "20 : [ 20.0, 20.0, 20.0, 20.0 ]\n"
                "30 : [ 30.0, 30.0, 30.0, 30.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    ASSERT_NE(nullptr, db);
    create_virtual_table(db.get());

    const std::string name = unique_filter_name("vlite_named_filter");
    const fs::path expected_path = named_bitset_filter_path(name);
    fs::remove(expected_path);
    ScopedPathCleanup cleanup{expected_path};

    const auto rows = query_results(db.get(),
        std::string("SELECT id, score FROM nn ")
        + "WHERE query = '20.0, 20.0, 20.0, 20.0' AND k = 3 "
        + "AND allowed_ids = (SELECT bitset_agg(id, '" + name + "') "
        + "                   FROM (SELECT 20 AS id UNION ALL SELECT 10))");

    ASSERT_EQ(2u, rows.size());
    std::vector<uint64_t> ids{rows[0].first, rows[1].first};
    std::sort(ids.begin(), ids.end());
    EXPECT_EQ((std::vector<uint64_t>{10, 20}), ids);
    EXPECT_TRUE(fs::exists(expected_path));
}

TEST_F(VliteTest, BitsetAggRejectsInconsistentNames) {
    SqliteDbPtr db = open_db_with_extension();
    ASSERT_NE(nullptr, db);

    const std::string first_name = unique_filter_name("vlite_first_filter");
    const std::string second_name = unique_filter_name("vlite_second_filter");
    const fs::path first_path = named_bitset_filter_path(first_name);
    const fs::path second_path = named_bitset_filter_path(second_name);
    fs::remove(first_path);
    fs::remove(second_path);
    ScopedPathCleanup first_cleanup{first_path};
    ScopedPathCleanup second_cleanup{second_path};

    expect_query_error(db.get(),
        std::string("SELECT bitset_agg(id, name) FROM (")
        + "SELECT 1 AS id, '" + first_name + "' AS name "
        + "UNION ALL "
        + "SELECT 2 AS id, '" + second_name + "' AS name)",
        "inconsistent bitset filter name");

    EXPECT_FALSE(fs::exists(first_path));
    EXPECT_FALSE(fs::exists(second_path));
}

TEST_F(VliteTest, BitsetAggAcceptsDescendingIds) {
    write_input("f32,4\n"
                "1 : [ 1.0, 1.0, 1.0, 1.0 ]\n"
                "2 : [ 2.0, 2.0, 2.0, 2.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    ASSERT_NE(nullptr, db);
    create_virtual_table(db.get());

    const auto rows = query_results(db.get(),
        "SELECT id, score FROM nn "
        "WHERE query = '2.0, 2.0, 2.0, 2.0' AND k = 2 "
        "AND allowed_ids = (SELECT bitset_agg(id) FROM (SELECT 2 AS id UNION ALL SELECT 1))");

    ASSERT_EQ(2u, rows.size());
    std::vector<uint64_t> ids{rows[0].first, rows[1].first};
    std::sort(ids.begin(), ids.end());
    EXPECT_EQ((std::vector<uint64_t>{1, 2}), ids);
}

TEST_F(VliteTest, AllowedIdsBitsetAggConstraintFiltersResults) {
    write_input("f32,4\n"
                "0 : [ 0.0, 0.0, 0.0, 0.0 ]\n"
                "1 : [ 1.0, 1.0, 1.0, 1.0 ]\n"
                "2 : [ 2.0, 2.0, 2.0, 2.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto baseline_rows = query_results(db.get(),
        "SELECT id, score FROM nn "
        "WHERE match_expr MATCH '2.1, 2.1, 2.1, 2.1' AND k = 3 "
        "ORDER BY score");

    const auto filtered_rows = query_results(db.get(),
        "SELECT id, score FROM nn "
        "WHERE match_expr MATCH '2.1, 2.1, 2.1, 2.1' AND k = 3 "
        "AND allowed_ids = (SELECT bitset_agg(id) FROM (SELECT 2 AS id UNION ALL SELECT 0)) "
        "ORDER BY score");

    ASSERT_EQ(3u, baseline_rows.size());
    ASSERT_EQ(2u, filtered_rows.size());
    std::vector<uint64_t> ids{filtered_rows[0].first, filtered_rows[1].first};
    std::sort(ids.begin(), ids.end());
    EXPECT_EQ((std::vector<uint64_t>{0, 2}), ids);
}

TEST_F(VliteTest, AllowedIdsBitsetAggConstraintSupportsNonZeroIds) {
    write_input("f32,4\n"
                "10 : [ 10.0, 10.0, 10.0, 10.0 ]\n"
                "20 : [ 20.0, 20.0, 20.0, 20.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto filtered_rows = query_results(db.get(),
        "SELECT id, score FROM nn "
        "WHERE query = '10.0, 10.0, 10.0, 10.0' AND k = 2 "
        "AND allowed_ids = (SELECT bitset_agg(id) FROM (SELECT 20 AS id UNION ALL SELECT 10))");

    ASSERT_EQ(2u, filtered_rows.size());
    std::vector<uint64_t> ids{filtered_rows[0].first, filtered_rows[1].first};
    std::sort(ids.begin(), ids.end());
    EXPECT_EQ((std::vector<uint64_t>{10, 20}), ids);
}

TEST_F(VliteTest, AllowedIdsEmptyBitsetMatchesNothing) {
    write_input("f32,4\n"
                "10 : [ 10.0, 10.0, 10.0, 10.0 ]\n"
                "20 : [ 20.0, 20.0, 20.0, 20.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto rows = query_results(db.get(),
        "SELECT id, score FROM nn "
        "WHERE query = '10.0, 10.0, 10.0, 10.0' AND k = 2 "
        "AND allowed_ids = (SELECT bitset_agg(id) FROM (SELECT 1 AS id WHERE 0))");

    EXPECT_TRUE(rows.empty());
}

TEST_F(VliteTest, AllowedIdsBitsetAggConstraintFiltersAcrossChunks) {
    constexpr uint64_t kCrossChunkId = (1ull << 20u) + 5u;
    write_input(
        std::string("f32,4\n")
        + "1 : [ 1.0, 1.0, 1.0, 1.0 ]\n"
        + std::to_string(kCrossChunkId) + " : [ 5.0, 5.0, 5.0, 5.0 ]\n"
        + "9 : [ 9.0, 9.0, 9.0, 9.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto rows = query_results(db.get(),
        "SELECT id, score FROM nn "
        "WHERE query = '5.0, 5.0, 5.0, 5.0' AND k = 3 "
        "AND allowed_ids = (SELECT bitset_agg(id) FROM (SELECT 1 AS id UNION ALL SELECT " +
            std::to_string(kCrossChunkId) + "))");

    ASSERT_EQ(2u, rows.size());
    std::vector<uint64_t> ids{rows[0].first, rows[1].first};
    std::sort(ids.begin(), ids.end());
    EXPECT_EQ((std::vector<uint64_t>{1u, kCrossChunkId}), ids);
}

TEST_F(VliteTest, AllowedIdsNullIsTreatedAsNoFilter) {
    write_input("f32,4\n"
                "10 : [ 10.0, 10.0, 10.0, 10.0 ]\n"
                "20 : [ 20.0, 20.0, 20.0, 20.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    const auto baseline_rows = query_results(db.get(),
        "SELECT id, score FROM nn "
        "WHERE query = '10.0, 10.0, 10.0, 10.0' AND k = 2 "
        "ORDER BY score");

    const auto null_rows = query_results(db.get(),
        "SELECT id, score FROM nn "
        "WHERE query = '10.0, 10.0, 10.0, 10.0' AND k = 2 "
        "AND allowed_ids = (SELECT CAST(NULL AS BLOB)) "
        "ORDER BY score");

    ASSERT_EQ(baseline_rows.size(), null_rows.size());
    for (size_t i = 0; i < baseline_rows.size(); ++i) {
        EXPECT_EQ(baseline_rows[i].first, null_rows[i].first);
        EXPECT_DOUBLE_EQ(baseline_rows[i].second, null_rows[i].second);
    }
}

TEST_F(VliteTest, AllowedIdsRejectsNonBlobValues) {
    write_input("f32,4\n"
                "10 : [ 10.0, 10.0, 10.0, 10.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    expect_query_error(db.get(),
        "SELECT id FROM nn "
        "WHERE query = '10.0, 10.0, 10.0, 10.0' AND k = 1 "
        "AND allowed_ids = 'not_a_blob'",
        "allowed_ids must be a BLOB or NULL");
}

TEST_F(VliteTest, AllowedIdsRejectsEmptyBlob) {
    write_input("f32,4\n"
                "10 : [ 10.0, 10.0, 10.0, 10.0 ]\n");
    create_dataset(DataType::f32, 4, 100, DistFunc::DOT);

    SqliteDbPtr db = open_db_with_extension();
    create_virtual_table(db.get());

    expect_query_error(db.get(),
        "SELECT id FROM nn "
        "WHERE query = '10.0, 10.0, 10.0, 10.0' AND k = 1 "
        "AND allowed_ids = X''",
        "allowed_ids BLOB must not be empty");
}

} // namespace
