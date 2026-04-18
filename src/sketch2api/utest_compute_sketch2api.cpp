// End-to-end tests for build-selected calc-engine behavior through sketch2api.

#include "internal.h"
#include "sketch2.h"

#include <cstdlib>
#include <experimental/scope>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include <gtest/gtest.h>

namespace {

#define ASSERT_OK(handle, call_expr) ASSERT_EQ(0, (call_expr)) << sk_error_message(handle)

constexpr const char* kScenarioEnv = "SKETCH2API_COMPUTE_SCENARIO";
constexpr const char* kConfigEnv = "SKETCH2_CONFIG";

#if SKETCH_CALC_ENGINE_HIGHWAY
constexpr const char* kCompiledEngine = "highway";
#elif SKETCH_CALC_ENGINE_NUMKONG
constexpr const char* kCompiledEngine = "numkong";
#else
#error "Exactly one calc engine must be compiled."
#endif

std::filesystem::path make_temp_dir() {
    const std::filesystem::path base = std::filesystem::temp_directory_path();
    std::filesystem::create_directories(base);
    std::string pattern = (base / "sketch2_compute_api_ut_XXXXXX").string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* const dir = mkdtemp(writable.data());
    EXPECT_NE(nullptr, dir);
    return dir == nullptr ? base / "sketch2_compute_api_ut_fallback" : std::filesystem::path(dir);
}

std::vector<uint64_t> api_knn(sk_handle_t* handle, const char* vec, unsigned int k) {
    uint64_t* ids = nullptr;
    size_t count = 0;
    EXPECT_EQ(0, sk_knn(handle, vec, k, &ids, &count)) << sk_error_message(handle);

    std::vector<uint64_t> out;
    if (ids != nullptr) {
        out.assign(ids, ids + count);
        sk_free(ids);
    }
    return out;
}

void set_env_or_unset(const char* name, const char* value) {
    if (value == nullptr || value[0] == '\0') {
        unsetenv(name);
        return;
    }
    ASSERT_EQ(0, setenv(name, value, 1));
}

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

void write_runtime_config(const std::filesystem::path& path) {
    std::ofstream out(path);
    ASSERT_TRUE(out.is_open());
    out << "[log]\n";
    out << "level=INFO\n";
    out << "[thread_pool]\n";
    out << "size=2\n";
}

void populate_dataset(sk_handle_t* handle, const char* dataset_name, const char* dist_func,
        const std::vector<std::pair<uint64_t, std::string>>& rows) {
    ASSERT_OK(handle, sk_create(handle, dataset_name, nullptr, 4, "f32", 1000, dist_func));
    ASSERT_OK(handle, sk_start_writing(handle));
    for (const auto& row : rows) {
        ASSERT_OK(handle, sk_write_vector(handle, row.first, row.second.c_str()));
    }
    ASSERT_OK(handle, sk_complete_writing(handle));
}

void expect_scan_results(sk_handle_t* handle, const char* dataset_name, const char* query,
        const std::vector<uint64_t>& expected_ids) {
    const std::vector<uint64_t> ids = api_knn(handle, query, expected_ids.size());
    ASSERT_EQ(expected_ids.size(), ids.size());
    EXPECT_EQ(expected_ids, ids);
    ASSERT_OK(handle, sk_close(handle));
    ASSERT_OK(handle, sk_drop(handle, dataset_name));
}

void run_compute_chain_assertions() {
    const char* scenario = std::getenv(kScenarioEnv);
    ASSERT_NE(nullptr, scenario);

    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path config_path = root / "sketch2.ini";
    std::experimental::scope_exit cleanup([&]() {
        std::filesystem::remove_all(root);
        unsetenv(kScenarioEnv);
        unsetenv(kConfigEnv);
    });

    bool write_config = false;
    bool use_missing_config_path = false;

    if (std::string_view(scenario) == "config_file_is_ignored_for_engine_selection") {
        write_config = true;
    } else if (std::string_view(scenario) == "missing_defaults_compiled") {
    } else if (std::string_view(scenario) == "missing_config_file_defaults_compiled") {
        use_missing_config_path = true;
    } else {
        FAIL() << "Unknown scenario: " << scenario;
    }

    if (write_config) {
        write_runtime_config(config_path);
        set_env_or_unset(kConfigEnv, config_path.c_str());
    } else if (use_missing_config_path) {
        const std::filesystem::path missing_path = root / "missing-sketch2.ini";
        set_env_or_unset(kConfigEnv, missing_path.c_str());
    } else {
        unsetenv(kConfigEnv);
    }

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(nullptr, handle);
    ASSERT_STREQ(kCompiledEngine, sketch2api::detail::sk_knn_engine_name_for_testing_());

    populate_dataset(handle, "l2_ds", "l2", {
        {10, "0.0, 0.0, 0.0, 0.0"},
        {20, "1.0, 1.0, 1.0, 1.0"},
        {30, "5.0, 5.0, 5.0, 5.0"},
    });
    expect_scan_results(handle, "l2_ds", "0.0, 0.0, 0.0, 0.0", {10, 20, 30});

    populate_dataset(handle, "cos_ds", "cos", {
        {10, "100.0, 1.0, 0.0, 0.0"},
        {20, "1.0, 1.0, 0.0, 0.0"},
        {30, "-1.0, 0.0, 0.0, 0.0"},
    });
    expect_scan_results(handle, "cos_ds", "1.0, 0.0, 0.0, 0.0", {10, 20, 30});

    sk_release_handle(handle);
}

void run_child_scenario(const char* scenario) {
    EnvVarGuard scenario_guard(kScenarioEnv);
    EnvVarGuard config_guard(kConfigEnv);

    ASSERT_EQ(0, setenv(kScenarioEnv, scenario, 1));

    pid_t pid = fork();
    ASSERT_NE(-1, pid);
    if (pid == 0) {
        execl("/proc/self/exe", "/proc/self/exe",
            "--gtest_filter=sketch2api_compute_chain.ChildScenario", nullptr);
        _exit(127);
    }

    int status = 0;
    ASSERT_NE(-1, waitpid(pid, &status, 0));
    ASSERT_TRUE(WIFEXITED(status));
    ASSERT_EQ(0, WEXITSTATUS(status));
}

} // namespace

TEST(sketch2api_compute_chain, ChildScenario) {
    const char* scenario = std::getenv(kScenarioEnv);
    if (scenario == nullptr || scenario[0] == '\0') {
        GTEST_SKIP() << "Scenario is only executed through the parent launcher.";
    }

    run_compute_chain_assertions();
}

TEST(sketch2api_compute_chain, ConfigFileDoesNotAffectCompiledEngine) {
    run_child_scenario("config_file_is_ignored_for_engine_selection");
}

TEST(sketch2api_compute_chain, MissingConfigDefaultsToCompiledEngine) {
    run_child_scenario("missing_defaults_compiled");
}

TEST(sketch2api_compute_chain, MissingConfigFileDefaultsToCompiledEngine) {
    run_child_scenario("missing_config_file_defaults_compiled");
}
