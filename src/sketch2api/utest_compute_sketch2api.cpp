// End-to-end tests for compute.engine selection through sketch2api.

#include "internal.h"
#include "sketch2.h"
#include "core/utils/compute_unit.h"

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
constexpr const char* kComputeEngineEnv = "SKETCH2_COMPUTE_ENGINE";

const char* unsupported_legacy_backend_name() {
    struct Candidate {
        const char* name;
        sketch2::ComputeBackendKind kind;
    };

    constexpr Candidate kCandidates[] = {
        {"avx512_vnni", sketch2::ComputeBackendKind::avx512_vnni},
        {"avx512f", sketch2::ComputeBackendKind::avx512f},
        {"avx2", sketch2::ComputeBackendKind::avx2},
        {"neon", sketch2::ComputeBackendKind::neon},
    };

    for (const Candidate& candidate : kCandidates) {
        if (!sketch2::ComputeUnit::is_supported(candidate.kind)) {
            return candidate.name;
        }
    }
    return nullptr;
}

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

void write_compute_config(const std::filesystem::path& path, const char* engine) {
    std::ofstream out(path);
    ASSERT_TRUE(out.is_open());
    out << "[compute]\n";
    out << "engine=" << engine << "\n";
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
        unsetenv(kComputeEngineEnv);
    });

    std::string expected_engine = "compute";
    bool write_config = false;
    bool use_missing_config_path = false;
    std::string config_engine;
    std::string env_engine;
    const char* unsupported_backend = unsupported_legacy_backend_name();
    ASSERT_NE(nullptr, unsupported_backend);

    if (std::string_view(scenario) == "config_highway") {
        write_config = true;
        config_engine = "highway";
        expected_engine = "highway";
    } else if (std::string_view(scenario) == "config_numkong") {
        write_config = true;
        config_engine = "numkong";
        expected_engine = "numkong";
    } else if (std::string_view(scenario) == "env_highway") {
        env_engine = "highway";
        expected_engine = "highway";
    } else if (std::string_view(scenario) == "env_numkong") {
        env_engine = "numkong";
        expected_engine = "numkong";
    } else if (std::string_view(scenario) == "env_overrides_config") {
        write_config = true;
        config_engine = "highway";
        env_engine = "numkong";
        expected_engine = "numkong";
    } else if (std::string_view(scenario) == "config_auto") {
        write_config = true;
        config_engine = "auto";
    } else if (std::string_view(scenario) == "env_auto") {
        env_engine = "auto";
    } else if (std::string_view(scenario) == "env_auto_overrides_config_highway") {
        write_config = true;
        config_engine = "highway";
        env_engine = "auto";
    } else if (std::string_view(scenario) == "env_auto_overrides_config_numkong") {
        write_config = true;
        config_engine = "numkong";
        env_engine = "auto";
    } else if (std::string_view(scenario) == "missing_defaults_compute") {
    } else if (std::string_view(scenario) == "invalid_is_advisory") {
        write_config = true;
        config_engine = "bogus";
    } else if (std::string_view(scenario) == "invalid_env_is_advisory") {
        env_engine = "bogus";
    } else if (std::string_view(scenario) == "unsupported_config_is_advisory") {
        write_config = true;
        config_engine = unsupported_backend;
    } else if (std::string_view(scenario) == "unsupported_env_is_advisory") {
        env_engine = unsupported_backend;
    } else if (std::string_view(scenario) == "missing_config_file_defaults_compute") {
        use_missing_config_path = true;
    } else if (std::string_view(scenario) == "env_overrides_missing_config_file") {
        use_missing_config_path = true;
        env_engine = "highway";
        expected_engine = "highway";
    } else {
        FAIL() << "Unknown scenario: " << scenario;
    }

    if (write_config) {
        write_compute_config(config_path, config_engine.c_str());
        set_env_or_unset(kConfigEnv, config_path.c_str());
    } else if (use_missing_config_path) {
        const std::filesystem::path missing_path = root / "missing-sketch2.ini";
        set_env_or_unset(kConfigEnv, missing_path.c_str());
    } else {
        unsetenv(kConfigEnv);
    }
    set_env_or_unset(kComputeEngineEnv, env_engine.empty() ? nullptr : env_engine.c_str());

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(nullptr, handle);
    ASSERT_STREQ(expected_engine.c_str(), sketch2api::detail::sk_knn_engine_name_for_testing_());

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
    EnvVarGuard engine_guard(kComputeEngineEnv);

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

TEST(sketch2api_compute_chain, ConfigHighway) {
    run_child_scenario("config_highway");
}

TEST(sketch2api_compute_chain, ConfigNumKong) {
    run_child_scenario("config_numkong");
}

TEST(sketch2api_compute_chain, EnvHighway) {
    run_child_scenario("env_highway");
}

TEST(sketch2api_compute_chain, EnvNumKong) {
    run_child_scenario("env_numkong");
}

TEST(sketch2api_compute_chain, EnvOverridesConfig) {
    run_child_scenario("env_overrides_config");
}

TEST(sketch2api_compute_chain, ConfigAutoUsesComputeScanner) {
    run_child_scenario("config_auto");
}

TEST(sketch2api_compute_chain, EnvAutoUsesComputeScanner) {
    run_child_scenario("env_auto");
}

TEST(sketch2api_compute_chain, EnvAutoOverridesConfigHighway) {
    run_child_scenario("env_auto_overrides_config_highway");
}

TEST(sketch2api_compute_chain, EnvAutoOverridesConfigNumKong) {
    run_child_scenario("env_auto_overrides_config_numkong");
}

TEST(sketch2api_compute_chain, MissingConfigDefaultsToComputeScanner) {
    run_child_scenario("missing_defaults_compute");
}

TEST(sketch2api_compute_chain, InvalidConfigRemainsAdvisory) {
    run_child_scenario("invalid_is_advisory");
}

TEST(sketch2api_compute_chain, InvalidEnvRemainsAdvisory) {
    run_child_scenario("invalid_env_is_advisory");
}

TEST(sketch2api_compute_chain, UnsupportedConfigRemainsAdvisory) {
    run_child_scenario("unsupported_config_is_advisory");
}

TEST(sketch2api_compute_chain, UnsupportedEnvRemainsAdvisory) {
    run_child_scenario("unsupported_env_is_advisory");
}

TEST(sketch2api_compute_chain, MissingConfigFileDefaultsToComputeScanner) {
    run_child_scenario("missing_config_file_defaults_compute");
}

TEST(sketch2api_compute_chain, EnvOverridesMissingConfigFile) {
    run_child_scenario("env_overrides_missing_config_file");
}
