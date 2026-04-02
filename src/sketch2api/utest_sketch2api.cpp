// Unit tests for the public sketch2api API.

#include "sketch2api.h"

#include "storage/input_generator.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include <gtest/gtest.h>

namespace {

#define ASSERT_OK(handle, call_expr) ASSERT_EQ(0, (call_expr)) << sk_error_message(handle)
#define EXPECT_OK(handle, call_expr) EXPECT_EQ(0, (call_expr)) << sk_error_message(handle)

std::filesystem::path make_temp_dir() {
    const std::filesystem::path base = std::filesystem::temp_directory_path();
    std::filesystem::create_directories(base);
    std::string pattern = (base / "sketch2_parasol_ut_XXXXXX").string();
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* const dir = mkdtemp(writable.data());
    EXPECT_NE(nullptr, dir);
    return dir == nullptr ? base / "sketch2_parasol_ut_fallback" : std::filesystem::path(dir);
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path);
    return std::string((std::istreambuf_iterator<char>(in)),
        std::istreambuf_iterator<char>());
}

std::string api_get(sk_handle_t* handle, uint64_t id) {
    char* value = nullptr;
    EXPECT_EQ(0, sk_get(handle, id, &value)) << sk_error_message(handle);
    if (value == nullptr) {
        return "";
    }

    std::string out(value);
    sk_free(value);
    return out;
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

} // namespace

TEST(sketch2api, create_open_close_drop_lifecycle) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);

    ASSERT_OK(handle, sk_create(handle, "dataset", nullptr, 4, "f32", 1000, "l1"));
    const std::filesystem::path dataset_dir = root / "dataset";
    EXPECT_TRUE(std::filesystem::exists(dataset_dir));
    EXPECT_TRUE(std::filesystem::exists(dataset_dir / "dataset.ini"));
    EXPECT_TRUE(std::filesystem::exists(dataset_dir / "dataset.lock"));

    const std::string ini = read_file(dataset_dir / "dataset.ini");
    EXPECT_NE(ini.find("[dataset]\n"), std::string::npos);
    EXPECT_NE(ini.find("dirs=" + (root / "dataset").string() + "\n"), std::string::npos);
    EXPECT_NE(ini.find("range_size=1000\n"), std::string::npos);
    EXPECT_NE(ini.find("dim=4\n"), std::string::npos);
    EXPECT_NE(ini.find("type=f32\n"), std::string::npos);
    EXPECT_NE(ini.find("dist_func=l1\n"), std::string::npos);

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "dataset"));

    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, reopen_restores_pending_wal_for_get_and_knn) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);

    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "l1"));
    const std::filesystem::path input_path = root / "vectors.txt";
    {
        std::ofstream out(input_path);
        out << "f32,4\n";
        out << "1 : [ 0.0, 0.0, 0.0, 0.0 ]\n";
        out << "2 : [ 10.0, 10.0, 10.0, 10.0 ]\n";
    }
    ASSERT_OK(handle, sk_load_file(handle, input_path.string().c_str()));
    ASSERT_OK(handle, sk_close(handle));

    ASSERT_OK(handle, sk_open(handle, "ds"));

    EXPECT_EQ("[ 0.00, 0.00, 0.00, 0.00 ]", api_get(handle, 1));

    const std::vector<uint64_t> ids = api_knn(handle, "0.0, 0.0, 0.0, 0.0", 2);
    ASSERT_EQ(2u, ids.size());
    EXPECT_EQ(1u, ids[0]);
    EXPECT_EQ(2u, ids[1]);

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));

    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, generate_stats_and_print_smoke) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);

    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "l1"));
    ASSERT_OK(handle, sk_generate(handle, 8, 10, 0));

    testing::internal::CaptureStdout();
    ASSERT_OK(handle, sk_stats(handle));
    const std::string stats_out = testing::internal::GetCapturedStdout();
    EXPECT_NE(stats_out.find("dataset:"), std::string::npos);
    EXPECT_NE(stats_out.find("Name: ds"), std::string::npos);
    EXPECT_NE(stats_out.find("Type: f32"), std::string::npos);
    EXPECT_NE(stats_out.find("Dist: l1"), std::string::npos);
    EXPECT_NE(stats_out.find("Dim: 4"), std::string::npos);
    EXPECT_NE(stats_out.find("Range: 1000"), std::string::npos);
    EXPECT_NE(stats_out.find(".data:"), std::string::npos);

    testing::internal::CaptureStdout();
    ASSERT_OK(handle, sk_print(handle));
    const std::string print_out = testing::internal::GetCapturedStdout();
    EXPECT_NE(print_out.find("10 : ["), std::string::npos);

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));

    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, generate_bin_creates_and_loads_binary_input) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);

    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "l1"));
    ASSERT_OK(handle, sk_generate_bin(handle, 8, 10, 0));

    EXPECT_NE(api_get(handle, 10).find("[ 10.1"), std::string::npos);

    const std::vector<uint64_t> ids = api_knn(handle, "10.0, 10.0, 10.0, 10.0", 1);
    ASSERT_EQ(1u, ids.size());
    EXPECT_EQ(10u, ids[0]);

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));

    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, load_file_accepts_binary_input) {
    const std::filesystem::path root = make_temp_dir();
    const std::filesystem::path input_path = root / "input.bin";

    sketch2::GeneratorConfig cfg;
    cfg.pattern_type = sketch2::PatternType::Sequential;
    cfg.count = 3;
    cfg.min_id = 20;
    cfg.type = sketch2::DataType::f32;
    cfg.dim = 4;
    cfg.max_val = 1000;
    cfg.binary = true;
    ASSERT_EQ(0, sketch2::generate_input_file(input_path.string(), cfg).code());

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);

    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "l1"));
    ASSERT_OK(handle, sk_load_file(handle, input_path.string().c_str()));

    EXPECT_NE(api_get(handle, 20).find("[ 20.1"), std::string::npos);

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));

    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, close_requires_open_dataset) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);

    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "l1"));
    EXPECT_OK(handle, sk_close(handle)); // sk_create() creates AND opens a dataset
    ASSERT_OK(handle, sk_open(handle, "ds"));
    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));

    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, create_rejects_invalid_distance_function) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);

    EXPECT_NE(0, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "cosine"));

    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, drop_waits_for_dataset_owner_lock) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "l1"));

    int pipefd[2];
    ASSERT_EQ(0, pipe(pipefd));

    const pid_t pid = fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        close(pipefd[0]);
        sk_handle_t* child = sk_new_handle(root.string().c_str());
        if (child == nullptr) {
            _exit(10);
        }
        if (sk_open(child, "ds") != 0) {
            _exit(11);
        }
        if (sk_generate(child, 1, 0, 0) != 0) {
            _exit(12);
        }
        const char ready = '1';
        if (write(pipefd[1], &ready, 1) != 1) {
            _exit(13);
        }
        usleep(300000);
        if (sk_close(child) != 0) {
            _exit(14);
        }
        sk_release_handle(child);
        close(pipefd[1]);
        _exit(0);
    }

    close(pipefd[1]);
    char ready = 0;
    ASSERT_EQ(1, read(pipefd[0], &ready, 1));
    close(pipefd[0]);

    const auto started = std::chrono::steady_clock::now();
    ASSERT_OK(handle, sk_drop(handle, "ds"));
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started);
    EXPECT_GE(elapsed.count(), 200);

    int status = 0;
    ASSERT_EQ(pid, waitpid(pid, &status, 0));
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(0, WEXITSTATUS(status));

    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, get_rejects_null_output_parameter) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "l1"));
    EXPECT_NE(0, sk_get(handle, 1, nullptr));

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));
    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, knn_rejects_null_output_parameters) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "l1"));
    EXPECT_NE(0, sk_knn(handle, "0.0, 0.0, 0.0, 0.0", 1, nullptr, nullptr));

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));
    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, staged_write_creates_data_and_removes_input_file) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "l1"));

    const std::filesystem::path input_path = root / "ds" / "ds.input";
    ASSERT_OK(handle, sk_start_writing(handle));
    EXPECT_TRUE(std::filesystem::exists(input_path));
    ASSERT_OK(handle, sk_write_vector(handle, 10, "10.1, 10.1, 10.1, 10.1"));
    ASSERT_OK(handle, sk_write_vector(handle, 11, "11.1 11.1 11.1 11.1"));
    ASSERT_OK(handle, sk_complete_writing(handle));

    EXPECT_FALSE(std::filesystem::exists(input_path));
    EXPECT_NE(api_get(handle, 10).find("[ 10.1"), std::string::npos);
    EXPECT_NE(api_get(handle, 11).find("[ 11.1"), std::string::npos);

    const std::vector<uint64_t> ids = api_knn(handle, "10.0, 10.0, 10.0, 10.0", 2);
    ASSERT_EQ(2u, ids.size());
    EXPECT_EQ(10u, ids[0]);
    EXPECT_EQ(11u, ids[1]);

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));
    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, staged_write_delete_hides_existing_item) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "l1"));

    ASSERT_OK(handle, sk_start_writing(handle));
    ASSERT_OK(handle, sk_write_vector(handle, 10, "10.1, 10.1, 10.1, 10.1"));
    ASSERT_OK(handle, sk_complete_writing(handle));

    ASSERT_OK(handle, sk_start_writing(handle));
    ASSERT_OK(handle, sk_write_deleted(handle, 10));
    ASSERT_OK(handle, sk_complete_writing(handle));

    char* value = nullptr;
    EXPECT_NE(0, sk_get(handle, 10, &value));
    EXPECT_EQ(nullptr, value);

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));
    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, staged_write_rejects_invalid_call_order) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "l1"));

    EXPECT_NE(0, sk_write_vector(handle, 10, "10.1, 10.1, 10.1, 10.1"));
    EXPECT_NE(0, sk_write_deleted(handle, 10));
    EXPECT_NE(0, sk_abort_writing(handle));
    EXPECT_NE(0, sk_complete_writing(handle));

    ASSERT_OK(handle, sk_start_writing(handle));
    EXPECT_NE(0, sk_start_writing(handle));

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));
    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, staged_abort_removes_input_file_and_allows_restart) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "l1"));

    const std::filesystem::path input_path = root / "ds" / "ds.input";

    ASSERT_OK(handle, sk_start_writing(handle));
    ASSERT_OK(handle, sk_write_vector(handle, 10, "10.1, 10.1, 10.1, 10.1"));
    ASSERT_TRUE(std::filesystem::exists(input_path));
    ASSERT_OK(handle, sk_abort_writing(handle));
    EXPECT_FALSE(std::filesystem::exists(input_path));

    ASSERT_OK(handle, sk_start_writing(handle));
    ASSERT_OK(handle, sk_write_vector(handle, 11, "11.1, 11.1, 11.1, 11.1"));
    ASSERT_OK(handle, sk_complete_writing(handle));

    char* value = nullptr;
    EXPECT_NE(0, sk_get(handle, 10, &value));
    EXPECT_EQ(nullptr, value);
    EXPECT_NE(api_get(handle, 11).find("[ 11.1"), std::string::npos);

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));
    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, staged_write_many_vectors_then_knn) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "l1"));

    ASSERT_OK(handle, sk_start_writing(handle));
    for (uint64_t id = 0; id < 100; ++id) {
        const std::string value = std::to_string(static_cast<float>(id)) + ", " +
            std::to_string(static_cast<float>(id)) + ", " +
            std::to_string(static_cast<float>(id)) + ", " +
            std::to_string(static_cast<float>(id));
        ASSERT_OK(handle, sk_write_vector(handle, id, value.c_str()));
    }
    ASSERT_OK(handle, sk_complete_writing(handle));

    const std::vector<uint64_t> ids = api_knn(handle, "0.0, 0.0, 0.0, 0.0", 3);
    ASSERT_EQ(3u, ids.size());
    EXPECT_EQ(0u, ids[0]);
    EXPECT_EQ(1u, ids[1]);
    EXPECT_EQ(2u, ids[2]);

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));
    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, staged_write_mixed_vectors_and_deletes) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "l1"));

    // First session: create all vectors
    ASSERT_OK(handle, sk_start_writing(handle));
    ASSERT_OK(handle, sk_write_vector(handle, 10, "0.0, 0.0, 0.0, 0.0"));
    ASSERT_OK(handle, sk_write_vector(handle, 20, "5.0, 5.0, 5.0, 5.0"));
    ASSERT_OK(handle, sk_write_vector(handle, 30, "10.0, 10.0, 10.0, 10.0"));
    ASSERT_OK(handle, sk_write_vector(handle, 40, "15.0, 15.0, 15.0, 15.0"));
    ASSERT_OK(handle, sk_complete_writing(handle));

    // Second session: delete one
    ASSERT_OK(handle, sk_start_writing(handle));
    ASSERT_OK(handle, sk_write_deleted(handle, 40));
    ASSERT_OK(handle, sk_complete_writing(handle));

    EXPECT_NE(api_get(handle, 10).find("[ 0"), std::string::npos);
    EXPECT_NE(api_get(handle, 20).find("[ 5"), std::string::npos);
    EXPECT_NE(api_get(handle, 30).find("[ 10"), std::string::npos);

    char* value = nullptr;
    EXPECT_NE(0, sk_get(handle, 40, &value));
    EXPECT_EQ(nullptr, value);

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));
    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, staged_write_multiple_sessions_accumulate) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "l1"));

    ASSERT_OK(handle, sk_start_writing(handle));
    ASSERT_OK(handle, sk_write_vector(handle, 10, "10.0, 10.0, 10.0, 10.0"));
    ASSERT_OK(handle, sk_complete_writing(handle));

    ASSERT_OK(handle, sk_start_writing(handle));
    ASSERT_OK(handle, sk_write_vector(handle, 20, "20.0, 20.0, 20.0, 20.0"));
    ASSERT_OK(handle, sk_complete_writing(handle));

    ASSERT_OK(handle, sk_start_writing(handle));
    ASSERT_OK(handle, sk_write_vector(handle, 30, "30.0, 30.0, 30.0, 30.0"));
    ASSERT_OK(handle, sk_complete_writing(handle));

    EXPECT_NE(api_get(handle, 10).find("[ 10"), std::string::npos);
    EXPECT_NE(api_get(handle, 20).find("[ 20"), std::string::npos);
    EXPECT_NE(api_get(handle, 30).find("[ 30"), std::string::npos);

    const std::vector<uint64_t> ids = api_knn(handle, "10.0, 10.0, 10.0, 10.0", 3);
    ASSERT_EQ(3u, ids.size());
    EXPECT_EQ(10u, ids[0]);

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));
    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, staged_write_rejects_null_and_empty_vector) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);
    ASSERT_OK(handle, sk_create(handle, "ds", nullptr, 4, "f32", 1000, "l1"));

    ASSERT_OK(handle, sk_start_writing(handle));
    EXPECT_NE(0, sk_write_vector(handle, 1, nullptr));
    EXPECT_NE(0, sk_write_vector(handle, 1, ""));

    EXPECT_OK(handle, sk_close(handle));
    EXPECT_OK(handle, sk_drop(handle, "ds"));
    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}

TEST(sketch2api, staged_write_without_open_dataset_fails) {
    const std::filesystem::path root = make_temp_dir();

    sk_handle_t* handle = sk_new_handle(root.string().c_str());
    ASSERT_NE(handle, nullptr);

    EXPECT_NE(0, sk_start_writing(handle));
    EXPECT_NE(0, sk_write_vector(handle, 1, "1.0, 1.0, 1.0, 1.0"));
    EXPECT_NE(0, sk_write_deleted(handle, 1));
    EXPECT_NE(0, sk_complete_writing(handle));

    sk_release_handle(handle);
    std::filesystem::remove_all(root);
}
