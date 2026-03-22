// Google Benchmark suite for compute kernels and scanner query paths.

#include <benchmark/benchmark.h>

#include "core/compute/compute_cos.h"
#include "core/compute/compute_l1.h"
#include "core/compute/compute_l2.h"
#include "core/compute/scanner.h"
#include "core/storage/data_reader.h"
#include "core/storage/data_writer.h"
#include "core/storage/dataset_node.h"
#include "core/storage/input_generator.h"
#include "core/utils/log.h"
#include "core/utils/singleton.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

namespace sketch2 {
namespace {

constexpr size_t kScannerVectorCountEssential = 8192;
constexpr size_t kScannerVectorCountExtended = 16384;
constexpr size_t kScannerAccumulatorAdditionsEssential = 256;
constexpr size_t kScannerAccumulatorAdditionsExtended = 512;
constexpr size_t kScannerAccumulatorDeletionsEssential = 128;
constexpr size_t kScannerAccumulatorDeletionsExtended = 256;
constexpr size_t kScannerRangeSize = 2048;

enum class ScannerMode : int64_t {
    reader = 0,
    dataset_persisted = 1,
    dataset_mixed = 2,
};

enum class BenchmarkProfile {
    essential,
    extended,
};

struct TempDir {
    fs::path path;

    explicit TempDir(const std::string& prefix) {
        const fs::path base = fs::temp_directory_path();
        std::string pattern = (base / (prefix + "_" + std::to_string(getpid()) + "_XXXXXX")).string();
        std::vector<char> buffer(pattern.begin(), pattern.end());
        buffer.push_back('\0');
        if (char* created = ::mkdtemp(buffer.data())) {
            path = created;
            return;
        }
        throw std::runtime_error("failed to create temporary benchmark directory");
    }

    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

class BackendScope {
public:
    BackendScope() : original_(get_singleton().compute_unit().kind()) {}
    BackendScope(const BackendScope&) = delete;
    BackendScope& operator=(const BackendScope&) = delete;
    BackendScope(BackendScope&&) = delete;
    BackendScope& operator=(BackendScope&&) = delete;

    ~BackendScope() {
        (void)Singleton::force_compute_unit_for_testing(original_);
    }

    bool force(ComputeBackendKind kind) {
        return Singleton::force_compute_unit_for_testing(kind);
    }

private:
    ComputeBackendKind original_;
};

const char* backend_name(ComputeBackendKind kind) {
    return ComputeUnit(kind).name();
}

const char* metric_name(DistFunc func) {
    switch (func) {
        case DistFunc::L1: return "L1";
        case DistFunc::L2: return "L2";
        case DistFunc::COS: return "COS";
        default: return "unknown";
    }
}

const char* scanner_mode_name(ScannerMode mode) {
    switch (mode) {
        case ScannerMode::reader: return "reader";
        case ScannerMode::dataset_persisted: return "dataset_persisted";
        case ScannerMode::dataset_mixed: return "dataset_mixed";
        default: return "unknown";
    }
}

BenchmarkProfile benchmark_profile() {
    const char* value = std::getenv("SKETCH2_GBENCH_PROFILE");
    if (!value || !*value) {
        return BenchmarkProfile::essential;
    }
    const std::string profile = value;
    if (profile == "extended" || profile == "full") {
        return BenchmarkProfile::extended;
    }
    return BenchmarkProfile::essential;
}

size_t scanner_vector_count() {
    return benchmark_profile() == BenchmarkProfile::essential ?
        kScannerVectorCountEssential : kScannerVectorCountExtended;
}

size_t scanner_accumulator_additions() {
    return benchmark_profile() == BenchmarkProfile::essential ?
        kScannerAccumulatorAdditionsEssential : kScannerAccumulatorAdditionsExtended;
}

size_t scanner_accumulator_deletions() {
    return benchmark_profile() == BenchmarkProfile::essential ?
        kScannerAccumulatorDeletionsEssential : kScannerAccumulatorDeletionsExtended;
}

const std::vector<int>& scanner_modes() {
    static const std::vector<int> modes = [] {
        std::vector<int> values;
        const char* env = std::getenv("SKETCH2_GBENCH_SCANNER_MODE");
        if (env && *env) {
            const std::string mode = env;
            if (mode == "reader") {
                throw std::runtime_error(
                    "SKETCH2_GBENCH_SCANNER_MODE=reader is no longer supported; "
                    "use dataset_persisted or dataset_mixed");
            } else if (mode == "dataset_persisted") {
                values.push_back(1);
            } else if (mode == "dataset_mixed") {
                values.push_back(2);
            } else {
                throw std::runtime_error(
                    "invalid SKETCH2_GBENCH_SCANNER_MODE: " + mode +
                    " (expected dataset_persisted or dataset_mixed)");
            }
        } else {
            values.push_back(1);
            values.push_back(2);
        }
        return values;
    }();
    return modes;
}

const std::vector<int>& benchmark_backends() {
    static const std::vector<int> backends = [] {
        std::vector<int> values;
        values.push_back(0);
#if defined(SKETCH_ENABLE_AVX2) && SKETCH_ENABLE_AVX2 && (defined(__x86_64__) || defined(__i386__))
        if (ComputeUnit::is_supported(ComputeBackendKind::avx2)) {
            values.push_back(1);
        }
#endif
#if defined(SKETCH_ENABLE_AVX512F) && SKETCH_ENABLE_AVX512F && (defined(__x86_64__) || defined(__i386__))
        if (ComputeUnit::is_supported(ComputeBackendKind::avx512f)) {
            values.push_back(2);
        }
#endif
#if defined(SKETCH_ENABLE_AVX512VNNI) && SKETCH_ENABLE_AVX512VNNI && (defined(__x86_64__) || defined(__i386__))
        if (ComputeUnit::is_supported(ComputeBackendKind::avx512_vnni)) {
            values.push_back(3);
        }
#endif
#if defined(__aarch64__)
        if (ComputeUnit::is_supported(ComputeBackendKind::neon)) {
            values.push_back(4);
        }
#endif
        return values;
    }();
    return backends;
}

ComputeBackendKind backend_from_arg(int64_t arg) {
    switch (arg) {
        case 0: return ComputeBackendKind::scalar;
        case 1: return ComputeBackendKind::avx2;
        case 2: return ComputeBackendKind::avx512f;
        case 3: return ComputeBackendKind::avx512_vnni;
        case 4: return ComputeBackendKind::neon;
        default: throw std::runtime_error("invalid backend benchmark argument");
    }
}

DistFunc metric_from_arg(int64_t arg) {
    switch (arg) {
        case 0: return DistFunc::L1;
        case 1: return DistFunc::L2;
        case 2: return DistFunc::COS;
        default: throw std::runtime_error("invalid metric benchmark argument");
    }
}

DataType type_from_arg(int64_t arg) {
    switch (arg) {
        case 0: return DataType::f32;
        case 1: return DataType::f16;
        case 2: return DataType::i16;
        default: throw std::runtime_error("invalid type benchmark argument");
    }
}

ScannerMode scanner_mode_from_arg(int64_t arg) {
    switch (arg) {
        case 0: return ScannerMode::reader;
        case 1: return ScannerMode::dataset_persisted;
        case 2: return ScannerMode::dataset_mixed;
        default: throw std::runtime_error("invalid scanner mode benchmark argument");
    }
}

void require_ok(const Ret& ret, const std::string& context) {
    if (ret.code() != 0) {
        throw std::runtime_error(context + ": " + ret.message());
    }
}

bool skip_if_unsupported(benchmark::State& state, ComputeBackendKind backend) {
    if (!ComputeUnit::is_supported(backend)) {
        state.SkipWithError("requested backend is not supported on this build/CPU");
        return true;
    }
    return false;
}

std::string make_label(ComputeBackendKind backend, DistFunc func, DataType type, size_t dim) {
    std::ostringstream out;
    out << "backend=" << backend_name(backend)
        << ",metric=" << metric_name(func)
        << ",type=" << data_type_to_string(type)
        << ",dim=" << dim;
    return out.str();
}

std::string make_scanner_label(ComputeBackendKind backend, DistFunc func, DataType type,
        size_t dim, size_t k, ScannerMode mode) {
    std::ostringstream out;
    out << make_label(backend, func, type, dim)
        << ",k=" << k
        << ",mode=" << scanner_mode_name(mode);
    return out.str();
}

std::string make_fixture_key(DistFunc func, DataType type, size_t dim, size_t count, bool mixed_state) {
    std::ostringstream out;
    out << static_cast<int>(func)
        << ':' << static_cast<int>(type)
        << ':' << dim
        << ':' << count
        << ':' << (mixed_state ? 1 : 0);
    return out.str();
}

std::vector<uint8_t> make_vector(DataType type, size_t dim, uint64_t seed) {
    switch (type) {
        case DataType::f32: {
            std::vector<uint8_t> bytes(dim * sizeof(float));
            auto* out = reinterpret_cast<float*>(bytes.data());
            for (size_t i = 0; i < dim; ++i) {
                const int32_t base = static_cast<int32_t>((seed * 131 + i * 17) % 401) - 200;
                out[i] = static_cast<float>(base) * 0.125f +
                    static_cast<float>((seed + i) % 7) * 0.03125f;
            }
            return bytes;
        }
        case DataType::f16: {
            std::vector<uint8_t> bytes(dim * sizeof(float16));
            auto* out = reinterpret_cast<float16*>(bytes.data());
            for (size_t i = 0; i < dim; ++i) {
                const int32_t base = static_cast<int32_t>((seed * 73 + i * 19) % 401) - 200;
                const float value = static_cast<float>(base) * 0.125f +
                    static_cast<float>((seed + i) % 5) * 0.0625f;
                out[i] = static_cast<float16>(value);
            }
            return bytes;
        }
        case DataType::i16: {
            std::vector<uint8_t> bytes(dim * sizeof(int16_t));
            auto* out = reinterpret_cast<int16_t*>(bytes.data());
            for (size_t i = 0; i < dim; ++i) {
                const int32_t value = static_cast<int32_t>((seed * 977 + i * 131) % 4096) - 2048;
                out[i] = static_cast<int16_t>(value);
            }
            return bytes;
        }
        default:
            throw std::runtime_error("unsupported data type");
    }
}

std::unique_ptr<ICompute> make_compute(DistFunc func) {
    switch (func) {
        case DistFunc::L1: return std::make_unique<ComputeL1>();
        case DistFunc::L2: return std::make_unique<ComputeL2>();
        case DistFunc::COS: return std::make_unique<ComputeCos>();
        default: throw std::runtime_error("unsupported metric");
    }
}

struct [[maybe_unused]] ReaderBenchmarkData {
    TempDir temp{"sketch2_bench_reader"};
    fs::path input_path = temp.path / "input.txt";
    fs::path data_path = temp.path / "data.bin";
    DataReader reader;
    std::vector<uint8_t> query;

    ReaderBenchmarkData(DistFunc func, DataType type, size_t dim, size_t count) {
        GeneratorConfig cfg{PatternType::Sequential, count, 0, type, dim, 1000};
        require_ok(generate_input_file(input_path.string(), cfg), "generate reader input");

        DataWriter writer;
        require_ok(writer.init(input_path.string(), data_path.string(), 0, 0, func == DistFunc::COS),
            "init reader writer");
        require_ok(writer.exec(), "write reader data");
        require_ok(reader.init(data_path.string()), "init reader");
        query = make_vector(type, dim, count + 7);
    }
};

struct DatasetBenchmarkData {
    TempDir temp{"sketch2_bench_dataset"};
    DatasetNode dataset;
    std::vector<uint8_t> query;
    size_t visible_count = 0;

    DatasetBenchmarkData(DistFunc func, DataType type, size_t dim, size_t count, bool mixed_state) {
        DatasetMetadata metadata;
        metadata.dirs = {temp.path.string()};
        metadata.type = type;
        metadata.dim = dim;
        metadata.dist_func = func;
        metadata.range_size = kScannerRangeSize;
        metadata.accumulator_size = static_cast<uint64_t>(count * dim * data_type_size(type)) + (1u << 20);
        require_ok(dataset.init(metadata), "init dataset");

        for (size_t id = 0; id < count; ++id) {
            const std::vector<uint8_t> vec = make_vector(type, dim, id);
            require_ok(dataset.add_vector(id, vec.data()), "add persisted dataset vector");
        }
        require_ok(dataset.store_accumulator(), "store persisted dataset vectors");
        visible_count = count;

        if (mixed_state) {
            for (size_t i = 0; i < scanner_accumulator_additions(); ++i) {
                const uint64_t id = static_cast<uint64_t>(count + i);
                const std::vector<uint8_t> vec = make_vector(type, dim, 100000 + id);
                require_ok(dataset.add_vector(id, vec.data()), "add accumulator dataset vector");
                ++visible_count;
            }
            for (size_t i = 0; i < scanner_accumulator_deletions(); ++i) {
                const uint64_t id = static_cast<uint64_t>(i * 2);
                require_ok(dataset.delete_vector(id), "delete dataset vector");
                --visible_count;
            }
        }

        query = make_vector(type, dim, count + 31);
    }
};

[[maybe_unused]] std::shared_ptr<ReaderBenchmarkData> get_reader_benchmark_data(
        DistFunc func, DataType type, size_t dim, size_t count) {
    static std::map<std::string, std::shared_ptr<ReaderBenchmarkData>> cache;
    const std::string key = make_fixture_key(func, type, dim, count, false);
    const auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    auto data = std::make_shared<ReaderBenchmarkData>(func, type, dim, count);
    cache.emplace(key, data);
    return data;
}

std::shared_ptr<DatasetBenchmarkData> get_dataset_benchmark_data(
        DistFunc func, DataType type, size_t dim, size_t count, bool mixed_state) {
    static std::map<std::string, std::shared_ptr<DatasetBenchmarkData>> cache;
    const std::string key = make_fixture_key(func, type, dim, count, mixed_state);
    const auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    auto data = std::make_shared<DatasetBenchmarkData>(func, type, dim, count, mixed_state);
    cache.emplace(key, data);
    return data;
}

void ApplyComputeArgs(benchmark::internal::Benchmark* benchmark) {
    const BenchmarkProfile profile = benchmark_profile();
    for (int backend : benchmark_backends()) {
        for (int metric = 0; metric <= 2; ++metric) {
            for (int type = 0; type <= 2; ++type) {
                if (profile == BenchmarkProfile::essential) {
                    benchmark->Args({backend, metric, type, 256});
                } else {
                    benchmark->Args({backend, metric, type, 64});
                    benchmark->Args({backend, metric, type, 256});
                    benchmark->Args({backend, metric, type, 1024});
                }
            }
        }
    }
}

void ApplyScannerArgs(benchmark::internal::Benchmark* benchmark) {
    const BenchmarkProfile profile = benchmark_profile();
    const std::vector<int>& modes = scanner_modes();
    // In the essential profile, run a compact representative set for supported
    // dataset modes.
    for (int backend : benchmark_backends()) {
        if (profile == BenchmarkProfile::essential) {
            for (int mode : modes) {
                benchmark->Args({backend, 1, 0, 256, 10, mode});
                benchmark->Args({backend, 1, 2, 256, 10, mode});
                benchmark->Args({backend, 2, 2, 256, 10, mode});
            }
        } else {
            for (int metric = 0; metric <= 2; ++metric) {
                for (int type = 0; type <= 2; ++type) {
                    for (int mode : modes) {
                        benchmark->Args({backend, metric, type, 128, 10, mode});
                        benchmark->Args({backend, metric, type, 256, 10, mode});
                        benchmark->Args({backend, metric, type, 256, 100, mode});
                    }
                }
            }
        }
    }
}

void ApplyDatasetScannerArgs(benchmark::internal::Benchmark* benchmark) {
    const BenchmarkProfile profile = benchmark_profile();
    if (profile == BenchmarkProfile::essential) {
        return;
    }
    const std::vector<int>& modes = scanner_modes();
    for (int backend : benchmark_backends()) {
        for (int metric = 0; metric <= 2; ++metric) {
            for (int type = 0; type <= 2; ++type) {
                for (int mode : modes) {
                    if (mode == 0) continue;
                    benchmark->Args({backend, metric, type, 128, 10, mode});
                    benchmark->Args({backend, metric, type, 256, 10, mode});
                    benchmark->Args({backend, metric, type, 256, 100, mode});
                }
            }
        }
    }
}

void BM_ComputeDistance(benchmark::State& state) {
    const ComputeBackendKind backend = backend_from_arg(state.range(0));
    const DistFunc func = metric_from_arg(state.range(1));
    const DataType type = type_from_arg(state.range(2));
    const size_t dim = static_cast<size_t>(state.range(3));

    state.SetLabel(make_label(backend, func, type, dim));
    if (skip_if_unsupported(state, backend)) {
        return;
    }

    BackendScope backend_scope;
    if (!backend_scope.force(backend)) {
        state.SkipWithError("failed to force benchmark backend");
        return;
    }

    const std::vector<uint8_t> lhs = make_vector(type, dim, 11);
    const std::vector<uint8_t> rhs = make_vector(type, dim, 29);
    std::unique_ptr<ICompute> compute = make_compute(func);

    for (auto _ : state) {
        double value = compute->dist(lhs.data(), rhs.data(), type, dim);
        benchmark::DoNotOptimize(value);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(dim));
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(2 * dim * data_type_size(type)));
}

void BM_ScannerFindIds(benchmark::State& state) {
    const ComputeBackendKind backend = backend_from_arg(state.range(0));
    const DistFunc func = metric_from_arg(state.range(1));
    const DataType type = type_from_arg(state.range(2));
    const size_t dim = static_cast<size_t>(state.range(3));
    const size_t k = static_cast<size_t>(state.range(4));
    const ScannerMode mode = scanner_mode_from_arg(state.range(5));

    state.SetLabel(make_scanner_label(backend, func, type, dim, k, mode));
    if (skip_if_unsupported(state, backend)) {
        return;
    }

    BackendScope backend_scope;
    if (!backend_scope.force(backend)) {
        state.SkipWithError("failed to force benchmark backend");
        return;
    }

    Scanner scanner;
    std::vector<uint64_t> result;
    result.reserve(k);
    size_t visible_count = 0;

    std::shared_ptr<DatasetBenchmarkData> dataset_data;
    const size_t vector_count = scanner_vector_count();
    switch (mode) {
        case ScannerMode::dataset_persisted:
            dataset_data = get_dataset_benchmark_data(func, type, dim, vector_count, false);
            visible_count = dataset_data->visible_count;
            break;
        case ScannerMode::dataset_mixed:
            dataset_data = get_dataset_benchmark_data(func, type, dim, vector_count, true);
            visible_count = dataset_data->visible_count;
            break;
        default:
            state.SkipWithError("invalid scanner mode");
            return;
    }

    for (auto _ : state) {
        const Ret ret = scanner.find(dataset_data->dataset, k, dataset_data->query.data(), result);
        if (ret.code() != 0) {
            state.SkipWithError(ret.message().c_str());
            break;
        }
        benchmark::DoNotOptimize(result.data());
        benchmark::ClobberMemory();
    }

    state.counters["queries/s"] = benchmark::Counter(static_cast<double>(state.iterations()),
        benchmark::Counter::kIsRate);
    state.counters["vectors/s"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * static_cast<double>(visible_count),
        benchmark::Counter::kIsRate);
    state.counters["components/s"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * static_cast<double>(visible_count) * static_cast<double>(dim),
        benchmark::Counter::kIsRate);
}

void BM_ScannerFindItems(benchmark::State& state) {
    const ComputeBackendKind backend = backend_from_arg(state.range(0));
    const DistFunc func = metric_from_arg(state.range(1));
    const DataType type = type_from_arg(state.range(2));
    const size_t dim = static_cast<size_t>(state.range(3));
    const size_t k = static_cast<size_t>(state.range(4));
    const ScannerMode mode = scanner_mode_from_arg(state.range(5));

    state.SetLabel(make_scanner_label(backend, func, type, dim, k, mode));
    if (skip_if_unsupported(state, backend)) {
        return;
    }

    if (mode == ScannerMode::reader) {
        state.SkipWithError("BM_ScannerFindItems only supports dataset benchmark modes");
        return;
    }

    BackendScope backend_scope;
    if (!backend_scope.force(backend)) {
        state.SkipWithError("failed to force benchmark backend");
        return;
    }

    const size_t vector_count = scanner_vector_count();
    std::shared_ptr<DatasetBenchmarkData> dataset_data = get_dataset_benchmark_data(
        func, type, dim, vector_count, mode == ScannerMode::dataset_mixed);

    Scanner scanner;
    std::vector<DistItem> result;
    result.reserve(k);
    const size_t visible_count = dataset_data->visible_count;

    for (auto _ : state) {
        const Ret ret = scanner.find_items(dataset_data->dataset, k, dataset_data->query.data(), result);
        if (ret.code() != 0) {
            state.SkipWithError(ret.message().c_str());
            break;
        }
        benchmark::DoNotOptimize(result.data());
        benchmark::ClobberMemory();
    }

    state.counters["queries/s"] = benchmark::Counter(static_cast<double>(state.iterations()),
        benchmark::Counter::kIsRate);
    state.counters["vectors/s"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * static_cast<double>(visible_count),
        benchmark::Counter::kIsRate);
    state.counters["components/s"] = benchmark::Counter(
        static_cast<double>(state.iterations()) * static_cast<double>(visible_count) * static_cast<double>(dim),
        benchmark::Counter::kIsRate);
}

benchmark::internal::Benchmark* RegisterScannerFindItemsBenchmark() {
    if (benchmark_profile() != BenchmarkProfile::extended) {
        return nullptr;
    }
    return benchmark::RegisterBenchmark("BM_ScannerFindItems", &BM_ScannerFindItems)
        ->Apply(ApplyDatasetScannerArgs)
        ->Unit(benchmark::kMillisecond);
}

BENCHMARK(BM_ComputeDistance)->Apply(ApplyComputeArgs)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ScannerFindIds)->Apply(ApplyScannerArgs)->Unit(benchmark::kMillisecond);
auto* const g_bench_scanner_find_items = RegisterScannerFindItemsBenchmark();

} // namespace
} // namespace sketch2

int main(int argc, char** argv) {
    sketch2::log::set_current_log_level(sketch2::log::LogLevel::Error);
    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}
