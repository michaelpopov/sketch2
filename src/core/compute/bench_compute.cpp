// Direct kernel benchmark for the compiled Highway backend.

#include "core/compute/compute_engine.h"
#include "core/compute/cosine_distance.h"
#include "core/compute/scanner_query_context.h"
#include "core/utils/shared_types.h"
#include "core/utils/singleton.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

using namespace sketch2;

namespace {

volatile double g_sink = 0.0;

struct CaseStats {
    std::string name;
    double avg_ns_per_call = 0.0;
    double min_ns_per_call = 0.0;
    double max_ns_per_call = 0.0;
};

struct Args {
    DistFunc dist = DistFunc::DOT;
    DataType type = DataType::f32;
    size_t dim = 256;
    size_t iterations = 200000;
    size_t warmup_iterations = 5000;
    size_t repeats = 7;
};

std::string json_escape(std::string_view input) {
    std::string out;
    out.reserve(input.size());
    for (char ch : input) {
        switch (ch) {
            case '\"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(ch) < 0x20u) {
                    // This benchmark only emits ASCII control chars here.
                    out += "?";
                } else {
                    out += ch;
                }
                break;
        }
    }
    return out;
}

template <typename T>
void fill_pair(std::vector<T>* a, std::vector<T>* b, uint32_t seed, size_t dim) {
    a->resize(dim);
    b->resize(dim);
    for (size_t i = 0; i < dim; ++i) {
        const int32_t ai = static_cast<int32_t>((i * 977 + seed * 131) % 65536) - 32768;
        const int32_t bi = static_cast<int32_t>((i * 733 + seed * 191) % 65536) - 32768;
        (*a)[i] = static_cast<T>(ai);
        (*b)[i] = static_cast<T>(bi);
    }
}

template <>
void fill_pair<float>(std::vector<float>* a, std::vector<float>* b, uint32_t seed, size_t dim) {
    a->resize(dim);
    b->resize(dim);
    for (size_t i = 0; i < dim; ++i) {
        const int32_t ai = static_cast<int32_t>((i * 17 + seed * 13) % 401) - 200;
        const int32_t bi = static_cast<int32_t>((i * 29 + seed * 7) % 401) - 200;
        (*a)[i] = static_cast<float>(ai) * 0.125f + static_cast<float>((i + seed) % 5) * 0.03125f;
        (*b)[i] = static_cast<float>(bi) * 0.125f - static_cast<float>((i + seed) % 3) * 0.0625f;
    }
}

template <>
void fill_pair<float16>(std::vector<float16>* a, std::vector<float16>* b, uint32_t seed, size_t dim) {
    a->resize(dim);
    b->resize(dim);
    for (size_t i = 0; i < dim; ++i) {
        const int32_t ai = static_cast<int32_t>((i * 17 + seed * 13) % 401) - 200;
        const int32_t bi = static_cast<int32_t>((i * 29 + seed * 7) % 401) - 200;
        (*a)[i] = static_cast<float16>(static_cast<float>(ai) * 0.125f +
                                       static_cast<float>((i + seed) % 5) * 0.03125f);
        (*b)[i] = static_cast<float16>(static_cast<float>(bi) * 0.125f -
                                       static_cast<float>((i + seed) % 3) * 0.0625f);
    }
}

template <typename Fn>
CaseStats benchmark_case(std::string name, size_t warmup_iterations, size_t iterations,
                         size_t repeats, Fn&& fn) {
    for (size_t i = 0; i < warmup_iterations; ++i) {
        g_sink += fn();
    }

    std::vector<double> samples;
    samples.reserve(repeats);
    for (size_t repeat = 0; repeat < repeats; ++repeat) {
        const auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < iterations; ++i) {
            g_sink += fn();
        }
        const auto end = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(end - start).count();
        samples.push_back(seconds * 1e9 / static_cast<double>(iterations));
    }

    const auto [min_it, max_it] = std::minmax_element(samples.begin(), samples.end());
    double total = 0.0;
    for (double sample : samples) {
        total += sample;
    }

    return CaseStats{
        std::move(name),
        total / static_cast<double>(samples.size()),
        *min_it,
        *max_it,
    };
}

Args parse_args(int argc, char** argv) {
    Args args;
    const std::string compiled_engine_name = compute_engine_name(compiled_compute_engine());
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        auto require_value = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + flag);
            }
            return argv[++i];
        };

        if (arg == "--engine") {
            // Backward-compatible flag: this build can benchmark only one engine.
            const std::string value = require_value("--engine");
            if (value != compiled_engine_name) {
                throw std::runtime_error(
                    "--engine does not match compiled engine: " + compiled_engine_name);
            }
        } else if (arg == "--dist") {
            args.dist = dist_func_from_string(require_value("--dist"));
        } else if (arg == "--type") {
            args.type = data_type_from_string(require_value("--type"));
        } else if (arg == "--dim") {
            args.dim = static_cast<size_t>(std::stoull(require_value("--dim")));
        } else if (arg == "--iterations") {
            args.iterations = static_cast<size_t>(std::stoull(require_value("--iterations")));
        } else if (arg == "--warmup-iterations") {
            args.warmup_iterations = static_cast<size_t>(std::stoull(require_value("--warmup-iterations")));
        } else if (arg == "--repeats") {
            args.repeats = static_cast<size_t>(std::stoull(require_value("--repeats")));
        } else {
            throw std::runtime_error(std::string("unknown argument: ") + std::string(arg));
        }
    }

    if (args.iterations == 0) {
        throw std::runtime_error("--iterations must be >= 1");
    }
    if (args.repeats == 0) {
        throw std::runtime_error("--repeats must be >= 1");
    }
    return args;
}

template <typename T>
std::pair<const uint8_t*, const uint8_t*> prepare_bytes(std::vector<T>* a, std::vector<T>* b, size_t dim) {
    fill_pair(a, b, 17, dim);
    return {
        reinterpret_cast<const uint8_t*>(a->data()),
        reinterpret_cast<const uint8_t*>(b->data()),
    };
}

std::vector<CaseStats> run_compute_bench(const Args& args, const uint8_t* a, const uint8_t* b) {
    const ComputeKernels kernels = resolve_compute_kernels(args.dist, args.type);
    std::vector<CaseStats> results;
    const bool is_dot = args.dist == DistFunc::DOT;
    results.push_back(benchmark_case(
        is_dot ? "dot" : "dist",
        args.warmup_iterations, args.iterations, args.repeats, [&] {
            return is_dot && kernels.dot ? kernels.dot(a, b, args.dim) : kernels.dist(a, b, args.dim);
        }));

    if (kernels.dot && !is_dot) {
        results.push_back(benchmark_case("dot", args.warmup_iterations, args.iterations, args.repeats, [&] {
            return kernels.dot(a, b, args.dim);
        }));
    }

    if (kernels.squared_norm) {
        results.push_back(benchmark_case(
            "squared_norm", args.warmup_iterations, args.iterations, args.repeats, [&] {
                return kernels.squared_norm(b, args.dim);
            }));
    }

    if (args.dist == DistFunc::COS && kernels.dot && kernels.squared_norm && kernels.dist_with_query_norm) {
        const double query_norm_sq = kernels.squared_norm(b, args.dim);
        const double query_inv_norm = query_inverse_norm(query_norm_sq);
        const double stored_inv_norm = query_inverse_norm(kernels.squared_norm(a, args.dim));
        results.push_back(benchmark_case(
            "dist_with_query_norm", args.warmup_iterations, args.iterations, args.repeats, [&] {
                return kernels.dist_with_query_norm(a, b, args.dim, query_norm_sq);
            }));
        results.push_back(benchmark_case(
            "dist_with_stored_norms", args.warmup_iterations, args.iterations, args.repeats, [&] {
                return finalize_cosine_distance_from_inverse_norms(
                    kernels.dot(a, b, args.dim), stored_inv_norm, query_inv_norm);
            }));
    }

    if (args.dist == DistFunc::L2 && kernels.dot && kernels.squared_norm) {
        const double query_norm_sq = kernels.squared_norm(b, args.dim);
        const double stored_norm_sq = kernels.squared_norm(a, args.dim);
        results.push_back(benchmark_case(
            "dist_with_stored_norms", args.warmup_iterations, args.iterations, args.repeats, [&] {
                return finalize_squared_l2_distance_from_squared_norms(
                    kernels.dot(a, b, args.dim), stored_norm_sq, query_norm_sq);
            }));
    }

    return results;
}

std::vector<CaseStats> run_benchmarks(const Args& args) {
    switch (args.type) {
        case DataType::f32: {
            std::vector<float> a;
            std::vector<float> b;
            const auto [av, bv] = prepare_bytes(&a, &b, args.dim);
            return run_compute_bench(args, av, bv);
        }
        case DataType::f16: {
            std::vector<float16> a;
            std::vector<float16> b;
            const auto [av, bv] = prepare_bytes(&a, &b, args.dim);
            return run_compute_bench(args, av, bv);
        }
        case DataType::i16: {
            std::vector<int16_t> a;
            std::vector<int16_t> b;
            const auto [av, bv] = prepare_bytes(&a, &b, args.dim);
            return run_compute_bench(args, av, bv);
        }
    }
    throw std::runtime_error("unsupported data type");
}

void print_json(const Args& args, const std::vector<CaseStats>& cases) {
    const std::string engine = compute_engine_name(compiled_compute_engine());
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "{\n";
    std::cout << "  \"engine\": \"" << json_escape(engine) << "\",\n";
    std::cout << "  \"dist\": \"" << json_escape(dist_func_to_string(args.dist)) << "\",\n";
    std::cout << "  \"type\": \"" << json_escape(data_type_to_string(args.type)) << "\",\n";
    std::cout << "  \"dim\": " << args.dim << ",\n";
    std::cout << "  \"iterations\": " << args.iterations << ",\n";
    std::cout << "  \"warmup_iterations\": " << args.warmup_iterations << ",\n";
    std::cout << "  \"repeats\": " << args.repeats << ",\n";
    std::cout << "  \"active_compute_backend\": \"" << json_escape(get_singleton().compute_unit().name()) << "\"";
    std::cout << ",\n  \"cases\": [\n";
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto& entry = cases[i];
        std::cout << "    {\n";
        std::cout << "      \"name\": \"" << json_escape(entry.name) << "\",\n";
        std::cout << "      \"avg_ns_per_call\": " << entry.avg_ns_per_call << ",\n";
        std::cout << "      \"min_ns_per_call\": " << entry.min_ns_per_call << ",\n";
        std::cout << "      \"max_ns_per_call\": " << entry.max_ns_per_call << '\n';
        std::cout << "    }";
        if (i + 1 != cases.size()) {
            std::cout << ",";
        }
        std::cout << '\n';
    }
    std::cout << "  ]\n";
    std::cout << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        initialize_compute_engine_runtime();
        const Args args = parse_args(argc, argv);
        const std::vector<CaseStats> cases = run_benchmarks(args);
        if (std::isnan(g_sink)) {
            throw std::runtime_error("unexpected NaN sink");
        }
        print_json(args, cases);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
