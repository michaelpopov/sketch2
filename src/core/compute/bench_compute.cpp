// Simple throughput benchmark for the public compute APIs under a forced backend.

#include "core/compute/compute_cos.h"
#include "core/compute/compute_l1.h"
#include "core/compute/compute_l2.h"
#include "core/utils/singleton.h"

#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sketch2;

namespace {

constexpr size_t kDim = 512;
constexpr size_t kIterations = 4000;
constexpr size_t kWarmupIterations = 200;
constexpr size_t kRepeats = 7;
volatile double g_sink = 0.0;

template <typename T>
void fill_pair(std::vector<T> *a, std::vector<T> *b, uint32_t seed) {
    a->resize(kDim);
    b->resize(kDim);
    for (size_t i = 0; i < kDim; ++i) {
        const int32_t ai = static_cast<int32_t>((i * 977 + seed * 131) % 65536) - 32768;
        const int32_t bi = static_cast<int32_t>((i * 733 + seed * 191) % 65536) - 32768;
        (*a)[i] = static_cast<T>(ai);
        (*b)[i] = static_cast<T>(bi);
    }
}

template <>
void fill_pair<float>(std::vector<float> *a, std::vector<float> *b, uint32_t seed) {
    a->resize(kDim);
    b->resize(kDim);
    for (size_t i = 0; i < kDim; ++i) {
        const int32_t ai = static_cast<int32_t>((i * 17 + seed * 13) % 401) - 200;
        const int32_t bi = static_cast<int32_t>((i * 29 + seed * 7) % 401) - 200;
        (*a)[i] = static_cast<float>(ai) * 0.125f + static_cast<float>((i + seed) % 5) * 0.03125f;
        (*b)[i] = static_cast<float>(bi) * 0.125f - static_cast<float>((i + seed) % 3) * 0.0625f;
    }
}

template <>
void fill_pair<float16>(std::vector<float16> *a, std::vector<float16> *b, uint32_t seed) {
    a->resize(kDim);
    b->resize(kDim);
    for (size_t i = 0; i < kDim; ++i) {
        const int32_t ai = static_cast<int32_t>((i * 17 + seed * 13) % 401) - 200;
        const int32_t bi = static_cast<int32_t>((i * 29 + seed * 7) % 401) - 200;
        (*a)[i] = static_cast<float16>(static_cast<float>(ai) * 0.125f +
                                       static_cast<float>((i + seed) % 5) * 0.03125f);
        (*b)[i] = static_cast<float16>(static_cast<float>(bi) * 0.125f -
                                       static_cast<float>((i + seed) % 3) * 0.0625f);
    }
}

template <typename Fn>
void run_case(const std::string &name, Fn &&fn) {
    for (size_t i = 0; i < kWarmupIterations; ++i) {
        g_sink += fn();
    }

    std::vector<double> samples_ns_per_call;
    samples_ns_per_call.reserve(kRepeats);
    for (size_t repeat = 0; repeat < kRepeats; ++repeat) {
        const auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < kIterations; ++i) {
            g_sink += fn();
        }
        const auto end = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(end - start).count();
        samples_ns_per_call.push_back(seconds * 1e9 / static_cast<double>(kIterations));
    }
    std::sort(samples_ns_per_call.begin(), samples_ns_per_call.end());
    const double median_ns_per_call = samples_ns_per_call[samples_ns_per_call.size() / 2];
    const double best_ns_per_call = samples_ns_per_call.front();
    const double median_calls_per_sec = 1e9 / median_ns_per_call;

    std::cout << std::left << std::setw(16) << name
              << " median_ns/call=" << std::setw(12) << std::fixed << std::setprecision(1) << median_ns_per_call
              << " best_ns/call=" << std::setw(12) << std::fixed << std::setprecision(1) << best_ns_per_call
              << " median_calls/s=" << std::setw(14) << std::fixed << std::setprecision(0) << median_calls_per_sec
              << '\n';
}

} // namespace

int main() {
    const char *requested = std::getenv("SKETCH2_COMPUTE_BACKEND");
    const std::string requested_backend = requested != nullptr ? requested : "auto";
    const std::string active_backend = get_singleton().compute_unit().name();
    if (requested != nullptr && requested_backend != "auto" && requested_backend != active_backend) {
        std::cerr << "requested_backend=" << requested_backend
                  << " active_backend=" << active_backend
                  << " error=backend_override_not_honored\n";
        return 2;
    }

    std::cout << "requested_backend=" << requested_backend
              << " active_backend=" << active_backend
              << " dim=" << kDim
              << " iterations=" << kIterations
              << " repeats=" << kRepeats
              << '\n';

    ComputeL1 l1;
    ComputeL2 l2;
    ComputeCos cos;

    std::vector<float> a_f32;
    std::vector<float> b_f32;
    fill_pair(&a_f32, &b_f32, 11);

    run_case("l1-f32", [&] {
        return l1.dist(reinterpret_cast<const uint8_t *>(a_f32.data()),
                       reinterpret_cast<const uint8_t *>(b_f32.data()),
                       DataType::f32, a_f32.size());
    });
    run_case("l2-f32", [&] {
        return l2.dist(reinterpret_cast<const uint8_t *>(a_f32.data()),
                       reinterpret_cast<const uint8_t *>(b_f32.data()),
                       DataType::f32, a_f32.size());
    });
    run_case("cos-f32", [&] {
        return cos.dist(reinterpret_cast<const uint8_t *>(a_f32.data()),
                        reinterpret_cast<const uint8_t *>(b_f32.data()),
                        DataType::f32, a_f32.size());
    });

    if (supports_f16()) {
        std::vector<float16> a_f16;
        std::vector<float16> b_f16;
        fill_pair(&a_f16, &b_f16, 17);

        run_case("l1-f16", [&] {
            return l1.dist(reinterpret_cast<const uint8_t *>(a_f16.data()),
                           reinterpret_cast<const uint8_t *>(b_f16.data()),
                           DataType::f16, a_f16.size());
        });
        run_case("l2-f16", [&] {
            return l2.dist(reinterpret_cast<const uint8_t *>(a_f16.data()),
                           reinterpret_cast<const uint8_t *>(b_f16.data()),
                           DataType::f16, a_f16.size());
        });
        run_case("cos-f16", [&] {
            return cos.dist(reinterpret_cast<const uint8_t *>(a_f16.data()),
                            reinterpret_cast<const uint8_t *>(b_f16.data()),
                            DataType::f16, a_f16.size());
        });
    } else {
        std::cout << "f16-unavailable\n";
    }

    std::vector<int16_t> a_i16;
    std::vector<int16_t> b_i16;
    fill_pair(&a_i16, &b_i16, 23);

    run_case("l1-i16", [&] {
        return l1.dist(reinterpret_cast<const uint8_t *>(a_i16.data()),
                       reinterpret_cast<const uint8_t *>(b_i16.data()),
                       DataType::i16, a_i16.size());
    });
    run_case("l2-i16", [&] {
        return l2.dist(reinterpret_cast<const uint8_t *>(a_i16.data()),
                       reinterpret_cast<const uint8_t *>(b_i16.data()),
                       DataType::i16, a_i16.size());
    });
    run_case("cos-i16", [&] {
        return cos.dist(reinterpret_cast<const uint8_t *>(a_i16.data()),
                        reinterpret_cast<const uint8_t *>(b_i16.data()),
                        DataType::i16, a_i16.size());
    });

    if (std::isnan(g_sink)) {
        std::cerr << "unexpected NaN sink\n";
        return 1;
    }
    return 0;
}
