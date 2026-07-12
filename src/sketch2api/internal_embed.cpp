// Implements the public C embedding API on top of the bundled llama.cpp
// (CPU-only). One sk_embedder owns one model + context pair; embedding calls
// are serialized on a per-embedder mutex, error state is thread-local.

#include "sketch2.h"

#include "core/utils/log.h"

#include "ggml-cpu.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__aarch64__) && defined(__linux__)
#include <sys/auxv.h>
#endif
#if defined(__linux__)
#include <sched.h>  // sched_getaffinity: the process's real CPU budget
#endif

using namespace sketch2::log;

struct sk_embedder {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    ggml_threadpool* pool = nullptr;
    const llama_vocab* vocab = nullptr;
    unsigned int dim = 0;
    unsigned int max_tokens = 0;
    bool is_encoder = false;
    bool normalize = true;
    bool truncate = false;
    // The llama context holds per-inference state: one forward pass at a time.
    std::mutex inference_mutex;
};

namespace {

// Default context cap; bounds KV-cache allocations for decoder models whose
// training context can be very large.
constexpr unsigned int kDefaultMaxContext = 4096;

// Thread-local, errno-style: threads sharing an embedder cannot clobber each
// other's failure details. Fixed buffer so recording an error never allocates
// or throws across the C ABI.
struct ThreadError {
    int code = 0;
    char message[256] = {};
};
thread_local ThreadError thread_error;

__attribute__((format(printf, 1, 2)))
void set_embed_error(const char* fmt, ...) noexcept {
    thread_error.code = -1;
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(thread_error.message, sizeof(thread_error.message), fmt, args);
    va_end(args);
}

#define EMBED_ERR(...) { \
    set_embed_error(__VA_ARGS__); \
    return -1; \
}

#define EMBED_DECL \
    if (embedder == nullptr) { \
        return -1; \
    } \
    thread_error.code = 0; \
    thread_error.message[0] = '\0';

// Route llama.cpp logging into the project logger; forward only complete lines.
void llama_log_bridge(ggml_log_level level, const char* text, void* /*user_data*/) {
    if (text == nullptr) {
        return;
    }
    std::string msg(text);
    while (!msg.empty() && (msg.back() == '\n' || msg.back() == '\r')) {
        msg.pop_back();
    }
    if (msg.empty()) {
        return;
    }
    switch (level) {
    case GGML_LOG_LEVEL_ERROR:
        LOG_ERROR << "llama: " << msg;
        break;
    case GGML_LOG_LEVEL_WARN:
        LOG_WARN << "llama: " << msg;
        break;
    default:
        LOG_DEBUG << "llama: " << msg;
        break;
    }
}

void backend_init_once() {
    static std::once_flag once;
    std::call_once(once, [] {
        llama_log_set(llama_log_bridge, nullptr);
        llama_backend_init();
    });
}

// ggml kernels are compiled for a fixed ISA set (see src/CMakeLists.txt) with
// no runtime dispatch; a CPU lacking a compiled-in extension would SIGILL.
// Check what this build requires (ggml_cpu_has_* are compile-time constants)
// against what the CPU exposes.
#if defined(__x86_64__) || defined(__i386__)
const char* missing_cpu_feature() {
    // __builtin_cpu_supports() only accepts literals, hence the macro.
#define SK_REQUIRE_FEATURE(compiled, name) \
    if ((compiled) && !__builtin_cpu_supports(name)) { \
        return name; \
    }
    SK_REQUIRE_FEATURE(ggml_cpu_has_sse3(),        "sse3")
    SK_REQUIRE_FEATURE(ggml_cpu_has_ssse3(),       "ssse3")
    SK_REQUIRE_FEATURE(ggml_cpu_has_avx(),         "avx")
    SK_REQUIRE_FEATURE(ggml_cpu_has_avx_vnni(),    "avxvnni")
    SK_REQUIRE_FEATURE(ggml_cpu_has_avx2(),        "avx2")
    SK_REQUIRE_FEATURE(ggml_cpu_has_fma(),         "fma")
    SK_REQUIRE_FEATURE(ggml_cpu_has_f16c(),        "f16c")
    SK_REQUIRE_FEATURE(ggml_cpu_has_bmi2(),        "bmi2")
    SK_REQUIRE_FEATURE(ggml_cpu_has_avx512(),      "avx512f")
    SK_REQUIRE_FEATURE(ggml_cpu_has_avx512_vbmi(), "avx512vbmi")
    SK_REQUIRE_FEATURE(ggml_cpu_has_avx512_vnni(), "avx512vnni")
    SK_REQUIRE_FEATURE(ggml_cpu_has_avx512_bf16(), "avx512bf16")
    SK_REQUIRE_FEATURE(ggml_cpu_has_amx_int8(),    "amx-int8")
#undef SK_REQUIRE_FEATURE
    return nullptr;
}
#elif defined(__aarch64__) && defined(__linux__)
// AT_HWCAP/AT_HWCAP2 bits, defined locally: some toolchain sysroots ship
// without asm/hwcap.h.
constexpr unsigned long kHwcapAsimdHp = 1UL << 10;  // FEAT_FP16 (vector)
constexpr unsigned long kHwcapAsimdDp = 1UL << 20;  // FEAT_DotProd
constexpr unsigned long kHwcapSve     = 1UL << 22;  // FEAT_SVE
constexpr unsigned long kHwcap2I8mm   = 1UL << 13;  // FEAT_I8MM
constexpr unsigned long kHwcap2Sme    = 1UL << 23;  // FEAT_SME

const char* missing_cpu_feature() {
    const unsigned long hwcap  = getauxval(AT_HWCAP);
    const unsigned long hwcap2 = getauxval(AT_HWCAP2);
    if (ggml_cpu_has_fp16_va() && !(hwcap & kHwcapAsimdHp)) {
        return "fp16";
    }
    if (ggml_cpu_has_dotprod() && !(hwcap & kHwcapAsimdDp)) {
        return "dotprod";
    }
    if (ggml_cpu_has_sve() && !(hwcap & kHwcapSve)) {
        return "sve";
    }
    if (ggml_cpu_has_matmul_int8() && !(hwcap2 & kHwcap2I8mm)) {
        return "i8mm";
    }
    if (ggml_cpu_has_sme() && !(hwcap2 & kHwcap2Sme)) {
        return "sme";
    }
    // +sve2 tuning builds are not covered: ggml exposes no ggml_cpu_has_sve2().
    return nullptr;
}
#else
// Other targets (e.g. Apple Silicon) have no mismatch to detect.
const char* missing_cpu_feature() {
    return nullptr;
}
#endif

// Error reporting for sk_embedder_open(), which has no handle to carry state;
// shares the thread-local slot.
__attribute__((format(printf, 2, 3)))
const char* set_open_error(const char** error_message_out, const char* fmt, ...) noexcept {
    thread_error.code = -1;
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(thread_error.message, sizeof(thread_error.message), fmt, args);
    va_end(args);
    if (error_message_out != nullptr) {
        *error_message_out = thread_error.message;
    }
    return thread_error.message;
}

// RAII owners so every early return and exception path in open() cleans up.
struct ModelDeleter {
    void operator()(llama_model* model) const { llama_model_free(model); }
};
struct ContextDeleter {
    void operator()(llama_context* ctx) const { llama_free(ctx); }
};
struct PoolDeleter {
    void operator()(ggml_threadpool* pool) const { ggml_threadpool_free(pool); }
};
struct MallocDeleter {
    void operator()(void* p) const { std::free(p); }
};
using ModelPtr = std::unique_ptr<llama_model, ModelDeleter>;
using ContextPtr = std::unique_ptr<llama_context, ContextDeleter>;
using PoolPtr = std::unique_ptr<ggml_threadpool, PoolDeleter>;
// Owns malloc()ed output buffers until handed to the caller (freed by sk_free).
using FloatsPtr = std::unique_ptr<float[], MallocDeleter>;

struct EmbedderOptions {
    enum llama_pooling_type pooling = LLAMA_POOLING_TYPE_UNSPECIFIED;
    bool normalize = true;
    bool truncate = false;
    unsigned int context = 0;   // 0 = default
    unsigned int threads = 0;   // 0 = default
};

bool parse_uint_option(const std::string& value, unsigned int* out) {
    if (value.empty() || value.size() > 9) {
        return false;
    }
    for (char c : value) {
        if (c < '0' || c > '9') {
            return false;
        }
    }
    *out = static_cast<unsigned int>(std::strtoul(value.c_str(), nullptr, 10));
    return *out > 0;
}

bool parse_options(const char* options, EmbedderOptions* out, std::string* error) {
    if (options == nullptr || options[0] == '\0') {
        return true;
    }
    std::string spec(options);
    size_t start = 0;
    while (start <= spec.size()) {
        size_t end = spec.find(',', start);
        if (end == std::string::npos) {
            end = spec.size();
        }
        std::string item = spec.substr(start, end - start);
        start = end + 1;
        if (item.empty()) {
            continue;
        }
        size_t eq = item.find('=');
        if (eq == std::string::npos) {
            *error = "invalid option '" + item + "': expected key=value";
            return false;
        }
        std::string key = item.substr(0, eq);
        std::string value = item.substr(eq + 1);
        if (key == "pooling") {
            if (value == "mean") {
                out->pooling = LLAMA_POOLING_TYPE_MEAN;
            } else if (value == "cls") {
                out->pooling = LLAMA_POOLING_TYPE_CLS;
            } else if (value == "last") {
                out->pooling = LLAMA_POOLING_TYPE_LAST;
            } else {
                *error = "invalid pooling '" + value + "': expected mean|cls|last";
                return false;
            }
        } else if (key == "normalize") {
            if (value != "0" && value != "1") {
                *error = "invalid normalize '" + value + "': expected 0 or 1";
                return false;
            }
            out->normalize = value == "1";
        } else if (key == "truncate") {
            if (value != "0" && value != "1") {
                *error = "invalid truncate '" + value + "': expected 0 or 1";
                return false;
            }
            out->truncate = value == "1";
        } else if (key == "context") {
            if (!parse_uint_option(value, &out->context)) {
                *error = "invalid context '" + value + "': expected a positive integer";
                return false;
            }
        } else if (key == "threads") {
            if (!parse_uint_option(value, &out->threads)) {
                *error = "invalid threads '" + value + "': expected a positive integer";
                return false;
            }
        } else {
            *error = "unknown option '" + key + "'";
            return false;
        }
    }
    return true;
}

// Tokenize one input, applying the per-input token limit and the truncation
// policy. Returns 0 on success, -1 after set_embed_error().
int tokenize_one(sk_embedder* embedder, const char* text, size_t index,
        std::vector<llama_token>* tokens_out) {
    std::vector<llama_token>& tokens = *tokens_out;
    const size_t text_size = std::strlen(text);
    if (text_size > INT32_MAX) {
        set_embed_error("input %zu is too large", index);
        return -1;
    }
    const int32_t text_len = static_cast<int32_t>(text_size);
    int32_t n_tokens = -llama_tokenize(
        embedder->vocab, text, text_len, nullptr, 0, true, true);
    if (n_tokens <= 0) {
        set_embed_error("failed to tokenize input %zu", index);
        return -1;
    }
    tokens.resize(static_cast<size_t>(n_tokens));
    if (llama_tokenize(embedder->vocab, text, text_len, tokens.data(), n_tokens,
            true, true) < 0) {
        set_embed_error("failed to tokenize input %zu", index);
        return -1;
    }
    if (n_tokens > static_cast<int32_t>(embedder->max_tokens)) {
        if (!embedder->truncate) {
            set_embed_error(
                "input %zu produces %d tokens, exceeding the limit of %u; "
                "shorten the input, raise context=, or pass truncate=1",
                index, n_tokens, embedder->max_tokens);
            return -1;
        }
        // Preserve the terminal special token ([SEP]/EOS): the model never
        // trained without it, and last-token pooling would otherwise pool an
        // arbitrary mid-text token.
        const llama_token last = tokens.back();
        const bool keep_terminal =
            (llama_vocab_get_add_eos(embedder->vocab)
                && last == llama_vocab_eos(embedder->vocab))
            || (llama_vocab_get_add_sep(embedder->vocab)
                && last == llama_vocab_sep(embedder->vocab));
        n_tokens = static_cast<int32_t>(embedder->max_tokens);
        tokens.resize(static_cast<size_t>(n_tokens));
        if (keep_terminal) {
            tokens[static_cast<size_t>(n_tokens - 1)] = last;
        }
    }
    return 0;
}

void normalize_row(float* row, unsigned int dim) {
    double sum = 0.0;
    for (unsigned int i = 0; i < dim; ++i) {
        sum += static_cast<double>(row[i]) * static_cast<double>(row[i]);
    }
    const double norm = std::sqrt(sum);
    if (norm > 0.0) {
        const float inv = static_cast<float>(1.0 / norm);
        for (unsigned int i = 0; i < dim; ++i) {
            row[i] *= inv;
        }
    }
}

struct BatchOwner {
    llama_batch batch;
    explicit BatchOwner(int32_t capacity)
        : batch(llama_batch_init(capacity, 0, 1)) {}
    ~BatchOwner() { llama_batch_free(batch); }
    BatchOwner(const BatchOwner&) = delete;
    BatchOwner& operator=(const BatchOwner&) = delete;
};

// Run one packed batch and write each sequence's pooled embedding into its
// input's row of out. seq_to_index maps batch sequence id -> input index.
int flush_batch(sk_embedder* embedder, const llama_batch& batch,
        const std::vector<size_t>& seq_to_index, float* out) {
    if (seq_to_index.empty()) {
        return 0;
    }
    llama_memory_clear(llama_get_memory(embedder->ctx), true);
    const int32_t status = embedder->is_encoder
        ? llama_encode(embedder->ctx, batch)
        : llama_decode(embedder->ctx, batch);
    if (status != 0) {
        set_embed_error(
            "model inference failed on inputs %zu..%zu (status %d)",
            seq_to_index.front(), seq_to_index.back(), status);
        return -1;
    }
    for (size_t s = 0; s < seq_to_index.size(); ++s) {
        const float* emb = llama_get_embeddings_seq(
            embedder->ctx, static_cast<llama_seq_id>(s));
        if (emb == nullptr) {
            set_embed_error(
                "model returned no pooled embedding for input %zu",
                seq_to_index[s]);
            return -1;
        }
        float* row = out + seq_to_index[s] * embedder->dim;
        std::memcpy(row, emb, sizeof(float) * embedder->dim);
        if (embedder->normalize) {
            normalize_row(row, embedder->dim);
        }
    }
    return 0;
}

// Embed count texts into out (count rows of dim floats), packing as many
// inputs per forward pass as fit the token budget and sequence limit.
int embed_many(sk_embedder* embedder, const char** texts, size_t count,
        float* out) {
    // n_batch == n_ubatch == n_ctx == max_tokens: any single valid input fits.
    const size_t token_capacity = embedder->max_tokens;
    const size_t seq_capacity = llama_n_seq_max(embedder->ctx);
    BatchOwner owner(static_cast<int32_t>(token_capacity));
    llama_batch& batch = owner.batch;
    batch.n_tokens = 0;
    std::vector<size_t> seq_to_index;
    std::vector<llama_token> tokens;
    for (size_t i = 0; i < count; ++i) {
        if (texts[i] == nullptr) {
            EMBED_ERR("texts[%zu] must not be NULL", i)
        }
        if (tokenize_one(embedder, texts[i], i, &tokens) != 0) {
            return -1;
        }
        if (static_cast<size_t>(batch.n_tokens) + tokens.size() > token_capacity
                || seq_to_index.size() == seq_capacity) {
            if (flush_batch(embedder, batch, seq_to_index, out) != 0) {
                return -1;
            }
            seq_to_index.clear();
            batch.n_tokens = 0;
        }
        const llama_seq_id seq = static_cast<llama_seq_id>(seq_to_index.size());
        for (size_t j = 0; j < tokens.size(); ++j) {
            const int32_t slot = batch.n_tokens + static_cast<int32_t>(j);
            batch.token[slot] = tokens[j];
            batch.pos[slot] = static_cast<llama_pos>(j);
            batch.n_seq_id[slot] = 1;
            batch.seq_id[slot][0] = seq;
            batch.logits[slot] = 1;
        }
        batch.n_tokens += static_cast<int32_t>(tokens.size());
        seq_to_index.push_back(i);
    }
    return flush_batch(embedder, batch, seq_to_index, out);
}

// CPUs actually available to this process (cpuset/taskset affinity), falling
// back to hardware_concurrency().
unsigned int available_cpu_count() {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        const int count = CPU_COUNT(&set);
        if (count > 0) {
            return static_cast<unsigned int>(count);
        }
    }
#endif
    const unsigned int hw = std::thread::hardware_concurrency();
    return hw != 0 ? hw : 4;
}

// How many pool threads the OS will actually grant: ggml_threadpool_new()
// aborts the process (GGML_ASSERT) on pthread_create failure, e.g. under a
// container pids limit or RLIMIT_NPROC. Park desired-1 probe threads (ggml
// spawns desired-1 secondaries) and count how many the OS allows. Returns
// [1, desired]; a one-thread pool spawns nothing and cannot abort.
unsigned int grantable_thread_count(unsigned int desired) noexcept {
    if (desired <= 1) {
        return 1;
    }
    std::mutex m;
    std::condition_variable cv;
    bool release = false;
    std::vector<std::thread> probes;
    try {
        probes.reserve(desired - 1);
        for (unsigned int i = 0; i + 1 < desired; ++i) {
            // Park so all requested slots are held simultaneously.
            probes.emplace_back([&m, &cv, &release] {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [&release] { return release; });
            });
        }
    } catch (...) {
        // std::thread throws when the OS refuses another thread.
    }
    {
        std::lock_guard<std::mutex> lk(m);
        release = true;
    }
    cv.notify_all();
    for (std::thread& t : probes) {
        t.join();
    }
    return static_cast<unsigned int>(probes.size()) + 1;
}

} // namespace

sk_embedder_t* sk_embedder_open(const char* model_path, const char* options,
        const char** error_message_out) {
    try {
        thread_error.code = 0;
        thread_error.message[0] = '\0';
        if (error_message_out != nullptr) {
            *error_message_out = nullptr;
        }
        if (model_path == nullptr || model_path[0] == '\0') {
            set_open_error(error_message_out, "model_path must not be empty");
            return nullptr;
        }
        EmbedderOptions opts;
        std::string option_error;
        if (!parse_options(options, &opts, &option_error)) {
            set_open_error(error_message_out, "%s", option_error.c_str());
            return nullptr;
        }

        if (const char* feature = missing_cpu_feature()) {
            set_open_error(error_message_out,
                "this build's embedding kernels require the '%s' CPU "
                "extension, which this CPU (or its hypervisor CPU model) "
                "does not expose", feature);
            return nullptr;
        }

        backend_init_once();

        llama_model_params model_params = llama_model_default_params();
        ModelPtr model(llama_model_load_from_file(model_path, model_params));
        if (model == nullptr) {
            set_open_error(error_message_out,
                "failed to load GGUF model from '%s'", model_path);
            return nullptr;
        }
        if (llama_model_has_encoder(model.get())
                && llama_model_has_decoder(model.get())) {
            set_open_error(error_message_out,
                "encoder-decoder models are not supported for embedding");
            return nullptr;
        }

        const int32_t n_ctx_train = llama_model_n_ctx_train(model.get());
        unsigned int n_ctx = opts.context != 0 ? opts.context : kDefaultMaxContext;
        if (n_ctx_train > 0 && n_ctx > static_cast<unsigned int>(n_ctx_train)) {
            n_ctx = static_cast<unsigned int>(n_ctx_train);
        }
        // Clamp threads= to the CPU budget: oversubscribing a CPU-bound pool
        // only hurts.
        const unsigned int cpu_budget = available_cpu_count();
        unsigned int n_threads = opts.threads != 0 ? opts.threads : cpu_budget;
        if (n_threads > cpu_budget) {
            n_threads = cpu_budget;
        }
        if (n_threads > GGML_MAX_N_THREADS) {
            n_threads = GGML_MAX_N_THREADS;
        }
        // Reduce to what the OS will grant (down to 1) instead of letting
        // ggml's GGML_ASSERT abort the process.
        const unsigned int grantable = grantable_thread_count(n_threads);
        if (grantable < n_threads) {
            LOG_WARN << "embedder: OS granted only " << grantable
                     << " of " << n_threads << " compute threads "
                     << "(container pids/RLIMIT_NPROC limit); using "
                     << grantable;
            n_threads = grantable;
        }

        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.embeddings = true;
        ctx_params.n_ctx = n_ctx;
        // Non-causal encoders require the whole sequence in one physical batch.
        ctx_params.n_batch = n_ctx;
        ctx_params.n_ubatch = n_ctx;
        ctx_params.n_threads = static_cast<int32_t>(n_threads);
        ctx_params.n_threads_batch = static_cast<int32_t>(n_threads);
        ctx_params.pooling_type = opts.pooling;
        // Pack several inputs per forward pass. kv_unified keeps n_ctx a
        // shared pool (llama otherwise divides it by n_seq_max); the cap must
        // not exceed n_batch — llama reserves n_seq_max output slots at init.
        ctx_params.n_seq_max = static_cast<uint32_t>(std::min<size_t>(
            {64, llama_max_parallel_sequences(), n_ctx}));
        ctx_params.kv_unified = true;

        // The pool owner is declared before ctx: owners unwind in reverse
        // order, and the context must be freed before its attached pool.
        PoolPtr pool;
        ContextPtr ctx(llama_init_from_model(model.get(), ctx_params));
        if (ctx == nullptr) {
            set_open_error(error_message_out,
                "failed to create a llama context for the model");
            return nullptr;
        }

        // Resident pool for the embedder's lifetime; poll=0 makes idle workers
        // sleep on a condvar instead of busy-polling.
        ggml_threadpool_params pool_params = ggml_threadpool_params_default(
            static_cast<int>(n_threads));
        pool_params.poll = 0;
        pool.reset(ggml_threadpool_new(&pool_params));
        if (pool == nullptr) {
            set_open_error(error_message_out, "failed to create the compute thread pool");
            return nullptr;
        }
        llama_attach_threadpool(ctx.get(), pool.get(), nullptr);

        // Rank pooling produces n_cls_out reranking scores, not an n_embd
        // vector; it has no valid mapping onto this API.
        const enum llama_pooling_type effective_pooling = llama_pooling_type(ctx.get());
        if (effective_pooling == LLAMA_POOLING_TYPE_NONE
                || effective_pooling == LLAMA_POOLING_TYPE_RANK) {
            set_open_error(error_message_out,
                effective_pooling == LLAMA_POOLING_TYPE_NONE
                    ? "model metadata selects pooling type 'none'; pass an "
                      "explicit pooling=mean|cls|last override"
                    : "model metadata selects pooling type 'rank'; reranking "
                      "models are not supported by the embedding API");
            return nullptr;
        }
        const int32_t n_embd = llama_model_n_embd_out(model.get());
        if (n_embd <= 0) {
            set_open_error(error_message_out, "model reports a non-positive embedding size");
            return nullptr;
        }

        auto embedder = std::make_unique<sk_embedder>();
        embedder->model = model.get();
        embedder->ctx = ctx.get();
        embedder->pool = pool.get();
        embedder->vocab = llama_model_get_vocab(model.get());
        embedder->dim = static_cast<unsigned int>(n_embd);
        embedder->max_tokens = n_ctx;
        embedder->is_encoder = llama_model_has_encoder(model.get())
            && !llama_model_has_decoder(model.get());
        embedder->normalize = opts.normalize;
        embedder->truncate = opts.truncate;
        // Nothing below can throw: hand ownership to the embedder.
        model.release();
        ctx.release();
        pool.release();
        return embedder.release();
    } catch (const std::exception& ex) {
        set_open_error(error_message_out, "%s", ex.what());
        return nullptr;
    } catch (...) {
        set_open_error(error_message_out, "unknown error while opening the model");
        return nullptr;
    }
}

void sk_embedder_close(sk_embedder_t* embedder) {
    if (embedder == nullptr) {
        return;
    }
    try {
        // The context must go before the pool it is attached to.
        llama_free(embedder->ctx);
        ggml_threadpool_free(embedder->pool);
        llama_model_free(embedder->model);
    } catch (...) {
        // Never propagate across the C ABI; still release the handle.
    }
    delete embedder;
}

int sk_embedder_dim(sk_embedder_t* embedder, unsigned int* dim_out) {
    EMBED_DECL
    if (dim_out == nullptr) {
        EMBED_ERR("dim_out must not be NULL")
    }
    *dim_out = embedder->dim;
    return 0;
}

int sk_embedder_max_tokens(sk_embedder_t* embedder, unsigned int* max_tokens_out) {
    EMBED_DECL
    if (max_tokens_out == nullptr) {
        EMBED_ERR("max_tokens_out must not be NULL")
    }
    *max_tokens_out = embedder->max_tokens;
    return 0;
}

int sk_embed_text(sk_embedder_t* embedder, const char* text,
        float** vec_out, size_t* dim_out) {
    if (vec_out != nullptr) {
        *vec_out = nullptr;
    }
    EMBED_DECL
    if (text == nullptr || vec_out == nullptr || dim_out == nullptr) {
        EMBED_ERR("text, vec_out, and dim_out must not be NULL")
    }
    try {
        FloatsPtr vec(static_cast<float*>(
            std::malloc(sizeof(float) * embedder->dim)));
        if (vec == nullptr) {
            EMBED_ERR("out of memory")
        }
        std::lock_guard<std::mutex> lock(embedder->inference_mutex);
        if (embed_many(embedder, &text, 1, vec.get()) != 0) {
            return -1;
        }
        *vec_out = vec.release();
        *dim_out = embedder->dim;
        return 0;
    } catch (const std::exception& ex) {
        EMBED_ERR("%s", ex.what())
    } catch (...) {
        EMBED_ERR("unknown error while embedding")
    }
}

int sk_embed_texts(sk_embedder_t* embedder, const char** texts, size_t count,
        float** vecs_out, size_t* dim_out) {
    if (vecs_out != nullptr) {
        *vecs_out = nullptr;
    }
    EMBED_DECL
    if (texts == nullptr || vecs_out == nullptr || dim_out == nullptr) {
        EMBED_ERR("texts, vecs_out, and dim_out must not be NULL")
    }
    if (count == 0) {
        EMBED_ERR("count must be greater than zero")
    }
    if (count > SIZE_MAX / (sizeof(float) * embedder->dim)) {
        EMBED_ERR("count is too large")
    }
    try {
        FloatsPtr vecs(static_cast<float*>(
            std::malloc(sizeof(float) * embedder->dim * count)));
        if (vecs == nullptr) {
            EMBED_ERR("out of memory")
        }
        std::lock_guard<std::mutex> lock(embedder->inference_mutex);
        if (embed_many(embedder, texts, count, vecs.get()) != 0) {
            return -1;
        }
        *vecs_out = vecs.release();
        *dim_out = embedder->dim;
        return 0;
    } catch (const std::exception& ex) {
        EMBED_ERR("%s", ex.what())
    } catch (...) {
        EMBED_ERR("unknown error while embedding")
    }
}

int sk_embedder_error(sk_embedder_t* embedder) {
    if (embedder == nullptr) {
        return -1;
    }
    return thread_error.code;
}

const char* sk_embedder_error_message(sk_embedder_t* embedder) {
    if (embedder == nullptr) {
        return "";
    }
    return thread_error.message;
}
