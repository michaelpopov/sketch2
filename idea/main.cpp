// emgen — command-line utility for generating text embeddings using llama.cpp
//
// Reads lines from stdin, generates a vector embedding for each line using a
// GGUF embedding model, and prints the first 4 components of each vector to
// stdout. Multiple lines are batched into a single decode call for efficiency:
// tokens from several input lines are packed into one batch with distinct
// sequence IDs, so the model processes them in parallel.
//
// llama.cpp libraries are loaded at runtime via dlopen (see llama-link.h),
// so the binary has no link-time dependency on them.
//
// Usage: emgen [-v] <lib_dir> <model.gguf>
//   -v       Enable verbose logging from llama.cpp (info + warnings)
//   lib_dir  Path to directory containing libllama.so and its dependencies
//   model    Path to a GGUF embedding model file

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "llama-link.h"

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Maximum number of tokens that can be submitted to llama_decode in one call.
static constexpr int MAX_BATCH_TOKENS = 512;

// Maximum number of independent sequences (input lines) in a single batch.
static constexpr int MAX_BATCH_SEQS = 64;

// Number of embedding components to print per line.
static constexpr int N_PRINT = 4;

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

static bool g_verbose = false;

// Custom log callback that suppresses llama.cpp's info/warning output unless
// the user passes -v. Errors are always printed so failures aren't silent.
static void log_callback(ggml_log_level level, const char * text, void * /*user_data*/) {
    if (g_verbose || level >= GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

struct args {
    bool         verbose;
    const char * lib_dir;
    const char * model_path;
};

static void print_usage(const char * prog) {
    std::fprintf(stderr, "Usage: %s [-v] <lib_dir> <model.gguf>\n", prog);
}

// Parse command-line arguments. Returns false and prints usage on error.
static bool parse_args(int argc, char ** argv, args & out) {
    out.verbose    = false;
    out.lib_dir    = nullptr;
    out.model_path = nullptr;

    int argi = 1;
    if (argi < argc && std::string(argv[argi]) == "-v") {
        out.verbose = true;
        argi++;
    }
    if (argi + 2 > argc) {
        print_usage(argv[0]);
        return false;
    }
    out.lib_dir    = argv[argi++];
    out.model_path = argv[argi];
    return true;
}

// ---------------------------------------------------------------------------
// Model and context initialization
// ---------------------------------------------------------------------------

// Silent progress callback — suppresses the dots printed during model loading.
// Returning true means "continue loading".
static bool silent_progress(float /*progress*/, void * /*user_data*/) {
    return true;
}

// Load a GGUF model from disk. Returns nullptr on failure.
static llama_model * load_model(const llama_api & api, const char * path) {
    llama_model_params params = api.model_default_params();
    if (!g_verbose) {
        params.progress_callback = silent_progress;
    }
    return api.model_load_from_file(path, params);
}

// Create an inference context configured for embedding extraction.
//
// Key settings:
//   - embeddings = true    required for get_embeddings_seq to work
//   - n_batch               max tokens per decode call (controls batching)
//   - n_seq_max             max distinct sequences per decode call
//   - n_ctx                 total context window; must be divisible by n_seq_max
//                           and large enough so each sequence gets at least
//                           n_ctx_train tokens (the model's training length)
static llama_context * create_context(const llama_api & api, llama_model * model) {
    llama_context_params params = api.context_default_params();
    params.embeddings = true;
    params.n_batch    = MAX_BATCH_TOKENS;
    params.n_seq_max  = MAX_BATCH_SEQS;
    // Give each sequence a full model-training-length context window.
    // This avoids the "n_ctx_seq < n_ctx_train" warning and ensures
    // n_ctx is divisible by n_seq_max (required by llama.cpp).
    params.n_ctx = MAX_BATCH_TOKENS * MAX_BATCH_SEQS;
    return api.init_from_model(model, params);
}

// ---------------------------------------------------------------------------
// Tokenization
// ---------------------------------------------------------------------------

// Tokenize a text string into a vector of token IDs.
// add_special=true inserts BOS/EOS tokens as the model expects.
// Returns an empty vector on failure.
static std::vector<llama_token> tokenize(const llama_api & api,
                                         const llama_vocab * vocab,
                                         const std::string & text) {
    int max_tokens = text.size() + 16;
    std::vector<llama_token> tokens(max_tokens);
    int n = api.tokenize(vocab, text.c_str(), text.size(),
                         tokens.data(), max_tokens,
                         /*add_special=*/true,
                         /*parse_special=*/false);
    if (n < 0) {
        return {};
    }
    tokens.resize(n);
    return tokens;
}

// ---------------------------------------------------------------------------
// Batch processing
// ---------------------------------------------------------------------------

// A pending input line together with its pre-computed tokens.
struct pending_seq {
    std::string              line;
    std::vector<llama_token> tokens;
};

// Pack all pending sequences into a single llama_batch and run decode.
// Each sequence is assigned a unique seq_id (0, 1, 2, ...) so the model can
// compute a separate pooled embedding per sequence. After decoding, retrieve
// each embedding with get_embeddings_seq(ctx, seq_id).
//
// On success, prints the first N_PRINT components of each embedding to stdout.
// On failure, logs an error and clears state so the next batch starts fresh.
static void flush_batch(const llama_api & api,
                        llama_context * ctx,
                        llama_batch & batch,
                        std::vector<pending_seq> & pending,
                        int & pending_token_count,
                        int n_embd) {
    if (pending.empty()) return;

    // Pack tokens from all pending sequences into the batch.
    // Each token gets:
    //   - pos:    position within its own sequence (starts at 0)
    //   - seq_id: which sequence this token belongs to
    //   - logits: 1 = produce output for this token (required for embeddings)
    batch.n_tokens = 0;
    for (int seq = 0; seq < (int)pending.size(); seq++) {
        const auto & toks = pending[seq].tokens;
        for (int i = 0; i < (int)toks.size(); i++) {
            int idx = batch.n_tokens;
            batch.token[idx]     = toks[i];
            batch.pos[idx]       = i;
            batch.n_seq_id[idx]  = 1;
            batch.seq_id[idx][0] = seq;
            batch.logits[idx]    = 1;
            batch.n_tokens++;
        }
    }

    if (api.decode(ctx, batch) != 0) {
        std::fprintf(stderr, "llama_decode failed for batch of %zu lines\n",
                     pending.size());
        api.memory_clear(api.get_memory(ctx), true);
        pending.clear();
        pending_token_count = 0;
        return;
    }

    // Retrieve the pooled embedding for each sequence and print it.
    int n_print = (n_embd < N_PRINT) ? n_embd : N_PRINT;
    for (int seq = 0; seq < (int)pending.size(); seq++) {
        const float * embd = api.get_embeddings_seq(ctx, seq);
        if (!embd) {
            std::fprintf(stderr, "Failed to get embeddings for line: %s\n",
                         pending[seq].line.c_str());
            continue;
        }
        for (int i = 0; i < n_print; i++) {
            if (i > 0) std::printf(" ");
            std::printf("%.6f", embd[i]);
        }
        std::printf("\n");
    }
    std::fflush(stdout);

    // Clear the KV cache so the next batch starts with a clean state.
    api.memory_clear(api.get_memory(ctx), true);
    pending.clear();
    pending_token_count = 0;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char ** argv) {
    args a;
    if (!parse_args(argc, argv, a)) {
        return 1;
    }
    g_verbose = a.verbose;

    // Dynamically load llama.cpp libraries from the specified directory.
    llama_api api;
    if (!llama_link_load(a.lib_dir, api)) {
        return 1;
    }

    api.log_set(log_callback, nullptr);
    api.backend_init();

    llama_model * model = load_model(api, a.model_path);
    if (!model) {
        std::fprintf(stderr, "Failed to load model: %s\n", a.model_path);
        api.backend_free();
        llama_link_unload();
        return 1;
    }

    const int n_embd = api.model_n_embd(model);

    llama_context * ctx = create_context(api, model);
    if (!ctx) {
        std::fprintf(stderr, "Failed to create context\n");
        api.model_free(model);
        api.backend_free();
        llama_link_unload();
        return 1;
    }

    const llama_vocab * vocab = api.model_get_vocab(model);
    llama_batch batch = api.batch_init(MAX_BATCH_TOKENS, 0, 1);

    std::vector<pending_seq> pending;
    int pending_token_count = 0;

    // Read lines from stdin and accumulate them into batches.
    // When the next line would exceed the token budget, flush the current
    // batch (decode + print) before adding the new line.
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }

        std::vector<llama_token> tokens = tokenize(api, vocab, line);
        if (tokens.empty()) {
            std::fprintf(stderr, "Tokenization failed for line: %s\n", line.c_str());
            continue;
        }
        int n_tokens = (int)tokens.size();

        if (n_tokens > MAX_BATCH_TOKENS) {
            std::fprintf(stderr, "Line too long (%d tokens, max %d): %s\n",
                         n_tokens, MAX_BATCH_TOKENS, line.c_str());
            continue;
        }

        // Flush if this line would overflow the batch token budget.
        if (pending_token_count + n_tokens > MAX_BATCH_TOKENS && !pending.empty()) {
            flush_batch(api, ctx, batch, pending, pending_token_count, n_embd);
        }

        pending.push_back({line, std::move(tokens)});
        pending_token_count += n_tokens;
    }

    // Process any remaining buffered lines.
    flush_batch(api, ctx, batch, pending, pending_token_count, n_embd);

    api.batch_free(batch);
    api.free(ctx);
    api.model_free(model);
    api.backend_free();
    llama_link_unload();

    return 0;
}
