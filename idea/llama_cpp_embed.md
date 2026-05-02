# Generating Text Embeddings with llama.cpp

This document explains how the `emgen` utility uses the llama.cpp framework to
generate text embeddings, and how its dynamic library loading mechanism works.
All code examples reference the actual source files of the utility.

---

## Part 1: How llama.cpp Embedding Generation Works

### Overview

llama.cpp is a C/C++ inference framework for running GGUF-format language
models. While it is primarily known for text generation, it also supports
**embedding extraction** — producing fixed-dimensional float vectors that
represent the semantic meaning of input text.

The embedding workflow has five stages:

    1. Initialize the backend
    2. Load a model and create a context
    3. Tokenize input text
    4. Decode (run the model's forward pass)
    5. Retrieve the embedding vectors

### 1.1 Backend Initialization

Before any other llama.cpp call, the compute backend must be initialized:

```cpp
// main.cpp:243
api.backend_init();
```

This sets up internal state for the tensor computation library (ggml) that
llama.cpp is built on. It must be called once at startup and paired with
`backend_free()` at shutdown.

### 1.2 Loading a Model

A GGUF model file is loaded into memory via `llama_model_load_from_file`.
The function returns an opaque `llama_model *` handle:

```cpp
// main.cpp:97-103
static llama_model * load_model(const llama_api & api, const char * path) {
    llama_model_params params = api.model_default_params();
    if (!g_verbose) {
        params.progress_callback = silent_progress;
    }
    return api.model_load_from_file(path, params);
}
```

`llama_model_params` controls GPU layer offloading, memory mapping, and
progress reporting. The default params produced by `model_default_params()`
are suitable for CPU-only inference. The `progress_callback` field is set to
a no-op function to suppress the progress dots that llama.cpp prints to stderr
during loading.

Once the model is loaded, its **embedding dimension** can be queried:

```cpp
// main.cpp:253
const int n_embd = api.model_n_embd(model);
```

This returns the number of floats in each embedding vector. It is a fixed
property of the model architecture (e.g. 384 for all-MiniLM-L6-v2, 768 for
nomic-embed-text). Every embedding the model produces will have exactly this
many components.

### 1.3 Creating an Inference Context

A context holds the working memory for inference. For embeddings, several
parameters must be set explicitly:

```cpp
// main.cpp:114-124
static llama_context * create_context(const llama_api & api, llama_model * model) {
    llama_context_params params = api.context_default_params();
    params.embeddings = true;
    params.n_batch    = MAX_BATCH_TOKENS;   // 512
    params.n_seq_max  = MAX_BATCH_SEQS;     // 64
    params.n_ctx = MAX_BATCH_TOKENS * MAX_BATCH_SEQS;  // 32768
    return api.init_from_model(model, params);
}
```

**Key parameters explained:**

- **`embeddings = true`** — This is required. Without it,
  `llama_get_embeddings_seq` returns NULL. It tells llama.cpp to compute and
  store embedding vectors during the forward pass rather than discarding them.

- **`n_batch`** (512) — The maximum number of tokens that can be submitted in
  a single `llama_decode` call. This is the token budget for batching multiple
  input lines together.

- **`n_seq_max`** (64) — The maximum number of independent sequences (input
  lines) that can be processed in a single batch. Each sequence gets a unique
  ID and produces its own pooled embedding.

- **`n_ctx`** (512 * 64 = 32768) — The total context window size. llama.cpp
  divides this evenly among sequences: each sequence gets `n_ctx / n_seq_max`
  = 512 tokens, which matches the training context length of most embedding
  models. Two constraints apply:
  - `n_ctx` must be divisible by `n_seq_max` (llama.cpp will round down and
    warn if not)
  - `n_ctx / n_seq_max` should be >= `n_ctx_train` (the model's training
    context length) to avoid a warning about underutilizing the model

### 1.4 Tokenization

Text must be converted to token IDs before the model can process it. The
vocabulary is obtained from the model, and `llama_tokenize` performs the
conversion:

```cpp
// main.cpp:133-147
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
```

**Parameters:**

- **`add_special = true`** — Inserts the model's special tokens (BOS/EOS) at
  the appropriate positions. Embedding models expect these tokens to be
  present.
- **`parse_special = false`** — Treats the input as plain text. Special token
  strings in the input (like `<s>`) are tokenized as regular text, not
  interpreted as control tokens.

The function returns a negative value if the output buffer is too small (the
absolute value indicates how many tokens would be needed). We allocate
`text.size() + 16` tokens as a conservative upper bound, since tokens are
typically 3-4 characters long.

### 1.5 Batching Multiple Sequences

This is the core optimization in `emgen`. Instead of running `llama_decode`
once per input line, multiple lines are packed into a single batch using
**sequence IDs** to keep them separate:

```cpp
// main.cpp:179-191
batch.n_tokens = 0;
for (int seq = 0; seq < (int)pending.size(); seq++) {
    const auto & toks = pending[seq].tokens;
    for (int i = 0; i < (int)toks.size(); i++) {
        int idx = batch.n_tokens;
        batch.token[idx]     = toks[i];
        batch.pos[idx]       = i;        // position within this sequence
        batch.n_seq_id[idx]  = 1;        // belongs to exactly 1 sequence
        batch.seq_id[idx][0] = seq;      // which sequence (0, 1, 2, ...)
        batch.logits[idx]    = 1;        // produce output for this token
        batch.n_tokens++;
    }
}
```

**The `llama_batch` struct has these arrays:**

| Field       | Purpose                                                    |
|-------------|------------------------------------------------------------|
| `token`     | Token ID                                                   |
| `pos`       | Position within its sequence (each sequence starts at 0)   |
| `n_seq_id`  | How many sequences this token belongs to (always 1 here)   |
| `seq_id`    | Which sequence(s) this token belongs to                    |
| `logits`    | Whether to produce output for this token (1 = yes)         |

Setting `logits = 1` for all tokens is required for embedding mode. Without
it, llama.cpp overrides the setting and prints a warning.

**Batching budget:** lines are accumulated until the next one would exceed
`MAX_BATCH_TOKENS` (512), then the batch is flushed:

```cpp
// main.cpp:292-294
if (pending_token_count + n_tokens > MAX_BATCH_TOKENS && !pending.empty()) {
    flush_batch(api, ctx, batch, pending, pending_token_count, n_embd);
}
```

For short inputs (10-30 tokens per sentence), this means 15-50 lines per
decode call. For longer inputs like RAG chunks (200-500 tokens), it drops to
1-2 per call. The batching advantage is proportional to how many sequences
fit within the token budget.

### 1.6 Decoding and Retrieving Embeddings

After packing the batch, a single `llama_decode` call runs the model's
forward pass on all sequences at once:

```cpp
// main.cpp:193
api.decode(ctx, batch)
```

The model computes a per-token hidden state for every token in the batch,
then **pools** (averages) the hidden states within each sequence to produce
one embedding vector per sequence. The pooling strategy is determined by the
model's metadata (typically mean pooling for embedding models).

The pooled embedding for each sequence is retrieved by its sequence ID:

```cpp
// main.cpp:204-205
const float * embd = api.get_embeddings_seq(ctx, seq);
```

This returns a pointer to `n_embd` contiguous floats inside the context's
internal memory. The pointer is valid until the next `llama_decode` call or
until the context is freed.

### 1.7 Memory Management Between Batches

After processing a batch, the context's internal memory (KV cache) must be
cleared before the next batch. Without this, state from previous sequences
would leak into new ones:

```cpp
// main.cpp:220
api.memory_clear(api.get_memory(ctx), true);
```

`llama_get_memory` returns a handle to the context's memory state, and
`llama_memory_clear` resets it. The `true` argument indicates a full clear.

### 1.8 Logging Control

llama.cpp generates substantial log output on stderr: model metadata, backend
info, memory allocation details, and warnings. The `llama_log_set` function
installs a custom callback that controls what gets printed:

```cpp
// main.cpp:46-49
static void log_callback(ggml_log_level level, const char * text, void *) {
    if (g_verbose || level >= GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}
```

Log levels are defined in ggml.h:

| Level                  | Value | Meaning                      |
|------------------------|-------|------------------------------|
| `GGML_LOG_LEVEL_DEBUG` | 1     | Detailed internal info       |
| `GGML_LOG_LEVEL_INFO`  | 2     | Normal operational messages  |
| `GGML_LOG_LEVEL_WARN`  | 3     | Potential issues             |
| `GGML_LOG_LEVEL_ERROR` | 4     | Failures                     |

In quiet mode (default), only errors are shown. The `-v` flag enables all
levels.

Note: the model loading progress indicator (dots printed to stderr) uses a
separate mechanism — the `progress_callback` field on `llama_model_params`.
It is suppressed by installing a no-op callback:

```cpp
// main.cpp:92-93
static bool silent_progress(float, void *) {
    return true;
}
```

Returning `true` means "continue loading"; returning `false` would abort.

### 1.9 Cleanup

Resources are freed in reverse order of creation:

```cpp
// main.cpp:304-308
api.batch_free(batch);       // free the batch token buffer
api.free(ctx);               // free the inference context
api.model_free(model);       // unload the model from memory
api.backend_free();          // shut down the compute backend
```

### 1.10 Complete Lifecycle Summary

```
backend_init
  |
  v
model_load_from_file  -->  model_n_embd (query dimensions)
  |                        model_get_vocab (get tokenizer)
  v
init_from_model (embeddings=true)
  |
  v
+-- for each batch of lines: --------+
|   tokenize each line                |
|   pack tokens into llama_batch      |
|   llama_decode (forward pass)       |
|   get_embeddings_seq for each seq   |
|   memory_clear (reset KV cache)     |
+-----------loop----------------------+
  |
  v
batch_free -> free -> model_free -> backend_free
```

---

## Part 2: Dynamic Loading of llama.cpp Libraries

### Problem

Normally, a C++ program that uses llama.cpp links directly against its shared
libraries at build time:

```makefile
LIBS = -lllama -lggml -lggml-base -lggml-cpu
```

This creates a hard dependency: the binary won't start unless the linker can
find these `.so` files at load time (via `LD_LIBRARY_PATH` or system paths).
This is inconvenient for a research utility where the libraries may be built
locally and not installed system-wide.

### Solution

The `emgen` utility uses **runtime dynamic loading** (`dlopen`/`dlsym`) to
load llama.cpp libraries from a directory specified as a command-line
argument. The binary itself links only against `-ldl` (the dynamic loader)
and has no dependency on llama.cpp at link time:

```
$ ldd emgen
    libstdc++.so.6 => /lib/...
    libc.so.6 => /lib/...
    (no libllama, no libggml)
```

### Architecture

The implementation is split across two files:

- **`llama-link.h`** — declares the `llama_api` struct and load/unload
  functions
- **`llama-link.cpp`** — implements the loading logic

### 2.1 The `llama_api` Struct

Instead of calling llama.cpp functions directly, all calls go through a
struct of function pointers:

```cpp
// llama-link.h:23-59
struct llama_api {
    // Backend lifecycle
    decltype(&llama_backend_init)          backend_init;
    decltype(&llama_backend_free)          backend_free;

    // Logging
    decltype(&llama_log_set)               log_set;

    // Model
    decltype(&llama_model_default_params)  model_default_params;
    decltype(&llama_model_load_from_file)  model_load_from_file;
    decltype(&llama_model_free)            model_free;
    decltype(&llama_model_n_embd)          model_n_embd;
    decltype(&llama_model_get_vocab)       model_get_vocab;

    // Context
    decltype(&llama_context_default_params) context_default_params;
    decltype(&llama_init_from_model)        init_from_model;
    decltype(&llama_free)                   free;

    // Batch
    decltype(&llama_batch_init)            batch_init;
    decltype(&llama_batch_free)            batch_free;

    // Tokenization
    decltype(&llama_tokenize)              tokenize;

    // Inference
    decltype(&llama_decode)                decode;

    // Memory
    decltype(&llama_get_memory)            get_memory;
    decltype(&llama_memory_clear)          memory_clear;

    // Embeddings
    decltype(&llama_get_embeddings_seq)    get_embeddings_seq;
};
```

Each field uses `decltype(&llama_xxx)` to automatically derive the correct
function pointer type from the declaration in `llama.h`. This guarantees
type safety — if the llama.cpp API changes, the compiler catches the
mismatch.

The header still includes `llama.h`, but only for type definitions
(`llama_model`, `llama_context`, `llama_batch`, `llama_model_params`, etc.).
The function declarations in `llama.h` are never linked against — they just
provide the types that `decltype` extracts.

### 2.2 Library Loading Order

llama.cpp is split into several shared libraries with dependencies between
them. They must be loaded in the correct order:

```cpp
// llama-link.cpp:31-36
static const char * lib_names[MAX_LIBS] = {
    "libggml-base.so",   // 1. low-level tensor operations
    "libggml.so",        // 2. ggml core (depends on ggml-base)
    "libggml-cpu.so",    // 3. CPU backend (depends on ggml, ggml-base)
    "libllama.so",       // 4. main library (depends on all of the above)
};
```

### 2.3 dlopen Flags

Each library is opened with `RTLD_LAZY | RTLD_GLOBAL`:

```cpp
// llama-link.cpp:76
lib_handles[i] = dlopen(path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
```

- **`RTLD_LAZY`** — Symbols are resolved when first called, not at load time.
  This makes startup faster since we only need a small subset of the symbols
  in each library.

- **`RTLD_GLOBAL`** — Symbols from this library are made available for
  resolving references in subsequently loaded libraries. This is critical:
  when `libllama.so` is loaded, it has unresolved references to functions in
  `libggml.so` and `libggml-base.so`. Without `RTLD_GLOBAL`, those references
  would fail with "undefined symbol" errors. This flag makes `dlopen`
  function like the regular dynamic linker, where all loaded libraries can
  see each other's symbols.

### 2.4 Symbol Resolution

After all libraries are loaded, function pointers are resolved from
`libllama.so` using a templated helper:

```cpp
// llama-link.cpp:45-58
template <typename T>
static bool resolve(void * handle, const char * name, T & out) {
    dlerror();   // clear any previous error
    void * sym = dlsym(handle, name);
    const char * err = dlerror();
    if (err) {
        std::fprintf(stderr, "llama-link: failed to resolve '%s': %s\n",
                     name, err);
        out = nullptr;
        return false;
    }
    out = reinterpret_cast<T>(sym);
    return true;
}
```

**Why `dlerror()` is called twice:**

1. Before `dlsym` — clears any stale error from a previous call. Without
   this, a leftover error string could be mistaken for a failure from the
   current `dlsym`.
2. After `dlsym` — checks if the symbol resolution actually failed. We check
   `dlerror()` rather than testing `sym == NULL` because `dlsym` can
   legitimately return NULL for symbols whose value is zero.

The template parameter `T` is deduced from the `llama_api` field being
assigned, so the `reinterpret_cast` converts the `void *` from `dlsym` to
the correct function pointer type.

All 18 symbols are resolved in a single pass. On failure, resolution
continues to report all missing symbols rather than stopping at the first:

```cpp
// llama-link.cpp:87-108
bool ok = true;
ok &= resolve(llama, "llama_backend_init",          api.backend_init);
ok &= resolve(llama, "llama_backend_free",          api.backend_free);
// ... (16 more)
ok &= resolve(llama, "llama_get_embeddings_seq",    api.get_embeddings_seq);
```

### 2.5 Unloading

Libraries are closed in **reverse dependency order** — dependents first, then
their dependencies:

```cpp
// llama-link.cpp:118-126
void llama_link_unload() {
    for (int i = MAX_LIBS - 1; i >= 0; i--) {
        if (lib_handles[i]) {
            dlclose(lib_handles[i]);
            lib_handles[i] = nullptr;
        }
    }
}
```

`libllama.so` is closed first (it depends on the others), then `libggml-cpu`,
then `libggml`, and finally `libggml-base`. This ensures no library is
unloaded while another still references its symbols. The NULL check makes the
function safe to call even if loading partially failed.

### 2.6 How main.cpp Uses It

At startup, the library directory is passed as a command-line argument:

```
./emgen ./llama-libs ./model.gguf
```

The loading happens before any llama.cpp function is called:

```cpp
// main.cpp:237-240
llama_api api;
if (!llama_link_load(a.lib_dir, api)) {
    return 1;
}
```

From this point on, all llama.cpp calls go through the `api` struct. For
example, what was previously `llama_backend_init()` becomes
`api.backend_init()`:

```cpp
// main.cpp:242-243
api.log_set(log_callback, nullptr);
api.backend_init();
```

The `api` struct is passed by const reference to every function that needs
llama.cpp access (`load_model`, `create_context`, `tokenize`, `flush_batch`).

### 2.7 Build Configuration

The Makefile reflects the absence of link-time dependencies:

```makefile
CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
INCLUDES = -Illama-include
LIBS     = -ldl

TARGET = emgen
SRCS   = main.cpp llama-link.cpp

$(TARGET): $(SRCS) llama-link.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(SRCS) $(LIBS)
```

The only library linked is `-ldl`, which provides `dlopen`, `dlsym`,
`dlclose`, and `dlerror`. The `llama-include` directory provides the header
files needed for type definitions at compile time, but no llama.cpp `.so`
files are needed at link time.

### 2.8 Trade-offs

**Advantages of dynamic loading:**

- No `LD_LIBRARY_PATH` needed at runtime — the library path is explicit
- The binary runs on systems without llama.cpp installed — it fails gracefully
  with a clear error message if the libraries aren't found
- Different library versions can be tested by pointing at different directories

**Disadvantages:**

- Slightly more complex code — every function call goes through a pointer
  indirection
- No compile-time verification that the library exports the expected symbols
  — a missing or renamed function is only caught at runtime
- If the llama.cpp API changes (function signature or struct layout), the
  program compiles but may crash at runtime due to ABI mismatch. With direct
  linking, the linker would catch some of these issues earlier
