// llama-link.cpp — runtime dynamic loading of llama.cpp shared libraries
//
// We use dlopen with RTLD_LAZY | RTLD_GLOBAL so that:
//   - RTLD_LAZY:  symbols are resolved on first use (faster startup)
//   - RTLD_GLOBAL: symbols from earlier libraries (ggml-base, ggml) are
//                  visible to later ones (libllama.so), satisfying their
//                  dependencies without needing LD_LIBRARY_PATH.
//
// Libraries must be loaded in dependency order:
//   1. libggml-base.so  — low-level tensor operations
//   2. libggml.so       — ggml core (depends on ggml-base)
//   3. libggml-cpu.so   — CPU backend (depends on ggml, ggml-base)
//   4. libllama.so      — the main library (depends on all of the above)
//
// All 18 API functions we use are exported by libllama.so.

#include "llama-link.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <dlfcn.h>

// ---------------------------------------------------------------------------
// Library handles
// ---------------------------------------------------------------------------

static constexpr int MAX_LIBS = 4;

// Names in dependency order. Loaded first-to-last, closed last-to-first.
static const char * lib_names[MAX_LIBS] = {
    "libggml-base.so",
    "libggml.so",
    "libggml-cpu.so",
    "libllama.so",
};

static void * lib_handles[MAX_LIBS] = {};

// ---------------------------------------------------------------------------
// Loading helpers
// ---------------------------------------------------------------------------

// Resolve a single symbol from a library handle. Returns false on failure.
template <typename T>
static bool resolve(void * handle, const char * name, T & out) {
    // Clear any previous error.
    dlerror();
    void * sym = dlsym(handle, name);
    const char * err = dlerror();
    if (err) {
        std::fprintf(stderr, "llama-link: failed to resolve '%s': %s\n", name, err);
        out = nullptr;
        return false;
    }
    out = reinterpret_cast<T>(sym);
    return true;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

bool llama_link_load(const char * lib_dir, llama_api & api) {
    std::memset(&api, 0, sizeof(api));

    std::string dir(lib_dir);
    // Ensure trailing slash.
    if (!dir.empty() && dir.back() != '/') {
        dir += '/';
    }

    // Load libraries in dependency order.
    for (int i = 0; i < MAX_LIBS; i++) {
        std::string path = dir + lib_names[i];
        lib_handles[i] = dlopen(path.c_str(), RTLD_LAZY | RTLD_GLOBAL);
        if (!lib_handles[i]) {
            std::fprintf(stderr, "llama-link: failed to load '%s': %s\n",
                         path.c_str(), dlerror());
            llama_link_unload();
            return false;
        }
    }

    // All API symbols live in libllama.so (the last loaded library).
    void * llama = lib_handles[MAX_LIBS - 1];
    bool ok = true;

    // Resolve each function pointer. We continue on failure to report all
    // missing symbols at once rather than stopping at the first one.
    ok &= resolve(llama, "llama_backend_init",          api.backend_init);
    ok &= resolve(llama, "llama_backend_free",          api.backend_free);
    ok &= resolve(llama, "llama_log_set",               api.log_set);
    ok &= resolve(llama, "llama_model_default_params",  api.model_default_params);
    ok &= resolve(llama, "llama_model_load_from_file",  api.model_load_from_file);
    ok &= resolve(llama, "llama_model_free",            api.model_free);
    ok &= resolve(llama, "llama_model_n_embd",          api.model_n_embd);
    ok &= resolve(llama, "llama_model_get_vocab",       api.model_get_vocab);
    ok &= resolve(llama, "llama_context_default_params", api.context_default_params);
    ok &= resolve(llama, "llama_init_from_model",       api.init_from_model);
    ok &= resolve(llama, "llama_free",                  api.free);
    ok &= resolve(llama, "llama_batch_init",            api.batch_init);
    ok &= resolve(llama, "llama_batch_free",            api.batch_free);
    ok &= resolve(llama, "llama_tokenize",              api.tokenize);
    ok &= resolve(llama, "llama_decode",                api.decode);
    ok &= resolve(llama, "llama_get_memory",            api.get_memory);
    ok &= resolve(llama, "llama_memory_clear",          api.memory_clear);
    ok &= resolve(llama, "llama_get_embeddings_seq",    api.get_embeddings_seq);

    if (!ok) {
        llama_link_unload();
        return false;
    }

    return true;
}

void llama_link_unload() {
    // Close in reverse order so dependents are closed before their dependencies.
    for (int i = MAX_LIBS - 1; i >= 0; i--) {
        if (lib_handles[i]) {
            dlclose(lib_handles[i]);
            lib_handles[i] = nullptr;
        }
    }
}
