// llama-link.h — dynamic loading interface for llama.cpp shared libraries
//
// Instead of linking against libllama.so at build time, this module loads the
// libraries at runtime from a user-specified directory using dlopen/dlsym.
// This lets the emgen binary run on systems where llama.cpp isn't installed
// system-wide — the user just points it at a directory containing the .so files.
//
// Usage:
//   llama_api api;
//   if (!llama_link_load("/path/to/libs", api)) { /* handle error */ }
//   api.backend_init();
//   ...
//   llama_link_unload();

#ifndef LLAMA_LINK_H
#define LLAMA_LINK_H

#include "llama.h"

// All llama.cpp functions used by emgen, as typed function pointers.
// Field names mirror the llama.cpp function names with the "llama_" prefix
// stripped for brevity (e.g. llama_backend_init -> backend_init).
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

// Load llama.cpp shared libraries from the given directory and resolve all
// function pointers into `api`. The directory must contain: libggml-base.so,
// libggml.so, libggml-cpu.so, and libllama.so.
//
// Returns true on success. On failure, prints diagnostics to stderr and
// returns false (no libraries are left loaded).
bool llama_link_load(const char * lib_dir, llama_api & api);

// Close all library handles opened by llama_link_load.
// Safe to call even if llama_link_load was never called or failed.
void llama_link_unload();

#endif // LLAMA_LINK_H
