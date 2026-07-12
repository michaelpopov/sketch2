// Unit tests for the sk_embedder_* embedding API. Error-path tests run
// unconditionally; end-to-end tests need a GGUF model (SKETCH2_EMBED_TEST_MODEL
// env var, or the fixture scripts/fetch-embed-test-model.sh caches in the
// build tree) and otherwise skip — unless SKETCH2_REQUIRE_EMBED_TESTS=1 (CI)
// turns the skip into a failure.

#include "sketch2.h"

#include <gtest/gtest.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

namespace {

double l2_norm(const float* vec, size_t dim) {
    double sum = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        sum += static_cast<double>(vec[i]) * static_cast<double>(vec[i]);
    }
    return std::sqrt(sum);
}

double cosine(const float* a, const float* b, size_t dim) {
    double dot = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
    }
    return dot / (l2_norm(a, dim) * l2_norm(b, dim));
}

const char* test_model_path() {
    const char* env = std::getenv("SKETCH2_EMBED_TEST_MODEL");
    if (env != nullptr && env[0] != '\0') {
        return env;
    }
#ifdef SKETCH2_EMBED_TEST_MODEL_DEFAULT
    std::FILE* f = std::fopen(SKETCH2_EMBED_TEST_MODEL_DEFAULT, "rb");
    if (f != nullptr) {
        std::fclose(f);
        return SKETCH2_EMBED_TEST_MODEL_DEFAULT;
    }
#endif
    return nullptr;
}

bool embed_tests_required() {
    const char* required = std::getenv("SKETCH2_REQUIRE_EMBED_TESTS");
    return required != nullptr && required[0] == '1';
}

// Resolves the test model; skips (local dev) or fails (CI) when absent.
// Expands to declarations — use at the top level of a test body.
#define REQUIRE_EMBED_TEST_MODEL(model_var)                                   \
    const char* model_var = test_model_path();                                \
    if (model_var == nullptr) {                                               \
        if (embed_tests_required()) {                                         \
            FAIL() << "SKETCH2_REQUIRE_EMBED_TESTS=1 but no embedding test "  \
                      "model is available; run "                              \
                      "scripts/fetch-embed-test-model.sh";                    \
        }                                                                     \
        GTEST_SKIP() << "no embedding test model; run "                       \
                        "scripts/fetch-embed-test-model.sh (or build the "    \
                        "fetch-embed-test-model target), or set "             \
                        "SKETCH2_EMBED_TEST_MODEL";                           \
    }

} // namespace

TEST(EmbedApi, OpenRejectsNullAndEmptyPath) {
    const char* message = nullptr;
    EXPECT_EQ(sk_embedder_open(nullptr, nullptr, &message), nullptr);
    ASSERT_NE(message, nullptr);
    EXPECT_NE(std::string(message).find("model_path"), std::string::npos);

    message = nullptr;
    EXPECT_EQ(sk_embedder_open("", nullptr, &message), nullptr);
    ASSERT_NE(message, nullptr);
}

TEST(EmbedApi, OpenRejectsMissingFile) {
    const char* message = nullptr;
    EXPECT_EQ(sk_embedder_open("/nonexistent/model.gguf", nullptr, &message), nullptr);
    ASSERT_NE(message, nullptr);
    // The CPU check runs before model loading, so on an unsupported CPU the
    // open fails with the CPU error before reaching the file: accept either.
    const std::string msg(message);
    EXPECT_TRUE(msg.find("failed to load") != std::string::npos
                || msg.find("CPU extension") != std::string::npos)
        << "unexpected message: " << msg;
}

TEST(EmbedApi, OpenRejectsGarbageFile) {
    std::string path = testing::TempDir() + "sk_embed_garbage.gguf";
    FILE* f = fopen(path.c_str(), "wb");
    ASSERT_NE(f, nullptr);
    const char junk[] = "this is not a gguf file";
    fwrite(junk, 1, sizeof(junk), f);
    fclose(f);

    const char* message = nullptr;
    EXPECT_EQ(sk_embedder_open(path.c_str(), nullptr, &message), nullptr);
    ASSERT_NE(message, nullptr);
    remove(path.c_str());
}

TEST(EmbedApi, OpenRejectsBadOptions) {
    const char* message = nullptr;
    EXPECT_EQ(sk_embedder_open("ignored.gguf", "pooling=banana", &message), nullptr);
    ASSERT_NE(message, nullptr);
    EXPECT_NE(std::string(message).find("pooling"), std::string::npos);

    message = nullptr;
    EXPECT_EQ(sk_embedder_open("ignored.gguf", "bogus_key=1", &message), nullptr);
    ASSERT_NE(message, nullptr);
    EXPECT_NE(std::string(message).find("unknown option"), std::string::npos);

    message = nullptr;
    EXPECT_EQ(sk_embedder_open("ignored.gguf", "context=abc", &message), nullptr);
    ASSERT_NE(message, nullptr);
    EXPECT_NE(std::string(message).find("context"), std::string::npos);
}

TEST(EmbedApi, NullHandleIsRejectedEverywhere) {
    unsigned int dim = 0;
    EXPECT_EQ(sk_embedder_dim(nullptr, &dim), -1);
    EXPECT_EQ(sk_embedder_max_tokens(nullptr, &dim), -1);
    float* vec = nullptr;
    size_t vec_dim = 0;
    EXPECT_EQ(sk_embed_text(nullptr, "x", &vec, &vec_dim), -1);
    EXPECT_EQ(sk_embedder_error(nullptr), -1);
    EXPECT_STREQ(sk_embedder_error_message(nullptr), "");
    sk_embedder_close(nullptr);
}

TEST(EmbedApi, EndToEndSingleText) {
    REQUIRE_EMBED_TEST_MODEL(model)
    const char* message = nullptr;
    sk_embedder_t* e = sk_embedder_open(model, nullptr, &message);
    ASSERT_NE(e, nullptr) << (message != nullptr ? message : "no message");

    unsigned int dim = 0;
    ASSERT_EQ(sk_embedder_dim(e, &dim), 0);
    EXPECT_GT(dim, 0u);
    unsigned int max_tokens = 0;
    ASSERT_EQ(sk_embedder_max_tokens(e, &max_tokens), 0);
    EXPECT_GT(max_tokens, 0u);

    float* vec = nullptr;
    size_t vec_dim = 0;
    ASSERT_EQ(sk_embed_text(e, "The quick brown fox jumps over the lazy dog.",
        &vec, &vec_dim), 0) << sk_embedder_error_message(e);
    ASSERT_NE(vec, nullptr);
    EXPECT_EQ(vec_dim, dim);
    // Default options L2-normalize the embedding.
    EXPECT_NEAR(l2_norm(vec, vec_dim), 1.0, 1e-3);
    sk_free(vec);

    sk_embedder_close(e);
}

TEST(EmbedApi, EndToEndBatchAndSimilarity) {
    REQUIRE_EMBED_TEST_MODEL(model)
    const char* message = nullptr;
    sk_embedder_t* e = sk_embedder_open(model, nullptr, &message);
    ASSERT_NE(e, nullptr) << (message != nullptr ? message : "no message");

    const char* texts[3] = {
        "A cat is sitting on the windowsill.",
        "A kitten rests by the window.",
        "The stock market fell sharply on Monday.",
    };
    float* vecs = nullptr;
    size_t dim = 0;
    ASSERT_EQ(sk_embed_texts(e, texts, 3, &vecs, &dim), 0)
        << sk_embedder_error_message(e);
    ASSERT_NE(vecs, nullptr);
    ASSERT_GT(dim, 0u);

    const double sim_related = cosine(vecs, vecs + dim, dim);
    const double sim_unrelated = cosine(vecs, vecs + 2 * dim, dim);
    EXPECT_GT(sim_related, sim_unrelated)
        << "related=" << sim_related << " unrelated=" << sim_unrelated;

    // Batched rows must match the same text embedded alone.
    for (int i = 0; i < 3; ++i) {
        float* single = nullptr;
        size_t single_dim = 0;
        ASSERT_EQ(sk_embed_text(e, texts[i], &single, &single_dim), 0)
            << sk_embedder_error_message(e);
        ASSERT_EQ(single_dim, dim);
        EXPECT_GT(cosine(single, vecs + i * dim, dim), 0.999)
            << "batched embedding of texts[" << i
            << "] diverges from the single-call embedding";
        sk_free(single);
    }
    sk_free(vecs);

    sk_embedder_close(e);
}

TEST(EmbedApi, EndToEndTruncationControl) {
    REQUIRE_EMBED_TEST_MODEL(model)
    std::string long_text;
    for (int i = 0; i < 200; ++i) {
        long_text += "many words repeated over and over ";
    }

    // Default: over-long input is an error.
    const char* message = nullptr;
    sk_embedder_t* e = sk_embedder_open(model, "context=32", &message);
    ASSERT_NE(e, nullptr) << (message != nullptr ? message : "no message");
    float* vec = nullptr;
    size_t dim = 0;
    EXPECT_EQ(sk_embed_text(e, long_text.c_str(), &vec, &dim), -1);
    EXPECT_NE(std::string(sk_embedder_error_message(e)).find("exceeding"),
        std::string::npos);
    sk_embedder_close(e);

    // truncate=1: same input succeeds.
    e = sk_embedder_open(model, "context=32,truncate=1", &message);
    ASSERT_NE(e, nullptr) << (message != nullptr ? message : "no message");
    ASSERT_EQ(sk_embed_text(e, long_text.c_str(), &vec, &dim), 0)
        << sk_embedder_error_message(e);
    ASSERT_NE(vec, nullptr);
    sk_free(vec);
    sk_embedder_close(e);
}

TEST(EmbedApi, EndToEndConcurrentSharedEmbedder) {
    REQUIRE_EMBED_TEST_MODEL(model)
    const char* message = nullptr;
    sk_embedder_t* e = sk_embedder_open(model, nullptr, &message);
    ASSERT_NE(e, nullptr) << (message != nullptr ? message : "no message");

    const char* texts[4] = {
        "A cat is sitting on the windowsill.",
        "The stock market fell sharply on Monday.",
        "Rain is expected across the coast tomorrow.",
        "The orchestra tuned their instruments before the concert.",
    };

    // Single-threaded reference embeddings.
    std::vector<std::vector<float>> refs(4);
    size_t dim = 0;
    for (int i = 0; i < 4; ++i) {
        float* vec = nullptr;
        size_t vec_dim = 0;
        ASSERT_EQ(sk_embed_text(e, texts[i], &vec, &vec_dim), 0)
            << sk_embedder_error_message(e);
        dim = vec_dim;
        refs[i].assign(vec, vec + vec_dim);
        sk_free(vec);
    }

    // Hammer one shared handle; results must match the reference.
    std::atomic<int> failures{0};
    std::vector<std::thread> workers;
    for (int t = 0; t < 4; ++t) {
        workers.emplace_back([&, t] {
            for (int iter = 0; iter < 8; ++iter) {
                float* vec = nullptr;
                size_t vec_dim = 0;
                if (sk_embed_text(e, texts[t], &vec, &vec_dim) != 0
                        || vec_dim != dim) {
                    ++failures;
                    return;
                }
                if (cosine(vec, refs[t].data(), dim) <= 0.999) {
                    ++failures;
                }
                sk_free(vec);
            }
        });
    }
    for (std::thread& w : workers) {
        w.join();
    }
    EXPECT_EQ(failures.load(), 0)
        << "concurrent embeddings failed or diverged from the reference";

    sk_embedder_close(e);
}

TEST(EmbedApi, ErrorStateIsThreadLocal) {
    REQUIRE_EMBED_TEST_MODEL(model)
    const char* message = nullptr;
    sk_embedder_t* e = sk_embedder_open(model, nullptr, &message);
    ASSERT_NE(e, nullptr) << (message != nullptr ? message : "no message");

    float* vec = nullptr;
    size_t dim = 0;
    EXPECT_EQ(sk_embed_text(e, nullptr, &vec, &dim), -1);
    EXPECT_EQ(sk_embedder_error(e), -1);
    const std::string failed_message = sk_embedder_error_message(e);
    EXPECT_FALSE(failed_message.empty());

    // A successful call on another thread must not disturb this thread's state.
    std::thread([&] {
        float* other_vec = nullptr;
        size_t other_dim = 0;
        EXPECT_EQ(sk_embed_text(e, "hello from another thread",
            &other_vec, &other_dim), 0) << sk_embedder_error_message(e);
        EXPECT_EQ(sk_embedder_error(e), 0);
        sk_free(other_vec);
    }).join();

    EXPECT_EQ(sk_embedder_error(e), -1);
    EXPECT_EQ(std::string(sk_embedder_error_message(e)), failed_message);

    sk_embedder_close(e);
}

TEST(EmbedApi, EndToEndLargeThreadCountIsClamped) {
    REQUIRE_EMBED_TEST_MODEL(model)
    // A pathological threads= must be clamped, not passed to ggml (whose pool
    // creation aborts the process); opening and inference must work.
    const char* message = nullptr;
    sk_embedder_t* e = sk_embedder_open(model, "threads=100000", &message);
    ASSERT_NE(e, nullptr) << (message != nullptr ? message : "no message");
    float* vec = nullptr;
    size_t dim = 0;
    ASSERT_EQ(sk_embed_text(e, "a clamped thread pool still embeds", &vec, &dim), 0)
        << sk_embedder_error_message(e);
    ASSERT_NE(vec, nullptr);
    sk_free(vec);
    sk_embedder_close(e);
}

TEST(EmbedApi, EndToEndBadInputs) {
    REQUIRE_EMBED_TEST_MODEL(model)
    const char* message = nullptr;
    sk_embedder_t* e = sk_embedder_open(model, nullptr, &message);
    ASSERT_NE(e, nullptr) << (message != nullptr ? message : "no message");

    float* vec = nullptr;
    size_t dim = 0;
    EXPECT_EQ(sk_embed_text(e, nullptr, &vec, &dim), -1);
    EXPECT_EQ(sk_embed_texts(e, nullptr, 1, &vec, &dim), -1);
    const char* texts[1] = {nullptr};
    EXPECT_EQ(sk_embed_texts(e, texts, 1, &vec, &dim), -1);
    EXPECT_NE(std::string(sk_embedder_error_message(e)).find("texts[0]"),
        std::string::npos);
    const char* ok_texts[1] = {"hello"};
    EXPECT_EQ(sk_embed_texts(e, ok_texts, 0, &vec, &dim), -1);
    // A count that wraps the allocation size must be rejected up front.
    EXPECT_EQ(sk_embed_texts(e, ok_texts, SIZE_MAX / 2, &vec, &dim), -1);
    EXPECT_NE(std::string(sk_embedder_error_message(e)).find("count"),
        std::string::npos);

    sk_embedder_close(e);
}
