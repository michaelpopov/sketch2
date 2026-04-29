// Unit tests for BitsetFileCache.

#include "bitset_file_cache.h"
#include "bitset_filter_control.h"
#include "utils/file_descriptor_guard.h"
#include "utils/shared_consts.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

namespace sketch2 {
namespace {

// Helper that owns a tempfile + an open O_RDONLY fd. The file is unlinked on
// destruction so a failing test does not leave temp files behind.
class TempBackedFile {
public:
    static TempBackedFile make(const char* tag, const std::string& content) {
        std::string tmpl = "/tmp/sketch2_utest_bitset_file_cache_";
        tmpl += tag;
        tmpl += "_XXXXXX";
        std::vector<char> writable(tmpl.begin(), tmpl.end());
        writable.push_back('\0');
        int wfd = ::mkstemp(writable.data());
        EXPECT_GE(wfd, 0) << "mkstemp failed";
        if (wfd >= 0) {
            ssize_t n = ::write(wfd, content.data(), content.size());
            EXPECT_EQ(static_cast<ssize_t>(content.size()), n);
            ::close(wfd);
        }
        std::string path(writable.data());
        FileDescriptorGuard fd(::open(path.c_str(), O_RDONLY));
        EXPECT_GE(fd.fd(), 0);
        return TempBackedFile(std::move(path), std::move(fd), content.size());
    }

    TempBackedFile(TempBackedFile&&) = default;
    TempBackedFile& operator=(TempBackedFile&&) = default;

    ~TempBackedFile() {
        if (!path_.empty()) {
            std::error_code ec;
            std::filesystem::remove(path_, ec);
        }
    }

    FileDescriptorGuard take_fd() { return std::move(fd_); }
    size_t size() const { return size_; }
    const std::string& path() const { return path_; }

    FileDescriptorGuard reopen() const {
        return FileDescriptorGuard(::open(path_.c_str(), O_RDONLY));
    }

private:
    TempBackedFile(std::string path, FileDescriptorGuard fd, size_t size)
        : path_(std::move(path)), fd_(std::move(fd)), size_(size) {}

    std::string path_;
    FileDescriptorGuard fd_;
    size_t size_ = 0;
};

class BitsetFileCacheTest : public ::testing::Test {
protected:
    void SetUp() override { bitset_file_cache().clear(); }
    void TearDown() override { bitset_file_cache().clear(); }
};

TEST_F(BitsetFileCacheTest, insert_then_contains_and_size) {
    auto file = TempBackedFile::make("alpha", "alpha-payload");
    const size_t expected_size = file.size();
    ASSERT_EQ(0, bitset_file_cache().insert(
        "alpha", file.take_fd(), expected_size).code());

    EXPECT_TRUE(bitset_file_cache().contains("alpha"));
    EXPECT_EQ(1u, bitset_file_cache().size());
}

TEST_F(BitsetFileCacheTest, contains_on_missing_returns_false) {
    EXPECT_FALSE(bitset_file_cache().contains("missing"));
    EXPECT_EQ(0u, bitset_file_cache().size());
}

TEST_F(BitsetFileCacheTest, acquire_on_miss_returns_null_storage) {
    std::unique_ptr<BitsetFilterStorage> out;
    EXPECT_EQ(0, bitset_file_cache().acquire("missing", &out).code());
    EXPECT_EQ(nullptr, out);
}

TEST_F(BitsetFileCacheTest, acquire_returns_independent_storage_with_same_data) {
    const std::string payload = "the-quick-brown-fox-the-quick-brown-fox";
    auto file = TempBackedFile::make("indep", payload);
    ASSERT_EQ(0, bitset_file_cache().insert(
        "k", file.take_fd(), payload.size()).code());

    std::unique_ptr<BitsetFilterStorage> first;
    std::unique_ptr<BitsetFilterStorage> second;
    ASSERT_EQ(0, bitset_file_cache().acquire("k", &first).code());
    ASSERT_EQ(0, bitset_file_cache().acquire("k", &second).code());
    ASSERT_NE(nullptr, first);
    ASSERT_NE(nullptr, second);

    // Each acquire produces a freshly-mapped, independent storage object.
    EXPECT_NE(first.get(), second.get());
    EXPECT_NE(first->data(), second->data());

    // The bytes match the payload.
    EXPECT_EQ(payload.size(), first->size());
    EXPECT_EQ(payload.size(), second->size());
    EXPECT_EQ(0, std::memcmp(first->data(), payload.data(), payload.size()));
    EXPECT_EQ(0, std::memcmp(second->data(), payload.data(), payload.size()));

    // Borrowed storage does not own a descriptor (cache retains it).
    EXPECT_EQ(-1, first->fd);
    EXPECT_EQ(-1, second->fd);
}

TEST_F(BitsetFileCacheTest, insert_replaces_existing_entry) {
    auto file_a = TempBackedFile::make("repl_a", "AAAA");
    auto file_b = TempBackedFile::make("repl_b", "BBBBBB");
    ASSERT_EQ(0, bitset_file_cache().insert("k", file_a.take_fd(), 4).code());
    ASSERT_EQ(0, bitset_file_cache().insert("k", file_b.take_fd(), 6).code());

    std::unique_ptr<BitsetFilterStorage> got;
    ASSERT_EQ(0, bitset_file_cache().acquire("k", &got).code());
    ASSERT_NE(nullptr, got);
    EXPECT_EQ(6u, got->size());
    EXPECT_EQ(0, std::memcmp(got->data(), "BBBBBB", 6));
    EXPECT_EQ(1u, bitset_file_cache().size());
}

TEST_F(BitsetFileCacheTest, insert_rejects_invalid_inputs) {
    auto file = TempBackedFile::make("inv", "data");
    EXPECT_NE(0, bitset_file_cache().insert("k", FileDescriptorGuard(), 4).code());
    EXPECT_NE(0, bitset_file_cache().insert("", file.take_fd(), 4).code());
    auto file2 = TempBackedFile::make("inv2", "data");
    EXPECT_NE(0, bitset_file_cache().insert("k", file2.take_fd(), 0).code());
    EXPECT_EQ(0u, bitset_file_cache().size());
}

TEST_F(BitsetFileCacheTest, remove_returns_true_on_hit_false_on_miss) {
    auto file = TempBackedFile::make("rm", "x");
    ASSERT_EQ(0, bitset_file_cache().insert("k", file.take_fd(), 1).code());
    EXPECT_TRUE(bitset_file_cache().remove("k"));
    EXPECT_FALSE(bitset_file_cache().remove("k"));
    EXPECT_EQ(0u, bitset_file_cache().size());
}

TEST_F(BitsetFileCacheTest, clear_empties_cache) {
    auto a = TempBackedFile::make("ca", "a");
    auto b = TempBackedFile::make("cb", "b");
    auto c = TempBackedFile::make("cc", "c");
    ASSERT_EQ(0, bitset_file_cache().insert("a", a.take_fd(), 1).code());
    ASSERT_EQ(0, bitset_file_cache().insert("b", b.take_fd(), 1).code());
    ASSERT_EQ(0, bitset_file_cache().insert("c", c.take_fd(), 1).code());
    EXPECT_EQ(3u, bitset_file_cache().size());

    bitset_file_cache().clear();
    EXPECT_EQ(0u, bitset_file_cache().size());
    EXPECT_FALSE(bitset_file_cache().contains("a"));
}

TEST_F(BitsetFileCacheTest, borrower_outlives_remove) {
    const std::string payload = "borrower-payload";
    auto file = TempBackedFile::make("borrow", payload);
    ASSERT_EQ(0, bitset_file_cache().insert(
        "k", file.take_fd(), payload.size()).code());

    std::unique_ptr<BitsetFilterStorage> borrower;
    ASSERT_EQ(0, bitset_file_cache().acquire("k", &borrower).code());
    ASSERT_NE(nullptr, borrower);

    EXPECT_TRUE(bitset_file_cache().remove("k"));
    EXPECT_FALSE(bitset_file_cache().contains("k"));

    // The borrower's mapping survives the eviction; the data remains intact.
    EXPECT_EQ(payload.size(), borrower->size());
    EXPECT_EQ(0, std::memcmp(borrower->data(), payload.data(), payload.size()));
}

TEST_F(BitsetFileCacheTest, insert_rejects_when_cache_is_full) {
    std::vector<TempBackedFile> files;
    files.reserve(kMaxLoadedBitsetCount);
    for (size_t i = 0; i < kMaxLoadedBitsetCount; ++i) {
        files.push_back(TempBackedFile::make("full", "x"));
        ASSERT_EQ(0, bitset_file_cache().insert(
            "k_" + std::to_string(i), files.back().take_fd(), 1).code()) << i;
    }
    EXPECT_EQ(kMaxLoadedBitsetCount, bitset_file_cache().size());

    auto overflow = TempBackedFile::make("overflow", "x");
    const Ret rejected = bitset_file_cache().insert(
        "overflow", overflow.take_fd(), 1);
    EXPECT_NE(0, rejected.code());
    EXPECT_NE(rejected.message().find("cache is full"), std::string::npos);
    EXPECT_EQ(kMaxLoadedBitsetCount, bitset_file_cache().size());
    EXPECT_FALSE(bitset_file_cache().contains("overflow"));

    auto replacement = TempBackedFile::make("replace", "y");
    EXPECT_EQ(0, bitset_file_cache().insert(
        "k_0", replacement.take_fd(), 1).code());
    EXPECT_EQ(kMaxLoadedBitsetCount, bitset_file_cache().size());

    EXPECT_TRUE(bitset_file_cache().remove("k_0"));
    auto fits = TempBackedFile::make("fits", "z");
    EXPECT_EQ(0, bitset_file_cache().insert(
        "overflow", fits.take_fd(), 1).code());
    EXPECT_TRUE(bitset_file_cache().contains("overflow"));
}

TEST_F(BitsetFileCacheTest, concurrent_acquire_remove_insert_does_not_crash) {
    constexpr int kThreads = 4;
    constexpr int kIterations = 500;

    // Pre-populate one shared backing file. Each iteration that inserts uses
    // a freshly-opened fd over this file so the cache always owns valid fds.
    auto shared_file = TempBackedFile::make("conc", "concurrent-payload");
    const size_t shared_size = shared_file.size();
    const std::string shared_path = shared_file.path();

    std::atomic<bool> start{false};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);

    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([t, &start, &shared_path, shared_size] {
            while (!start.load(std::memory_order_acquire)) {}
            for (int i = 0; i < kIterations; ++i) {
                const std::string name =
                    "k_" + std::to_string(t) + "_" + std::to_string(i % 8);
                if ((i & 3) == 0) {
                    bitset_file_cache().remove(name);
                } else if ((i & 3) == 1) {
                    std::unique_ptr<BitsetFilterStorage> out;
                    (void)bitset_file_cache().acquire(name, &out);
                } else {
                    FileDescriptorGuard fd(::open(shared_path.c_str(), O_RDONLY));
                    if (fd.fd() >= 0) {
                        (void)bitset_file_cache().insert(
                            name, std::move(fd), shared_size);
                    }
                }
            }
        });
    }
    start.store(true, std::memory_order_release);
    for (auto& th : threads) {
        th.join();
    }
    SUCCEED();
}

} // namespace
} // namespace sketch2
