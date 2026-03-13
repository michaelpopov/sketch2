// Unit tests for the thread-pool task submission and shutdown behavior.

#include "thread_pool.h"

#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace sketch2;

namespace {

struct ChunkResult {
    size_t count = 0;
};

ChunkResult process_chunk(size_t chunk_index) {
    ChunkResult result;
    result.count = chunk_index;
    return result;
}

} // namespace

TEST(ThreadPoolTest, Basics) {
    const size_t num_chunks = 10;
    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 4;
    }

    ThreadPool pool(num_threads);
    std::vector<std::future<ChunkResult>> futures;
    futures.reserve(num_chunks);

    size_t expected_total = 0;
    for (size_t i = 0; i < num_chunks; ++i) {
        expected_total += i;
        futures.push_back(pool.submit([i] {
            return process_chunk(i);
        }));
    }

    size_t actual_total = 0;
    for (auto& future : futures) {
        actual_total += future.get().count;
    }

    ASSERT_EQ(expected_total, actual_total);
}

TEST(ThreadPoolTest, ZeroThreadsFallsBackToOneWorker) {
    ThreadPool pool(0);
    std::future<int> future = pool.submit([] {
        return 7;
    });

    ASSERT_EQ(7, future.get());
}

TEST(ThreadPoolTest, PropagatesTaskExceptionsThroughFuture) {
    ThreadPool pool(1);
    std::future<void> future = pool.submit([] {
        throw std::runtime_error("boom");
    });

    EXPECT_THROW(future.get(), std::runtime_error);
}

TEST(ThreadPoolTest, SupportsMoveOnlyArguments) {
    ThreadPool pool(1);
    auto payload = std::make_unique<int>(42);

    std::future<int> future = pool.submit(
        [](std::unique_ptr<int> value) {
            return *value;
        },
        std::move(payload));

    ASSERT_EQ(42, future.get());
    ASSERT_EQ(nullptr, payload);
}

TEST(ThreadPoolTest, DestructorWaitsForQueuedTasks) {
    std::atomic<int> completed {0};

    {
        ThreadPool pool(2);
        for (int i = 0; i < 8; ++i) {
            pool.submit([&completed] {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                completed.fetch_add(1, std::memory_order_relaxed);
            });
        }
    }

    ASSERT_EQ(8, completed.load(std::memory_order_relaxed));
}

TEST(ThreadPoolTest, SubmitAfterShutdownThrows) {
    ThreadPool pool(1);
    pool.shutdown();

    EXPECT_THROW(
        {
            auto future = pool.submit([] {
                return 1;
            });
            (void)future;
        },
        std::runtime_error);
}

TEST(ThreadPoolTest, ShutdownIsIdempotent) {
    ThreadPool pool(1);
    pool.shutdown();
    pool.shutdown();

    SUCCEED();
}

TEST(ThreadPoolTest, WaitAllBlocksUntilTasksFinish) {
    ThreadPool pool(2);
    std::atomic<int> completed {0};

    for (int i = 0; i < 6; ++i) {
        pool.submit([&completed] {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            completed.fetch_add(1, std::memory_order_relaxed);
        });
    }

    pool.wait_all();
    ASSERT_EQ(6, completed.load(std::memory_order_relaxed));
}

TEST(ThreadPoolTest, WaitAllDoesNotStopPool) {
    ThreadPool pool(1);

    pool.submit([] {
        return 1;
    });
    pool.wait_all();

    std::future<int> future = pool.submit([] {
        return 2;
    });

    ASSERT_EQ(2, future.get());
}

TEST(ThreadPoolTest, ConcurrentSubmittersDoNotLoseTasks) {
    ThreadPool pool(4);
    constexpr int producer_count = 4;
    constexpr int tasks_per_producer = 250;
    std::atomic<int> completed {0};
    std::vector<std::thread> producers;
    producers.reserve(producer_count);

    for (int producer = 0; producer < producer_count; ++producer) {
        producers.emplace_back([&pool, &completed] {
            for (int task = 0; task < tasks_per_producer; ++task) {
                pool.submit([&completed] {
                    completed.fetch_add(1, std::memory_order_relaxed);
                });
            }
        });
    }

    for (auto& producer : producers) {
        producer.join();
    }

    pool.wait_all();
    ASSERT_EQ(producer_count * tasks_per_producer, completed.load(std::memory_order_relaxed));
}

TEST(ThreadPoolTest, ShutdownStopsAdmissionAndStillFinishesRunningWork) {
    ThreadPool pool(4);
    std::atomic<int> completed {0};
    std::atomic<bool> keep_submitting {true};
    std::atomic<int> rejected {0};
    std::vector<std::thread> producers;

    for (int i = 0; i < 4; ++i) {
        producers.emplace_back([&pool, &completed, &keep_submitting, &rejected] {
            while (keep_submitting.load(std::memory_order_relaxed)) {
                try {
                    pool.submit([&completed] {
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        completed.fetch_add(1, std::memory_order_relaxed);
                    });
                } catch (const std::runtime_error&) {
                    rejected.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
            }
        });
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    pool.shutdown();
    keep_submitting.store(false, std::memory_order_relaxed);

    for (auto& producer : producers) {
        producer.join();
    }

    EXPECT_GT(completed.load(std::memory_order_relaxed), 0);
    EXPECT_GT(rejected.load(std::memory_order_relaxed), 0);
    EXPECT_THROW(
        {
            auto future = pool.submit([] {
                return 1;
            });
            (void)future;
        },
        std::runtime_error);
}
