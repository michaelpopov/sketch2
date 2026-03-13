// Defines a small fixed-size thread pool for running submitted work items.
//
// This header provides a simple queue-based worker pool:
//   sketch2::ThreadPool pool(4);
//   auto future = pool.submit([] { return 42; });
//   pool.wait_all();
//
// Typical usage is to keep a pool alive across a phase of parallel work,
// collect futures only when results are needed, and otherwise use wait_all()
// as a barrier before moving to the next stage.
//
// Important caveats:
// - shutdown() permanently stops the pool. After shutdown starts, submit()
//   throws std::runtime_error and the pool cannot be reused.
// - wait_all() only drains currently submitted work; it does not stop workers
//   or reject future submissions.
// - The task queue currently stores std::function<void()> wrappers, so submit()
//   is convenient but not allocation-free.
// - Worker tasks execute on background threads with no built-in cancellation,
//   priority, or affinity control. Callers are responsible for making task
//   bodies thread-safe and for handling their own higher-level coordination.

#pragma once
#include <functional>
#include <condition_variable>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace sketch2 {

class ThreadPool {
public:
    explicit ThreadPool(std::size_t numThreads)
        : stop_(false)
    {
        if (numThreads == 0) {
            numThreads = 1;
        }

        for (std::size_t i = 0; i < numThreads; ++i) {
            workers_.emplace_back([this] {
                workerLoop();
            });
        }
    }

    // Non-copyable
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Movable if you really want to, but simplest is to delete for now
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;

    ~ThreadPool() noexcept { shutdown(); }

    // Stop accepting new work, wait for already submitted tasks to finish, and
    // then join all worker threads. The pool cannot be reused afterwards.
    void shutdown() {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (stop_) {
                return;
            }
            stop_ = true;
        }
        cv_.notify_all();
        wait_all();
        for (auto& t : workers_) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    void wait_all() {
        std::unique_lock<std::mutex> lock(mutex_);
        idle_cv_.wait(lock, [this] {
            return tasks_.empty() && active_tasks_ == 0;
        });
    }

    // Capture the callable and arguments into a tuple so submission preserves
    // move-only payloads and exact forwarding semantics without relying on std::bind.
    template <typename F, typename... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>>
    {
        using R = std::invoke_result_t<F, Args...>;

        auto task = std::make_shared<std::packaged_task<R()>>(
            [func = std::forward<F>(f), args_tuple = std::make_tuple(std::forward<Args>(args)...)]() mutable -> R {
                return std::apply(
                    [&func](auto&&... inner_args) -> R {
                        return std::invoke(std::move(func), std::forward<decltype(inner_args)>(inner_args)...);
                    },
                    std::move(args_tuple));
            }
        );

        std::future<R> fut = task->get_future();

        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (stop_) {
                throw std::runtime_error("ThreadPool::submit on stopped pool");
            }

            tasks_.emplace([task]() {
                (*task)();
            });
        }
        cv_.notify_one();

        return fut;
    }

private:
    std::vector<std::thread>        workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex                      mutex_;
    std::condition_variable         cv_;
    std::condition_variable         idle_cv_;
    std::size_t                     active_tasks_ = 0;
    bool                            stop_;

    void workerLoop() {
        for (;;) {
            std::function<void()> task;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] {
                    return stop_ || !tasks_.empty();
                });

                if (stop_ && tasks_.empty()) {
                    return;
                }

                task = std::move(tasks_.front());
                tasks_.pop();
                ++active_tasks_;
            }

            task();

            {
                std::unique_lock<std::mutex> lock(mutex_);
                --active_tasks_;
                if (tasks_.empty() && active_tasks_ == 0) {
                    idle_cv_.notify_all();
                }
            }
        }
    }
};

} // namespace sketch2
