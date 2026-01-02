// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __FDAPDE_EXECUTION_THREADPOOL_H__
#define __FDAPDE_EXECUTION_THREADPOOL_H__

#include "header_check.h"

namespace fdapde {

// forward decl
class Worker;

class ThreadPool {
    static constexpr int buffer_size = 4096;

    // a copy/movable vector of 64-byte aligned std::atomic<T>
    template <typename T> struct atomic_vector {
       private:
        struct alignas(64) atomic_t {
            atomic_t() : v_() { }
            atomic_t(const std::atomic<T>& v) : v_(v.load()) { }
            atomic_t(const atomic_t& other) : v_(other.v_.load()) { }
            atomic_t& operator=(const atomic_t& other) {
                v_ = other.v_.load();
                return *this;
            }
            void store(const T& value, std::memory_order order = std::memory_order_seq_cst) { v_.store(value, order); }
            T load(std::memory_order order = std::memory_order_seq_cst) const noexcept { return v_.load(order); }
            const std::atomic<T>& operator*() const { return v_; }
            std::atomic<T>& operator*() { return v_; }
           private:
            std::atomic<T> v_;
        };
        std::vector<atomic_t> data_;
       public:
        explicit atomic_vector(int size) : data_(size) { }
        template <typename T_>
            requires(std::is_convertible_v<T, T_>)
        atomic_vector(int size, T_ value) : data_(size) {
            for (int i = 0; i < size; ++i) { data_[i].store(value); }
        }
        // observers
        int size() const { return data_.size(); }
        const std::atomic<T>& operator[](int i) const {
            fdapde_assert(i >= 0 && i < size());
            return *data_[i];
        }
        // modifiers
        std::atomic<T>& operator[](int i) {
            fdapde_assert(i >= 0 && i < size());
            return *data_[i];
        }
    };

    // random stealing: returns a random worker among the ones having at least one job
    struct steal_policy {
        explicit steal_policy(int n) : rng_(std::random_device {}()) { idxs_.resize(n); }

        std::optional<int> pick(const atomic_vector<int>& workload) {
            int i = 0;
            for (int k = 0; k < workload.size(); k++) {
                if (workload[k].load(std::memory_order_acquire) > 0) { idxs_[i++] = k; }
            }
            if (i == 0) { return std::nullopt; }
            if (i == 1) { return idxs_[0]; }
            std::uniform_int_distribution<> random_int(0, i - 1);
            return idxs_[random_int(rng_)];
        }
       private:
        std::vector<int> idxs_;
        mutable std::mt19937 rng_;
    };
    steal_policy steal_policy_;

    // round_robin worker selection: returns the next worker index in round-robin order
    struct spawn_policy {
        explicit spawn_policy(int n) : idx_(0), n_workers_(n) { }

        int next() {
            int curr = idx_.load(std::memory_order_acquire);
            int next = (curr + 1) % n_workers_;
            do {
                idx_ = next;
            } while (!idx_.compare_exchange_strong(
              next, (next + 1) % n_workers_, std::memory_order_acquire, std::memory_order_release));
            return next;
        }
       private:
        std::atomic<int> idx_;
        int n_workers_;
    };
    spawn_policy spawn_policy_;

    // submits and schedules one task
    template <typename Task> void submit_task_(Task&& task) {
        int worker_id = spawn_policy_.next();
        // submit task
        workers_[worker_id]->submit_task(internals::task_handle(task, worker_id));
        workload_[worker_id].fetch_add(1, std::memory_order_release);
        // notify worker, if waiting
        workers_[worker_id]->wake_up();
        return;
    }

    using pool_t = std::vector<std::shared_ptr<internals::worker>>;

    pool_t workers_;                // worker-pool
    const int n_workers_;           // number of workers in the pool
    std::latch init_latch_;         // workers synchronization barrier at startup
    atomic_vector<int> workload_;   // number of assigned tasks per worker
    std::atomic<int> task_count_;   // overall number of active tasks
    // joining logic
    std::mutex join_m_;
    std::condition_variable join_cv_;
   public:
    // constructor
    ThreadPool() : ThreadPool(std::thread::hardware_concurrency()) { }
    explicit ThreadPool(int size) :
        steal_policy_(size), spawn_policy_(size), n_workers_(size), init_latch_(size + 1), workload_(size, 0) {
        workers_.reserve(n_workers_);
        internals::tls_worker_id = 0;   // set main thread worker id to zero
        // start workers
        for (int i = 0; i < n_workers_; i++) {
            workers_.emplace_back(std::make_shared<internals::worker>(1 + i, buffer_size, this));
            workload_[i].store(0, std::memory_order_release);
        }
        // wait workers to be ready, avoids threadpool destruction before worker construction
        init_latch_.arrive_and_wait();
    }
    // observers
    int n_workers() const { return n_workers_; }
    const std::atomic<int>& task_count() const { return task_count_; }
  
    // submits f(args...) for asynchronous execution. returns a std::future holding the result
    template <typename F, typename... Args>
    [[nodiscard]] auto submit(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
        // type-erase task type to send to the threadpool
        using ret_t = decltype(f(args...));
        std::shared_ptr<std::packaged_task<ret_t()>> packaged_task =
          std::make_shared<std::packaged_task<ret_t()>>([f_ = f, ... args_ = args]() mutable { return f_(args_...); });
        std::function<void()> task = [packaged_task]() { (*packaged_task)(); };

        submit_task_(task);
        return packaged_task->get_future();
    };
    // execute f(args...) asynchronously. doesn't wait for any result
    template <typename F, typename... Args> void execute(F&& f, Args&&... args) {
        auto task = [f_ = f, ... args_ = args]() mutable { f_(args_...); };
        submit_task_(task);
        return;
    }


    // woorker coordination utilities

    void on_worker_ready() { init_latch_.arrive_and_wait(); }
    void on_task_acquire(Task* task) { workload_[task->allocation_pool()].fetch_sub(1, std::memory_order_release); }
    void on_task_complete(Task*) {
        return;
        // take some decision
        // dealloca il task
        // rischedula (risubmitta il parent se le sue dipendenze sono soddisfatte)
    }
    bool can_resume(int worker_id) { return workload_[worker_id].load() > 0; }

    std::pair<std::optional<Task*>, std::optional<int>> try_steal(int worker_id) {
        std::optional<int> j = steal_policy_.pick(workload_);
        if (j && (*j != worker_id)) {
            auto task = workers_[*j]->try_steal();
            if (task) { return std::make_pair(task, j); }
        }
        return std::make_pair(std::nullopt, std::nullopt);
    }

    // executes the range [begin, end) in parallel, submits one task per iteration
    // template <typename F, typename Iterator>
    //     requires(std::is_same_v<std::invoke_result_t<F, Iterator>, void> && requires(Iterator it) { ++it; })
    // void parallel_for(Iterator begin, Iterator end, F&& f) {
    //     for (Iterator j = begin; j != end; ++j) { execute(f, j); }
    //     join();
    // }
    // // executes the range [begin, end) in parallel with custom increment logic, submits one task per iteration
    // template <typename F, typename Iterator>
    //     requires(
    //       std::is_same_v<std::invoke_result_t<F, Iterator>, void> && requires(Iterator it, Iterator jt) { it += jt; })
    // void parallel_for(Iterator begin, Iterator end, std::function<Iterator(Iterator)> inc, F&& f) {
    //     for (Iterator j = begin; j < end; j += inc(j)) { execute(f, j); }
    //     join();
    // }

    // // splits [begin, end) into contiguous chunks of size grain_size and submits one task per chunk
    // template <typename F, typename Iterator>
    //     requires(
    //       std::is_same_v<std::invoke_result_t<F, Iterator>, void> &&
    //       requires(Iterator it, Iterator jt, int k) {
    //           { it - jt } -> std::convertible_to<int>;
    //           it += k;
    //           ++it;
    //           k < it;
    //       })
    // void parallel_for(Iterator begin, Iterator end, int grain_size, F&& f) {
    //     fdapde_assert(grain_size > 0);
    //     const int n = end - begin;
    //     if (n == 0) return;   // nothing to loop

    //     grain_size = (grain_size < n) ? grain_size : n;
    //     std::atomic<int> group_ptr {1};
    //     for (Iterator j = begin; j < end; j += grain_size) {
    //         Iterator k = ((j + grain_size) < end) ? (j + grain_size) : end;
    //         // send job for [i, k) blocked range execution
    //         auto loop_body = [&, k_ = k, j_ = j, f_ = f]() {
    //             for (Iterator it = j_; it < k_; ++it) { f_(it); }
    //             if (group_ptr.fetch_sub(1, std::memory_order_acq_rel) == 1) {
    //                 // last chunk of this parallel_for task group
    //                 std::unique_lock<std::mutex> lock(join_m_);
    //                 join_cv_.notify_all();
    //             }
    //         };

    // 	    // this must be implemented as a tree-shaped task graph, and sent in execution to the pool
	    
    //         submit_task_(loop_body);
    //         group_ptr.fetch_add(1, std::memory_order_release);
    //     }
    //     // return fast if main thread is already the last
    //     if (group_ptr.fetch_sub(1, std::memory_order_acq_rel) == 1) { return; }
    //     // wait for this task group
    //     std::unique_lock<std::mutex> lock(join_m_);
    //     join_cv_.wait(lock, [&] { return group_ptr.load(std::memory_order_acquire) == 0; });
    //     return;
    // }
    // // splits container range into contiguous chunks whose size is adaptively chosen. submits one task per chunk
    // template <typename Container, typename F> void parallel_for_each(Container&& c, F&& f) {
    //     using iterator_t = typename std::decay_t<Container>::iterator;
    //     iterator_t begin = c.begin();
    //     iterator_t end = c.end();
    //     const int n = std::distance(begin, end);
    //     if (n == 0) return;   // nothing to loop

    //     int grain_size = std::max(1.0, double(n) / (4 * workers_.size()));   // TBB auto-partitioning
    //     for (iterator_t j = begin; j < end; j += grain_size) {
    //         iterator_t k = ((j + grain_size) < end) ? (j + grain_size) : end;
    //         // send job for [i, k) blocked range execution
    //         execute([=, f_ = f, this]() mutable {
    //             for (iterator_t it = j; it < k; ++it) { f_(*it); }   // pass dereferenced iterator
    //         });
    //     }
    //     join();
    // }

    // // waits until all submitted tasks complete
    // void join() {
    //     //         // work-helping: instead of blocking the caller, temporarily use it as member of the pool
    //     //         // while (n_tasks_.load(std::memory_order_acquire) > 0) {
    //     //         //     bool busy = false;
    //     //         //     for (int i = 0; i < n_workers_; ++i) {
    //     //         //         auto task = workers_[i]->try_steal();
    //     //         //         if (task) {
    //     //         //             execute_task_(task, i);
    //     //         //             busy = true;
    //     //         //             break;
    //     //         //         }
    //     //         //     }
    //     //         //     if (!busy && n_tasks_.load(std::memory_order_acquire) > 0) {
    //     //                 // std::unique_lock<std::mutex> lock(join_m_);
    //     //                 // join_cv_.wait(lock, [&] { return n_tasks_.load(std::memory_order_acquire) == 0; });
    //     //         //     }
    //     //         // }
    //     // while (n_tasks_.load(std::memory_order_acquire) > 0) {
    //     //         std::this_thread::yield();
    //     //     }
    //     //       // n_tasks_.store(0);
    //     //         return;
    // }
    // stops all running threads. not yet completed tasks are losts. explicitly call join() to wait for all sent tasks
    void stop() {
        for (auto& worker : workers_) { worker->stop(); }
        for (auto& worker : workers_) { worker->join(); }
        return;
    }
    // destructor
    ~ThreadPool() { stop(); }
};

// namespace internals {

// struct threadpool_executor {
//     static auto& instance() {
//         // intentionally leaked to guarantee threadpool teardown at program termination
//         static ThreadPool<>* tp = new ThreadPool<>();
//         return *tp;
//     }
// };

// }   // namespace internals

// // public API

// number of available threads
// int num_threads() { return internals::threadpool_executor::instance().n_workers(); }
// logical identifier of running thread
int this_thread_id() noexcept { return internals::tls_worker_id; }
// // executes the range [begin, end) in parallel, submits one task per iteration
// template <typename Iterator, typename F> void parallel_for(Iterator begin, Iterator end, F&& f) {
//     internals::threadpool_executor::instance().parallel_for(begin, end, f);
// }
// // executes the range [begin, end) in parallel with custom increment logic, submits one task per iteration
// template <typename Iterator, typename F>
// void parallel_for(Iterator begin, Iterator end, std::function<Iterator(Iterator)> inc, F&& f) {
//     internals::threadpool_executor::instance().parallel_for(begin, end, inc, f);
// }
// // splits [begin, end) into contiguous chunks of size grain_size and submits one task per chunk
// template <typename Iterator, typename F> void parallel_for(Iterator begin, Iterator end, int grain_size, F&& f) {
//     internals::threadpool_executor::instance().parallel_for(begin, end, grain_size, f);
// }
// // parallel range-for over container
// template <typename Container, typename F> void parallel_for_each(Container&& c, F&& f) {
//     internals::threadpool_executor::instance().parallel_for_each(c, f);
// }

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_THREADPOOL_H__
