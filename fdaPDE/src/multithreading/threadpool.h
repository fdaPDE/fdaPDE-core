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

#ifndef __FDAPDE_THREADPOOL_H__
#define __FDAPDE_THREADPOOL_H__
#include "header_check.h"

namespace fdapde {

namespace internals {
// concept for parallel_for with an iterator that traverses the range (relaxes the requirement compared to a random
// access iterator)
template <typename It>
concept parallel_iterator = requires(It a, It b, int n) {
    { a += n } -> std::same_as<It&>;
    { a - b } -> std::convertible_to<int>;
};
}   // namespace internals

// Template parameter for threadpool:
//-struct per StealingStrategy: max_load_stealing, random_stealing, top_half_random_stealing
//-struct per SchedulingStrategy: round_robin_scheduling, least_loaded_scheduling

struct max_load_stealing {
    // Returns the index of the worker with the largest job count.
    // If all counts are zero, returns -1 to indicate that no work can be stolen.
    int pick(std::deque<std::atomic<int>>& count_job, int n_worker) const {
        int worker_indx = 0;
        int max_elem = count_job[0].load(std::memory_order_acquire);
        int current_el = 0;
        for (int j = 1; j < n_worker; j++) {
            current_el = count_job[j].load(std::memory_order_acquire);
            if (current_el > max_elem) {
                worker_indx = j;
                max_elem = current_el;
            }
        }
        if (max_elem == 0) { return -1; }
        return worker_indx;
    };
};

struct random_stealing {
    mutable std::mt19937 gen_;
    random_stealing() : gen_(std::random_device {}()) {};

    // return random int in  [0, size-1]
    int random_int(int size) const {
        std::uniform_int_distribution<> distrib(0, size - 1);
        return distrib(gen_);
    };
    // Returns a random index among the workers that have at least one job.
    // If no worker has jobs, returns -1.
    int pick(std::deque<std::atomic<int>>& count_job, int n_worker) const {
        std::vector<int> indxs;   // vector of index of worker busy
        for (int k = 0; k < n_worker; k++) {
            if (count_job[k].load(std::memory_order_acquire) > 0) { indxs.push_back(k); }
        }
        int size = indxs.size();
        switch (size) {   // avoid random generation if not need it
        case 0:
            return -1;
        case 1:
            return indxs[0];
        default:
            return indxs[random_int(size)];
        }
    };
};

struct top_half_random_stealing {
    mutable std::mt19937 gen_;

    top_half_random_stealing() : gen_(std::random_device {}()) {};

    // return random int in  [0, size-1]
    int random_int(int size) const {
        std::uniform_int_distribution<> distrib(0, size - 1);
        return distrib(gen_);
    };
    // Returns a random index among the workers in the top half of the busiest ones.
    // If no worker has jobs, returns -1.
    int pick(std::deque<std::atomic<int>>& count_job, int n_worker) const {
        std::vector<std::pair<int, int>> indxs;   // vector of pair: (index worker busy, number of job its queue)
        int tmp_count_job = 0;
        for (int k = 0; k < n_worker; k++) {
            tmp_count_job = count_job[k].load(std::memory_order_acquire);
            if (tmp_count_job > 0) { indxs.emplace_back(k, tmp_count_job); }
        }
        int size = indxs.size();
        if (size == 0) { return -1; }
        // sort from mostbusy to least
        std::sort(indxs.begin(), indxs.end(), [](std::pair<int, int>& a, std::pair<int, int>& b) {
            return a.second > b.second;
        });
        if (size < 4) {   // avoid random generation if not need it
            return indxs[0].first;
        } else {
            return indxs[random_int(size / 2)].first;
        }
    };
};

struct round_robin_scheduling {
    int indx_ = 0;
    // Returns the next worker index in round-robin order.
    // After reaching the last worker, wraps around to the first worker (index 0).
    int pick(std::deque<std::atomic<int>>& count_job, int n_worker) {
        int index = indx_;
        (indx_ == n_worker - 1) ? (indx_ = 0) : (indx_++);   // index_++
        return index;
    }
};

struct least_loaded_scheduling {
    // Returns the index of the worker with the smallest job count.
    // If all counts are zero, returns 0
    int pick(std::deque<std::atomic<int>>& count_job, int n_worker) const {
        int worker_indx = 0;
        int min_elem = count_job[0].load(std::memory_order_acquire);
        int current_el = 0;
        for (int j = 1; j < n_worker; j++) {
            current_el = count_job[j].load(std::memory_order_acquire);
            if (current_el < min_elem) {
                worker_indx = j;
                min_elem = current_el;
            }
        }
        return worker_indx;
    };
};

template <typename SchedulingStrategy = round_robin_scheduling, typename StealingStrategy = random_stealing>
class threadpool {
    using job = std::function<void()>;
   public:
    // worker in threadpool. wrap a synchro_queue and a std::thread
    class worker {
       private:
        int indx_;   // Index of the worker in the threadpool's vector of workers. Uniquely identifies the worker within
                     // the threadpool.
        fdapde::synchro_queue<job, fdapde::relaxed> sync_queue_;
        std::thread t_;
        bool stop_ = false;                    // Set to true when the worker should stop working (finish workerLoop()).
        mutable std::mutex m_;                 // worker's mutex
        mutable std::condition_variable cv_;   // Condition variable to suspend the thread when there are no jobs to
                                               // execute, avoiding wasted CPU resources.
       public:
        friend class threadpool;
        worker(int n, void (threadpool::*worker_loop)(int), threadpool* Th, int idx) :
            indx_(idx),
            sync_queue_(n),
            t_(
              worker_loop, Th,
              idx) {};   // Starts the std::thread t_ with the member function workerLoop() of the threadpool.
        // wraps
        std::unique_lock<std::mutex> get_lock() const {
            std::unique_lock<std::mutex> loc(m_);
            return loc;
        }
        void notify() const { cv_.notify_one(); }
        void stop() { stop_ = true; }
        void start() { stop_ = false; }
        void join() { t_.join(); }
        bool push_front(job& fun) {
            return sync_queue_.push_front(fun);
        };   // note: reference in input because is only a wrap-function (intermediate step), than synchro_queue store a
             // copy
        bool push_back(job& fun) { return sync_queue_.push_back(fun); };
        std::optional<job> pop_front() { return sync_queue_.pop_front(); };
        std::optional<job> pop_back() { return sync_queue_.pop_back(); };
    };
   private:
    std::vector<std::shared_ptr<worker>> workers_;   // vector of pointers because worker not movable, not copiable
    std::deque<std::atomic<int>> count_job_;         // deque because atomic<int> not movable
    int n_worker_;
    int queue_size_;
    mutable std::shared_mutex
      m_threadpool_;   // Thread pool's mutex. Shared because only the threadpool writes to active_, and workers read it
                       // before starting the main while-loop in workerLoop().
    std::condition_variable_any cv_threadpool_;   // Condition variable used to ensure that all workers have been
                                                  // initialized before starting workerLoop().
    bool active_ = false;                         // Set to true after the threadpool has initialized all workers.
    std::unordered_map<std::thread::id, int>
      map_thread_worker_;   // map of : thread_ID of worker's thread, index of worker in workers_
    mutable std::shared_mutex
      m_map_;   // Mutex to protect access to map_thread_worker_. Shared because each worker writes only once during
                // registration at the beginning of workerLoop(), and is only read afterwards during task execution.
    StealingStrategy steal_policy_;
    SchedulingStrategy schedule_policy_;

    // Helper function used in workerLoop(). indx = index of the worker from whom the job j was taken. Returns true if
    // the job was executed, false if j is std::nullopt.
    bool try_do_(std::optional<job> j, int indx) {
        if (j) {
            (j.value())();
            count_job_[indx].fetch_sub(1, std::memory_order_release);
            return true;
        }
        return false;
    }
   public:
    friend class worker;
    // n = size of synchro_queue, k = number of workers
    threadpool(int size_queue, int n_worker) : n_worker_(n_worker), queue_size_(size_queue) {
        std::unique_lock<std::shared_mutex> loc(m_threadpool_);
        workers_.reserve(n_worker);
        // Initializes workers_ and count_job_.
        for (int i = 0; i < n_worker; i++) {
            workers_.emplace_back(std::make_shared<worker>(size_queue, &threadpool::worker_loop, this, i));
            count_job_.emplace_back(0);
        }
        active_ = true;
        loc.unlock();
        cv_threadpool_.notify_all();   // notify waiting workers
    };

    // constructor default number of worker
    threadpool(int size_queue) : threadpool(size_queue, std::thread::hardware_concurrency()) { }

    // default constructor
    threadpool() : threadpool(256) { }

    ~threadpool() {
        // Non-blocking but unreliable waiting
        while (n_jobs() > 0) { std::this_thread::yield(); }
        // Blocking but reliable waiting
        for (int i = 0; i < n_worker_; i++) {
            while (!workers_[i]->sync_queue_.empty()) { }
        }
        // let's finish all worker_loop
        for (int j = 0; j < n_worker_; j++) {
            std::unique_lock<std::mutex> loc(workers_[j]->get_lock());
            workers_[j]->stop();
            workers_[j]->notify();
        }
        for (int j = 0; j < n_worker_; j++) { workers_[j]->join(); }
    }

    // getter
    int n_workers() const { return n_worker_; };
    // Returns the index of the worker in workers_ that executed it.
    int index_worker() const {
        std::shared_lock<std::shared_mutex> loc(
          m_map_);   // lock to avoid data_race causes by write at the beginning of workerloop()
        return map_thread_worker_.at(std::this_thread::get_id());
    }

    void worker_loop(int i) {   // i = index of worker in workers_
        // Wait until all workers have been initialized.
        std::shared_lock<std::shared_mutex> lock_shared(m_threadpool_);
        cv_threadpool_.wait(lock_shared, [this]() { return active_; });
        lock_shared.unlock();

        // Worker registers itself in map_thread_worker in a thread-safe manner.
        std::unique_lock<std::shared_mutex> loc_map(m_map_);
        map_thread_worker_[std::this_thread::get_id()] = i;
        loc_map.unlock();

        int indx_steal = -1;   // index to steal from
        while (!workers_[i]->stop_) {
            if (try_do_(workers_[i]->pop_front(), i)) {
                continue;   // Non-blocking behavior when a job is found in its queue.
            }
            // try steal
            indx_steal = steal_policy_.pick(count_job_, n_worker_);
            if (indx_steal != -1) {
                try_do_(workers_[indx_steal]->pop_back(), indx_steal);
                continue;   // Non-blocking behavior when a job is found in other queue.
            }
            // chek for job whit mutex lock, eventualy go to sleep
            std::unique_lock<std::mutex> loc(workers_[i]->m_);
            workers_[i]->cv_.wait(loc, [&]() {
                return count_job_[i].load(std::memory_order_acquire) > 0 || n_jobs() > 0 || workers_[i]->stop_;
            });
            loc.unlock();
        }
    };

    // return count of all jobs in threadpool
    int n_jobs() const {
        int somma = 0;
        for (int i = 0; i < n_worker_; i++) somma += count_job_[i].load(std::memory_order_acquire);
        return somma;
    }

    // Wraps a generic function in a job and submits it to the thread pool. Returns a future associated with the return
    // value of the function.
    template <typename F, typename... Args> auto send(F&& f, Args&&... args) -> std::future<decltype(f(args...))> {
        // wrap
        using return_type = decltype(f(args...));
        std::shared_ptr<std::packaged_task<return_type()>> ptr_task =
          std::make_shared<std::packaged_task<return_type()>>(
            [fun = f, ... args_catturati = args]() mutable { return fun(args_catturati...); });
        std::future<return_type> fut = ptr_task->get_future();
        job j = [ptr_task]() { (*ptr_task)(); };

        int indx_worker = schedule_policy_.pick(
          count_job_, n_worker_);   // Selects the index of the worker to which the job should be sent.
        std::unique_lock<std::mutex> lock(workers_[indx_worker]->get_lock());   // lock worker's mutex
        bool flag = workers_[indx_worker]->push_back(j);
        while (!flag) { flag = workers_[indx_worker]->push_back(j); }      // Ensures the job is sent.
        count_job_[indx_worker].fetch_add(1, std::memory_order_release);   // Increments the job counter for the worker.
        lock.unlock();                                                     // unlock worker's mutex
        workers_[indx_worker]->cv_.notify_one();   // Notifies the waiting workers to whom the job was sent.
        return fut;
    };

    // Definition of granularity:= number of iterations of the for-loop in a single job
    // Parallel_for overloads:
    // parallel_for(Index, Index, F&&) --> F represents a single job (granularity = 1). Index can be int or an iterator
    // type parallel_for(Index, Index, F&&, granularity) --> F is the body function of the for-loop. Granularity is
    // provided as input; if granularity <= 0, default granularity is used. Default granularity is set so that each
    // worker receives max one job. Index can be int or an iterator satisfying the parallel_iterator concept
    // parallel_for(int, int, F&&, function<int(int)> incr) --> Uses a custom increment function to traverse the range.
    // F represents a single job (granularity = 1) parallel_for(int, int, F&&, vector<int>) --> F is the body function
    // of the for-loop. The range is divided into vect.size() jobs, with the granularity of job j given by vect[j]
    // parallel_for(Iterator, Iterator, F&&, granularity) --> F is the body function of the for-loop. Granularity is
    // provided as input. Iterator does not satisfy the parallel_iterator concept

    template <typename F, typename Index>
        requires std::is_same_v<std::invoke_result_t<F, Index>, void>
    void parallel_for(Index start, Index end, F&& f) {
        using return_type = void;
        std::vector<std::future<return_type>> ret_fut;   // Vector of futures for the jobs submitted to the threadpool
        if constexpr (std::is_integral_v<Index>) {
            ret_fut.reserve(end - start);
        }   // Reserve space if Index is an integer (not if Index is an iterator type, because the iterator may not
            // support operator-)
        for (Index j = start; j != end; ++j) {
            ret_fut.emplace_back(this->send(f, j));
        }   // Send function f with input j to the threadpool
        for (std::future<void>& fut : ret_fut) {
            fut.get();
        }   // Wait for all futures to ensure the entire for-loop has executed
        return;
    }

    template <typename F, typename Index, typename... Args>
        requires std::is_same_v<std::invoke_result_t<F, Index, int, Args&...>, void> &&
                 (std::is_integral_v<Index> || internals::parallel_iterator<Index>)
    void parallel_for(Index start, Index end, F&& f, int granularity, Args... args) {
        using return_type = void;
        int range = (end - start);
        if (granularity > range) {   // set granularity = range (all for-loop in single job)
            granularity = range;
        }
        int n_job = range / std::max(1, granularity);
        if (granularity <= 0) {   // set default value of granularity
            if (range < n_worker_) {
                granularity = 1;
                n_job = range;
            } else {
                granularity = range / n_worker_;
                n_job = n_worker_;   // ensures max 1 job per worker
            }
        }
        std::vector<int> it_add;           // Vector of additional iterations for the first job of each worker
        int resto = range % granularity;   // number of extra iteartions
        if (resto > 0) {
            if ((n_job % n_worker_) != 0) {   // At least one worker has fewer jobs than worker0, so do not redistribute
                                              // extra iterations; assign as a single job
                n_job++;
            } else {   // All workers have the same number of jobs. Distribute extra iterations to the first job of each
                       // worker
                int iter_add = resto / n_worker_;
                int resto_di_resto =
                  resto % n_worker_;   // All workers take iter_add extra iterations, and the remaining ones are
                                       // distributed (+1) to the first resto_di_resto workers
                for (int w = 0; w < resto_di_resto; w++) { it_add.push_back(iter_add + 1); }
                for (int w = resto_di_resto; w < n_worker_; w++) { it_add.push_back(iter_add); }
            }
        }
        std::vector<std::future<return_type>> ret_fut;
        ret_fut.reserve(n_job);

        Index stop_local = start;
        // Send the first jobs with extra iterations. if it_add.size() == 0 skip this for
        for (int j = 0; j < it_add.size(); j++) {
            stop_local += (granularity + it_add[j]);
            ret_fut.emplace_back(this->send([start, stop = stop_local, fun = f, ... args = args, this]() mutable {
                int index_worker = this->index_worker();
                for (Index k = start; k != stop; ++k) { fun(k, index_worker, args...); }
            }));
            start = stop_local;
        }
        if (it_add.size() == n_job) {   // Avoid reaching the last send, which would submit an empty job
            for (std::future<void>& fut : ret_fut) { fut.get(); }   // get futures
            return;
        }
        // Send the remaining jobs, except for the last one
        for (int j = it_add.size(); j < n_job - 1; j++) {
            stop_local = granularity + start;
            ret_fut.emplace_back(this->send([start, stop = stop_local, fun = f, ... args = args, this]() mutable {
                int index_worker = this->index_worker();
                for (Index k = start; k != stop; ++k) { fun(k, index_worker, args...); }
            }));
            start = stop_local;
        }
        // Send the last job separately to avoid branching in the previous loop, because the last job may have a
        // different granularity
        ret_fut.emplace_back(this->send([start, end, fun = f, ... args = args, this]() mutable {
            int index_worker = this->index_worker();
            for (Index it = start; it != end; ++it) { fun(it, index_worker, args...); }
        }));
        for (std::future<void>& fut : ret_fut) { fut.get(); }   // get futures
        return;
    }

    template <typename F>
        requires std::is_same_v<std::invoke_result_t<F, int>, void>
    void parallel_for(int start, int end, F&& f, std::function<int(int)> incr) {
        using return_type = void;
        std::vector<std::future<return_type>> ret_fut;
        ret_fut.reserve(end - start);
        for (int j = start; j < end; j = incr(j)) { ret_fut.emplace_back(this->send(f, j)); }
        for (std::future<void>& fut : ret_fut) { fut.get(); }
        return;
    }

    template <typename F, typename... Args>
        requires std::is_same_v<std::invoke_result_t<F, int, int, Args&...>, void>
    void parallel_for(
      int start, int end, F&& f, std::vector<int> granularities,
      Args... args) {   // Copy of the granularity vector; safer in multithreading
        using return_type = void;
        if (std::reduce(granularities.cbegin(), granularities.cend(), 0) != (end - start)) {
            std::cerr << "The sum of elements in granularities must equal the range (end - start)" << std::endl;
            return;
        }
        int stop_local = start;
        size_t vect_size = granularities.size();
        std::vector<std::future<return_type>> ret_fut;
        ret_fut.reserve(vect_size);
        for (size_t j = 0; j < vect_size; j++) {
            stop_local += granularities[j];
            ret_fut.emplace_back(this->send([start, stop = stop_local, fun = f, ... args = args, this]() mutable {
                int index_worker = this->index_worker();
                for (int k = start; k < stop; k++) { fun(k, index_worker, args...); }
            }));
            start = stop_local;
        }
        for (std::future<void>& fut : ret_fut) { fut.get(); }
        return;
    }

    // Parallel reduce: wrap std::reduce so that each worker applies std::reduce on its sub‑range, then the partial
    // results from all workers are combined with a final reduction. Input: begin and end delimit the range, init is the
    // initial value for the accumulator (sum -> 0, dot -> 1, min -> inf, max -> -inf), operation is a binary operation.
    // Output: the reduced value.
    template <typename Iterator, typename Value_type, typename F>
    Value_type reduce(Iterator begin, Iterator end, Value_type init, F&& operation) {
        // Aligned to avoid false sharing. Each worker loads only its own AlignedValue into the cache line and does not
        // see modifications to adjacent AlignedValues made by other workers.
        struct alignas(64) AlignedValue_type {
            AlignedValue_type(Value_type v) : value(v) {};
            Value_type value;
        };
        std::vector<AlignedValue_type> worker_partial_result(n_worker_, init);

        int range = (end - begin);
        int granularity = range / n_worker_;   // Base granularity value for all workers
        int plusone = range % n_worker_;       // Number of workers that should reduce granularity + 1 values
        std::vector<std::future<void>> ret_fut;
        ret_fut.reserve(n_worker_);
        for (int i = 0; i < plusone; i++) {
            ret_fut.emplace_back(this->send(
              [&worker_partial_result, init, operation, i, granularity = granularity + 1](Iterator begin_local) {
                  Iterator end_local = begin_local + granularity;
                  worker_partial_result[i].value = std::reduce(begin_local, end_local, init, operation);
              },
              begin));
            begin += (granularity + 1);
        }
        for (int i = plusone; i < n_worker_; i++) {
            ret_fut.emplace_back(this->send(
              [&worker_partial_result, init, operation, i, granularity](Iterator begin_local) {
                  Iterator end_local = begin_local + granularity;
                  worker_partial_result[i].value = std::reduce(begin_local, end_local, init, operation);
              },
              begin));
            begin += granularity;
        }
        for (std::future<void>& fut : ret_fut) {
            fut.get();
        }   // Ensures that the reduce operation of all workers has been completed
        // Final reduce
        Value_type ret = worker_partial_result[0].value;
        for (int i = 1; i < n_worker_; i++) { ret = operation(ret, worker_partial_result[i].value); }
        return ret;
    }
};
}   // namespace fdapde
#endif
