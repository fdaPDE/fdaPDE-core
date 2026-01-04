
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

