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

#ifndef __FDAPDE_EXECUTION_FUTURE_H__
#define __FDAPDE_EXECUTION_FUTURE_H__

#include "header_check.h"

namespace fdapde {

template <typename T> class future {
    static_assert(std::is_void_v<T> || (std::is_object_v<T> && std::is_move_constructible_v<T>));
   public:
    struct state_type {
        state_type() : executor_(internals::threaded_executor_instance()) { }

        // retain only while the producer or another task still holds a pending completion
        void register_task() noexcept { pending_.fetch_add(1, std::memory_order_relaxed); }
        // decrease the reference count, signaling as complete a previously retained operation
        void complete_task() noexcept {
            if (pending_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                {
                    std::lock_guard<std::mutex> lock(m_);
                    ready_cv_.notify_all();
                }
                executor_->notify_all_join();
            }
        }
        bool ready() const noexcept { return pending_.load(std::memory_order_acquire) == 0; }

        // externally invoked to set a valid state for the future
        template <typename... Args>
            requires(
              (std::is_void_v<T> && sizeof...(Args) == 0) ||
              (!std::is_void_v<T> && std::is_constructible_v<T, Args...>))
        void set_value(Args&&... args) {
            fdapde_assert(pending_.load(std::memory_order_acquire) != 0);
            std::lock_guard<std::mutex> lock(m_);
            if (value_) { throw std::future_error(std::future_errc::promise_already_satisfied); }
            value_.emplace(std::forward<Args>(args)...);
        }
        // externally invoked to set an error state for the future
        void set_error(std::exception_ptr error) {
            fdapde_assert(error && pending_.load(std::memory_order_acquire) != 0);
            std::lock_guard<std::mutex> lock(m_);
            if (!error_) { error_ = std::move(error); }
        }
       private:
        friend class future<T>;
        using value_type = std::conditional_t<std::is_void_v<T>, bool, T>;
        internals::threaded_executor* executor_;

        std::atomic<std::size_t> pending_ {1};   // initalized with sentinel, number of pending completions
        std::exception_ptr error_;               // error state if asynchronous computation yield an exception state
        std::optional<value_type> value_;        // result of the asynchronous call
        std::condition_variable ready_cv_;
        bool consumed_ = false;
        std::mutex m_;
    };

    // constructors
    future() noexcept = default;
    explicit future(std::shared_ptr<state_type> state) : state_(std::move(state)) {
        if (state_) {
            std::lock_guard<std::mutex> lock(state_->m_);
            if (std::exchange(state_->consumed_, true)) {
                throw std::future_error(std::future_errc::future_already_retrieved);
            }
        }
    }
    // delete copy-semantic, retain move-semantic
    future(const future&) = delete;
    future& operator=(const future&) = delete;
    future(future&&) noexcept = default;
    future& operator=(future&&) noexcept = default;

    // checks if the future has a shared state
    bool valid() const noexcept { return bool(state_); }
    // wait for the result to become available
    void wait() const {
        if (!state_) { throw std::future_error(std::future_errc::no_state); }
        if (state_->ready()) { return; }   // nothing to wait
        // send the caller in active_join if possible, otherwise put it in passive wait
        auto attachment_handle = state_->executor_->try_attach();
        state_->executor_->active_join(this_thread_id(), [&] { return !state_->ready(); });
    }
    // waits for the result, returns if it is not available for the specified timeout duration
    template <typename Rep, typename Period>
    std::future_status wait_for(const std::chrono::duration<Rep, Period>& timeout) const {
        if (!state_) { throw std::future_error(std::future_errc::no_state); }
        std::unique_lock<std::mutex> lock(state_->m_);
        return state_->ready_cv_.wait_for(lock, timeout, [&] { return state_->ready(); }) ?
                 std::future_status::ready :
                 std::future_status::timeout;
    }
    // waits for the result, returns if it is not available until specified time point has been reached
    template <typename Clock, typename Duration>
    std::future_status wait_until(const std::chrono::time_point<Clock, Duration>& deadline) const {
        if (!state_) { throw std::future_error(std::future_errc::no_state); }
        std::unique_lock<std::mutex> lock(state_->m_);
        return state_->ready_cv_.wait_until(lock, deadline, [&] { return state_->ready(); }) ?
                 std::future_status::ready :
                 std::future_status::timeout;
    }
    // return the result if available, otherwise waits for the result, throw if the async computation has failed
    T get() {
        wait();
        auto state = std::move(state_);
	// rethrow if computation has failed
        if (state->error_) { std::rethrow_exception(state->error_); }
        if constexpr (!std::is_void_v<T>) {
            if (!state->value_) { throw std::future_error(std::future_errc::broken_promise); }
	    // return result
            return std::move(*state->value_);
        }
    }
   private:
    std::shared_ptr<state_type> state_;
};

template <typename T = void> auto make_future() {
    auto state = std::make_shared<typename future<T>::state_type>();
    return std::pair {future<T>(state), state};
}

// a task_handle with an attached future object for asynchronous execution
template <typename Signature> class packaged_task;
template <typename R, typename... Args> class packaged_task<R(Args...)> {
    using state_type = typename future<R>::state_type;
    using return_type = R;
    using arguments_type = std::optional<std::tuple<Args...>>;

    template <typename Callable> struct task_payload {
        task_payload(std::shared_ptr<state_type> state, Callable task) :
            state_(std::move(state)), task_(std::move(task)) {
            if constexpr (sizeof...(Args) == 0) { arguments_.emplace(); }
        }

        void operator()() {
            // packaged_task can be invoked only once
            if (std::exchange(invoked_, true)) { throw std::future_error(std::future_errc::promise_already_satisfied); }
            try {
                // call task_ dispatching whether it accepts no argument or not
                if constexpr (std::is_void_v<R>) {
                    std::apply(task_, std::move(*arguments_));
                    state_->set_value();
                } else {
                    state_->set_value(std::apply(task_, std::move(*arguments_)));
                }
            } catch (...) {
                state_->set_error(std::current_exception());
            }
	    // signal future completion
            state_->complete_task();
        }

        ~task_payload() {
            if (!invoked_) {
                state_->set_error(std::make_exception_ptr(std::future_error(std::future_errc::broken_promise)));
                state_->complete_task();
            }
        }

        std::shared_ptr<state_type> state_;
        Callable task_;
        arguments_type arguments_;
        bool invoked_ = false;
    };
   public:

    template <typename Callable>
        requires(std::is_invocable_r_v<R, std::decay_t<Callable>&, Args...>)
    explicit packaged_task(Callable&& task) : state_(std::make_shared<state_type>()) {
        // the state's initial pending reservation belongs to this payload
        auto payload = std::make_shared<task_payload<std::decay_t<Callable>>>(state_, std::forward<Callable>(task));
        arguments_ = std::addressof(payload->arguments_);
        task_.emplace([payload]() { (*payload)(); });
    }
    packaged_task(const packaged_task&) = delete;
    packaged_task& operator=(const packaged_task&) = delete;

    // return a fdapde::future associated with the promised result
    future<R> get_future() { return future<R>(state_); }
    // executes the stored task
    void operator()(Args... args) {
        if (!task_) { throw std::future_error(std::future_errc::promise_already_satisfied); }
        if constexpr (sizeof...(Args) != 0) { arguments_->emplace(std::forward<Args>(args)...); }
        auto task = *task_;
        task_.reset();
        task.run();
	return;
    }
    // return the task_handle associated with this packaged_task
    const internals::task_handle& get_task() const requires(sizeof...(Args) == 0) {
        if (!task_) { throw std::future_error(std::future_errc::no_state); }
        return *task_;
    }
   private:
    std::shared_ptr<state_type> state_;
    std::optional<internals::task_handle> task_;
    arguments_type* arguments_ = nullptr;   // points into the payload owned by task_
};

template <typename Callable>
packaged_task(Callable&&)
  -> packaged_task<std::invoke_result_t<std::decay_t<Callable>&>()>;

template <typename R, typename... Args>
packaged_task(R (*)(Args...)) -> packaged_task<R(Args...)>;

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_FUTURE_H__
