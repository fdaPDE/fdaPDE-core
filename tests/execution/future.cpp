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

#include <fdaPDE/execution.h>
#include <gtest/gtest.h>

#include <stdexcept>

TEST(execution, future) {
    // future and packaged_task: packaged task public interface.
    {
        using namespace std::chrono_literals;
        fdapde::packaged_task task([value = std::make_unique<int>(42)] { return *value; });
        static_assert(std::is_same_v<decltype(task), fdapde::packaged_task<int()>>);
        static_assert(!std::is_copy_constructible_v<decltype(task)>);
        auto result = task.get_future();
        static_assert(std::is_same_v<decltype(result), fdapde::future<int>>);
        EXPECT_THROW(task.get_future(), std::future_error);
        EXPECT_EQ(result.wait_for(0ms), std::future_status::timeout);
        task();
        EXPECT_EQ(result.wait_for(0ms), std::future_status::ready);
        EXPECT_THROW(task(), std::future_error);
        EXPECT_EQ(result.get(), 42);

        fdapde::packaged_task completed([] { });
        completed();
        EXPECT_NO_THROW(completed.get_future().get());
    }
    // future and packaged_task: abandoned packaged task reports broken promise.
    {
        using namespace std::chrono_literals;
        fdapde::future<int> result;
        auto resource = std::make_shared<int>(42);
        std::weak_ptr<int> observer = resource;
        {
            fdapde::packaged_task task([resource = std::move(resource)] { return *resource; });
            result = task.get_future();
        }
        EXPECT_TRUE(observer.expired());
        ASSERT_EQ(result.wait_for(0ms), std::future_status::ready);
        try {
            result.get();
            FAIL() << "missing broken_promise error";
        } catch (const std::future_error& error) {
            EXPECT_EQ(error.code(), std::make_error_code(std::future_errc::broken_promise));
        }
        EXPECT_FALSE(result.valid());
    }
    // future and packaged_task: packaged task gets native handle.
    {
        using namespace std::chrono_literals;
        fdapde::packaged_task<int()> packaged([value = std::make_unique<int>(42)] { return *value; });
        static_assert(
          std::is_same_v<decltype(std::as_const(packaged).get_task()), const fdapde::internals::task_handle&>);
        auto task = packaged.get_task();
        static_assert(std::is_same_v<decltype(task), fdapde::internals::task_handle>);
        auto copied = packaged.get_task();
        auto result = packaged.get_future();
        EXPECT_EQ(result.wait_for(0ms), std::future_status::timeout);
        task.run();
        EXPECT_EQ(result.wait_for(0ms), std::future_status::ready);
        EXPECT_THROW(copied.run(), std::future_error);
        EXPECT_EQ(result.get(), 42);
    }
    // future and packaged_task: packaged task readiness does not release native captures.
    {
        auto resource = std::make_shared<int>(42);
        std::weak_ptr<int> observer = resource;
        {
            fdapde::packaged_task<int()> packaged([held = std::move(resource)] { return *held; });
            auto task = packaged.get_task();
            auto copy = task;
            auto result = packaged.get_future();
            task.run();
            EXPECT_EQ(result.get(), 42);
            EXPECT_FALSE(observer.expired());
            EXPECT_THROW(copy.run(), std::future_error);
        }
        EXPECT_TRUE(observer.expired());
    }
    // future and packaged_task: packaged task signature arguments.
    {
        fdapde::packaged_task<int(int, int)> sum([](int lhs, int rhs) { return lhs + rhs; });
        auto value = sum.get_future();
        sum(19, 23);
        EXPECT_EQ(value.get(), 42);
        EXPECT_THROW(sum(1, 2), std::future_error);

        int count = 19, increment = 23;
        fdapde::packaged_task<void(int&, const int&)> update([](int& target, const int& amount) { target += amount; });
        auto completion = update.get_future();
        update(count, increment);
        completion.get();
        EXPECT_EQ(count, 42);

        fdapde::packaged_task<std::unique_ptr<int>(std::unique_ptr<int>)> move_value(
          [offset = std::make_unique<int>(2)](std::unique_ptr<int> input) {
              *input += *offset;
              return input;
          });
        auto moved = move_value.get_future();
        move_value(std::make_unique<int>(40));
        EXPECT_EQ(*moved.get(), 42);

        fdapde::packaged_task<std::unique_ptr<int>(std::unique_ptr<int>&&)> forward_value(
          [](std::unique_ptr<int>&& input) { return std::move(input); });
        auto forwarded = forward_value.get_future();
        auto input = std::make_unique<int>(42);
        forward_value(std::move(input));
        EXPECT_FALSE(input);
        EXPECT_EQ(*forwarded.get(), 42);

        auto function = +[](int lhs, int rhs) { return lhs + rhs; };
        fdapde::packaged_task inferred(function);
        static_assert(std::is_same_v<decltype(inferred), fdapde::packaged_task<int(int, int)>>);
        auto inferred_value = inferred.get_future();
        inferred(19, 23);
        EXPECT_EQ(inferred_value.get(), 42);
    }
    // future and packaged_task: packaged task signature errors.
    {
        fdapde::packaged_task<int(int)> failure([](int) -> int { throw std::runtime_error("expected"); });
        auto result = failure.get_future();
        EXPECT_NO_THROW(failure(42));
        EXPECT_NO_THROW(result.wait());
        EXPECT_THROW(result.get(), std::runtime_error);

        fdapde::future<void> abandoned;
        {
            fdapde::packaged_task<void(int&)> task([](int&) { });
            abandoned = task.get_future();
        }
        EXPECT_THROW(abandoned.get(), std::future_error);
    }
    // future: empty state errors and move ownership.
    {
        using namespace std::chrono_literals;
        fdapde::future<int> empty;
        EXPECT_FALSE(empty.valid());
        EXPECT_THROW(empty.wait(), std::future_error);
        EXPECT_THROW(empty.wait_for(0ms), std::future_error);
        EXPECT_THROW(empty.wait_until(std::chrono::steady_clock::now()), std::future_error);
        EXPECT_THROW(empty.get(), std::future_error);
        fdapde::packaged_task task([] { return 42; });
        auto original = task.get_future();
        auto moved = std::move(original);
        EXPECT_FALSE(original.valid());
        EXPECT_TRUE(moved.valid());
        EXPECT_EQ(moved.wait_until(std::chrono::steady_clock::now()), std::future_status::timeout);
        task();
        empty = std::move(moved);
        EXPECT_FALSE(moved.valid());
        EXPECT_EQ(empty.wait_for(0ms), std::future_status::ready);
        EXPECT_EQ(empty.get(), 42);
        EXPECT_FALSE(empty.valid());
        EXPECT_THROW(empty.get(), std::future_error);
    }
    // make future value and completion.
    {
        using namespace std::chrono_literals;
        auto [result, state] = fdapde::make_future<std::unique_ptr<int>>();
        static_assert(std::is_same_v<decltype(result), fdapde::future<std::unique_ptr<int>>>);
        static_assert(std::is_same_v<decltype(state), std::shared_ptr<decltype(result)::state_type>>);
        EXPECT_TRUE(result.valid());
        EXPECT_FALSE(state->ready());
        EXPECT_EQ(result.wait_for(0ms), std::future_status::timeout);
        EXPECT_THROW((fdapde::future<std::unique_ptr<int>>(state)), std::future_error);
        state->register_task();
        state->set_value(std::make_unique<int>(42));
        state->complete_task();
        EXPECT_EQ(result.wait_until(std::chrono::steady_clock::now()), std::future_status::timeout);
        fdapde::parallel_execute([state] { state->complete_task(); });
        EXPECT_EQ(*result.get(), 42);
        EXPECT_TRUE(state->ready());
        EXPECT_FALSE(result.valid());
        fdapde::parallel_join();
    }
    // make future void and error.
    {
        auto [completion, state] = fdapde::make_future<>();
        static_assert(std::is_same_v<decltype(completion), fdapde::future<void>>);
        state->set_value();
        state->complete_task();
        EXPECT_NO_THROW(completion.get());

        auto [failure, error_state] = fdapde::make_future<int>();
        error_state->set_error(std::make_exception_ptr(std::runtime_error("expected")));
        error_state->complete_task();
        EXPECT_NO_THROW(failure.wait());
        EXPECT_THROW(failure.get(), std::runtime_error);
        EXPECT_FALSE(failure.valid());
    }
}
