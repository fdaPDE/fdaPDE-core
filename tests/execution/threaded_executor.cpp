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
#include <array>

TEST(execution, threaded_executor) {
    // threaded_executor: configured sizes preserve async progress and global join.
    {
        for (int size : {1, 2, 4}) {
            EXPECT_EXIT(
              {
                  fdapde::parallel_set_num_threads(size);
                  bool valid = fdapde::parallel_get_num_threads() == size;
                  auto parent = fdapde::parallel_async([] { return fdapde::parallel_async([] { return 42; }).get(); });
                  valid = valid && parent.get() == 42;
                  std::atomic<int> completed {0};
                  for (int i = 0; i < 128; ++i) {
                      fdapde::parallel_execute([&] {
                          ++completed;
                          fdapde::parallel_execute([&] { ++completed; });
                      });
                  }
                  fdapde::parallel_join();
                  std::exit(valid && completed.load() == 256 ? 0 : 1);
              },
              ::testing::ExitedWithCode(0), "");
        }
    }
    // threaded_executor: async arguments, values and errors.
    {
        auto value = fdapde::parallel_async([](int lhs, int rhs) { return lhs + rhs; }, 19, 23);
        static_assert(std::is_same_v<decltype(value), fdapde::future<int>>);
        EXPECT_EQ(value.get(), 42);

        int calls = 0;
        auto completion = fdapde::parallel_async([](int& count) { ++count; }, std::ref(calls));
        static_assert(std::is_same_v<decltype(completion), fdapde::future<void>>);
        completion.get();
        EXPECT_EQ(calls, 1);
        static_assert(std::is_void_v<decltype(fdapde::parallel_execute([] { }))>);

        auto moved = fdapde::parallel_async(
          [offset = std::make_unique<int>(2)](std::unique_ptr<int>& input) {
              *input += *offset;
              return std::move(input);
          },
          std::make_unique<int>(40));
        EXPECT_EQ(*moved.get(), 42);

        auto failure = fdapde::parallel_async([]() -> int { throw std::runtime_error("expected"); });
        EXPECT_NO_THROW(failure.wait());
        EXPECT_TRUE(failure.valid());
        EXPECT_THROW(failure.get(), std::runtime_error);
        EXPECT_FALSE(failure.valid());

        struct throwing_value {
            throwing_value() = default;
            throwing_value(throwing_value&&) { throw std::runtime_error("result construction failed"); }
        };
        auto construction_failure = fdapde::parallel_async([] { return throwing_value {}; });
        EXPECT_THROW(construction_failure.get(), std::runtime_error);
    }
    // threaded_executor: release captures after success and failure.
    {
        using namespace std::chrono_literals;
        for (bool fail : {false, true}) {
            auto resource = std::make_shared<int>(42);
            std::weak_ptr<int> observer = resource;
            auto result = fdapde::parallel_async([resource = std::move(resource), fail] {
                if (fail) { throw std::runtime_error("expected"); }
                return *resource;
            });
            ASSERT_EQ(result.wait_until(std::chrono::steady_clock::now() + 2s), std::future_status::ready);
            fdapde::parallel_join();
            EXPECT_TRUE(observer.expired());
            if (fail) {
                EXPECT_THROW(result.get(), std::runtime_error);
            } else {
                EXPECT_EQ(result.get(), 42);
            }
        }
    }
    // threaded_executor: discarding a future does not cancel work.
    {
        using namespace std::chrono_literals;
        std::promise<void> started, release;
        auto gate = release.get_future().share();
        auto result = fdapde::parallel_async([&] {
            started.set_value();
            gate.wait();
        });
        const auto running = started.get_future().wait_for(2s);
        EXPECT_EQ(running, std::future_status::ready);
        EXPECT_EQ(result.wait_for(0ms), std::future_status::timeout);
        result = fdapde::future<void> {};
        release.set_value();
        fdapde::parallel_join();
    }
    // threaded_executor: nested async waits make progress.
    {
        std::vector<fdapde::future<int>> parents;
        for (int i = 0; i < 16; ++i) {
            parents.push_back(
              fdapde::parallel_async([i] { return fdapde::parallel_async([i] { return i + 1; }).get(); }));
        }
        for (int i = 0; i < 16; ++i) EXPECT_EQ(parents[i].get(), i + 1);
        fdapde::parallel_join();
    }
    // threaded_executor: join includes descendants from concurrent publishers.
    {
        std::atomic<int> completed {0};
        std::vector<std::thread> publishers;
        for (int producer = 0; producer < 4; ++producer) {
            publishers.emplace_back([&] {
                for (int i = 0; i < 512; ++i) {
                    fdapde::parallel_execute([&] {
                        ++completed;
                        fdapde::parallel_execute([&] { ++completed; });
                    });
                }
            });
        }
        for (auto& publisher : publishers) publisher.join();
        fdapde::parallel_join();
        EXPECT_EQ(completed.load(), 4096);
    }
    // threaded_executor: idle workers resume for repeated bursts.
    {
        using namespace std::chrono_literals;
        for (int round = 0; round < 16; ++round) {
            std::this_thread::sleep_for(1ms);
            auto result = fdapde::parallel_async([round] { return round; });
            EXPECT_EQ(result.wait_for(2s), std::future_status::ready);
            EXPECT_EQ(result.get(), round);
            fdapde::parallel_join();
        }
    }
}

// repeated model phases can reuse one capped executor without resetting frozen configuration
TEST(execution, capped_executor_initialization) {
    // invalid caps leave configuration mutable, and subsequent calls retain one pool across threads and budgets
    EXPECT_EXIT(
      ([] {
          bool valid = true;
          for (int cap : {0, -1}) {
              bool rejected = false;
              try {
                  fdapde::parallel_initialize(cap);
              } catch (const std::invalid_argument&) { rejected = true; }
              valid = valid && rejected;
          }
          fdapde::parallel_set_num_threads(4);
          valid = valid && fdapde::parallel_initialize(2) == 2;
          valid = valid && fdapde::parallel_get_num_threads() == 2;
          std::array<int, 4> sizes {};
          std::vector<std::thread> callers;
          for (int i = 0; i < 4; ++i) {
              callers.emplace_back([&, i] { sizes[i] = fdapde::parallel_initialize(4); });
          }
          for (auto& caller : callers) caller.join();
          valid = valid && std::all_of(sizes.begin(), sizes.end(), [](int size) { return size == 2; });
          valid = valid && fdapde::parallel_initialize(2) == 2;
          bool rejected = false;
          try {
              fdapde::parallel_initialize(1);
          } catch (const std::logic_error&) { rejected = true; }
          valid = valid && rejected;
          bool invalid_rejected = false;
          try {
              fdapde::parallel_initialize(0);
          } catch (const std::invalid_argument&) { invalid_rejected = true; }
          valid = valid && invalid_rejected && fdapde::parallel_get_num_threads() == 2;
          auto result = fdapde::parallel_async([] { return 42; });
          valid = valid && result.get() == 42;
          fdapde::parallel_join();
          std::exit(valid ? 0 : 1);
      }()),
      ::testing::ExitedWithCode(0), "");

    // a larger cap preserves a smaller configured pool and its existing single-worker async progress
    EXPECT_EXIT(
      ([] {
          fdapde::parallel_set_num_threads(1);
          bool valid = fdapde::parallel_initialize(4) == 1;
          auto result = fdapde::parallel_async([] { return fdapde::parallel_async([] { return 42; }).get(); });
          valid = valid && result.get() == 42 && fdapde::parallel_get_num_threads() == 1;
          fdapde::parallel_join();
          std::exit(valid ? 0 : 1);
      }()),
      ::testing::ExitedWithCode(0), "");

    // a pool started through the existing async api remains usable after a tighter cap is rejected
    EXPECT_EXIT(
      ([] {
          fdapde::parallel_set_num_threads(3);
          auto initial = fdapde::parallel_async([] { return 42; });
          bool valid = initial.get() == 42;
          bool rejected = false;
          try {
              fdapde::parallel_initialize(2);
          } catch (const std::logic_error&) { rejected = true; }
          valid = valid && rejected && fdapde::parallel_initialize(4) == 3;
          auto after = fdapde::parallel_async([] { return 43; });
          valid = valid && after.get() == 43 && fdapde::parallel_get_num_threads() == 3;
          fdapde::parallel_join();
          std::exit(valid ? 0 : 1);
      }()),
      ::testing::ExitedWithCode(0), "");
}
