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

namespace {
/// @brief changes only the subprocess test environment using the hosting platform's api
void set_concurrency_test_environment(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value ? value : "");
#else
    if (value)
        setenv(name, value, 1);
    else
        unsetenv(name);
#endif
}
}   // namespace

// scheduler budgets are independent of nested numerical-library thread settings
TEST(execution, scheduler_concurrency_budget) {
    // each subprocess compares detection with its real affinity bound and explicit scheduler reservation
    EXPECT_EXIT(
      {
          set_concurrency_test_environment("SLURM_CPUS_PER_TASK", nullptr);
          set_concurrency_test_environment("SLURM_CPUS_ON_NODE", nullptr);
          set_concurrency_test_environment("NCPUS", nullptr);
          set_concurrency_test_environment("OMP_NUM_THREADS", nullptr);
          const auto unrestricted = fdapde::available_concurrency();
          set_concurrency_test_environment("OMP_NUM_THREADS", "1");
          bool valid = fdapde::available_concurrency() == unrestricted;
          set_concurrency_test_environment("NCPUS", "20");
          valid = valid && fdapde::available_concurrency() == std::min<std::size_t>(20, unrestricted);
          set_concurrency_test_environment("SLURM_CPUS_ON_NODE", "16");
          set_concurrency_test_environment("SLURM_CPUS_PER_TASK", "3");
          valid = valid && fdapde::available_concurrency() == std::min<std::size_t>(3, unrestricted);
          set_concurrency_test_environment("SLURM_CPUS_PER_TASK", "invalid");
          valid = valid && fdapde::available_concurrency() == std::min<std::size_t>(20, unrestricted);
          set_concurrency_test_environment("NCPUS", nullptr);
          valid = valid && fdapde::available_concurrency() == std::min<std::size_t>(16, unrestricted);
          set_concurrency_test_environment("SLURM_CPUS_ON_NODE", nullptr);
          for (const char* invalid : {"", "0", "-1", " 2", "2x", "9999999999999999999999"}) {
              set_concurrency_test_environment("NCPUS", invalid);
              valid = valid && fdapde::available_concurrency() == unrestricted;
          }
          std::_Exit(valid ? 0 : 1);
      },
      ::testing::ExitedWithCode(0), "");
#ifdef __linux__
    // affinity remains an upper bound even when a scheduler advertises more CPUs
    EXPECT_EXIT(
      {
          set_concurrency_test_environment("SLURM_CPUS_PER_TASK", nullptr);
          set_concurrency_test_environment("SLURM_CPUS_ON_NODE", nullptr);
          set_concurrency_test_environment("NCPUS", "20");
          cpu_set_t allowed;
          bool valid = sched_getaffinity(0, sizeof(allowed), &allowed) == 0;
          int cpu = 0;
          while (cpu < CPU_SETSIZE && !CPU_ISSET(cpu, &allowed)) ++cpu;
          cpu_set_t selected;
          CPU_ZERO(&selected);
          if (cpu < CPU_SETSIZE) CPU_SET(cpu, &selected);
          valid = valid && sched_setaffinity(0, sizeof(selected), &selected) == 0;
          valid = valid && fdapde::available_concurrency() == 1;
          std::_Exit(valid ? 0 : 1);
      },
      ::testing::ExitedWithCode(0), "");
#endif
}
