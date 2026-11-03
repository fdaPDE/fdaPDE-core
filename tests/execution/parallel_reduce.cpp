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

#include <forward_list>
#include <stdexcept>
#include <string>

TEST(execution, parallel_reduce) {
    // parallel_reduce: preserves order and applies seed once.
    {
        const auto check = [](auto dispatcher) {
            for (int count : {0, 1, 2, 3, 17, 113}) {
                std::vector<std::string> values;
                std::string expected = "seed:";
                for (int i = 0; i < count; ++i) {
                    values.push_back(std::to_string(i) + ",");
                    expected += values.back();
                }
                EXPECT_EQ(
                  fdapde::parallel_reduce(
                    values.begin(), values.end(), std::string("seed:"), std::plus<> {}, dispatcher),
                  expected);
            }
            const int factors[] {2, 3, 4};
            EXPECT_EQ(fdapde::parallel_reduce(factors, factors + 3, 5, std::multiplies<> {}, dispatcher), 120);
            const std::forward_list<std::string> words {"a", "b", "c", "d", "e"};
            EXPECT_EQ(
              fdapde::parallel_reduce(words.begin(), words.end(), std::string("seed:"), std::plus<> {}, dispatcher),
              "seed:abcde");
        };
        check(fdapde::static_dispatch);
        check(fdapde::adaptive_dispatch);
    }
    // parallel_reduce: supports movable nondefault result.
    {
        const auto check = [](auto dispatcher) {
            struct total {
                int value;
                total(int value) : value(value) { }
                total(total&&) = default;
                total& operator=(total&&) = default;
                total(const total&) = delete;
                total& operator=(const total&) = delete;
            };
            const std::vector<int> values(97, 1);
            auto redux = [](total lhs, auto&& rhs) {
                if constexpr (std::is_same_v<std::decay_t<decltype(rhs)>, total>) {
                    lhs.value += rhs.value;
                } else {
                    lhs.value += rhs;
                }
                return lhs;
            };
            const auto result = fdapde::parallel_reduce(values.begin(), values.end(), total(7), redux, dispatcher);
            EXPECT_EQ(result.value, 104);
        };
        check(fdapde::static_dispatch);
        check(fdapde::adaptive_dispatch);
    }
    // parallel_reduce: propagates task and aggregation errors.
    {
        const auto check = [](auto dispatcher) {
            const std::vector<int> values(128 * fdapde::parallel_get_num_threads(), 1);
            auto fail = [](int, int) -> int { throw std::runtime_error("partial failed"); };
            EXPECT_THROW(
              fdapde::parallel_reduce(values.begin(), values.end(), 0, fail, dispatcher), std::runtime_error);
            const std::vector<std::string> words {"a", "b", "c", "d", "e"};
            auto combine = [](std::string lhs, const std::string& rhs) {
                if (lhs.starts_with("seed:")) { throw std::runtime_error("aggregation failed"); }
                return lhs + rhs;
            };
            EXPECT_THROW(
              fdapde::parallel_reduce(words.begin(), words.end(), std::string("seed:"), combine, dispatcher),
              std::runtime_error);
            EXPECT_EQ(
              fdapde::parallel_reduce(values.begin(), values.end(), 7, std::plus<> {}, dispatcher),
              7 + int(values.size()));
        };
        check(fdapde::static_dispatch);
        check(fdapde::adaptive_dispatch);
    }
    // parallel_reduce: algorithms allow nested waits.
    {
        const auto check = [](auto dispatcher) {
            std::vector<int> values(97);
            auto outer = fdapde::parallel_async([&] {
                fdapde::parallel_for_each(values, [](int& value) { ++value; }, dispatcher).get();
                return fdapde::parallel_reduce(values.begin(), values.end(), 7, std::plus<> {}, dispatcher);
            });
            EXPECT_EQ(outer.get(), 104);
        };
        check(fdapde::static_dispatch);
        check(fdapde::adaptive_dispatch);
    }
}
