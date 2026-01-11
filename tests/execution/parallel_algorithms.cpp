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

#include <gtest/gtest.h>   // testing framework
using namespace fdapde;

TEST(execution, parallel_for) {
    std::vector<double> v1;
    v1.resize(1000000);
    for (int i = 0; i < int(v1.size()); ++i) { v1[i] = 1; }
    std::vector<double> v2;
    v2.resize(1000000);
    for (int i = 0; i < int(v2.size()); ++i) { v2[i] = 2; }
    std::vector<double> v3;
    v3.resize(1000000);
    std::fill(v3.begin(), v3.end(), 0);

    int batch_size = 10000;
    fdapde::parallel_for(0, v1.size(), batch_size, [&](int i) { v3[i] += (v1[i] + v2[i]); });
    for (double val : v3) { EXPECT_EQ(val, 3.0); }

    // default batch_size
    fdapde::parallel_for(0, v1.size(), [&](int i) { v3[i] += (v1[i] + v2[i]); });
    for (double val : v3) { EXPECT_EQ(val, 6.0); }

    // nested parallel_for (check each iteration is eventually considered)
    int outer_size = 10;
    int inner_size = 100;
    std::vector<int> v4(inner_size * outer_size, 0);
    std::atomic<int> counter;

    fdapde::parallel_for(0, outer_size, [&](int i) {
        fdapde::parallel_for(0, inner_size, [&](int j) { v4[i * inner_size + j] = 1; });
        counter.fetch_add(1, std::memory_order_release);
    });

    for (int val : v4) { EXPECT_EQ(val, 1); }   // all iterations considered
    EXPECT_EQ(counter.load(), outer_size);      // all outer itereations waited for the inner iteration and proceed

    // nesting a paralle_reduce inside a parallel_for
    std::vector<int> v5(10000);
    std::mt19937 rng(std::random_device {}());
    std::uniform_int_distribution<> dist(1, 1000000);
    int expected_max = 0;
    for (int i = 0, n = v5.size(); i < n; ++i) {
        int tmp = dist(rng);
        if (tmp > expected_max) { expected_max = tmp; }
        v5[i] = tmp;
    }
    std::vector<int> v6(parallel_get_num_threads(), 0);
    // parallel computation of maximum
    fdapde::parallel_for(0, v5.size() / 100, 100, [&](int i) {
        int partial_max =
          fdapde::parallel_reduce(v5.begin() + (i * 100), v5.begin() + ((i + 1) * 100 - 1), int(0), [](int a, int b) {
              return std::max(a, b);
          });
        v6[this_thread_id()] = std::max(v6[this_thread_id()], partial_max);
    });

    int computed_max = 0;
    for(int val : v6) { computed_max = std::max(computed_max, val); }
    EXPECT_EQ(computed_max, expected_max);

    // parallel for with custom stepping logic
    std::vector<int> v7(10000, 0);
    fdapde::parallel_for(0, v7.size(), [&](int i) { v7[i] = 1; }, [](int i) { return i + 2; });
    // only even indices must be 1
    for (int i = 0, n = v7.size(); i < n; ++i) {
        if (i % 2 == 0) {
            EXPECT_EQ(v7[i], 1);
        } else {
            EXPECT_EQ(v7[i], 0);
        }
    }
}

TEST(execution, parallel_for_each) {
    std::vector<double> v;
    v.resize(10000);
    for(int i = 0, n = v.size(); i < n; ++i) { v[i] = 0; }

    fdapde::parallel_for_each(v, [&](auto& i) { i += 1; });   // check all items are seen exactly once
    for(int i = 0, n = v.size(); i < n; ++i) { EXPECT_EQ(v[i], 1); }
}

TEST(execution, parallel_reduce) {
    std::vector<double> v;
    v.resize(10000);
    for(int i = 0, n = v.size(); i < n; ++i) { v[i] = 1; }

    // compute parallel sum
    int sum = fdapde::parallel_reduce(v.begin(), v.end(), int(0), [](auto a, auto b) { return a + b; });
    EXPECT_EQ(sum, int(v.size()));
}

