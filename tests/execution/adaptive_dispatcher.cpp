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

TEST(execution, adaptive_dispatcher) {
    // adaptive_dispatch: custom algorithm ranges, completion and exceptions.
    {
        struct algorithm {
            std::vector<std::atomic<int>>& visits;
            int& queries;
            bool fail;
            std::pair<int, int> range() const {
                ++queries;
                return {-7, int(visits.size()) - 7};
            }
            void run(int begin, int end) {
                for (int i = begin; i < end; ++i) { ++visits[i + 7]; }
                if (fail) throw std::runtime_error("range body failed");
            }
        };
        for (int count : {0, 1, 3, 17, 257, 1025}) {
            for (bool fail : {false, true}) {
                std::vector<std::atomic<int>> visits(count);
                int queries = 0;
                auto result = fdapde::adaptive_dispatch.run(algorithm {visits, queries, fail});
                EXPECT_NO_THROW(result.wait());
                EXPECT_EQ(queries, 1);
                for (const auto& visit : visits) EXPECT_EQ(visit.load(), 1);
                if (fail && count != 0) {
                    EXPECT_THROW(result.get(), std::runtime_error);
                } else {
                    EXPECT_NO_THROW(result.get());
                }
            }
        }
    }
}
