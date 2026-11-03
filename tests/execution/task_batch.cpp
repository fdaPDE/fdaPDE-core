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

TEST(execution, task_batch) {
    // task_batch: empty.
    {
        using namespace std::chrono_literals;
        int prepared = 0;
        auto result = fdapde::internals::task_batch::run(0, [&] {
            ++prepared;
            return [] { };
        });
        EXPECT_EQ(prepared, 0);
        EXPECT_TRUE(result.valid());
        EXPECT_EQ(result.wait_for(0ms), std::future_status::ready);
        result.get();
        EXPECT_FALSE(result.valid());
    }
    // task_batch: body error is retrieved by get.
    {
        std::atomic<int> completed {0};
        int prepared = 0;
        auto result = fdapde::internals::task_batch::run(13, [&] {
            return [&, id = prepared++] {
                if (id == 0) { throw std::runtime_error("body failed"); }
                ++completed;
            };
        });
        EXPECT_NO_THROW(result.wait());
        EXPECT_EQ(completed.load(), 12);
        EXPECT_THROW(result.get(), std::runtime_error);
        EXPECT_FALSE(result.valid());
        fdapde::parallel_join();
    }
    // task_batch: discarded task batch future keeps task captures alive.
    {
        std::latch release(1);
        std::atomic<int> completed {0};
        auto resource = std::make_shared<int>(42);
        std::weak_ptr<int> observer = resource;
        {
            auto result = fdapde::internals::task_batch::run(2, [resource, &release, &completed] {
                return [resource, &release, &completed] {
                    release.wait();
                    EXPECT_EQ(*resource, 42);
                    ++completed;
                };
            });
            resource.reset();
        }
        EXPECT_FALSE(observer.expired());
        release.count_down();
        fdapde::parallel_join();
        EXPECT_EQ(completed.load(), 2);
        EXPECT_TRUE(observer.expired());
    }
}
