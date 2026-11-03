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

#include <array>

TEST(execution, task_handle) {
    // task_handle: handle lifetime.
    {
        using fdapde::internals::task_handle;
        int calls = 0;
        auto trivial = [&] { ++calls; };
        static_assert(std::is_trivially_destructible_v<decltype(trivial)>);
        task_handle first(trivial), second(first), third(std::move(first));
        second.run();
        third.run();
        first = second;
        first.run();
        EXPECT_EQ(calls, 3);
        auto resource = std::make_shared<int>(42);
        std::weak_ptr<int> observer = resource;
        {
            task_handle owned([held = resource] { EXPECT_EQ(*held, 42); });
            resource.reset();
            task_handle copied(owned), moved(std::move(owned));
            moved.run();
            copied.run();
            EXPECT_FALSE(observer.expired());
        }
        EXPECT_TRUE(observer.expired());
        // a trivial heap-backed callable still needs its allocation deleted
        struct alignas(128) heap_callable {
            int* calls;
            void operator()() { ++*calls; }
        };
        {
            task_handle heap(heap_callable {&calls});
            task_handle copy(heap);
            heap.run();
            copy.run();
        }
        EXPECT_EQ(calls, 5);
    }
    // task_handle: handle execution data.
    {
        using namespace fdapde::internals;
        const execution_data data {3, 7};
        int calls = 0;
        auto body = [&](const execution_data& current) {
            EXPECT_EQ(current.exec_context, 3);
            EXPECT_EQ(current.orig_context, 7);
            ++calls;
        };
        task_handle first(body), copied(first), moved(std::move(first));
        copied.run(data);
        moved.run(data);
        first = copied;
        first.run(data);
        struct alignas(128) heap_task {
            decltype(body) function;
            void operator()(const execution_data& current) { function(current); }
        };
        task_handle heap(heap_task {body}), heap_copy(heap), heap_move(std::move(heap));
        heap_copy.run(data);
        heap_move.run(data);
        EXPECT_EQ(calls, 5);
    }
    // task_handle: invocation signature.
    {
        using namespace fdapde::internals;
        struct contextual_task {
            std::atomic<int>* calls;
            void operator()(const execution_data&) & { ++*calls; }
        };
        struct plain_task {
            std::atomic<int>* calls;
            void operator()() & { ++*calls; }
        };
        struct rvalue_task {
            void operator()(const execution_data&) && { }
        };
        struct rvalue_data_task {
            void operator()(execution_data&&) { }
        };
        static_assert(is_task_wrappable_v<contextual_task> && is_task_wrappable_v<plain_task>);
        static_assert(!is_task_wrappable_v<rvalue_task> && !is_task_wrappable_v<rvalue_data_task>);
        static_assert(!std::is_constructible_v<task_handle, rvalue_task>);
        task_handle defaults([](const execution_data& data) {
            EXPECT_EQ(data.exec_context, external_thread);
            EXPECT_EQ(data.orig_context, external_thread);
        });
        defaults.run();
        std::atomic<int> calls {0};
        task_handle contextual(contextual_task {&calls}), plain(plain_task {&calls});
        contextual.run();
        plain.run();
        EXPECT_EQ(calls.load(), 2);
    }
}
