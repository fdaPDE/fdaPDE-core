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

#ifndef __FDAPDE_EXECUTION_TASK_GRAPH_H__
#define __FDAPDE_EXECUTION_TASK_GRAPH_H__

#include "header_check.h"

namespace fdapde {
namespace internals {

struct task_graph {
    template <typename TaskGraph> void run(threaded_executor_impl* executor, TaskGraph& graph) {
        using task_pointer = typename threaded_executor_impl::task_pointer;
        const int num_nodes = graph.nodes();
        if (num_nodes == 0) return;

	std::vector<task_pointer> task_table(num_nodes, nullptr);
	std::vector<int> runnable_indices;
	runnable_indices.reserve(num_nodes / 2);
        std::atomic<int> local_task_count {1};

        // load TaskGraph to stable executor memory
        for (int i = 0; i < num_nodes; ++i) {
            int w_id = i % num_threads();
	    const auto& node = graph.node(i);
	    // wrap user task to enable active join logic
            auto wrapped_task = [user_task = node.task(), &local_task_count]() mutable {
                user_task.run();
                local_task_count.fetch_sub(1, std::memory_order_release);
            };
	    task_table[node.id()] = executor->allocate_task(w_id, std::move(wrapped_task));
            if (node.in_degree() == 0) { runnable_indices.push_back(node.id()); }
            local_task_count.fetch_add(1, std::memory_order_release);
        }
	// connect stable pointers
        for (int i = 0; i < num_nodes; ++i) {
            const auto& node = graph.node(i);
            for (auto successor : node.successors()) {
                task_table[node.id()]->add_inverse_dep(task_table[successor->id()]);
            }
        }
        local_task_count.fetch_sub(1, std::memory_order_release);
        // send runnable tasks to execution (dependent tasks will be pulled by the executor autonomously)
        executor->expect_tasks(num_nodes);
        for (int i : runnable_indices) {
            int w_id = i % num_threads();
            executor->enqueue_task(w_id, task_table[i]);
        }
        executor->notify_all();
        executor->active_join(this_worker_id(), [&] {
            // help the pool while the task group is not fully consumed
            return local_task_count.load(std::memory_order_acquire) > 0;
        });
        return;
    }
};

}   // namespace internals

class TaskGraph {
    struct node_type {
        // constructor
        node_type() noexcept : task_(nullptr), id_(0), succ_(), pred_() { }
        template <typename Task_>
        node_type(Task_&& task, int id) :
            task_(new internals::task_handle(std::forward<Task_>(task))), id_(id), succ_(), pred_() { }
        // observers
        int id() const { return id_; }
        const internals::task_handle& task() const { return *task_; }
        int out_degree() const { return succ_.size(); }
        int in_degree() const { return pred_.size(); }
        const std::vector<node_type*>& successors() const { return succ_; }
        const std::vector<node_type*>& predecessors() const { return pred_; }
        // modifiers
        internals::task_handle& task() { return *task_; }
        template <typename... TaskNodes>
            requires(std::is_same_v<std::decay_t<TaskNodes>, node_type> && ...)
        void after(TaskNodes&&... nodes) {
            // wires this node with the supplied dependencies
            internals::for_each_index_and_args<sizeof...(nodes)>(
              [&]<int Ns_, typename TaskNode_>(const TaskNode_& node) {
                  pred_.push_back(&node);
		  node.succ_.push_back(this);
              },
              nodes...);
        }
        ~node_type() = default;
       private:
        internals::task_handle* task_;
        int id_;
        std::vector<node_type*> succ_, pred_;
    };
    // detect cycles
    // topological sort
    // iterators
    // modifiers
   public:
    TaskGraph() : adjacency_() { }

    // observers
    std::size_t nodes() const { return adjacency_.size(); }
    std::size_t edges() const {
        std::size_t edges_ = 0;
        for (node_type* ptr : adjacency_) { edges_ += ptr->out_degree(); }
        return edges_;
    }

  // devono ritornare un node-handle??
    const node_type& node(std::size_t i) const {
        fdapde_assert(i < nodes());
        return *adjacency_[i];
    }
    node_type& node(std::size_t i) {
        fdapde_assert(i < nodes());
        return *adjacency_[i];
    }
    // modifiers
    template <typename Task_> node_type& task(Task_&& task) {
        node_type* ptr = new node_type(std::forward<Task_>(task), adjacency_.size());
        adjacency_.push_back(ptr);
        return *ptr;
    }

    ~TaskGraph() {
        for (node_type* n : adjacency_) { delete n; }
    }
    std::vector<node_type*> adjacency_ {};
};

void parallel_execute(TaskGraph& tg) {
    return internals::threaded_executor::instance().execute(internals::task_graph(), tg);
}

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_TASK_GRAPH_H__
