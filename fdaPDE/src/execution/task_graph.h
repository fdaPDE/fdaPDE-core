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

class TaskGraph {
    struct node {
        template <typename Task_>
        node(Task_&& task, int pos) :
            task_(new internals::task_handle(std::forward<Task_>(task))), pos_(pos), succ_() { }

        internals::task_handle* task_;
        int pos_;   // position of this node in adjacency list
        std::vector<node*> succ_;

        template <typename... Tasks>
            requires(std::is_same_v<std::decay_t<Tasks>, node> && ...)
        void after(Tasks&&... tasks) {
            internals::for_each_index_and_args<sizeof...(tasks)>(
              [&]<int Ns_, typename Task_>(const Task_& t) {
                  succ_.push_back(std::addressof(t));
		  t.task_->required_by().push_back(task_); 
                  task_->ref_count_fetch_add(1, std::memory_order_release);
              },
              tasks...);
        }
        ~node() = default;
    };

    // detect cycles
    // topological sort
    // iterators
    // modifiers

  public:
    TaskGraph() : adjacency_() {}

    // observers
    std::size_t nodes() const { return adjacency_.size(); }
    std::size_t edges() const {
        std::size_t edges_ = 0;
        for (node* ptr : adjacency_) { edges_ += ptr->succ_.size(); }
        return edges_;
    }

    // modifiers
    template <typename Task_> node& task(Task_&& task) {
        node* ptr = new node(std::forward<Task_>(task), adjacency_.size());
        adjacency_.push_back(ptr);
        return *(adjacency_.back());
    }

    ~TaskGraph() {
        for (node* n : adjacency_) { delete n; }
    }
    std::vector<node*> adjacency_;
};

    // execute a TaskGraph object
    // void execute(const TaskGraph& tg) {
    //     const size_type num_nodes = tg.nodes();
    //     if (num_nodes == 0) return;

    //     std::unordered_map<task_pointer, task_pointer> task_map;
    //     std::vector<int> worker_vec(num_nodes);
    //     std::vector<task_pointer> ready_tasks;
    //     // copy task graph to stable memory
    //     for (size_type i = 0; i < num_nodes; ++i) {
    //         int w_id = scheduling_policy_.pick();
    //         worker_vec[i] = w_id;
    //         task_pointer old_ptr = tg.adjacency_[i]->task_;
    //         task_pointer new_ptr = workers_[w_id]->allocate_task(*old_ptr);
    //         task_map[old_ptr] = new_ptr;
    //     }
    //     // replace old dependency pointers with stable worker-local pointers
    //     for (const auto& [_, new_ptr] : task_map) {
    //         for (auto& old_ptr : new_ptr->required_by()) { old_ptr = task_map[old_ptr]; }
    //         if (new_ptr->runnable()) { ready_tasks.push_back(new_ptr); }
    //     }
    //     // notify workers and start execution
    //     std::unique_lock<std::mutex> lock(m_);
    //     task_count_ += num_nodes;   // increase task count
    //     lock.unlock();
    //     for (size_type i = 0; i < ready_tasks.size(); ++i) { workers_[worker_vec[i]]->enqueue_task(ready_tasks[i]); }
    //     cv_.notify_all();
    //     return;
    // }
  
// void parallel_execute(const TaskGraph& tg) {
//     internals::threadpool_executor::instance().execute(tg);
//     internals::threadpool_executor::instance().join();  
// }

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_TASK_GRAPH_H__
