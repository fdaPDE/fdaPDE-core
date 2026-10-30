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

// a set of tasks with DAG-type dependencies
struct task_graph {
    template <typename TaskGraph> static void run(threaded_executor_impl* executor, TaskGraph& graph) {
        fdapde_assert(!graph.has_cycles(), "task graphs must be acyclic");
        using task_submission_handle = typename threaded_executor_impl::task_submission_handle;
        const int num_nodes = graph.n_nodes();
        if (num_nodes == 0) return;
        const int num_threads = executor->size();

        struct node_state {
            std::atomic<int> pending {0};
            task_submission_handle task;
            std::vector<int> successors;
        };
        std::vector<node_state> task_table(num_nodes);
        std::vector<task_submission_handle> sources;
        sources.reserve(num_nodes / 2);
        std::atomic<int> remaining {num_nodes};
        std::mutex error_mutex;
        std::exception_ptr error;
        auto attachment = executor->try_attach();
        const int caller_id = this_thread_id();

        try {
            // prepare every task and dependency before any task becomes visible to workers
            for (const auto& node : graph) {
                const int node_id = node.id();
                task_table[node_id].pending.store(node.in_degree(), std::memory_order_relaxed);
                task_table[node_id].successors = node.succ_ids();
                auto wrapped_task = [user_task = std::optional {node.task()}, &task_table, &remaining, &error_mutex,
                                     &error, executor, node_id]() mutable {
                    try {
                        user_task->run();
                    } catch (...) {
                        std::lock_guard<std::mutex> lock(error_mutex);
                        if (!error) error = std::current_exception();
                    }
                    // publish successors before completing this task so join cannot observe an empty executor
                    for (int successor_id : task_table[node_id].successors) {
                        auto& successor = task_table[successor_id];
                        if (successor.pending.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                            executor->submit_task(successor.task);
                        }
                    }
                    user_task.reset();
                    if (remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) executor->notify_all_join();
                };
                // any predecessor may publish this task, so allocate on another slot to reserve mailbox storage
                int worker_id = node_id % num_threads;
                if (caller_id >= 0 && worker_id >= caller_id) { ++worker_id; }
                task_table[node_id].task = executor->allocate_task(worker_id, std::move(wrapped_task));
                if (node.in_degree() == 0) { sources.push_back(task_table[node_id].task); }
            }
            executor->submit_task(sources);
        } catch (...) {
            for (auto& state : task_table) {
                if (state.task) { state.task->discard(); }
            }
            throw;
        }
        executor->active_join(this_thread_id(), [&] { return remaining.load(std::memory_order_acquire) != 0; });
        if (error) std::rethrow_exception(error);
        return;
    }
};

}   // namespace internals

class TaskGraph {
    // internal graph node owning task and connectivity informations
    struct node_type {
        node_type() noexcept : task(), succ(), pred(), id(0) { }
        template <typename Task_>
        node_type(Task_&& task_, int id_) : task(std::forward<Task_>(task_)), succ(), pred(), id(id_) { }

        internals::task_handle task;
        std::vector<int> succ;
        std::vector<int> pred;
        int id;
    };

    // stable proxy to graph node
    template <typename TaskGraph_> struct node_proxy {
        // constructor
        node_proxy(TaskGraph_* graph, int id) : graph_(graph), id_(id) { }
        // observers
        const std::vector<int>& succ_ids() const { return graph_->nodes_[id_].succ; }
        const std::vector<int>& pred_ids() const { return graph_->nodes_[id_].pred; }
        int in_degree() const { return pred_ids().size(); }
        int out_degree() const { return succ_ids().size(); }
        const internals::task_handle& task() const { return graph_->nodes_[id_].task; }
        int id() const { return id_; }
        // query
        bool is_source() const { return in_degree() == 0; }
        bool is_sink() const { return out_degree() == 0; }
        // modifiers
        template <typename... TaskNodes>
            requires(std::is_same_v<std::decay_t<TaskNodes>, node_proxy> && ...)
        void precedes(TaskNodes&&... nodes) {
            (graph_->add_edge(*this, std::forward<TaskNodes>(nodes)), ...);
        }
        template <typename... TaskNodes>
            requires(std::is_same_v<std::decay_t<TaskNodes>, node_proxy> && ...)
        void succeeds(TaskNodes&&... nodes) {
            (graph_->add_edge(std::forward<TaskNodes>(nodes), *this), ...);
        }
       private:
        friend class TaskGraph;
        TaskGraph_* graph_;
        int id_;
    };

    // perform topological sorting of the TaskGraph by Khan's algorithm
    std::vector<int> topological_sort_() const {
        const int n_nodes = nodes_.size();
        std::vector<int> in_degrees(n_nodes, 0);
        std::queue<int> queue;

        for (int i = 0; i < n_nodes; ++i) {
            in_degrees[i] = node(i).in_degree();
            if (in_degrees[i] == 0) {   // push nodes with no dependencies
                queue.push(i);
            }
        }

        std::vector<int> order;
        while (!queue.empty()) {
            int curr = queue.front();
            queue.pop();
            order.push_back(curr);

            const auto current = node(curr);
            for (int next : current.succ_ids()) {
                in_degrees[next]--;   // decrease dependency count
                if (in_degrees[next] == 0) { queue.push(next); }
            }
        }
        return order;
    }
   public:
    using reference = node_proxy<TaskGraph>;
    using const_reference = node_proxy<const TaskGraph>;
    // constructor
    TaskGraph() : nodes_() { }
    // observers
    int n_nodes() const { return nodes_.size(); }
    int n_edges() const {
        return std::accumulate(
          nodes_.begin(), nodes_.end(), 0, [](int sum, const node_type& node) { return sum + node.succ.size(); });
    }
    reference node(int i) { return reference(this, i); }
    const_reference node(int i) const { return const_reference(this, i); }
    reference operator[](int i) { return reference(this, i); }
    const_reference operator[](int i) const { return const_reference(this, i); }
    // modifiers
    template <typename Task_>
        requires(std::is_invocable_v<std::decay_t<Task_>&> && std::is_copy_constructible_v<std::decay_t<Task_>>)
    reference add_node(Task_&& task) {
        int i = nodes_.size();
        nodes_.emplace_back(std::forward<Task_>(task), i);
        return node_proxy(this, i);
    }
    template <typename... Tasks_>
        requires(
          sizeof...(Tasks_) > 1 && (std::is_invocable_v<std::decay_t<Tasks_>&> && ...) &&
          (std::is_copy_constructible_v<std::decay_t<Tasks_>> && ...))
    auto add_node(Tasks_&&... tasks) {
        return std::make_tuple(add_node(std::forward<Tasks_>(tasks))...);
    }
    template <typename TaskGraph_> void add_edge(node_proxy<TaskGraph_> from, node_proxy<TaskGraph_> to) {
        fdapde_assert(from.graph_ == this && to.graph_ == this, "connecting nodes from different task graphs");
        nodes_[from.id()].succ.push_back(to.id());
        nodes_[to.id()].pred.push_back(from.id());
        return;
    }
    // diagnostic
    bool has_cycles() const { return topological_sort_().size() != nodes_.size(); }
    // iterator support
   private:
    template <typename TaskGraph_> struct iterator_type {
        using iterator_category = std::random_access_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = std::conditional_t<std::is_const_v<TaskGraph_>, const_reference, reference>;

        iterator_type(TaskGraph_* graph, int index) : graph_(graph), index_(index) { }

        value_type operator*() { return value_type(graph_, index_); }
        value_type operator[](difference_type n) const { return value_type(graph_, index_ + n); }
        iterator_type& operator++() {
            ++index_;
            return *this;
        }
        iterator_type& operator--() {
            --index_;
            return *this;
        }
        iterator_type& operator+=(difference_type n) {
            index_ += n;
            return *this;
        }
        iterator_type& operator-=(difference_type n) {
            index_ -= n;
            return *this;
        }
        friend iterator_type operator+(iterator_type it, difference_type n) { return it += n; }
        friend iterator_type operator-(iterator_type it, difference_type n) { return it -= n; }

        friend difference_type operator-(iterator_type lhs, iterator_type rhs) { return lhs.index_ - rhs.index_; }

        friend bool operator==(iterator_type lhs, iterator_type rhs) { return lhs.index_ == rhs.index_; }
        friend auto operator<=>(iterator_type lhs, iterator_type rhs) { return lhs.index_ <=> rhs.index_; }
       private:
        TaskGraph_* graph_;
        int index_;
    };
   public:
    using iterator = iterator_type<TaskGraph>;
    using const_iterator = iterator_type<const TaskGraph>;

    iterator begin() { return iterator(this, 0); }
    iterator end() { return iterator(this, nodes_.size()); }
    const_iterator begin() const { return const_iterator(this, 0); }
    const_iterator end() const { return const_iterator(this, nodes_.size()); }
   private:
    std::vector<node_type> nodes_ {};
};

inline void parallel_execute(TaskGraph& tg) {
    auto& executor = internals::threaded_executor::instance();
    return internals::task_graph::run(std::addressof(executor), tg);
}

}   // namespace fdapde

#endif   // __FDAPDE_EXECUTION_TASK_GRAPH_H__
