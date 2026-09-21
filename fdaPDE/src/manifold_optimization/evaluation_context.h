/// @details This file is part of fdaPDE, a C++ library for physics-informed
/// @details spatial and functional data analysis.
//
/// @details This program is free software: you can redistribute it and/or modify
/// @details it under the terms of the GNU General Public License as published by
/// @details the Free Software Foundation, either version 3 of the License, or
/// @details (at your option) any later version.
//
/// @details This program is distributed in the hope that it will be useful,
/// @details but WITHOUT ANY WARRANTY; without even the implied warranty of
/// @details MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
/// @details GNU General Public License for more details.
//
/// @details You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __FDAPDE_MANIFOLD_EVALUATION_CONTEXT_H__
#define __FDAPDE_MANIFOLD_EVALUATION_CONTEXT_H__

#include "header_check.h"

namespace fdapde {
namespace manifold {

/// @brief provides an empty default workspace for problems without retained intermediates
struct EmptyWorkspace { };

/// @brief keeps independent current and trial evaluations, each bound to one candidate generation
/// @details each Evaluation generation belongs to exactly one manifold point
/// @details the context deliberately stores no point key: reset a slot before binding it
/// to another point, promote an accepted trial, and reset a rejected trial
template <std::movable Gradient, std::default_initializable Workspace = EmptyWorkspace>
    requires std::movable<Workspace>
class EvaluationContext {
   public:
    /// @brief stores the cost, gradient and problem workspace for one immutable candidate
    class Evaluation {
        friend class EvaluationContext;

        std::size_t generation_;
        std::optional<double> cost_;
        std::optional<Gradient> gradient_;
        Workspace workspace_;

        /// @brief binds an initially empty evaluation to a candidate generation
        explicit Evaluation(std::size_t generation) : generation_(generation) { }
        /// @brief clears all quantities before binding a slot to another candidate
        void reset(std::size_t generation) {
            generation_ = generation;
            cost_.reset();
            gradient_.reset();
            workspace_ = Workspace {};
        }
       public:
        /// @brief returns the candidate generation associated with this cache slot
        std::size_t generation() const { return generation_; }
        /// @brief borrows the objective cached for this candidate generation
        const std::optional<double>& cost() const& { return cost_; }
        /// @brief prevents borrowing storage from a temporary object
        const std::optional<double>& cost() && = delete;
        /// @brief prevents borrowing storage from a temporary object
        const std::optional<double>& cost() const&& = delete;
        /// @brief borrows the tangent gradient cached for this candidate generation
        const std::optional<Gradient>& gradient() const& { return gradient_; }
        /// @brief prevents borrowing storage from a temporary object
        const std::optional<Gradient>& gradient() && = delete;
        /// @brief prevents borrowing storage from a temporary object
        const std::optional<Gradient>& gradient() const&& = delete;
        /// @brief borrows intermediates belonging exclusively to this candidate generation
        Workspace& workspace() & { return workspace_; }
        /// @brief borrows intermediates bound to this candidate generation
        const Workspace& workspace() const& { return workspace_; }
        /// @brief prevents borrowing storage from a temporary object
        Workspace& workspace() && = delete;
        /// @brief prevents borrowing storage from a temporary object
        const Workspace& workspace() const&& = delete;

        /// @brief stores the evaluated objective for the current slot generation
        void set_cost(double cost) noexcept { cost_ = cost; }
        /// @brief stores the evaluated tangent gradient for the current slot generation
        void set_gradient(Gradient gradient) { gradient_ = std::move(gradient); }
    };
   private:
    std::size_t next_generation_ = 2;
    Evaluation current_ {0};
    Evaluation trial_ {1};

    /// @brief allocates a new generation identifier without wrapping the counter
    std::size_t fresh_generation_() {
        fdapde_assert(
          next_generation_ != std::numeric_limits<std::size_t>::max(), std::overflow_error,
          "Evaluation-context generation range exhausted.");
        return next_generation_++;
    }
   public:
    /// @brief borrows the current candidate evaluation
    Evaluation& current() & { return current_; }
    /// @brief borrows the current candidate evaluation
    const Evaluation& current() const& { return current_; }
    /// @brief prevents borrowing storage from a temporary object
    Evaluation& current() && = delete;
    /// @brief prevents borrowing storage from a temporary object
    const Evaluation& current() const&& = delete;
    /// @brief borrows the trial candidate evaluation
    Evaluation& trial() & { return trial_; }
    /// @brief borrows the trial candidate evaluation
    const Evaluation& trial() const& { return trial_; }
    /// @brief prevents borrowing storage from a temporary object
    Evaluation& trial() && = delete;
    /// @brief prevents borrowing storage from a temporary object
    const Evaluation& trial() const&& = delete;

    /// @brief invalidates the current objective, gradient and relative frames
    void reset_current() { current_.reset(fresh_generation_()); }
    /// @brief invalidates the trial objective, gradient and relative frames
    void reset_trial() { trial_.reset(fresh_generation_()); }
    /// @brief moves the accepted trial quantities to current and clears the next trial slot
    void promote_trial() {
        const std::size_t generation = fresh_generation_();
        current_ = std::move(trial_);
        trial_.reset(generation);
    }
};

}   // namespace manifold
}   // namespace fdapde

#endif   // __FDAPDE_MANIFOLD_EVALUATION_CONTEXT_H__
