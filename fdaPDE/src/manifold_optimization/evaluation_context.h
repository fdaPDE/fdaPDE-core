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

#ifndef __FDAPDE_MANIFOLD_EVALUATION_CONTEXT_H__
#define __FDAPDE_MANIFOLD_EVALUATION_CONTEXT_H__

#include "header_check.h"

namespace fdapde {
namespace manifold {

struct EmptyWorkspace { };

// each Evaluation generation belongs to exactly one manifold point
// the context deliberately stores no point key: reset a slot before binding it
// to another point, promote an accepted trial, and reset a rejected trial
template <std::movable Gradient, std::default_initializable Workspace = EmptyWorkspace>
    requires std::movable<Workspace>
class EvaluationContext {
   public:
    class Evaluation {
        friend class EvaluationContext;

        std::size_t generation_;
        std::optional<double> cost_;
        std::optional<Gradient> gradient_;
        Workspace workspace_;

        explicit Evaluation(std::size_t generation) : generation_(generation) { }
        void reset(std::size_t generation) {
            generation_ = generation;
            cost_.reset();
            gradient_.reset();
            workspace_ = Workspace {};
        }
       public:
        std::size_t generation() const { return generation_; }
        const std::optional<double>& cost() const& { return cost_; }
        const std::optional<double>& cost() && = delete;
        const std::optional<double>& cost() const&& = delete;
        const std::optional<Gradient>& gradient() const& { return gradient_; }
        const std::optional<Gradient>& gradient() && = delete;
        const std::optional<Gradient>& gradient() const&& = delete;
        Workspace& workspace() & { return workspace_; }
        const Workspace& workspace() const& { return workspace_; }
        Workspace& workspace() && = delete;
        const Workspace& workspace() const&& = delete;

        void set_cost(double cost) noexcept { cost_ = cost; }
        void set_gradient(Gradient gradient) { gradient_ = std::move(gradient); }
    };
   private:
    std::size_t next_generation_ = 2;
    Evaluation current_ {0};
    Evaluation trial_ {1};

    std::size_t fresh_generation_() {
        if (next_generation_ == std::numeric_limits<std::size_t>::max()) {
            throw std::overflow_error("Evaluation-context generation range exhausted.");
        }
        return next_generation_++;
    }
   public:
    Evaluation& current() & { return current_; }
    const Evaluation& current() const& { return current_; }
    Evaluation& current() && = delete;
    const Evaluation& current() const&& = delete;
    Evaluation& trial() & { return trial_; }
    const Evaluation& trial() const& { return trial_; }
    Evaluation& trial() && = delete;
    const Evaluation& trial() const&& = delete;

    void reset_current() { current_.reset(fresh_generation_()); }
    void reset_trial() { trial_.reset(fresh_generation_()); }
    void promote_trial() {
        const std::size_t generation = fresh_generation_();
        current_ = std::move(trial_);
        trial_.reset(generation);
    }
};

}   // namespace manifold
}   // namespace fdapde

#endif   // __FDAPDE_MANIFOLD_EVALUATION_CONTEXT_H__
