// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef __FDAPDE_GOLDEN_SECTION_SEARCH_H__
#define __FDAPDE_GOLDEN_SECTION_SEARCH_H__

#include <stdexcept>

#include "header_check.h"

namespace fdapde {

/** @brief bracket a scalar minimum from one starting point and refine it without derivatives */
class GoldenSectionSearch {
    using vector_t = Eigen::Matrix<double, 1, 1>;
    int max_evaluations_, n_iter_ = 0, direction_ = 0;
    double tolerance_, score_tolerance_, value_ = INFINITY;
    vector_t optimum_;
    std::map<double, double> values_;
    std::string status_ = "not_started";
   public:
    static constexpr bool gradient_free = true;
    static constexpr int static_input_size = 1;
    vector_t x_curr;
    double obj_curr = INFINITY;
    /** @brief set the evaluation budget, coordinate resolution and relative tail plateau tolerance */
    GoldenSectionSearch(int max_evaluations = 20, double tolerance = .05, double score_tolerance = 1e-4) :
        max_evaluations_(max_evaluations), tolerance_(tolerance), score_tolerance_(score_tolerance) {
        fdapde_strong_assert(
          max_evaluations >= 3, std::invalid_argument, "scalar search requires at least three evaluations");
        fdapde_strong_assert(
          std::isfinite(tolerance) && tolerance > 0 && std::isfinite(score_tolerance) && score_tolerance >= 0,
          std::invalid_argument,
          "scalar search tolerances must be finite and nonnegative, with positive coordinate tolerance");
    }
    /** @brief search locally without bounds and select a minimizer or the nearest equivalent plateau point */
    template <typename Objective>
    const vector_t& optimize(Objective& objective, const vector_t& initial, double step = 1.) {
        fdapde_strong_assert(
          initial.allFinite() && std::isfinite(step) && step > 0, std::invalid_argument,
          "scalar search requires a finite initial point and positive finite step");
        values_.clear();
        n_iter_ = 0;
        direction_ = 0;
        value_ = INFINITY;
        optimum_ = initial;
        status_ = "evaluation_limit";
        auto evaluate = [&](double x) {
            if (!std::isfinite(x)) return double(INFINITY);
            if (auto found = values_.find(x); found != values_.end()) return found->second;
            if (values_.size() >= std::size_t(max_evaluations_)) return double(INFINITY);
            x_curr[0] = x;
            obj_curr = objective(x_curr);
            if (!std::isfinite(obj_curr)) obj_curr = INFINITY;
            values_.emplace(x, obj_curr);
            if (obj_curr < value_) {
                optimum_ = x_curr;
                value_ = obj_curr;
            }
            return obj_curr;
        };
        auto plateau = [&] {
            const double threshold =
              value_ + score_tolerance_ * std::max(std::abs(value_), std::numeric_limits<double>::min());
            for (const auto& [x, score] : values_)
                if (score <= threshold && std::abs(x - initial[0]) < std::abs(optimum_[0] - initial[0]))
                    optimum_[0] = x;
            value_ = values_.at(optimum_[0]);
            status_ = "score_plateau";
        };
        constexpr double ratio = .6180339887498948482;
        double a = initial[0] - step, b = initial[0], c = initial[0] + step;
        double fb = evaluate(b), fa = evaluate(a), fc = evaluate(c);
        if (!std::isfinite(value_)) {
            status_ = "no_finite_value";
            return optimum_;
        }
        if (fa == fb && fb == fc) {
            plateau();
            return optimum_;
        }
        if (!(fb <= fa && fb <= fc)) {
            const double direction = fa < fc ? -1. : 1.;
            direction_ = direction < 0 ? -1 : 1;
            a = b;
            b += direction * step;
            fb = direction < 0 ? fa : fc;
            int flat_steps = 0;
            // expand only along the improving tail, retaining every completed objective evaluation
            while (values_.size() < std::size_t(max_evaluations_)) {
                step /= ratio;
                c = b + direction * step;
                if (!std::isfinite(c)) {
                    status_ = "numerical_limit";
                    return optimum_;
                }
                fc = evaluate(c);
                ++n_iter_;
                const double scale = std::max({std::abs(fb), std::abs(fc), std::numeric_limits<double>::min()});
                flat_steps = std::isfinite(fb) && std::isfinite(fc) && std::abs(fb - fc) <= score_tolerance_ * scale ?
                               flat_steps + 1 :
                               0;
                if (flat_steps >= 2) {
                    plateau();
                    return optimum_;
                }
                if (fc >= fb && flat_steps == 0) break;
                a = b;
                b = c;
                fb = fc;
                if (flat_steps) step = ratio;
            }
            if (c == b || fc < fb || values_.size() >= std::size_t(max_evaluations_)) return optimum_;
        }
        double left = std::min(a, c), right = std::max(a, c);
        if (right - left <= tolerance_) {
            status_ = "interval_tolerance";
            return optimum_;
        }
        double x1 = right - ratio * (right - left), x2 = left + ratio * (right - left);
        double f1 = evaluate(x1), f2 = evaluate(x2);
        // each contraction needs one new evaluation; the other interior point is reused
        while (right - left > tolerance_ && values_.size() < std::size_t(max_evaluations_)) {
            const double width = right - left;
            if (f1 <= f2) {
                right = x2;
                x2 = x1;
                f2 = f1;
                x1 = right - ratio * (right - left);
                f1 = evaluate(x1);
            } else {
                left = x1;
                x1 = x2;
                f1 = f2;
                x2 = left + ratio * (right - left);
                f2 = evaluate(x2);
            }
            if (right - left >= width) {
                status_ = "numerical_limit";
                return optimum_;
            }
            ++n_iter_;
        }
        if (right - left <= tolerance_) status_ = "interval_tolerance";
        return optimum_;
    }
    /** @brief return the selected coordinate, preferring proximity to the initial point on an equivalent plateau */
    const vector_t& optimum() const { return optimum_; }
    /** @brief return the selected objective value, or infinity when all evaluations failed */
    double value() const { return value_; }
    /** @brief return the number of bracketing and contraction steps */
    int n_iter() const { return n_iter_; }
    /** @brief return the cached coordinate and objective pairs for this run */
    const std::map<double, double>& values() const { return values_; }
    /** @brief distinguish interval resolution, a flat tail and an exhausted evaluation budget */
    const std::string& status() const { return status_; }
    /** @brief return the bracketing tail direction, or zero when the initial probes already bracketed a minimum */
    int direction() const { return direction_; }
};

}   // namespace fdapde
#endif
