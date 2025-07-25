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

#ifndef __FDAPDE_NELDER_MEAD_H__
#define __FDAPDE_NELDER_MEAD_H__

#include "header_check.h"

namespace fdapde {

template <int N> class NelderMead {
   private:
    using vector_t = std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, 1>, Eigen::Matrix<double, N, 1>>;
    using matrix_t =
      std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, Dynamic>, Eigen::Matrix<double, N, N>>;

    vector_t optimum_;
    double value_;     // objective value at optimum
    int max_iter_;     // maximum number of iterations before forced stop
    double tol_;       // tolerance on error before forced stop
    int n_iter_ = 0;   // current iteration number
    int seed_;         // employed for random centroid perturbation
    double alpha_;     // reflexion coeff
    double beta_;      // expension coeff
    double gamma_;     // outer contraction coeff
    double delta_;     // inner contraction coeff

    Eigen::Matrix<double, Dynamic, Dynamic> simplex_;   // simplex vertices
    std::vector<double> vertices_values_;               // objective values at simplex vertices
    std::vector<int> vertices_rank_;                    // simplex vertices sorted from lower to higher objective value
   public:
    vector_t x_curr;
    double obj_curr;
    static constexpr bool gradient_free = true;
    static constexpr int static_input_size = N;
    // constructors
    NelderMead() = default;
    NelderMead(int max_iter, double tol, int seed = fdapde::random_seed) :
        max_iter_(max_iter), tol_(tol), seed_(seed) { }

    template <typename ObjectiveT, typename... Callbacks>
    vector_t optimize(ObjectiveT&& objective, const vector_t& x0, Callbacks&&... callbacks) {
        fdapde_static_assert(
          std::is_same<decltype(std::declval<ObjectiveT>().operator()(vector_t())) FDAPDE_COMMA double>::value,
          INVALID_CALL_TO_OPTIMIZE__OBJECTIVE_FUNCTOR_NOT_ACCEPTING_VECTORTYPE);
        fdapde_assert(x0.rows() > 0);
        std::tuple<Callbacks...> callbacks_ {callbacks...};
        double dims = x0.rows();
        bool stop = false;
        bool shrink = false;
        n_iter_ = 0;
	vector_t centroid;
	// adaptive parameters initialization
        // Gao, F., Han, L. Implementing the Nelder-Mead simplex algorithm with adaptive parameters. Comput Optim Appl
        // 51, 259–277 (2012). https://doi.org/10.1007/s10589-010-9329-3
        alpha_ = 1.0;
        beta_  = 1.0 + 2. / dims;
        gamma_ = 0.75 - 1. / (2 * dims);
        delta_ = 1.0 - 1. / dims;
        // simplex initialization
        // Wessing, S. Proper initialization is crucial for the Nelder–Mead simplex search. Optim Lett 13, 847–856
        // (2019). https://doi.org/10.1007/s11590-018-1284-4
        double c_h = std::min(std::max(x0.template lpNorm<Eigen::Infinity>(), 1.0), 10.0);
	simplex_ = x0.replicate(1, dims + 1);
        simplex_.block(0, 0, dims, dims) += vector_t::Constant(dims, c_h).asDiagonal();
        simplex_.col(dims).array() += c_h * (1.0 - std::sqrt(dims + 1)) / dims;
        for (int i = 0; i < dims + 1; ++i) {
            x_curr = simplex_.col(i);
            obj_curr = objective(simplex_.col(i));
            stop |= internals::exec_eval_hooks(*this, objective, callbacks_);
            vertices_values_.push_back(obj_curr);
        }
        for (int i = 0; i < dims + 1; ++i) { vertices_rank_.push_back(i); }
        // sort vertices according to their objective value
        std::sort(vertices_rank_.begin(), vertices_rank_.end(), [&](int a, int b) {
            return vertices_values_[a] < vertices_values_[b];
        });
		
        while (n_iter_ < max_iter_ && !stop) {
            // centroid calculation (using the dims best vertices)
            centroid.setZero();
            for (int i = 0; i < dims; ++i) { centroid += simplex_.col(vertices_rank_[i]); }
            centroid /= dims;
            // Fajfar, I., Bűrmen, Á. & Puhan, J. The Nelder–Mead simplex algorithm with perturbed centroid for
            // high-dimensional function optimization. Optim Lett 13, 1011–1025 (2019).
            // https://doi.org/10.1007/s11590-018-1306-2
	    vector_t perturbed_centroid = centroid;
            if (dims >= 10) {
                vector_t random_delta = internals::gaussian_matrix(1, dims, 1.0, seed_);
                random_delta /= random_delta.norm();
                perturbed_centroid +=
                  0.1 * random_delta * (simplex_.col(vertices_rank_[0]) - simplex_.col(vertices_rank_[dims])).norm();
            }
            vector_t xr = perturbed_centroid + alpha_ * (perturbed_centroid - simplex_.col(vertices_rank_[dims]));

            // simplex update
            shrink = false;
            double xr_val = objective(xr);   // (perturbed) centroid objective value
            double fb_val = vertices_values_[vertices_rank_[0]];
            double fw_val = vertices_values_[vertices_rank_[dims]];
            double sw_val = vertices_values_[vertices_rank_[dims - 1]];
	    x_curr = xr;
	    obj_curr = xr_val;
            stop |= internals::exec_eval_hooks(*this, objective, callbacks_);
	    
            if (fb_val <= xr_val && xr_val < sw_val) {
                // reflection
                simplex_.col(vertices_rank_[dims]) = xr;
                vertices_values_[vertices_rank_[dims]] = xr_val;
            } else if (xr_val < fb_val) {
                // expansion
                vector_t xe = perturbed_centroid + beta_ * (xr - perturbed_centroid);
                double xe_val = objective(xe);
                x_curr = xe;
                obj_curr = xe_val;
                stop |= internals::exec_eval_hooks(*this, objective, callbacks_);
                simplex_.col(vertices_rank_[dims]) = xe_val < xr_val ? xe : xr;
                vertices_values_[vertices_rank_[dims]] = xe_val < xr_val ? xe_val : xr_val;
            } else if (sw_val <= xr_val && xr_val < fw_val) {
                // outer contraction
                vector_t xoc = centroid + gamma_ * (xr - centroid);
                double xoc_val = objective(xoc);
                x_curr = xoc;
                obj_curr = xoc_val;
                stop |= internals::exec_eval_hooks(*this, objective, callbacks_);
                if (xoc_val <= xr_val) {
                    simplex_.col(vertices_rank_[dims]) = xoc;
                    vertices_values_[vertices_rank_[dims]] = xoc_val;
                } else {
                    shrink = true;
                }
            } else {
                // inner contraction
                vector_t xic = centroid - gamma_ * (xr - centroid);
                double xic_val = objective(xic);
                x_curr = xic;
                obj_curr = xic_val;
                stop |= internals::exec_eval_hooks(*this, objective, callbacks_);
                if (xic_val < fb_val) {
                    simplex_.col(vertices_rank_[dims]) = xic;
                    vertices_values_[vertices_rank_[dims]] = xic_val;
                } else {
                    shrink = true;
                }
            }
            // shrink
            if (shrink) {
                auto best_vertex = simplex_.col(vertices_rank_[0]);
                for (int i = 1; i < dims + 1; ++i) {
                    simplex_.col(vertices_rank_[i]) =
                      best_vertex + delta_ * (simplex_.col(vertices_rank_[i]) - best_vertex);
                }
            }

            // prepare next iteration
            double mean = std::accumulate(vertices_values_.begin(), vertices_values_.end(), 0.0) / (dims + 1);
            double std_dev =
              std::accumulate(vertices_values_.begin(), vertices_values_.end(), 0.0, [&](double a, double b) {
                  return a + (std::pow(b - mean, 2));
              });
            std_dev = std::sqrt(std_dev / dims);
            stop = std_dev < tol_;
            // sort vertices according to their objective value
            std::sort(vertices_rank_.begin(), vertices_rank_.end(), [&](int a, int b) {
                return vertices_values_[a] < vertices_values_[b];
            });
	    stop |= internals::exec_stop_if(*this, objective);
            n_iter_++;
        }
        optimum_ = simplex_.col(vertices_rank_[0]);
        value_ = objective(optimum_);
        return optimum_;
    }
    // getters
    const vector_t& optimum() const { return optimum_; }
    double value() const { return value_; }
    int n_iter() const { return n_iter_; }
};

}   // namespace fdapde

#endif   // __FDAPDE_NELDER_MEAD_H__
