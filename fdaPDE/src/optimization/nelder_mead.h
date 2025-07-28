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

// Implementation of Nelder-Mead algorithm for gradient-free unconstrained nonlinear optimization
template <int N> class NelderMead {
private:
    using vector_t = std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, 1>, Eigen::Matrix<double, N, 1>>;
    using matrix_t = std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, Dynamic>, Eigen::Matrix<double, N, N>>;

    vector_t optimum_;   // Argmin of the optimum
    double value_;       // objective value at optimum
    int max_iter_ = 500; // maximum number of iterations before forced stop
    int n_iter_ = 0;     // current iteration number
    double tol_ = 1e-5;  // tolerance on error before forced stop

    matrix_t simplex_;               // Edges of the simplex
    vector_t vertices_values_;       // Value of each edge
    std::vector<int> vertices_rank_; // Index into the simplex vector, sorted from best to worst
    
    double alpha_ = 0.0; // Reflexion coeff
    double beta_  = 0.0; // Expension coeff
    double gamma_ = 0.0; // Outer contraction coeff
    double delta_ = 0.0; // Inner contraction coeff

public:
    static constexpr bool gradient_free = true;
    static constexpr int static_input_size = N;
    vector_t x_curr;
    double obj_curr;

public:
    // constructors
    NelderMead() = default;

    NelderMead(int max_iter, double tol): max_iter_(max_iter), tol_(tol)
    {}

    template <typename ObjectiveT, typename... Callbacks>
    vector_t optimize(ObjectiveT&& objective, const vector_t& x0, Callbacks&&... callbacks) {
        fdapde_static_assert(
          std::is_same<decltype(std::declval<ObjectiveT>().operator()(vector_t())) FDAPDE_COMMA double>::value,
          INVALID_CALL_TO_OPTIMIZE__OBJECTIVE_FUNCTOR_NOT_ACCEPTING_VECTORTYPE);
        
        fdapde_assert(x0.rows() > 0);

        std::tuple<Callbacks...> callbacks_ {callbacks...};
        
        bool stop = false;
        bool require_shrink = false;
        n_iter_ = 0;
        vector_t zero;
        const int dimension = x0.rows();
        fdapde_assert(dimension >= 1);
        if constexpr (N == Dynamic) {
	        zero = vector_t::Zero(x0.rows());
        } else {
	        zero = vector_t::Zero();
        }
        vector_t centroid = zero; // centroid
        vector_t xr = zero;       // initial next guess
        vector_t tmp = zero;      // temporary vec for inner, outer contractions, etc.

        // Gao, F., Han, L. Implementing the Nelder-Mead simplex algorithm with adaptive parameters.
        // Comput Optim Appl 51, 259–277 (2012). https://doi.org/10.1007/s10589-010-9329-3
        alpha_ = 1.0;
        beta_  = 1 + 2.0/(double)dimension;
        gamma_ = 0.75 - 1.0/(double)(2*dimension);
        delta_ = 1.0 - 1.0/(double)dimension;

        // Initialise the simplex given x0
        // Wessing, S. Proper initialization is crucial for the Nelder–Mead simplex search.
        // Optim Lett 13, 847–856 (2019). https://doi.org/10.1007/s11590-018-1284-4
        {
            double infnty_norm = x0.cwiseAbs().maxCoeff();
            double scale_factor = std::min(std::max(infnty_norm, 1.0), 10.0);
            double last_coeff = scale_factor * (1.0 - (double)std::sqrt(dimension+1))/(double)(dimension);

            // Initialises the vectors with cached values
            simplex_ = x0.rowwise().replicate(dimension+1);
            vertices_values_ = vector_t::Constant(dimension+1, 1, std::numeric_limits<double>::max());
            vertices_rank_.clear();
            vertices_rank_.resize(dimension+1, 0);

            for(int i = 0; i < dimension; ++i) {
                simplex_(i, i) += scale_factor;
                vertices_rank_[i+1] = i+1;
                simplex_(i, dimension) += last_coeff;
            }
        }

        fdapde_assert(vertices_values_.size() == simplex_.cols());
        fdapde_assert(vertices_rank_.size() == simplex_.cols());
        fdapde_assert(x0.rows()+1 == simplex_.cols());

        // Compute the vertices's values
        for(int i = 0; i < simplex_.cols(); ++i)
            vertices_values_[i] = objective(simplex_.col(i));
        
        // Sort the vertices according to their objective value
        std::sort(vertices_rank_.begin(), vertices_rank_.end(), [&](int a, int b) {
            return vertices_values_[a] < vertices_values_[b];
        });

        while(!stop && n_iter_ < max_iter_) {
            // Centroid calculation
            centroid.setZero(); // centroid of the [dimension] best vertices
            for(int i = 0; i < dimension; ++i) {
                centroid += simplex_.col(vertices_rank_[i]);
            }
            centroid /= (double)(dimension);
            
            xr = centroid + alpha_*(centroid - simplex_.col(vertices_rank_[dimension]));

            // Cache the values used for the if statements
            require_shrink = false;
            const double xr_val = objective(xr);
            const double best_val = vertices_values_[vertices_rank_[0]];
            const double worst_val = vertices_values_[vertices_rank_[dimension]];
            const double second_worst_val = vertices_values_[vertices_rank_[dimension-1]];

            stop |= internals::exec_eval_hooks(*this, objective, callbacks_);

            // Compute the new simplex
            if( best_val <= xr_val && xr_val < second_worst_val ) {
                // Reflexion
                simplex_.col(vertices_rank_[dimension]) = xr;
                vertices_values_[vertices_rank_[dimension]] = xr_val;
            } else if( xr_val < best_val ) {
                // Expansion
                tmp = centroid + beta_*(xr - centroid);
                double xe_val = objective(tmp);
                stop |= internals::exec_eval_hooks(*this, objective, callbacks_);
                if(xe_val < xr_val) {
                    simplex_.col(vertices_rank_[dimension]) = tmp;
                    vertices_values_[vertices_rank_[dimension]] = xe_val;
                } else {
                    simplex_.col(vertices_rank_[dimension]) = xr;
                    vertices_values_[vertices_rank_[dimension]] = xr_val;
                }
            } else if( second_worst_val <= xr_val < worst_val ) {
                // Outer Contraction
                tmp = centroid + gamma_ * (xr - centroid);
                double xoc_val = objective(tmp);
                stop |= internals::exec_eval_hooks(*this, objective, callbacks_);
                if (xoc_val <= xr_val) {
                    simplex_.col(vertices_rank_[dimension]) = tmp;
                    vertices_values_[vertices_rank_[dimension]] = xoc_val;
                } else {
                    require_shrink = true;
                }
            } else {
                // Inner Contraction
                tmp = centroid - gamma_ * (xr - centroid);
                double xic_val = objective(tmp);
                stop |= internals::exec_eval_hooks(*this, objective, callbacks_);
                if( xic_val < best_val) {
                    simplex_.col(vertices_rank_[dimension]) = tmp;
                    vertices_values_[vertices_rank_[dimension]] = xic_val;
                } else {
                    require_shrink = true;
                }
            }

            // Shrink
            if(require_shrink) {
                const auto &best_vertex = simplex_.col(vertices_rank_[0]);
                for(int i = 1; i < dimension + 1; ++i) {
                    simplex_.col(vertices_rank_[i]) = best_vertex + delta_ * (simplex_.col(vertices_rank_[i]) - best_vertex);
                    vertices_values_[vertices_rank_[i]] = objective(simplex_.col(vertices_rank_[i]));
                }
            }

            // Sort the vertices according to their objective value
            std::sort(vertices_rank_.begin(), vertices_rank_.end(), [&](int a, int b) {
                return vertices_values_[a] < vertices_values_[b];
            });

            // Stoping criterion
	        stop |= internals::exec_stop_if(*this, objective);

            // I did not use std::accumulate because I beleive that it can become
            // very slow depending on the container & formula
            const double value_mean = vertices_values_.mean();
            double value_var = 0.0;
            for(double val: vertices_values_) {
                const double x = val - value_mean;
                value_var += x*x;
            }
            value_var /= (double)vertices_values_.rows();

            stop |= std::sqrt(value_var) <= tol_;
            ++n_iter_;
        }
        
        optimum_ = simplex_.col(vertices_rank_[0]);
        value_ = objective(optimum_);
        return optimum_;
    }
    
    vector_t optimum() const { return optimum_; }
    double value() const { return value_; }
    int n_iter() const { return n_iter_; }
};

}   // namespace fdapde

#endif   // __FDAPDE_NELDER_MEAD_H__
