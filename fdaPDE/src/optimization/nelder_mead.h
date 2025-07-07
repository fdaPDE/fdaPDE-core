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

    vector_t optimum_; // Argmin of the optimum
    double value_;     // objective value at optimum
    int max_iter_;     // maximum number of iterations before forced stop
    int n_iter_ = 0;   // current iteration number
    double tol_;       // tolerance on error before forced stop

    std::normal_distribution<double> normal_dist_ {0.0, 1.0};
    std::mt19937 rng_;

    std::vector<vector_t> simplex_;       // Edges of the simplex
    std::vector<double> vertices_values_; // Value of each edge
    std::vector<int> vertices_rank_;      // Index into the simplex vector, sorted from best to worst
    
    double alpha_ = 0.0; // Reflexion coeff
    double beta_  = 0.0; // Expension coeff
    double gamma_ = 0.0; // Outer contraction coeff
    double delta_ = 0.0; // Inner contraction coeff

    // Wessing, S. Proper initialization is crucial for the Nelder–Mead simplex search.
    // Optim Lett 13, 847–856 (2019). https://doi.org/10.1007/s11590-018-1284-4
    void init_simplex_(const vector_t &x0) {
        const int dimension = x0.rows();
        double infnty_norm = x0.cwiseAbs().maxCoeff();
        double scale_factor = std::min(std::max(infnty_norm, 1.0), 10.0);
        double last_coeff = scale_factor * (1.0 - (double)std::sqrt(dimension+1))/(double)(dimension);

        // Initialises the vectors with cached values
        simplex_.clear();
        vertices_rank_.clear();
        vertices_values_.clear();
        vertices_rank_.resize(dimension+1, 0);
        vertices_values_.resize(dimension+1, std::numeric_limits<double>::max());
        simplex_.resize(dimension+1, x0);

        for(int i = 0; i < dimension; ++i) {
            simplex_[i][i] += scale_factor;
            vertices_rank_[i+1] = i+1;
            simplex_[dimension][i] += last_coeff;
        }

        simplex_[dimension][dimension-1] += last_coeff;
    }

public:

    // constructors
    NelderMead() = default;

    NelderMead(int max_iter, double tol, unsigned int seed = 0u):
        max_iter_(max_iter),
        tol_(tol),
        rng_(seed){
    }

    template <typename ObjectiveT, typename... Functor>
        requires(sizeof...(Functor) < 2) && ((requires(Functor f, double value) { f(value); }) && ...)
    vector_t optimize(ObjectiveT&& objective, const vector_t& x0, Functor&&... func) {
        fdapde_static_assert(
            std::is_same<decltype(std::declval<ObjectiveT>().operator()(vector_t())) FDAPDE_COMMA double>::value,
            INVALID_CALL_TO_OPTIMIZE__OBJECTIVE_FUNCTOR_NOT_ACCEPTING_VECTORTYPE);
            
        bool done = false;
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

        // Gao, F., Han, L. Implementing the Nelder-Mead simplex algorithm with adaptive parameters.
        // Comput Optim Appl 51, 259–277 (2012). https://doi.org/10.1007/s10589-010-9329-3
        alpha_ = 1.0;
        beta_ = 1 + 2.0/(double)dimension;
        gamma_ = 0.75- 1.0/(double)(2*dimension);
        delta_ = 1.0 - 1.0/(double)dimension;

        // Initialise the simplex given x0
        init_simplex_(x0);

        fdapde_assert(vertices_values_.size() == simplex_.size());
        fdapde_assert(vertices_rank_.size() == simplex_.size());
        fdapde_assert(x0.rows()+1 == simplex_.size());

        // Compute the vertices's values
        for(int i = 0; i < simplex_.size(); ++i)
            vertices_values_[i] = objective(simplex_[i]);

        while(!done && n_iter_ < max_iter_) {
            // Sort the vertices according to their objective value
            std::sort(vertices_rank_.begin(), vertices_rank_.end(), [&](int a, int b) {
                return vertices_values_[a] < vertices_values_[b];
            });

            // Centroid calculation
            vector_t centroid = zero; // centroid of the [dimension] best vertices
            vector_t random_vect = zero;
            for(int i = 0; i < dimension; ++i) {
                centroid += simplex_[vertices_rank_[i]];
                if(dimension >= 10) {
                    random_vect[i] = normal_dist_(rng_);
                }
            }
            centroid /= (double)(dimension);
            if(dimension >= 10) {
                random_vect /= random_vect.norm();
            }
            
            // Perturbation of the centroid to enhance performance for large dimensions
            // Optim Lett 13, 1011–1025 (2019). https://doi.org/10.1007/s11590-018-1306-2
            vector_t perturbed_centroid = centroid;
            if(dimension >= 10) {
                perturbed_centroid += 0.1 * random_vect * (simplex_[vertices_rank_[0]] - simplex_[vertices_rank_[dimension]]).norm();
            }
            vector_t xr = perturbed_centroid + alpha_*(perturbed_centroid - simplex_[vertices_rank_[dimension]]);

            // Cache the values used for the if statements
            require_shrink = false;
            const double xr_val = objective(xr);
            const double best_val = vertices_values_[vertices_rank_[0]];
            const double worst_val = vertices_values_[vertices_rank_[dimension]];
            const double second_worst_val = vertices_values_[vertices_rank_[dimension-1]];

            // Compute the new simplex
            if( best_val <= xr_val && xr_val < second_worst_val ) {
                // Reflexion
                simplex_[vertices_rank_[dimension]] = xr;
                vertices_values_[vertices_rank_[dimension]] = xr_val;
            } else if( xr_val < best_val ) {
                // Expansion
                vector_t xe = perturbed_centroid + beta_*(xr - perturbed_centroid);
                double xe_val = objective(xe);
                if(xe_val < xr_val) {
                    simplex_[vertices_rank_[dimension]] = xe;
                    vertices_values_[vertices_rank_[dimension]] = xe_val;
                } else {
                    simplex_[vertices_rank_[dimension]] = xr;
                    vertices_values_[vertices_rank_[dimension]] = xr_val;
                }
            } else if( second_worst_val <= xr_val < worst_val ) {
                // Outer Contraction
                vector_t xoc = centroid + gamma_ * (xr - centroid);
                double xoc_val = objective(xoc);
                if (xoc_val <= xr_val) {
                    simplex_[vertices_rank_[dimension]] = xoc;
                    vertices_values_[vertices_rank_[dimension]] = xoc_val;
                } else {
                    require_shrink = true;
                }
            } else {
                // Inner Contraction
                vector_t xic = centroid - gamma_ * (xr - centroid);
                double xic_val = objective(xic);
                if( xic_val < best_val) {
                    simplex_[vertices_rank_[dimension]] = xic;
                    vertices_values_[vertices_rank_[dimension]] = xic_val;
                } else {
                    require_shrink = true;
                }
            }

            // Shrink
            if(require_shrink) {
                const vector_t &best_vertex = simplex_[vertices_rank_[0]];
                for(int i = 1; i < dimension + 1; ++i) {
                    simplex_[vertices_rank_[i]] = best_vertex + delta_ * (simplex_[vertices_rank_[i]] - best_vertex);
                }
            }

            // Stoping criterion
            double value_mean = 0.0;
            for(double val: vertices_values_) {
                value_mean += val;
            }
            value_mean /= (double)vertices_values_.size();

            double std_dev = 0.0;
            for(double val: vertices_values_) {
                double x = val - value_mean;
                std_dev += x*x;
            }
            std_dev /= (double)(dimension);
            std_dev = std::sqrt(std_dev);

            if(std_dev <= tol_) {
                done = true;
            }

            ++n_iter_;
        }
        
        optimum_ = simplex_[0];
        value_ = objective(optimum_);
        return optimum_;
    }
    
    vector_t optimum() const { return optimum_; }
    double value() const { return value_; }
    int n_iter() const { return n_iter_; }
};

}   // namespace fdapde

#endif   // __FDAPDE_NELDER_MEAD_H__
