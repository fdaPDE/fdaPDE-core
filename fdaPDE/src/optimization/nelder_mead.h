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

/**
 * @brief Implementation of a perturbed version of the Nelder–Mead algorithm as analysed in [Fajfar, I., Bűrmen, Á. & Puhan, J. The Nelder–Mead simplex algorithm with perturbed centroid for high-dimensional function optimization. Optim Lett 13, 1011–1025 (2019)]
 * 
 * @tparam N dimension of the problem
 */
template <int N> class NelderMead {
private:
    using vector_t = std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, 1>, Eigen::Matrix<double, N, 1>>;
    using matrix_t = std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, Dynamic>, Eigen::Matrix<double, N, N>>;

    vector_t optimum_; // Argmin of the optimum
    double value_;     // objective value at optimum
    int max_iter_;     // maximum number of iterations before forced stop
    int n_iter_ = 0;   // current iteration number
    double tol_;       // tolerance on error before forced stop

    std::vector<vector_t> simplex_;       // Edges of the simplex
    std::vector<double> vertices_values_; // Value of each edge
    std::vector<int> vertices_rank_;      // Index into the simplex vector, sorted from best to worst
    
    double alpha_ = 1.0; // Reflexion coeff
    double beta_  = 2.0; // Expension coeff
    double gamma_ = 0.5; // Outer contraction coeff
    double delta_ = 0.5; // Inner contraction coeff

    void init_simplex_(const vector_t &x0) {
        // Wessing, S. Proper initialization is crucial for the Nelder–Mead simplex search.
        // Optim Lett 13, 847–856 (2019). https://doi.org/10.1007/s11590-018-1284-4


        const int dimension = x0.rows();

        // Initialises the vectors with cached values
        vertices_rank_.resize(dimension+1, 0);
        vertices_values_.resize(dimension+1, std::numeric_limits<double>::max());

        for(int i = 0; i < dimension+1; ++i) {
            // vertices_rank_[i] = ;
        }
    }

public:

    // constructors
    NelderMead() = default;

    /**
     * @brief Construct a new NelderMead instance
     * 
     * @param max_iter maximum number of iterations performed
     * @param memory_size size of the memory used to approximate the inverse hessian matrix in the LBFGH alg.
     * @param tol tolerance used as the stoping criterion (|\nabla f_k|_2 < tol)
     */
    NelderMead(int max_iter, double tol)
        requires(sizeof...(Args) != 0):
        max_iter_(max_iter),
        memory_size_(memory_size),
        tol_(tol){
        fdapde_assert(memory_size_ >= 0);
    }

    /**
     * @brief Minimizes a function using the L-BFGS method
     * 
     * @tparam ObjectiveT Type of the function to be minimized
     * @param objective function to be minimized
     * @param x0 initial point
     * @param func
     */
    template <typename ObjectiveT, typename... Functor>
        requires(sizeof...(Functor) < 2) && ((requires(Functor f, double value) { f(value); }) && ...)
    vector_t optimize(ObjectiveT&& objective, const vector_t& x0, Functor&&... func) {
        fdapde_static_assert(
            std::is_same<decltype(std::declval<ObjectiveT>().operator()(vector_t())) FDAPDE_COMMA double>::value,
            INVALID_CALL_TO_OPTIMIZE__OBJECTIVE_FUNCTOR_NOT_ACCEPTING_VECTORTYPE);
            
        bool done = false;
        vector_t zero;
        const int dimension = x0.rows();
        fdapde_assert(dimension >= 1);
        if constexpr (N == Dynamic) {   // inv_hessian approximated with identity matrix
	        zero = vector_t::Zero(x0.rows());
        } else {
	        zero = vector_t::Zero();
        }

        // Initialise the simplex given x0
        init_simplex_(x0);

        // Compute the vertices's values
        for(int i = 0; i < simplex_.size(); ++i)
            vertices_values[i] = objective(simplex_[i]);

        while(!done) {
            // Sort the vertices according to their objective value
            std::sort(vertices_rank_.begin(), vertices_rank_.end(), [&](int a, int b) {
                return vertices_values_[a] < vertices_values_[b];
            });

            // Centroid calculation

            // Fajfar, I., Bűrmen, Á. & Puhan, J. The Nelder–Mead simplex algorithm with perturbed centroid for high-dimensional function optimization.
            // Optim Lett 13, 1011–1025 (2019). https://doi.org/10.1007/s11590-018-1306-2
            vector_t centroid = zero; // centroid of the [dimension] best vertices
            for(int i = 0; i < dimension; ++i)
                centroid += simplex_[vertices_rank_[i]];
            centroid /= (double)(dimension);

            vector_t xr = centroid + alpha_*(centroid - simplex_[vertices_rank_[dimension-1]]);

            // Cache the values used for the if statements
            const double xr_value = objective(xr);
            const double best_value = vertices_values_[vertices_rank_[0]];
            const double worst_value = vertices_values_[vertices_rank_[dimension]];
            const double second_worst_value = vertices_values_[vertices_rank_[dimension-1]];

            // Compute the new simplex
            if( best_value <= xr_value && xr_value < second_worst_value ) {
                // Reflexion
                simplex_[vertices_rank_[dimension]] = xr;
                vertices_values_[vertices_rank_[dimension]] = xr_value;
            } else if( xr_value < best_value ) {
                // Expansion
                vector_t xe = centroid + beta_*(xr - centroid);
                double xe_val = objective(xe);
                if(xe_vel < xr_value) {
                    simplex_[vertices_rank_[dimension]] = xe;
                    vertices_values_[vertices_rank_[dimension]] = xe_value;
                } else {
                    simplex_[vertices_rank_[dimension]] = xr;
                    vertices_values_[vertices_rank_[dimension]] = xr_value;
                }
            } else if( second_worst_value <= xr_value < worst_value ) {
                // Outer Contraction
                vector_t xoc = c + gamma_ * (xr - c);
                double xoc_val = objective(xoc);
                if (xoc_val ≤ xr_val) {
                    simplex_[vertices_rank_[dimension]] = xoc;
                    vertices_values_[vertices_rank_[dimension]] = xoc_value;
                }
            } else {
                // Inner Contraction
                vector_t xic = c - delta_ * (xr - c);
                double xic_val = objective(xic);
                if( xic_val < best_val) {
                    simplex_[vertices_rank_[dimension]] = xic;
                    vertices_values_[vertices_rank_[dimension]] = xic_value;
                }
            }

            // Shrink
            const vector_t &best_vertex = simplex_[vertices_rank_[0]];
            for(int i = 1; i < dimension + 1; ++i) {
                simplex_[vertices_rank_[i]] = best_vertex + delta_ * (simplex_[vertices_rank_[i]] - best_vertex);
            }

            // Stoping criterion
            double value_mean = 0.0;
            for(double val: vertices_values_) {
                value_mean += val;
            }
            value_mean /= (double)simplex_.size();

            double variance = 0.0;
            for(double val: vertices_values_) {
                double x = val - value_mean;
                variance += x*x;
            }
            variance /= (double)(dimension);

            if(variance <= tol_) {
                done = true;
            }
        }
        
        optimum_ = simplex_[0];
        value_ = objective(optimum_);
        return optimum_;
    }
    
    /**
     * @brief Returns the argmin of the last call to [optimize]
     * 
     * @return vector_t 
     */
    vector_t optimum() const { return optimum_; }

    /**
     * @brief Returns the minimum of the last call to [optimize]
     * 
     * @return double 
     */
    double value() const { return value_; }

    /**
     * @brief Returns the number of iterations performed by the method on the last call to [optimize]
     * 
     * @return int 
     */
    int n_iter() const { return n_iter_; }
};

}   // namespace fdapde

#endif   // __FDAPDE_NELDER_MEAD_H__