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

    std::normal_distribution<double> normal_dist_ {0.0, 1.0};
    std::mt19937 rng_;

    std::vector<vector_t> simplex_;       // Edges of the simplex
    std::vector<double> vertices_values_; // Value of each edge
    std::vector<int> vertices_rank_;      // Index into the simplex vector, sorted from best to worst
    
    double alpha_ = 1.0; // Reflexion coeff
    double beta_  = 2.0; // Expension coeff
    double gamma_ = 0.5; // Outer contraction coeff
    double delta_ = 0.5; // Inner contraction coeff

    void init_simplex_(const vector_t &x0) {
        const int dimension = x0.rows();

        // Initialises the vectors with cached values
        vertices_rank_.resize(dimension+1, 0);
        vertices_values_.resize(dimension+1, std::numeric_limits<double>::max());
        simplex_.resize(dimension+1, x0);

        for(int i = 0; i < dimension; ++i) {
            double tho = std::abs(simplex_[i+1][i]) < tol_ ? 0.00025: 0.05; 
            simplex_[i+1][i] += tho;
            vertices_rank_[i+1] = i+1;
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
    NelderMead(int max_iter, double tol):
        max_iter_(max_iter),
        tol_(tol){
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
                random_vect[i] = normal_dist_(rng_);
            }
            centroid /= (double)(dimension);
            random_vect /= random_vect.norm();
            
            // Perturbation of the centroid to enhance performance for large dimensions
            // Optim Lett 13, 1011–1025 (2019). https://doi.org/10.1007/s11590-018-1306-2
            vector_t perturbed_centroid = centroid + 0.1 * random_vect * (simplex_[vertices_rank_[0]] - simplex_[vertices_rank_[dimension]]).norm();
            vector_t xr = perturbed_centroid + alpha_*(perturbed_centroid - simplex_[vertices_rank_[dimension]]);

            // Cache the values used for the if statements
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
                }
            } else {
                // Inner Contraction
                vector_t xic = centroid - gamma_ * (xr - centroid);
                double xic_val = objective(xic);
                if( xic_val < best_val) {
                    simplex_[vertices_rank_[dimension]] = xic;
                    vertices_values_[vertices_rank_[dimension]] = xic_val;
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

            std::cout << "---------- STEP " << n_iter_ << " ----------\n";
            std::cout << "Variance: " << variance << "\n";
            for(int i = 0; i < simplex_.size(); ++i) {
                std::cout << "Simplex " << i << ": ";
                std::cout << simplex_[i] << "\n\n";
            }

            for(int i = 0; i < simplex_.size(); ++i) {
                std::cout << "Rank n " << i << ": ";
                std::cout << vertices_rank_[i];
                std::cout << ". With value: " << vertices_values_[vertices_rank_[i]] << "\n\n";
            }

            ++n_iter_;
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