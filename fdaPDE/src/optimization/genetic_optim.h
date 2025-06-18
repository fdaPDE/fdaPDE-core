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

#ifndef __FDAPDE_GENETIC_OPTIM_H__
#define __FDAPDE_GENETIC_OPTIM_H__

#include "header_check.h"

namespace fdapde {

template <int N, typename... Args> class GeneticOptim {
private:
    using vector_t = std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, 1>, Eigen::Matrix<double, N, 1>>;
    using matrix_t = std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, Dynamic>, Eigen::Matrix<double, N, N>>;

    std::tuple<Args...> callbacks_;
    vector_t optimum_;
    double value_;                     // objective value at optimum
    int max_iter_;                     // maximum number of iterations before forced stop
    int n_iter_ = 0;                   // current iteration number
    double tol_;                       // tolerance on error before forced stop
    double variance_;                  // update step
    int population_size_;              // The size of any given generation
    unsigned seed_;

public:
    std::vector<vector_t> population;
    std::vector<double> population_fitness;
    std::mt19937 rng;

public:
    // constructors
    GeneticOptim() = default;

    // GeneticOptim(int max_iter, double tol, double variance, int population_size, unsigned seed = 0)
    //     requires(sizeof...(Args) != 0):
    //     seed_(seed),
    //     max_iter_(max_iter),
    //     population_size_(population_size),
    //     population(population_size, vector_t{}),
    //     population_fitness(population_size, std::numeric_limits<double>::max()),
    //     tol_(tol), variance_(variance),
    //     rng(seed) {
    //     assert(population_size_ >= 0);
    // }

    GeneticOptim(int max_iter, double tol, double variance, int population_size, unsigned seed, Args&&... callbacks)
        requires(sizeof...(Args) != 0):
        callbacks_(std::make_tuple(std::forward<Args>(callbacks)...)),
        max_iter_(max_iter),
        population_size_(population_size),
        population(population_size, vector_t{}),
        population_fitness(population_size, std::numeric_limits<double>::max()),
        tol_(tol), variance_(variance),
        rng(seed) {
        assert(population_size_ >= 0);
    }

    // copy semantic
    GeneticOptim(const GeneticOptim& other) :
        callbacks_(other.callbacks_),
        population_size_(other.population_size_),
        population_fitness(other.population_fitness),
        max_iter_(other.max_iter_),
        seed_(other.seed_),
        tol_(other.tol_),
        population(other.population),
        variance_(other.variance_){
    }
    
    GeneticOptim& operator=(const GeneticOptim& other) {
        max_iter_ = other.max_iter_;
        tol_ = other.tol_;
        population_fitness = other.population_fitness;
        variance_ = other.variance_;
        population_size_ = other.population_size_;
        callbacks_ = other.callbacks_;
        seed_ = other.seed_;
        population = other.population;
        return *this;
    }

    template <typename ObjectiveT, typename... Functor>
        requires(sizeof...(Functor) < 2) && ((requires(Functor f, double value) { f(value); }) && ...)
    vector_t optimize(ObjectiveT&& objective, const vector_t& x0, Functor&&... func) {
        fdapde_static_assert(
            std::is_same<decltype(std::declval<ObjectiveT>().operator()(vector_t())) FDAPDE_COMMA double>::value,
            INVALID_CALL_TO_OPTIMIZE__OBJECTIVE_FUNCTOR_NOT_ACCEPTING_VECTORTYPE);
        
        // whether or not the callbacks's stoping criteria are met
        bool stop = false;
        value_ = std::numeric_limits<double>::max();
        n_iter_ = 0;

        // Initialize the state
        for(int i = 0; i < population_size_; ++i) {
            population[i] = x0;
        }

        // In the case of this algorithm, post_update... is the mutation process
        // so we apply it before the main loop
        stop |= execute_post_update_step(*this, objective, callbacks_);
        
        while (n_iter_ < max_iter_ && !stop) {
            // Selection
            for(int i = 0; i < population_size_; ++i)
                population_fitness[i] = objective(population[i]);

            stop |= execute_pre_update_step(*this, objective, callbacks_);
            stop |= execute_post_update_step(*this, objective, callbacks_);
            
            // Update the current variance
            int current_best = 0;
            for(int i = 1; i < population_size_; ++i)
            if(population_fitness[i] < population_fitness[current_best])
            current_best = i;
        
            stop |= execute_stopping_criterion(*this, objective);
            
            ++n_iter_;
            value_ = population_fitness[current_best];
            optimum_ = population[current_best];
        }

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

#endif   // __FDAPDE_GENETIC_OPTIM_H__
