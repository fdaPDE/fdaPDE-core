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

template <int N> class GeneticOptim {
private:
    using vector_t = std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, 1>, Eigen::Matrix<double, N, 1>>;
    using matrix_t = std::conditional_t<N == Dynamic, Eigen::Matrix<double, Dynamic, Dynamic>, Eigen::Matrix<double, N, N>>;

    vector_t optimum_;
    double value_ = std::numeric_limits<double>::max(); // objective value at optimum
    int max_iter_ = 500;        // maximum number of iterations before forced stop
    int n_iter_ = 0;            // current iteration number
    double tol_ = 1e-2;         // tolerance
    int population_size_ = 100; // The size of any given generation
    unsigned seed_ = 0;         // Seed for the RNG
    int static_since_ = 0;      // number of iterations that passed since last optimum value change
    int no_improvement_limit_ = 0;

public:
    static constexpr bool gradient_free = true;
    static constexpr int static_input_size = N;

    std::vector<vector_t> population;
    std::vector<double> population_fitness;
    std::mt19937 rng;

   vector_t x_curr;
   double obj_curr;

public:
    // constructors
    GeneticOptim() = default;

    GeneticOptim(
        int max_iter,
        double tol,
        int no_improvement_limit,
        int population_size,
        unsigned seed = 0) :
        seed_(seed),
        no_improvement_limit_(no_improvement_limit),
        max_iter_(max_iter),
        tol_(tol),
        population_size_(population_size),
        population(population_size, vector_t{}),
        population_fitness(population_size, std::numeric_limits<double>::max()),
        rng(seed) {
        fdapde_assert(population_size_ > 0);
    }

    // copy semantic
    GeneticOptim(const GeneticOptim& other) :
        population_size_(other.population_size_),
        tol_(other.tol_),
        no_improvement_limit_(other.no_improvement_limit_),
        population_fitness(other.population_fitness),
        max_iter_(other.max_iter_),
        seed_(other.seed_),
        population(other.population){
    }
    
    GeneticOptim& operator=(const GeneticOptim& other) {
        max_iter_ = other.max_iter_;
        tol_ = other.tol_;
        no_improvement_limit_ = other.no_improvement_limit_;
        population_fitness = other.population_fitness;
        population_size_ = other.population_size_;
        seed_ = other.seed_;
        population = other.population;
        return *this;
    }

    template <typename ObjectiveT, typename... Callbacks>
    vector_t optimize(ObjectiveT&& objective, const vector_t& x0, Callbacks&&... callbacks) {
        fdapde_static_assert(
          std::is_same<decltype(std::declval<ObjectiveT>().operator()(vector_t())) FDAPDE_COMMA double>::value,
          INVALID_CALL_TO_OPTIMIZE__OBJECTIVE_FUNCTOR_NOT_CALLABLE_AT_VECTOR_TYPE);
        fdapde_assert(population_size_ == population.size());
        fdapde_assert(population_size_ == population_fitness.size());
        
        std::tuple<Callbacks...> callbacks_ {callbacks...};

        // whether or not the callbacks's stoping criteria are met
        bool stop = false;
        value_ = std::numeric_limits<double>::max();
        n_iter_ = 0;
        static_since_ = 0;

        // Initialize the state
        for(int i = 0; i < population_size_; ++i) {
            population[i] = x0;
        }

        // Reset the state of all callbacks and sync their state with the 
        // parameters in GeneticOptim
        stop |= internals::exec_sync_hooks(*this, callbacks_);

        // In the case of this algorithm, post_update... is the mutation process
        // so we apply it before the main loop
        stop |= internals::exec_mutate_hooks(*this, callbacks_);
        
        while (n_iter_ < max_iter_ && !stop) {
            // Compute the fitness for selection
            for(int i = 0; i < population_size_; ++i) {
                population_fitness[i] = objective(population[i]);
            }

            stop |= internals::exec_select_hooks(*this, callbacks_);
            stop |= internals::exec_mutate_hooks(*this, callbacks_);
            
            // Compute argmax of the population fitness
            int current_best = 0;
            for(int i = 1; i < population_size_; ++i) {
                if(population_fitness[i] < population_fitness[current_best])
                    current_best = i;
            }

            // TODO: use tolerance?
            if( std::abs(value_ - population_fitness[current_best]) < tol_) {
                ++static_since_;
            }
            
            // Stoping condition: haven't changed for a while
            stop |= internals::exec_stop_if(*this, objective);
            stop |= static_since_ > no_improvement_limit_;
            
            // Update
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
