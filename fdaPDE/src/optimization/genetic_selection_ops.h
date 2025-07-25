
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

#ifndef __FDAPDE_GENETIC_SELECTION_OPS_H__
#define __FDAPDE_GENETIC_SELECTION_OPS_H__

#include "header_check.h"

namespace fdapde {

template <int N>
class RankSelection {
private:
    using vector_t = Eigen::Matrix<double, N, 1>;
    using matrix_t = Eigen::Matrix<double, N, 1>;

    std::vector<int> population_order_;
    std::uniform_real_distribution<> distribution_{0.0, 1.0};
    std::vector<double> cdf_;
    matrix_t new_population_;

    /**
     * @brief Given a random number between 0 and 1, returns the index of selected rank with the probability of rank i \pi_i = (amax - (amax-amin)*(i-1)/(m-1))/m
     * 
     * @param sample random uniform sample between 0 and 1
     * @return int the rank associated with the sample
     */
    int index_of_sample(double sample) {
        int upper_bound = cdf_.size() - 1;
        int lower_bound = 0;

        // Handle edge cases
        if (sample <= cdf_[0]) {
            return 0;
        }
        if (sample > cdf_[upper_bound]) {
            return upper_bound;
        }

        // Binary search to find the appropriate interval
        while (lower_bound < upper_bound) {
            int middle = lower_bound + (upper_bound - lower_bound) / 2;
            if (cdf_[middle] < sample) {
                lower_bound = middle + 1;
            } else {
                upper_bound = middle;
            }
        }

        return upper_bound;
    }

public:
    // constructors
    RankSelection() = default;

    template <typename Opt> bool sync_hook(Opt& opt) {
        population_order_.clear();
        cdf_.clear();

        population_order_.reserve(opt.population.size());
        new_population_ = opt.population;
        cdf_.reserve(opt.population.size());

        constexpr double amax = 1.2;
        constexpr double amin = 2.0 - amax;
        double probability_sum = 0.0;
        for(int i = 0; i < opt.population.cols(); ++i) {
            double pi_i = (
                (amax - (double)(amax - amin)*(i-1)
                /
                (double)(opt.population.cols()-1))/(double)(opt.population.cols())
            );

            probability_sum += pi_i;
            cdf_.push_back(probability_sum);
            population_order_.push_back(i);
        }
        return false;
    }

    template <typename Opt> bool select_hook(Opt& opt) {
        std::sort(
            population_order_.begin(),
            population_order_.end(),
            [&](int a, int b) {
                return opt.population_fitness(a) < opt.population_fitness(b);
            }
        );

        for(int i = 0; i < opt.population.cols(); ++i) {
            int selected = index_of_sample(distribution_(opt.rng));
            new_population_.col(i) = (opt.population.col(population_order_[selected]));
        }
        opt.population = new_population_;

        return false;
    }
};

class BinaryTournamentSelection {
private:
    std::uniform_int_distribution<int> distribution_;

public:
    // constructors
    BinaryTournamentSelection(int population_size = 10):
        distribution_(0,population_size-1)
    {}

    template <typename Opt> bool sync_hook(Opt& opt) {
        fdapde_assert(opt.population.size() >= 2);
        distribution_ = std::uniform_int_distribution<int>{0,static_cast<int>(opt.population.cols())-1};
        return false;
    }

    template <typename Opt> bool select_hook(Opt& opt) {
        fdapde_assert(distribution_.max() == opt.population.cols() - 1);
        fdapde_assert(distribution_.min() == 0);

        // Perform binary tournament selection
        for(int i = 0; i < opt.population.cols(); i += 1) {
            int id_first = distribution_(opt.rng);
            int id_second = distribution_(opt.rng);
            if(id_first == id_second)
                continue;

            double fit_first = opt.population_fitness(id_first);
            double fit_second = opt.population_fitness(id_second);
            if(fit_first <= fit_second) {
                opt.population.col(id_second) = opt.population.col(id_first);
            } else {
                opt.population.col(id_first) = opt.population.col(id_second);
            }
        }
        return false;
    }
};

}   // namespace fdapde

#endif   // __FDAPDE_GENETIC_SELECTION_OPS_H__