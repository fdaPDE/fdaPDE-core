
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

#ifndef __FDAPDE_RANK_SELECTION_H__
#define __FDAPDE_RANK_SELECTION_H__

#include "header_check.h"

namespace fdapde {

template <typename VectorType>
class RankSelection {
private:
    std::vector<int> population_order_;
    std::uniform_real_distribution<> distribution_{0.0, 1.0};
    std::vector<double> cdf_;
    std::vector<VectorType> new_population_;

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

    template <typename Opt> void reset_step(Opt& opt) {
        population_order_.clear();
        cdf_.clear();
        new_population_.clear();

        population_order_.reserve(opt.population.size());
        new_population_.reserve(opt.population.size());
        cdf_.reserve(opt.population.size());

        constexpr double amax = 1.2;
        constexpr double amin = 2.0 - amax;
        double probability_sum = 0.0;
        for(int i = 0; i < opt.population.size(); ++i) {
            double pi_i = (amax - (double)(amax - amin)*(i-1)/(double)(opt.population.size()-1))/(double)(opt.population.size());
            probability_sum += pi_i;
            cdf_.push_back(probability_sum);
            population_order_.push_back(i);
        }
    }

    template <typename Opt, typename Obj> bool pre_update_step(Opt& opt, Obj& obj) {
        std::sort(
            population_order_.begin(),
            population_order_.end(),
            [&](int a, int b) {
                return opt.population_fitness[a] < opt.population_fitness[b];
            }
        );

        new_population_.clear();
        for(int i = 0; i < opt.population.size(); ++i) {
            int selected = index_of_sample(distribution_(opt.rng));
            new_population_.push_back(opt.population[population_order_[selected]]);
        }
        opt.population = new_population_;

        return false;
    }
};

}   // namespace fdapde

#endif   // __FDAPDE_RANK_SELECTION_H__