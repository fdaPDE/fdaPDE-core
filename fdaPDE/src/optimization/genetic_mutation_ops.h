
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

#ifndef __FDAPDE_GENETIC_MUTATION_OPS_H__
#define __FDAPDE_GENETIC_MUTATION_OPS_H__

#include "header_check.h"

namespace fdapde {

class GaussianMutation {
private:
    double initial_variance_ = 2.5;
    double variance_ = 2.5;
    double multiplier_ = 0.95;
    std::normal_distribution<double> normal_dist_{0.0,1.0};

public:
    // constructors
    GaussianMutation(double initial_variance = 2.5, double multiplier = 0.95):
        initial_variance_(initial_variance),
        variance_(initial_variance),
        multiplier_(multiplier)
    {}

    template <typename Opt> bool sync_hook(Opt& opt) {
        variance_ = initial_variance_;
        return false;
    }

    template <typename Opt> bool mutate_hook(Opt& opt) {
        // Perform binary tournament selection
        for(int j = 0; j < opt.population.cols(); ++j) {
            for(int i = 0; i < opt.population.rows(); ++i)
                opt.population(i,j) += normal_dist_(opt.rng) * variance_;
        }


        variance_ *= multiplier_;
        return false;
    }
};

class CrossoverMutation {
private:
    std::uniform_int_distribution<int> distribution_{0,10};

public:
    // constructors
    CrossoverMutation() = default;

    template <typename Opt> bool sync_hook(Opt& opt) {
        distribution_ = std::uniform_int_distribution<int>(0, static_cast<int>(opt.population.rows()));
        return false;
    }

    template <typename Opt> bool mutate_hook(Opt& opt) {
        for(int i = 0; i < opt.population.cols() - 1; i += 2) {
            auto &parent_1 = opt.population.col(i);
            auto &parent_2 = opt.population.col(i+1);
            int k = distribution_(opt.rng);
            for(int j = k; j < opt.population.rows(); ++j) {
                double coeff = (parent_1[j] + parent_2[j])/2.0;
                parent_1[j] = coeff;
                parent_2[j] = coeff;
            }
        }
        return false;
    }
};

}   // namespace fdapde

#endif   // __FDAPDE_GENETIC_MUTATION_OPS_H__