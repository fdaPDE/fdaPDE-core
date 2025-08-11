
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
        variance_ = initial_variance_ * opt.population.rowwise().norm().mean();
        return false;
    }

    template <typename Opt> bool mutate_hook(Opt& opt) {
        opt.population += internals::gaussian_matrix(opt.population.rows(), opt.population.cols(), variance_, opt.seed_);
        variance_ *= multiplier_;
        return false;
    }
};

class CrossoverMutation {
private:
    std::uniform_real_distribution<double> distribution_{0.0,1.0};
    std::uniform_int_distribution<int> indices_distribution_ {0,1};
    double mutation_probability_ = 0.3;

public:
    // constructors
    CrossoverMutation(double mutation_probability = 0.3) : mutation_probability_(mutation_probability) {}

    template <typename Opt> bool sync_hook(Opt& opt) {
        indices_distribution_ = std::uniform_int_distribution<int>(0, opt.population.cols()-1);
        return false;
    }

    template <typename Opt> bool mutate_hook(Opt& opt) {
        for(int j = 0; j < opt.population.cols() - 1; j += 2) {
            // Find a random pair of "parents"
            int partner_idx = indices_distribution_(opt.rng);

            for(int i = 0; i < opt.population.rows(); ++i) {
                // With a certain probability, swap
                if( distribution_(opt.rng) < mutation_probability_) {
                    double coeff = (opt.population(i, j) + opt.population(i, partner_idx))/2.0;
                    opt.population(i, j) = coeff;
                }
            }
        }
        return false;
    }
};

}   // namespace fdapde

#endif   // __FDAPDE_GENETIC_MUTATION_OPS_H__