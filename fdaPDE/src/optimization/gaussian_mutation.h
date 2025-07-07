
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

#ifndef __FDAPDE_GAUSSIAN_MUTATION_H__
#define __FDAPDE_GAUSSIAN_MUTATION_H__

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

    template <typename Opt> void reset_step(Opt& opt) {
        initial_variance_ = opt.initial_variance;
        variance_ = initial_variance_;
    }

    template <typename Opt, typename Obj> bool post_update_step(Opt& opt, Obj& obj) {
        // Perform binary tournament selection
        for(auto &vec: opt.population) {
            for(int i = 0; i < vec.rows(); ++i)
                vec[i] += normal_dist_(opt.rng) * variance_;
        }

        variance_ *= multiplier_;
        return false;
    }
};

}   // namespace fdapde

#endif   // __FDAPDE_GAUSSIAN_MUTATION_H__