
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

class RankSelection {
private:
    std::vector<int> population_order_;

public:
    // constructors
    RankSelection(int population_size = 10) {
        population_order_.reserve(population_size);
        for(int i = 0; i < population_size; ++i)
            population_order_.push_back(i);
    }

    template <typename Opt> void reset_step(Opt& opt) {
        population_order_.clear();
        population_order_.reserve(opt.population.size());
        for(int i = 0; i < opt.population.size(); ++i)
            population_order_.push_back(i);
    }

    template <typename Opt, typename Obj> bool pre_update_step(Opt& opt, Obj& obj) {
        std::sort(
            population_order_.begin(),
            population_order_.end(),
            [&](int a, int b) {
                return opt.population_fitness[a] > opt.population_fitness[b];
            }
        );


        return false;
    }
};

}   // namespace fdapde

#endif   // __FDAPDE_RANK_SELECTION_H__