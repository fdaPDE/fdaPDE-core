
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

#ifndef __FDAPDE_BINARY_TOURNAMENT_SELECTION_H__
#define __FDAPDE_BINARY_TOURNAMENT_SELECTION_H__

#include "header_check.h"

namespace fdapde {

class BinaryTournamentSelection {
private:
    std::uniform_int_distribution<int> distribution_;

public:
    // constructors
    BinaryTournamentSelection(int population_size = 10):
        distribution_(0,population_size-1)
    {}

    template <typename Opt> void reset_step(Opt& opt) {
        assert(opt.population.size() >= 2);
        distribution_ = std::uniform_int_distribution<int>{0,static_cast<int>(opt.population.size())-1};
    }

    template <typename Opt, typename Obj> bool pre_update_step(Opt& opt, Obj& obj) {
        assert(distribution_.max() == opt.population.size() - 1);

        // Perform binary tournament selection
        for(int i = 0; i < opt.population.size(); i += 1) {
            int id_first = distribution_(opt.rng);
            int id_second = distribution_(opt.rng);
            if(id_first == id_second)
                continue;

            double fit_first = opt.population_fitness[id_first];
            double fit_second = opt.population_fitness[id_second];
            if(fit_first <= fit_second) {
                opt.population[id_second] = opt.population[id_first];
            } else {
                opt.population[id_first] = opt.population[id_second];
            }
        }
        return false;
    }
};

}   // namespace fdapde

#endif   // __FDAPDE_BINARY_TOURNAMENT_SELECTION_H__