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

#ifndef __SPLINE_ASSEMBLER_H__
#define __SPLINE_ASSEMBLER_H__

#include "../pde/assembler.h"
#include "../utils/integration/integrator.h"
#include "../utils/integration/integrator_tables.h"
#include "../utils/symbols.h"
#include "../fields/field_ptrs.h"

namespace fdapde {
namespace core {

// spline assembler (B = SplineBasis<R> for some order R)
template <typename D, typename B, typename I> class Assembler<SPLINE, D, B, I> {
   private:
    const D& mesh_;         // 1D mesh
    const I& integrator_;   // quadrature rule
    B basis_;               // spline basis over mesh_
   public:
    Assembler(const D& mesh, const I& integrator) : mesh_(mesh), integrator_(integrator), basis_(mesh.nodes()) {};

    // discretization methods
    template <typename E> SpMatrix<double> discretize_operator(const E& op) {
        constexpr int R = B::order;
        std::size_t M = basis_.size();
        std::vector<fdapde::Triplet<double>> triplet_list;
        SpMatrix<double> discretization_matrix;

        // properly preallocate memory to avoid reallocations
        triplet_list.reserve(M * M);
        discretization_matrix.resize(M, M);

        // prepare space for bilinear form components
        using BasisType = typename B::ElementType;
        BasisType buff_psi_i, buff_psi_j;   // basis functions \psi_i, \psi_j
        // prepare buffer to be sent to bilinear form
        auto mem_buffer = std::make_tuple(ScalarPtr(&buff_psi_i), ScalarPtr(&buff_psi_j));

        // start assembly loop (exploit local support of spline basis)
        for (std::size_t i = 0; i < M; ++i) {
            buff_psi_i = basis_[i];
            for (std::size_t j = 0; j <= (E::is_symmetric ? i : M); ++j) {
                buff_psi_j = basis_[j];
                auto f = op.integrate(mem_buffer);   // let the compiler deduce the type of the expression template!

                // perform integration of f over interval [knots[j], knots[i+R+1]]
                double value = 0;
                for (std::size_t k = j; k <= i + R; ++k) {
                    value += integrator_.template integrate<decltype(f)>(basis_.knots()[k], basis_.knots()[k + 1], f);
                }
                triplet_list.emplace_back(i, j, value);
            }
        }
        // finalize construction
        discretization_matrix.setFromTriplets(triplet_list.begin(), triplet_list.end());
        discretization_matrix.makeCompressed();

        if constexpr (E::is_symmetric)
            return discretization_matrix.selfadjointView<Eigen::Lower>();
        else
            return discretization_matrix;
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __SPLINE_ASSEMBLER_H__
