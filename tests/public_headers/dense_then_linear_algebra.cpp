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

// preserve the aggregate inclusion order exercised by this translation unit
// clang-format off
#include <fdaPDE/dense_linear_algebra.h>
#include <fdaPDE/linear_algebra.h>
// clang-format on

// the Eigen dynamic vectors retain their vector classification after both aggregates are loaded
static_assert(fdapde::internals::is_vector_like_v<Eigen::VectorXd>);
// the Eigen dynamic matrices remain distinct from vectors after both aggregates are loaded
static_assert(!fdapde::internals::is_vector_like_v<Eigen::MatrixXd>);

int dense_header_order();
// the separate translation unit must preserve the final coefficient across the two include orders
int main() { return dense_header_order() == 4 ? 0 : 1; }
