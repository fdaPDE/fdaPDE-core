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

#include <fdaPDE/linear_algebra.h>

#include <limits>
#include <stdexcept>

namespace {

namespace native = fdapde::linalg;

using fixed_matrix = native::Matrix<double, 3, 3>;
using fixed_spd = native::SPDMatrix<double, 3, 3>;
using dynamic_spd = native::SPDMatrix<double, fdapde::Dynamic, fdapde::Dynamic>;

template <typename Exception, typename Function> bool throws(Function&& function) {
    try {
        function();
    } catch (const Exception&) { return true; } catch (...) {
        return false;
    }
    return false;
}

fixed_matrix identity() {
    fixed_matrix result;
    result.set_zero();
    for (int i = 0; i < 3; ++i) result(i, i) = 2.0;
    return result;
}

}   // namespace

int main() {
    namespace native = fdapde::linalg;

    native::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> empty(0, 0);
    if (!throws<std::invalid_argument>([&] { dynamic_spd value(empty, native::checked); })) return 1;

    native::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> nonsquare(2, 3);
    nonsquare.set_zero();
    if (!throws<std::invalid_argument>([&] { dynamic_spd value(nonsquare, native::checked); })) return 2;

    fixed_matrix asymmetric = identity();
    asymmetric(0, 1) = 1.0;
    if (!throws<std::invalid_argument>([&] { fixed_spd value(asymmetric, native::checked); })) return 3;

    fixed_matrix indefinite = identity();
    indefinite(0, 0) = -1.0;
    if (!throws<std::domain_error>([&] { fixed_spd value(indefinite, native::checked); })) return 4;

    fixed_matrix nonfinite = identity();
    nonfinite(0, 0) = std::numeric_limits<double>::infinity();
    if (!throws<std::invalid_argument>([&] { fixed_spd value(nonfinite, native::checked); })) return 5;

    native::Matrix<double, 4, 4> near_limit_asymmetric;
    near_limit_asymmetric.set_zero();
    for (int i = 0; i < 4; ++i) near_limit_asymmetric(i, i) = 1.0e308;
    near_limit_asymmetric(0, 1) = 1.0e307;
    if (!throws<std::invalid_argument>(
          [&] { native::SPDMatrix<double, 4, 4> value(near_limit_asymmetric, native::checked); })) {
        return 6;
    }

    const native::IdentityMatrix<double, fdapde::Dynamic, fdapde::Dynamic> too_large(46341, 46341);
    if (!throws<std::length_error>([&] { dynamic_spd value(too_large, native::checked); })) return 7;
    const auto too_large_symmetric = too_large.template as_symmetric<native::Lower>();
    if (!throws<std::length_error>([&] { static_cast<void>(native::matrix_exp(too_large_symmetric)); })) return 8;

    const fixed_matrix initial = identity();
    fixed_spd point(initial, native::checked);
    if (!throws<std::domain_error>([&] { point.assign(indefinite, native::checked); })) return 9;
    if (point.rows() != 3 || point(0, 0) != 2.0 || point(1, 1) != 2.0 || point(2, 2) != 2.0) return 10;

    const fixed_spd unchecked_indefinite(indefinite, native::unchecked);
    if (!throws<std::domain_error>([&] { static_cast<void>(native::matrix_log(unchecked_indefinite)); })) return 11;
    if (!throws<std::domain_error>([&] { static_cast<void>(native::matrix_inverse_sqrt(unchecked_indefinite)); })) {
        return 12;
    }

    native::SymmetricMatrix<double, 3, 3> overflow;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j <= i; ++j) overflow(i, j) = 0.0;
    }
    for (int i = 0; i < 3; ++i) overflow(i, i) = 1000.0;
    if (!throws<std::domain_error>([&] { static_cast<void>(native::matrix_exp(overflow)); })) return 13;

    return 0;
}
