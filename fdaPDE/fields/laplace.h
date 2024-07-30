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

#ifndef __LAPLACE_H__
#define __LAPLACE_H__

#include "scalar_field.h"

namespace fdapde {

template <typename Derived> class Laplace : public ScalarBase<Derived::StaticInnerSize, Laplace<Derived>> {
   private:
    using Base = ScalarBase<Derived::StaticInnerSize, Laplace<Derived>>;
    typename internals::ref_select<const Derived>::type derived_;
   public:
    using VectorType = typename Derived::VectorType;
    using Scalar = typename Derived::Scalar;
    static constexpr int StaticInnerSize = Derived::StaticInnerSize;
    static constexpr int NestAsRef = 0;
    Laplace(const Derived& derived) requires(StaticInnerSize == Dynamic)
        : Base(derived.inner_size()), derived_(derived) { }
    explicit constexpr Laplace(const Derived& derived) requires(StaticInnerSize != Dynamic)
        : Base(), derived_(derived) { }
    constexpr double operator()(const VectorType& p) const {
        Scalar res = 0;
        for (int i = 0; i < Base::inner_size(); ++i) { res += derived_.partial(i, i)(p); }
        return res;
    }
    constexpr int static_inner_size() const { return StaticInnerSize; }
    template <typename... Args> Laplace<Derived>& forward(Args&&... args) {
        derived_.forward(std::forward<Args>(args)...);
        return *this;
    }
};

template <int Size, typename Derived> constexpr Laplace<Derived> laplace(const ScalarBase<Size, Derived>& expr) {
    return Laplace<Derived>(expr.derived());
}

}   // namespace fdapde

#endif   // __LAPLACE_H__
