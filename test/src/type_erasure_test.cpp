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

#include <fdaPDE/utils.h>
#include <gtest/gtest.h>   // testing framework

#include <cstddef>
using fdapde::BindingsList;
using fdapde::interface_signature;
using fdapde::method_signature;
using fdapde::TypeErasure;

// define interface of an abstract shape concept
struct Shapable {
    // function pointer bindings to assigned type T
    template <typename T> using Bindings = BindingsList<&T::area>;

    struct Interface : public interface_signature<
      method_signature<double(void)>   // area method signature
      > {
        double area() { return fdapde::invoke<0>(*this); }   // forward to T::area
    };
};
// define type to which arbitrary types implementing the Shapable concept can be assigned
using Shape = TypeErasure<Shapable>;
// define function which takes Shape objects in input
int compute_area(Shape s) { return s.area(); }

// implementations of shapable concept (no inheritance!)
struct Square {
    double l_ = 0;
    Square(double l) : l_(l) {};
    int area() { return l_ * l_; }
};
struct Triangle {
    double b_ = 0;
    double h_ = 0;
    Triangle(double b, double h) : b_(b), h_(h) {};
    int area() { return b_ * h_ / 2; }
};

TEST(type_erasure_test, basics) {
    EXPECT_TRUE(compute_area(Square(2))     == 4);
    EXPECT_TRUE(compute_area(Triangle(2,2)) == 2);
}
