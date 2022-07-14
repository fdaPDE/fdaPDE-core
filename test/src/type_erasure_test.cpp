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

// define interface of an abstract shape concept
struct IShape {
    // function pointer bindings to assigned type T
    template <typename T> using fn_ptrs = fdapde::mem_fn_ptrs<&T::area>;

    // interface
    int area() const { return fdapde::invoke<int, 0>(*this); }   // forward to T::area
};
// define type to which arbitrary types implementing the Shapable concept can be assigned
using Shape = fdapde::erase<fdapde::heap_storage, IShape>;
// define function which takes Shape objects in input
int compute_area(Shape s) { return s.area(); }

// implementations of shapes (no inheritance!)
struct Square {
    double l_ = 0;
    Square() = default;
    Square(double l) : l_(l) {};
    int area() const { return l_ * l_; }
    std::string draw() const { return "square"; }
    std::string print() const { return std::to_string(area()); }
};
struct Triangle {
    double b_ = 0, h_ = 0;
    Triangle() = default;
    Triangle(double b, double h) : b_(b), h_(h) {};
    int area() const { return b_ * h_ / 2; }
    std::string draw() const { return "triangle"; }
    std::string print() const { return std::to_string(area()); }
};

TEST(type_erasure_test, basics) {
    EXPECT_TRUE(compute_area(Square(2)) == 4);
    EXPECT_TRUE(compute_area(Triangle(2, 2)) == 2);
}

TEST(type_erasure_test, container) {
    // create a container of shapes
    std::vector<Shape> shape_vector(2);
    shape_vector[0] = Square(2);
    shape_vector[1] = Triangle(2, 2);
    EXPECT_TRUE(compute_area(shape_vector[0]) == 4);
    EXPECT_TRUE(compute_area(shape_vector[1]) == 2);
}

TEST(type_erasure_test, copy_assign) {
    Shape s1 = Square(2);
    Shape s2 = Triangle(4, 4);
    EXPECT_TRUE(compute_area(s1) != compute_area(s2));
    s2 = s1;
    EXPECT_TRUE(compute_area(s1) == compute_area(s2));
}

// interface of something which can be draw
struct IDrawable {
    // function pointer bindings to assigned type T
    template <typename T> using fn_ptrs = fdapde::mem_fn_ptrs<&T::draw, &T::print>;
    // interface
    std::string draw()  const { return fdapde::invoke<std::string, 0>(*this); }
    std::string print() const { return fdapde::invoke<std::string, 1>(*this); }
};
using DrawableShape = fdapde::erase<fdapde::heap_storage, IShape, IDrawable>;   // a shape which can be draw
TEST(type_erasure_test, inheritance) {
    DrawableShape d = Square(2);
    EXPECT_TRUE(d.area() == 4);          // d exposes the IShape interface, and correctly forward to its implementation
    EXPECT_TRUE(d.draw() == "square");   // d also exposes the IDrawable interface

    d = Triangle(4, 4);   // triangle is also a shape, which can be draw
    EXPECT_TRUE(d.area() == 8);
    EXPECT_TRUE(d.draw() == "triangle");
}

// test order of interfaces doesn't matter
using DrawableShape_Reversed = fdapde::erase<fdapde::heap_storage, IDrawable, IShape>;
TEST(type_erasure_test, inheritance_order_does_not_matter) {
    DrawableShape_Reversed d = Square(2);
    EXPECT_TRUE(d.area() == 4);
    EXPECT_TRUE(d.draw() == "square");

    d = Triangle(4, 4);
    EXPECT_TRUE(d.area() == 8);
    EXPECT_TRUE(d.draw() == "triangle");
}

// test composability of interfaces
using Draw = fdapde::erase<fdapde::heap_storage, IDrawable>; // something which can be draw

struct Composed {
  Composed(const Shape& s) : s_(s) { }
  Shape s_ {};

  std::string draw()  const { return std::to_string(s_.area()); }
  std::string print() const { return std::to_string(s_.area() + 2); } // something useless
};

TEST(type_erasure_test, composition) {
  Shape s1 = Square(2);

  Draw d = Composed(s1);  // store s1 inside d
  EXPECT_TRUE(d.draw() == "4");
}

TEST(type_erasure_test, non_owning_storage) {
  using NonOwningDrawable = fdapde::erase<fdapde::non_owning_storage, IDrawable>;
  
  Shape s1 = Square(2);
  Draw d = Composed(s1);
  NonOwningDrawable non_owning_d = d;
  EXPECT_TRUE(non_owning_d.draw() == "4");
  // copy a non_owning object
  NonOwningDrawable non_owning_d2 = non_owning_d;
  EXPECT_TRUE(non_owning_d2.draw() == "4");
}

// test parameter forwarding
struct Value {   // just a type wrapping an integral type
  double value = 0;
};

// something which can hold a value
struct IKeep {
  // function pointer bindings to assigned type T
  template <typename T> using fn_ptrs = fdapde::mem_fn_ptrs<&T::set, &T::set_value, &T::get>;
  // interface
  void set(double x) { return fdapde::invoke<void, 0>(*this, x); }
  void set_value(Value x) { return fdapde::invoke<void, 1>(*this, x); }
  double get() const { return fdapde::invoke<double, 2>(*this); }
};
using Keeper = fdapde::erase<fdapde::heap_storage, IKeep>;

// implementation of Keeper
struct Keep {
  double x_ = 0;
  void set(double x) { x_ = x; }
  void set_value(Value x) { x_ = x.value; }
  double get() { return x_; }
};

TEST(type_erasure_test, forward_arguments) {
  Keeper k = Keep();
  k.set(4);
  EXPECT_TRUE(k.get() == 4);
  k.set_value(Value{20});
  EXPECT_TRUE(k.get() == 20);
}
