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
#include <gtest/gtest.h>   // testing framework
using namespace fdapde;

TEST(linear_algebra, matrix) {
    // static-sized
    {
        // construct empty
        constexpr Matrix<double, 2, 2> M0;
        static_assert(M0.rows() == 2);
        static_assert(M0.cols() == 2);
        static_assert(M0.size() == 4);

        // construct from C-array
        constexpr Matrix<double, 2, 3> M1({1, 2, 3, 4, 5, 6});
        static_assert(M1.rows() == 2);
        static_assert(M1.cols() == 3);
        static_assert(M1.size() == 6);
        static_assert([M1]() {
            bool v = true;
            int k = 1;
            for (int i = 0; i < M1.rows(); ++i) {
                for (int j = 0; j < M1.cols(); ++j) {
                    if (M1(i, j) != k++) { v = false; }
                }
            }
            return v;
        }());

        // construct empty and assign
        Matrix<double, 2, 3> M2;
        M2 = M1;
        EXPECT_EQ(M2.rows(), M1.rows());
        EXPECT_EQ(M2.cols(), M1.cols());
        EXPECT_EQ(M2.size(), M1.size());
        EXPECT_EQ(M2, M1);

        // copy-construct
        Matrix<double, 2, 3> M3 = M1;
        EXPECT_EQ(M3.rows(), M1.rows());
        EXPECT_EQ(M3.cols(), M1.cols());
        EXPECT_EQ(M3.size(), M1.size());
        EXPECT_EQ(M3, M1);

        // value-construct
        constexpr Matrix<int, 3, 3> M4(6);
        static_assert(M4.rows() == 3);
        static_assert(M4.cols() == 3);
        static_assert(M4.size() == 9);
        static_assert([M4]() {
            bool v = true;
            for (int i = 0; i < M4.rows(); ++i) {
                for (int j = 0; j < M4.cols(); ++j) { v &= M4(i, j) == 6 ? true : false; }
            }
            return v;
        }());

        // construct from vector-like container
        std::vector<double> vec(9);
        for (int i = 0, n = vec.size(); i < n; ++i) { vec[i] = i; }
        Matrix<double, 3, 3> M5(vec);
        for (int i = 0; i < M5.rows(); ++i) {
            for (int j = 0; j < M5.cols(); ++j) { EXPECT_EQ(M5(i, j), vec[3 * i + j]); }
        }

        // static-construct
        constexpr Matrix<int, 4, 4> M6 = Matrix<int, 4, 4>::Zero();
        static_assert(M6 == Matrix<int, 4, 4>::Zero());
        constexpr Matrix<int, 4, 4> M7 = Matrix<int, 4, 4>::Ones();
        static_assert(M7 == Matrix<int, 4, 4>::Ones());

        // const-access
        EXPECT_EQ(M1(0, 0), 1);
        static_assert(M1(0, 0) == 1);
        // non-const access
        M2(1, 1) = 10;
        EXPECT_EQ(M2(1, 1), 10);
    }

    // dynamic-sized
    {
        // construct empty
        Matrix<double, Dynamic, Dynamic> M0;
        EXPECT_EQ(M0.rows(), 0);
        EXPECT_EQ(M0.cols(), 0);
        EXPECT_EQ(M0.size(), 0);

        // construct empty and resize
        Matrix<double, Dynamic, Dynamic> M1;
        M1.resize(3, 3);   // allocate memory
        EXPECT_EQ(M1.rows(), 3);
        EXPECT_EQ(M1.cols(), 3);
        EXPECT_EQ(M1.size(), 9);
        for (int i = 0; i < M1.rows(); ++i) {
            for (int j = 0; j < M1.cols(); ++j) { EXPECT_EQ(M1(i, j), 0); }
        }

        // construct with sizes
        Matrix<double, Dynamic, Dynamic> M2(5, 5);
        EXPECT_EQ(M2.rows(), 5);
        EXPECT_EQ(M2.cols(), 5);
        EXPECT_EQ(M2.size(), 25);
        for (int i = 0; i < M2.rows(); ++i) {
            for (int j = 0; j < M2.cols(); ++j) { EXPECT_EQ(M2(i, j), 0); }
        }

        // value-construct
        Matrix<double, Dynamic, Dynamic> M3(5, 5, 1.0);
        EXPECT_EQ(M3.rows(), 5);
        EXPECT_EQ(M3.cols(), 5);
        EXPECT_EQ(M3.size(), 25);
        for (int i = 0; i < M3.rows(); ++i) {
            for (int j = 0; j < M3.cols(); ++j) { EXPECT_EQ(M3(i, j), 1.0); }
        }

        // static-construct
        Matrix<int, Dynamic, Dynamic> M4 = Matrix<int, Dynamic, Dynamic>::Zero(10, 10);
        for (int i = 0; i < M4.rows(); ++i) {
            for (int j = 0; j < M4.cols(); ++j) { EXPECT_EQ(M4(i, j), 0); }
        }
        Matrix<int, Dynamic, Dynamic> M5 = Matrix<int, Dynamic, Dynamic>::Ones(10, 10);
        for (int i = 0; i < M5.rows(); ++i) {
            for (int j = 0; j < M5.cols(); ++j) { EXPECT_EQ(M5(i, j), 1); }
        }

        // assignement
        M1 = M2;   // dynamic-sized to dynamic-sized
        EXPECT_EQ(M1.rows(), M2.rows());
        EXPECT_EQ(M1.cols(), M2.cols());
        EXPECT_EQ(M1.size(), M2.size());
        EXPECT_EQ(M1, M2);

        constexpr Matrix<double, 2, 3> M6({1, 2, 3, 4, 5, 6});
        M1 = M6;   // static-sized to dynamic-sized
        EXPECT_EQ(M1.rows(), M6.rows());
        EXPECT_EQ(M1.cols(), M6.cols());
        EXPECT_EQ(M1.size(), M6.size());
        EXPECT_EQ(M1, M6);

        // const access
        EXPECT_EQ(M1(1, 1), 5);
        // non-const access
        M1(1, 1) = 4;
        EXPECT_EQ(M1(1, 1), 4);
    }
}

TEST(linear_algebra, vector) {
    // static-sized
    {
        // construct empty
        constexpr Vector<double, 3> v0;
        static_assert(v0.rows() == 3);
        static_assert(v0.cols() == 1);
        static_assert(v0.size() == 3);

        // construct from C-array
        constexpr Vector<double, 6> v1({1, 2, 3, 4, 5, 6});
        static_assert(v1.rows() == 6);
        static_assert(v1.cols() == 1);
        static_assert(v1.size() == 6);
        static_assert([v1]() {
            bool v = true;
            for (int i = 0; i < v1.size(); ++i) v &= v1[i] == (i + 1) ? true : false;
            return v;
        }());

        // point constructors
        constexpr Vector<double, 1> p1(1);
        static_assert(p1[0] == 1);
        constexpr Vector<double, 2> p2(1, 2);
        static_assert(p2[0] == 1 && p2[1] == 2);
        constexpr Vector<double, 3> p3(1, 2, 3);
        static_assert(p3[0] == 1 && p3[1] == 2 && p3[2] == 3);

        // construct empty and assign
        Vector<double, 6> v2;
        v2 = v1;
        EXPECT_EQ(v2.rows(), v1.rows());
        EXPECT_EQ(v2.size(), v1.size());
        EXPECT_EQ(v2, v1);

        // copy-construct
        Vector<double, 6> v3 = v2;
        EXPECT_EQ(v3.rows(), v2.rows());
        EXPECT_EQ(v3.size(), v2.size());
        EXPECT_EQ(v3, v2);

        // construct from vector-like container
        std::vector<int> vec(6);
        for (int i = 0, n = vec.size(); i < n; ++i) { vec[i] = i; }
        Vector<int, 6> v4(vec);
        for (int i = 0; i < v4.size(); ++i) { EXPECT_EQ(v4[i], vec[i]); }

        // const access
        EXPECT_EQ(v1[0], 1);
        static_assert(v1[0] == 1);
        // non-const access
        v3[0] = 10;
        EXPECT_EQ(v3[0], 10);

        // range-for
        int i = 0;
        for (const auto& value : v3) { EXPECT_EQ(value, v3[i++]); }
        for (auto& value : v3) { value = 5; }
        for (const auto& value : v3) { EXPECT_EQ(value, 5); }
        static_assert(std::accumulate(v1.begin(), v1.end(), 0) == 21);   // constexpr begin/end
    }

    // dynamic-sized
    {
        // construct empty
        Vector<double, Dynamic> v0;
        EXPECT_EQ(v0.rows(), 0);
        EXPECT_EQ(v0.cols(), 1);
        EXPECT_EQ(v0.size(), 0);

        // construct empty and resize
        Vector<double, Dynamic> v1;
        v1.resize(10);   // allocate memory
        EXPECT_EQ(v1.rows(), 10);
        EXPECT_EQ(v1.size(), 10);
        for (int i = 0; i < v1.size(); ++i) { EXPECT_EQ(v1[i], 0); }

        // construct with sizes
        Vector<double, Dynamic> v2(5);
        EXPECT_EQ(v2.rows(), 5);
        EXPECT_EQ(v2.size(), 5);
        for (int i = 0; i < v2.size(); ++i) { EXPECT_EQ(v2[i], 0); }

        // value-construct
        Vector<double, Dynamic> v3(10, 5.0);
        EXPECT_EQ(v3.rows(), 10);
        EXPECT_EQ(v3.size(), 10);
        for (int i = 0; i < v3.size(); ++i) { EXPECT_EQ(v3[i], 5.0); }

        // construct from vector-like container
        std::vector<double> vec(9);
        for (int i = 0, n = vec.size(); i < n; ++i) { vec[i] = i; }
        Vector<double, Dynamic> v4(vec);
        for (int i = 0; i < v4.size(); ++i) { EXPECT_EQ(v4[i], vec[i]); }

        // static-construct
        Vector<int, Dynamic> v5 = Vector<int, Dynamic>::Zero(5);
        for (int i = 0; i < v5.size(); ++i) { EXPECT_EQ(v5[i], 0); }
        Vector<int, Dynamic> v6 = Vector<int, Dynamic>::Ones(5);
        for (int i = 0; i < v6.size(); ++i) { EXPECT_EQ(v6[i], 1); }
        Vector<double, Dynamic> v7 = Vector<double, Dynamic>::LinSpaced(10, 0, 1);
        for (double i = 0; i < v7.size(); ++i) { EXPECT_EQ(v7[i], i * (1./9)); }

        // const access
        EXPECT_EQ(v1[0], 0);
        // non-const access
        v3[0] = 10;
        EXPECT_EQ(v3[0], 10);

        // range-for
        for (const auto& value : v5) { EXPECT_EQ(value, 0); }
        for (auto& value : v5) { value = 5; }
        for (const auto& value : v5) { EXPECT_EQ(value, 5); }
    }
}
