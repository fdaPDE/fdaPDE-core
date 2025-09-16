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
#include <fdaPDE/utility.h>

using namespace fdapde;

TEST(matrix_test, MatrixMap_Initialization) {

    using Scalar = double;
    constexpr int len = 4, rows = 2, cols = 2;
    constexpr int size = len * cols * rows;

    {
        Scalar data[size] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        MatrixMap<MatrixView<Scalar, rows, cols>> map_raw(data, size);
        std::cout << "MatrixMap<MatrixView> from raw linear memory\n" << std::endl;
        std::cout << map_raw << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
    }

    {
        Eigen::Matrix<Scalar, len, rows * cols, Eigen::RowMajor> data;
        data << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16;
        MatrixMap<MatrixView<Scalar, rows, cols>> map_raw(data);
        std::cout << "MatrixMap<MatrixView> from EigenMatrix\n" << std::endl;
        std::cout << map_raw << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
    }

    {
        std::array<Scalar, size> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        MatrixMap<MatrixView<Scalar, rows, cols>> map_raw(data);
        std::cout << "MatrixMap<MatrixView> from std::array\n" << std::endl;
        std::cout << map_raw << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
    }

    {
        std::vector<Scalar> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        MatrixMap<MatrixView<Scalar, rows, cols>> map_raw(data);
        std::cout << "MatrixMap<MatrixView> from std::vector\n" << std::endl;
        std::cout << map_raw << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
    }

}

TEST(matrix_test, MatrixMap_StorageOrder) {

    using Scalar = double;
    constexpr int len = 4, rows = 2, cols = 2;
    constexpr int size = len * cols * rows;

    // create ColMajor Eigen Matrix
    Scalar data[size] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    Eigen::Matrix<Scalar, len, rows*cols, Eigen::RowMajor> emat_row(data);
    Eigen::Matrix<Scalar, len, rows*cols, Eigen::ColMajor> emat_col(emat_row);

    {
        MatrixMap<Matrix<Scalar, rows, cols>, ColMajor> map_raw(emat_col);
        std::cout << "MatrixMap<Matrix> ColMajor\n" << std::endl;
        std::cout << map_raw << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
        std::cout << "Extract first column (transposed)" << std::endl;
        std::cout << map_raw.col(0).transpose() << std::endl;
        std::cout << std::endl;
        std::cout << "Check that it changes the original data" << std::endl;
        map_raw.col(0)[0] = 42;
        std::cout << emat_col << std::endl;
        std::cout << std::endl;
    }

}