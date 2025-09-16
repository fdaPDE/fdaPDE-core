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

#ifndef __FDAPDE_MATRIX_MAP_H__
#define __FDAPDE_MATRIX_MAP_H__

#include "header_check.h"

namespace fdapde {

namespace internals {
template<typename T>
concept HasSizeAndPtrData =
    requires(T t) {
    { t.size() } -> std::convertible_to<std::size_t>;
    { t.data() } -> std::convertible_to<typename T::value_type*>;
    };
}

template<typename ReturnType_, int StorageOrder_ = RowMajor>
requires(StorageOrder_ == RowMajor || (StorageOrder_ == ColMajor && !internals::is_view_v<ReturnType_>))
struct MatrixMap {
    using ReturnType = ReturnType_;
    using Scalar = typename ReturnType::Scalar;
    static constexpr int ReturnTypeStorageSize = ReturnType::StorageSize;
    static constexpr int StorageOrder = StorageOrder_;
    #ifdef __FDAPDE_HAS_EIGEN__
    template<typename T>
    using ColMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
    #endif


    // default constructors
    constexpr MatrixMap() = delete;

    // constructors from linear data
    constexpr explicit MatrixMap(Scalar* ptr_table, const int size) : n_entities_(size/ReturnTypeStorageSize), ptr_data_(ptr_table) {
        fdapde_assert(n_entities_ % ReturnTypeStorageSize == 0);
    }

    // constructors from storage type (that implements data() and size())
    template<typename StorageType>
    requires(internals::HasSizeAndPtrData<StorageType>)
    constexpr explicit MatrixMap(StorageType& table) : n_entities_(table.size()/ReturnTypeStorageSize),  ptr_data_(table.data()) {
        fdapde_assert(table.size() % ReturnTypeStorageSize == 0);
    }

    // const access to matrices
    constexpr ReturnType operator[](const int i) const
    requires(StorageOrder == RowMajor){
        assert(i >=0 && i < n_entities());
        return ReturnType(ptr_data_ + i*ReturnTypeStorageSize);
    }
    constexpr ReturnType operator[](const int i) const
    requires(StorageOrder == ColMajor){
        assert(i >=0 && i < n_entities());
        std::array<Scalar, ReturnTypeStorageSize> data;
        for (int j = 0; j < ReturnTypeStorageSize; ++j) data[j] = ptr_data_[j*n_entities() + i];
        return ReturnType(data);
    }
    // non-const access to matrices
    constexpr ReturnType& operator[](const int i) {
        fdapde_static_assert(StorageOrder == RowMajor, NON_CONST_ACCESS_IS_FOR_ROW_MAJOR_STORAGE_ONLY);
        fdapde_assert(i >=0 && i < n_entities());
        return ReturnType(ptr_data_ + i*ReturnTypeStorageSize);
    }

    // columns access (ColMajor only)
    #ifdef __FDAPDE_HAS_EIGEN__
    constexpr ColMap<Scalar> col(const int i) {
        fdapde_assert(i >= 0 && i < ReturnTypeStorageSize);
        return ColMap<Scalar>(ptr_data_ + i * n_entities(), n_entities());
    }
    #endif

    // dimension
    [[nodiscard]] constexpr int n_entities() const { return n_entities_; }

    // send matrix map to ostream (this is not constexpr evaluable)
    friend std::ostream& operator<<(std::ostream& os, const MatrixMap<ReturnType, StorageOrder>& map) {
        int n = map.n_entities();
        if (n <= 3) {
            for (int i = 0; i < n; ++i) {
                os << map[i];
                if (i + 1 < n) os << "\n\n";
            }
        } else {
            os << map[0] << "\n\n"
               << map[1] << "\n\n"
               << "...\n\n"
               << map[n - 1];
        }
        return os;
    }
private:
    int n_entities_ = 0;
    Scalar* ptr_data_ = nullptr;

};

}

#endif   // _FDAPDE_MATRIX_H__