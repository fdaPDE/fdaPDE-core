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

#ifndef __FE_MAP_H__
#define __FE_MAP_H__

#include "../fields/scalar_field.h"
#include "../fields/meta.h"
#include "fe_function.h"

namespace fdapde {
  
// anytime you compose a trial or test function with a functor which is not callable at fe_assembler_packet, we wrap it
// into a FeMap, a fe_assembler_packet callable type encoding the functor evaluated at a fixed set of (quadrature) nodes
template <typename Derived_>
struct FeMap :
    public std::conditional_t<
      std::is_base_of_v<fdapde::ScalarBase<Derived_::StaticInputSize, Derived_>, Derived_>,
      fdapde::ScalarBase<Derived_::StaticInputSize, FeMap<Derived_>>,
      fdapde::MatrixBase<Derived_::StaticInputSize, FeMap<Derived_>>> {
   private:
    using OutputType = decltype(std::declval<Derived_>().operator()(std::declval<typename Derived_::InputType>()));
    static constexpr bool is_scalar =
      std::is_base_of_v<fdapde::ScalarBase<Derived_::StaticInputSize, Derived_>, Derived_>;
  using Derived = std::decay_t<Derived_>;
   public:
    using Base = std::conditional_t<
      std::is_base_of_v<fdapde::ScalarBase<Derived::StaticInputSize, Derived>, Derived>,
      fdapde::ScalarBase<Derived::StaticInputSize, FeMap<Derived>>,
      fdapde::MatrixBase<Derived::StaticInputSize, FeMap<Derived>>>;
    using InputType = internals::fe_assembler_packet<Derived::StaticInputSize>;
    using Scalar = double;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = Derived::XprBits | bilinear_bits::compute_physical_quad_nodes;
    static constexpr int ReadOnly = 1;
    static constexpr int Rows = []() { if constexpr(is_scalar) return 1; else return Derived::Rows; }();
    static constexpr int Cols = []() { if constexpr(is_scalar) return 1; else return Derived::Cols; }();

    constexpr FeMap() = default;
    constexpr FeMap(const Derived_& xpr) : xpr_(&xpr) { }
    template <typename CellIterator>
    void init(
      std::unordered_map<const void*, DMatrix<double>>& buff, const DMatrix<double>& nodes,
      [[maybe_unused]] CellIterator begin, [[maybe_unused]] CellIterator end) const {
        const void* ptr = reinterpret_cast<const void*>(xpr_);
        if (buff.find(ptr) == buff.end()) {
            DMatrix<double> mapped(nodes.rows(), Rows * Cols);
            if constexpr (is_scalar) {
                for (int i = 0, n = nodes.rows(); i < n; ++i) { mapped(i, 0)  = xpr_->operator()(nodes.row(i)); }
            } else {
                for (int i = 0, n = nodes.rows(); i < n; ++i) { mapped.row(i) = xpr_->operator()(nodes.row(i)); }
            }
            buff[ptr] = mapped;
            map_ = &buff[ptr];
        } else {
            map_ = &buff[ptr];
        }
    }
    // fe assembler evaluation
    constexpr OutputType operator()(const InputType& fe_packet) const {
        if constexpr (is_scalar) {
            return map_->operator()(fe_packet.quad_node_id, 0);
        } else {
            return map_->row(fe_packet.quad_node_id);
        }
    }
    constexpr const Derived& derived() const { return xpr_; }
    constexpr int input_size() const { return StaticInputSize; }
   private:
    const Derived* xpr_;
    mutable const DMatrix<Scalar>* map_;
};

// FeMap specialization for FeFunction types
template <typename FeSpace>
struct FeMap<FeFunction<FeSpace>> :
    public fdapde::ScalarBase<FeSpace::local_dim, FeMap<FeFunction<FeSpace>>> {
   private:
    using Derived = FeFunction<FeSpace>;
   public:
    using Base = ScalarBase<Derived::StaticInputSize, FeMap<Derived>>;
    using InputType = internals::fe_assembler_packet<Derived::StaticInputSize>;
    using Scalar = double;
    static constexpr int StaticInputSize = Derived::StaticInputSize;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = Derived::XprBits | bilinear_bits::compute_physical_quad_nodes;

    FeMap() = default;
    FeMap(const Derived& xpr) : xpr_(&xpr) { }
    // fast subscribe routine which bypasses the point location step (quadrature nodes are not arbitrary points)
    template <typename CellIterator>
    void init(
      std::unordered_map<const void*, DMatrix<double>>& buff, const DMatrix<double>& nodes, CellIterator begin,
      CellIterator end) const {
        const void* ptr = reinterpret_cast<const void*>(xpr_);
        if (buff.find(ptr) == buff.end()) {
            DMatrix<double> mapped(nodes.rows(), 1);
	    int n_cells = end.index() - begin.index();
            int n_quad_nodes = nodes.rows() / n_cells;
	    int cell_id = begin.index();
            for (int i = 0, n = nodes.rows(); i < n; ++i) {
                // map node to reference cell and evaluate
                auto cell = xpr_->fe_space().dof_handler().cell(cell_id + i / n_quad_nodes);
                SVector<FeSpace::local_dim> ref_node = cell.invJ() * (nodes.row(i).transpose() - cell.node(0));
                DVector<int> active_dofs = cell.dofs();
                Scalar value = 0;
                for (int i = 0, n = xpr_->fe_space().n_basis(); i < n; ++i) {
                    value += xpr_->coeff()[active_dofs[i]] * xpr_->fe_space().eval(i, ref_node);
                }
                mapped(i, 0) = value;
            }
            buff[ptr] = mapped;
            map_ = &buff[ptr];
        } else {
            map_ = &buff[ptr];
        }
    }
    // fe assembler evaluation
    constexpr Scalar operator()(const InputType& fe_packet) const {
        return map_->operator()(fe_packet.quad_node_id, 0);
    }
    constexpr const Derived& derived() const { return xpr_; }
    constexpr int input_size() const { return StaticInputSize; }
   private:
    const Derived* xpr_;
    mutable const DMatrix<Scalar>* map_;
};

}   // namespace fdapde

#endif   // __FE_MAP_H__
