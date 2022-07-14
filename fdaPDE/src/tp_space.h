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

#ifndef __FDAPDE_TP_SPACE_H__
#define __FDAPDE_TP_SPACE_H__

namespace fdapde {

// tensor product space
template <typename... FunctionSpace>
class TpSpace {
    fdapde_static_assert(sizeof...(FunctionSpace) > 1, NOTHING_TO_TENSORIZE__USE_THE_FUNCTIONAL_SPACE_DIRECTLY_INSTEAD);
    // tensorized dof space manager
    template <typename... DofHandler>
        requires(sizeof...(DofHandler) > 1)
    class dof_handler_t {
       public:
        using index_t = int;
        static constexpr int tp_order = sizeof...(DofHandler);
        struct CellType {
            CellType() = default;
            CellType(dof_handler_t* tp_dof_handler, const std::array<index_t, tp_order>& idxs) :
                tp_dof_handler_(tp_dof_handler), idxs_(idxs) { }

            std::vector<index_t> dofs() const { return tp_dof_handler_->active_dofs(idxs_); }
           private:
            dof_handler_t* tp_dof_handler_;
            std::array<index_t, tp_order> idxs_;
        };

        dof_handler_t() : tp_dof_handler_(), n_dofs_(0) { }
        dof_handler_t(const DofHandler&... dof_handler) :
            tp_dof_handler_(std::make_tuple(std::addressof(dof_handler)...)) {
            n_dofs_ = 1;
            offset_[0] = 1;
            int i = 0;
	    // compute overall number of dofs and offset_ vector
            std::apply(
              [&](auto... ts) {
                  (
                    [&]() {
                        n_dofs_ *= ts->n_dofs();
                        n_dofs_dof_handler_[i] = ts->n_dofs();
                        offset_[i + 1] = offset_[i] * ts->n_dofs();
                        i++;
                    }(),
                    ...);
              },
              tp_dof_handler_);
        }

        template <typename... IndexType>
            requires(sizeof...(IndexType) == tp_order) && (std::is_convertible_v<IndexType, index_t> && ...)
        CellType cell(IndexType&&... idxs) {
            return CellType(this, std::array<index_t, tp_order> {static_cast<index_t>(idxs)...});
        }
        // compute dof table
        Eigen::Matrix<double, Dynamic, Dynamic> dofs() const {
            int n_cells = 1;
            Eigen::Matrix<double, Dynamic, Dynamic> dof_table(n_cells, n_dofs_per_cell());
            std::array<index_t, tp_order> multi_index;
            multi_index.fill(0);
            for (int i = 0; i < n_cells; ++i) {
                std::vector<index_t> cell_dof = active_dofs(multi_index);
                for (int j = 0; j < n_dofs_per_cell(); ++j) { dof_table(i, j) = cell_dof[j]; }
                // increase multi-index
                multi_index[0]++;
                int j = 0;
                while (multi_index[j] >= n_dofs_dof_handler_[j]) {
                    multi_index[j] = 0;
                    multi_index[++j]++;
                }
            }
            return dof_table;
        }
        index_t n_dofs() const { return n_dofs_; }
        index_t n_dofs_per_cell() const {
            index_t n_dofs_per_cell_ = 1;
            std::apply(
              [&](auto... ts) { ([&]() { n_dofs_per_cell_ *= ts->n_dofs_per_cell(); }(), ...); }, tp_dof_handler_);
            return n_dofs_per_cell_;
        }
        // is_dof_on_boundary()
        // n_boundary_dofs()
        template <typename... IndexType>
            requires(sizeof...(IndexType) == tp_order) && (std::is_convertible_v<IndexType, index_t> && ...)
        std::vector<index_t> active_dofs(IndexType&&... idxs) const {
            return dof_tensorize_(std::array<index_t, tp_order> {static_cast<index_t>(idxs)...});
        }
        template <typename IndexType>
            requires(internals::is_subscriptable<IndexType, index_t>)
        std::vector<index_t> active_dofs(const IndexType& idx) const {
            return dof_tensorize_(idx);
        }

        operator bool() const { return n_dofs_ != 0; }
        // cell_iterator
       private:
        // given a multi-index (idx_1, idx_2, ..., idx_{tp_order}), compute the dof tensorization
        template <typename IndexType>
            requires(internals::is_subscriptable<IndexType, index_t>)
        std::vector<index_t> dof_tensorize_(const IndexType& idx) const {
            if constexpr (tp_order == 2) {   // optimized case for order 2 tensor product
                std::vector<index_t> lhs_active_dofs;
                std::vector<index_t> rhs_active_dofs;
                std::get<0>(tp_dof_handler_)->active_dofs(idx[0], lhs_active_dofs);
                std::get<1>(tp_dof_handler_)->active_dofs(idx[1], rhs_active_dofs);
                index_t n_lhs_dofs = lhs_active_dofs.size();
                index_t n_rhs_dofs = rhs_active_dofs.size();
                // perform dof tensor product
                std::vector<index_t> tp_dofs;
                tp_dofs.reserve(n_dofs_per_cell());
                int lhs_offset = std::get<0>(tp_dof_handler_)->n_dofs();
                for (int i = 0; i < n_rhs_dofs; ++i) {
                    for (int j = 0; j < n_lhs_dofs; ++j) {
                        tp_dofs.push_back(lhs_active_dofs[j] + lhs_offset * rhs_active_dofs[i]);
                    }
                }
                return tp_dofs;
            } else {
                // localize active dofs for each dof handler
                std::array<std::vector<index_t>, tp_order> active_dofs;
                index_t i = 0;
                std::apply(
                  [&](auto... ts) {
                      (
                        [&]() {
                            ts->active_dofs(idx[i], active_dofs[i]);
                            i++;
                        }(),
                        ...);
                  },
                  tp_dof_handler_);
                // perform dof tensor product
                std::vector<index_t> tp_dofs;
                tp_dofs.reserve(n_dofs_per_cell());
                std::array<index_t, tp_order> multi_index;
                multi_index.fill(0);
                for (int i = 0; i < n_dofs_per_cell(); ++i) {
                    tp_dofs.push_back(internals::apply_index_pack<tp_order>(
                      [&]<int... Ns_>() { return ((active_dofs[Ns_][multi_index[Ns_]] * offset_[Ns_]) + ... + 0); }));
                    // increase multi-index
                    multi_index[0]++;
                    int j = 0;
                    while (multi_index[j] >= tp_order) {
                        multi_index[j] = 0;
                        multi_index[++j]++;
                    }
                }
                return tp_dofs;
            }
        }
        std::tuple<std::add_pointer_t<const DofHandler>...> tp_dof_handler_;
        index_t n_dofs_;
        std::array<index_t, tp_order + 1> offset_;
        std::array<index_t, tp_order> n_dofs_dof_handler_;
    };
   public:
    using FunctionSpaces = std::tuple<FunctionSpace...>;
    using DofHandlerType = dof_handler_t<typename FunctionSpace::DofHandlerType...>;
    using discretization_category = std::tuple<typename FunctionSpace::discretization_category...>;
    static constexpr int tp_order = sizeof...(FunctionSpace);
    static constexpr std::array<int, tp_order> local_dims {FunctionSpace::local_dim...};
    static constexpr std::array<int, tp_order> embed_dims {FunctionSpace::embed_dim...};
    static constexpr std::array<int, tp_order> sobolev_regularity {FunctionSpace::sobolev_regularity...};

    TpSpace() noexcept = default;
    TpSpace(const FunctionSpace&... function_space) :
        function_spaces_(std::make_tuple(function_space...)), dof_handler_(function_space.dof_handler()...) { }

    constexpr int n_shape_functions() const {
        return accumulate_(
          1, [&](int& n_shape_functions_, const auto& ts) { n_shape_functions_ *= ts.n_shape_functions(); });
    }
    constexpr int n_dofs() const { return dof_handler_.n_dofs(); }
    constexpr const std::tuple<FunctionSpace...>& function_spaces() const { return function_spaces_; }
    const DofHandlerType& dof_handler() const { return dof_handler_; }
    DofHandlerType& dof_handler() { return dof_handler_; }
    template <typename... InputType>
        requires(sizeof...(InputType) == sizeof...(FunctionSpace)) && (requires(FunctionSpace f, int i, InputType p) {
                    { f.eval_shape_value(i, p) };
                } && ...)
    constexpr auto eval_shape_value(int i, InputType&&... p) {
        double val = 1.0;
        std::apply([&](const auto&... ts) { ([&]() { val *= ts.eval_shape_value(i, p); }(), ...); }, function_spaces_);
        return val;
    }
    // evaluate value of the i-th shape function given point (p_1, ..., p_n) (requires point location)
    template <typename... InputType>
        requires((requires(FunctionSpace f, int i, InputType p) {
                     { f.eval_cell_value(i, p) };
                 }) && ...) && (sizeof...(InputType) == sizeof...(FunctionSpace))
    auto eval_cell_value(int i, InputType&&... p) const {
        double val = 1.0;
        std::apply([&](const auto&... ts) { ([&]() { val *= ts.eval_cell_value(i, p); }(), ...); }, function_spaces_);
        return val;
    }
    // basis function evaluation, given id of cell containing each point
    template <typename... InputType>
        requires((internals::is_pair_v<InputType> &&
                  requires(FunctionSpace f, int i, InputType p) {
                      { f.eval_cell_value(i, p.first, p.second) };
                  }) &&
                 ...) &&
                (sizeof...(InputType) == sizeof...(FunctionSpace))
    auto eval_cell_value(int i, InputType&&... pair) const {
        double val = 1.0;
        std::apply(
          [&](const auto&... ts) { ([&]() { val *= ts.eval_cell_value(i, pair.first, pair.second); }(), ...); },
          function_spaces_);
        return val;
    }
   private:
    template <typename T_, typename F_> T_ accumulate_(T_ val, F_ f) const {
        T_ tmp = val;
        std::apply([&](const auto&... ts) { (f(tmp, ts), ...); }, function_spaces_);
        return tmp;
    }
    std::tuple<FunctionSpace...> function_spaces_;
    DofHandlerType dof_handler_;
};

// an element of a tensor product space
template <typename TpSpace_>
class TpFunction :
    public ScalarFieldBase<
      []() {
          std::array<int, std::decay_t<TpSpace_>::tp_order> local_dims = std::decay_t<TpSpace_>::local_dims;
          return std::accumulate(local_dims.begin(), local_dims.end(), 0);
      }(),
      TpFunction<TpSpace_>> {
   private:
    // multi-index data structure for tensor product expansion
    template <typename T, int Size> struct multi_index_t {
        using iterator = typename std::array<T, Size>::iterator;
        using const_iterator = typename std::array<T, Size>::const_iterator;
        constexpr multi_index_t() : curr_(0), size_(0) {
            limits_.fill(0);
            index_.fill(0);
        }
        constexpr multi_index_t(const std::array<T, Size>& limits) :
            limits_(limits),
            curr_(0),
            size_(std::accumulate(limits.begin(), limits.end(), 1, std::multiplies<index_t>())) {
            index_.fill(0);
        }
        // observers
        constexpr const T& operator[](T i) const { return index_[i]; }
        constexpr std::size_t size() const { return index_.size(); }
        constexpr const T& front() const { return index_[0]; }
        constexpr const T& back() const { return index_[Size - 1]; }
        operator bool() const { return curr_ < size_; }
        constexpr const std::array<T, Size>& limits() const { return limits_; }
        // modifiers
        constexpr T& operator[](T i) { return index_[i]; }
        multi_index_t& operator++() {
            int i = 0;
            index_[i]++;
            while (i < Size && index_[i] >= limits_[i]) {
                index_[i] = 0;
                index_[++i]++;
            }
	    curr_++;
            return *this;
        }
        void reset() { index_.fill(0); }
        constexpr T& front() { return index_[0]; }
        constexpr T& back() { return index_[Size - 1]; }
        // iterators
        iterator begin() { return index_.begin(); }
        iterator end() { return index_.end() - 1; }
        const_iterator begin() const { return index_.cbegin(); }
        const_iterator end() const { return index_.cend() - 1; }
       private:
        std::array<T, Size + 1> index_;
        std::array<T, Size> limits_;
        int curr_, size_;
    };
   public:
    using TpSpace = std::decay_t<TpSpace_>;
    using FunctionSpaces = typename TpSpace::FunctionSpaces;
    using index_t = int;
    static constexpr int tp_order = TpSpace::tp_order;
    static constexpr int StaticInputSize = []() {
        std::array<int, tp_order> dims = TpSpace::embed_dims;
        return std::accumulate(dims.begin(), dims.end(), 0);
    }();
    static constexpr std::array<index_t, tp_order> static_offset = []() {
        std::array<index_t, tp_order> offset_;
        std::array<int, tp_order> dims = TpSpace::embed_dims;
        offset_[0] = 0;
        for (int i = 1; i < tp_order; ++i) { offset_[i] = offset_[i - 1] + dims[i - 1]; }
        return offset_;
    }();
    static constexpr int Rows = 1;
    static constexpr int Cols = 1;
    static constexpr int NestAsRef = 1;
    static constexpr int XprBits = 0;
    using InputType = Eigen::Matrix<double, StaticInputSize, 1>;
    using Scalar = double;
    using OutputType = double;
    
    TpFunction() = default;
    explicit TpFunction(TpSpace_& tp_space) : tp_space_(&tp_space) {
        coeff_ = Eigen::Matrix<double, Dynamic, 1>::Zero(tp_space_->n_dofs());
    }
    TpFunction(TpSpace_& tp_space, const Eigen::Matrix<double, Dynamic, 1>& coeff) :
        tp_space_(&tp_space), coeff_(coeff) {
        fdapde_assert(coeff.size() > 0 && coeff.size() == tp_space_->n_dofs());
    }
    template <typename... InputType_>
        requires(sizeof...(InputType_) == tp_order)
    OutputType operator()(InputType_&&... p) const {
        // perform point location to identify id of cell containing each point
        std::array<index_t, tp_order> cell_ids;
        internals::for_each_index_and_args<tp_order>(
          [&]<int Ns_, typename InputType__>(const InputType__& p) {
              cell_ids[Ns_] = std::get<Ns_>(tp_space_->function_spaces()).triangulation().locate(p);
          },
          p...);
        // identify active_dofs
	std::vector<index_t> active_dofs = tp_space_->dof_handler().active_dofs(cell_ids);
        // evaluate tensor product basis system	
        OutputType value = 0;
        for (int i = 0, n = active_dofs.size(); i < n; ++i) {
            value += coeff_[active_dofs[i]] *
                     internals::apply_index_pack<tp_order>([&]<int... Ns_>() -> decltype(auto) {
                         return tp_space_->eval_cell_value(i, std::make_pair(cell_ids[Ns_], p)...);
                     });
        }
        return value;
    }
    // optimized point grid evaluation
    template <typename... InputType_>
        requires(sizeof...(InputType_) == tp_order) &&
                (std::is_same_v<std::decay_t<InputType_>, Eigen::Matrix<double, Dynamic, Dynamic>> && ...)
    std::vector<OutputType> grid_eval(InputType_&&... grids) {
        std::array<Eigen::Matrix<index_t, Dynamic, 1>, tp_order> grid_ids;
        std::array<std::vector<Scalar>, tp_order> shape_values;   // evaluation of basis system at grid points
        std::array<index_t, tp_order> outer_limits, inner_limits;
	// initialization
        internals::for_each_index_and_args<tp_order>(
          [&]<int Ns_, typename InputType__>(const InputType__& grid) {
              using function_space_t = std::tuple_element_t<Ns_, FunctionSpaces>;
              fdapde_assert(grid.rows() > 0 && grid.cols() == function_space_t::embed_dim);
              const auto& function_space = std::get<Ns_>(tp_space_->function_spaces());
              index_t n_shape_functions = function_space.n_shape_functions();
              index_t n_points = grid.rows();
              // perform point location
              grid_ids[Ns_] = function_space.triangulation().locate(grid);
              outer_limits[Ns_] = n_points;
              inner_limits[Ns_] = n_shape_functions;
              // evaluate shape functions at grid points
              // shape_values has a matrix structure having as i-th row [\psi_1(p_i), \psi_2(p_i), ..., \psi_n(p_i)]
              shape_values[Ns_].resize(n_points * n_shape_functions);
              for (index_t i = 0; i < n_points; ++i) {
                  for (index_t j = 0; j < n_shape_functions; ++j) {
                      shape_values[Ns_][i * n_shape_functions + j] = function_space.eval_cell_value(
                        j, grid_ids[Ns_][i], Eigen::Matrix<Scalar, function_space_t::embed_dim, 1>(grid.row(i)));
                  }
              }
          },
          grids...);
        // evaluation
        std::vector<OutputType> result(
          std::accumulate(outer_limits.begin(), outer_limits.end(), 1, std::multiplies<index_t>()), 0);
        multi_index_t<index_t, tp_order> outer_index(outer_limits);
        multi_index_t<index_t, tp_order> inner_index(inner_limits);
        multi_index_t<index_t, tp_order> cell_ids;
        for (index_t i = 0; i < tp_order; ++i) { cell_ids[i] = grid_ids[i][0]; }
        index_t i = 0;
        while (outer_index) {
            std::vector<index_t> active_dofs = tp_space_->dof_handler().active_dofs(cell_ids);
            inner_index.reset();
            // evaluate tensor basis expansion
            for (index_t j = 0, n = active_dofs.size(); j < n; ++j, ++inner_index) {
                result[i] +=
                  coeff_[active_dofs[j]] * internals::apply_index_pack<tp_order>([&]<int... Ns_>() {
                      return (shape_values[Ns_][outer_index[Ns_] * inner_limits[Ns_] + inner_index[Ns_]] * ... * 1);
                  });
            }
            ++outer_index;
            for (index_t j = 0; j < tp_order; ++j) { cell_ids[j] = grid_ids[j][outer_index[j]]; }
            i++;
        }
	return result;
    }
    OutputType operator()(const InputType& p) const {
        return internals::apply_index_pack<tp_order>([&]<int... Ns_>() {
            // split p in its components (p1, p2, ..., pn) and forward to tensorized call operator 
            return operator()(Eigen::Matrix<double, FunctionSpaces::embed_dims[Ns_], 1>(
              p.middleRows(static_offset[Ns_], FunctionSpaces::embed_dims[Ns_]))...);
        });
    }
    // observers
    const Eigen::Matrix<double, Dynamic, 1> coeff() const { return coeff_; }
    constexpr TpSpace& tp_space() { return *tp_space_; }
    constexpr const TpSpace& tp_space() const { return *tp_space_; }
    constexpr int rows() const { return Rows; }
    constexpr int cols() const { return Cols; }
    constexpr int input_size() const { return StaticInputSize; }
    void set_coeff(const Eigen::Matrix<double, Dynamic, 1>& coeff) { coeff_ = coeff; }
    // linear algebra
    friend constexpr TpFunction<TpSpace_> operator+(TpFunction<TpSpace_>& lhs, TpFunction<TpSpace_>& rhs) {
        return TpFunction<TpSpace_>(lhs.tp_space(), lhs.coeff() + rhs.coeff());
    }
    friend constexpr TpFunction<TpSpace_> operator-(TpFunction<TpSpace_>& lhs, TpFunction<TpSpace_>& rhs) {
        return TpFunction<TpSpace_>(lhs.tp_space(), lhs.coeff() - rhs.coeff());
    }
    // assignment from expansion coeff vector
    TpFunction& operator=(const Eigen::Matrix<double, Dynamic, 1>& coeff) {
        fdapde_assert(coeff.size() > 0 && coeff.size() == tp_space_->n_dofs());
        coeff_ = coeff;
        return *this;
    }
    // integrate (compute \int_{...} of this field)
   private:
    TpSpace* tp_space_;
    Eigen::Matrix<double, Dynamic, 1> coeff_;
};

}   // namespace fdapde

#endif   // __FDAPDE_TP_SPACE_H__
