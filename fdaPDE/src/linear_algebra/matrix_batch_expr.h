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

#ifndef __FDAPDE_LINALG_MATRIX_BATCH_EXPR_H__
#define __FDAPDE_LINALG_MATRIX_BATCH_EXPR_H__

#include <array>
#include <functional>
#include <span>

#include "header_check.h"

namespace fdapde {

/// @brief declares the common interface for deferred batch evaluation
template <typename Derived> class MatrixBatchExpr;
namespace internals {

/// @brief identifies native batch owners and deferred batch expressions
template <typename T>
concept matrix_batch_expression = std::derived_from<std::remove_cvref_t<T>, MatrixBatchExpr<std::remove_cvref_t<T>>>;

/// @brief accepts persistent operands and temporary nodes without borrowing expiring owners
template <typename T>
concept batch_operand =
  (matrix_batch_expression<T> || matrix_expression<T> || std::is_arithmetic_v<std::remove_cvref_t<T>>) &&
  !is_owning_rvalue_expression_v<T>;

/// @brief limits eager batch transforms to the native sequential and parallel execution policies
template <typename Policy>
concept BatchExecutionPolicy = std::same_as<Policy, execution_seq_t> || std::same_as<Policy, execution_par_t>;

/// @brief requires the owning row-major element and native views used by contiguous matrix batches
template <typename T>
concept batch_matrix_owner = matrix_expression<T> && requires {
    typename T::View;
    typename T::ConstView;
    requires T::NestAsRef != 0;
    requires T::StorageOrder == RowMajor;
    requires std::is_arithmetic_v<typename T::Scalar> && !std::is_const_v<typename T::Scalar>;
};

/// @brief fills an owning batch directly from independent indexed values with joined optional parallel execution
template <typename Result, BatchExecutionPolicy ExecutionPolicy, typename Function>
MatrixBatch<Result>
generate_matrix_batch(std::size_t count, int rows, int cols, ExecutionPolicy policy, const Function& function);

/// @brief infers nonempty expression shape once and materializes every entry with joined optional parallel execution
template <typename Result, BatchExecutionPolicy ExecutionPolicy, typename Source>
MatrixBatch<Result> evaluate_matrix_batch(const Source& source, ExecutionPolicy policy);

/// @brief borrows persistent batch operands and owns temporary expression nodes
template <typename Arg>
using batch_nested_t =
  std::conditional_t<std::is_lvalue_reference_v<Arg>, const std::remove_reference_t<Arg>&, std::remove_cvref_t<Arg>>;

/// @brief stores scalar operands by value while applying native lifetime rules to matrices and batches
template <typename Arg>
using batch_operand_nested_t =
  std::conditional_t<std::is_arithmetic_v<std::remove_cvref_t<Arg>>, std::remove_cvref_t<Arg>, batch_nested_t<Arg>>;

/// @brief resolves matrix expression results to owners while retaining diagonal and symmetric structure
template <typename Result, bool Numeric = std::is_arithmetic_v<std::remove_cvref_t<Result>>> struct batch_result_owner {
    using Xpr = std::remove_cvref_t<Result>;
    using Scalar = std::remove_cv_t<typename Xpr::Scalar>;
    /// @brief chooses the narrowest native owner that preserves the returned matrix structure
    static auto owner_type_() {
        if constexpr (SPDLike<Xpr>)
            return std::type_identity<SPDMatrix<Scalar, Xpr::Rows, typename Xpr::CachePolicy>> {};
        else if constexpr (RotationLike<Xpr>)
            return std::type_identity<RotationMatrix<Scalar, Xpr::Rows, typename Xpr::CachePolicy>> {};
        else if constexpr (is_orthogonal_matrix_v<Xpr> && std::floating_point<Scalar>)
            return std::type_identity<OrthogonalMatrix<Scalar, Xpr::Rows>> {};
        else if constexpr (NativeSymmetricLike<Xpr>)
            return std::type_identity<SymmetricMatrix<Scalar, Xpr::Rows, typename Xpr::CachePolicy>> {};
        else if constexpr (is_diagonal_matrix_v<Xpr>)
            return std::type_identity<DiagonalMatrix<Scalar, Xpr::Rows>> {};
        else if constexpr (is_symmetric_matrix_v<Xpr>)
            return std::type_identity<SymmetricMatrix<Scalar, Xpr::Rows>> {};
        else if constexpr (is_skew_symmetric_matrix_v<Xpr>)
            return std::type_identity<SkewSymmetricMatrix<Scalar, Xpr::Rows>> {};
        else
            return std::type_identity<Matrix<Scalar, Xpr::Rows, Xpr::Cols>> {};
    }
    using type = typename decltype(owner_type_())::type;
};
/// @brief represents a scalar map result as a native one-by-one matrix
template <typename Result> struct batch_result_owner<Result, true> {
    using type = Matrix<std::remove_cvref_t<Result>, 1, 1>;
};

/// @brief passes a scalar or read-only matrix view while its source owner remains alive
template <typename Value> auto batch_const_view(const Value& value) {
    if constexpr (std::is_arithmetic_v<Value>)
        return value;
    else if constexpr (Value::NestAsRef == 0)
        return value;
    else {
        using View = typename Value::ConstView;
        if constexpr (SPDLike<Value> || RotationLike<Value> || NativeSymmetricLike<Value>)
            return value.view();
        else if constexpr (is_orthogonal_matrix_v<Value>)
            return value.view();
        else if constexpr (is_diagonal_matrix_v<Value>) {
            if constexpr (Value::Rows == Dynamic)
                return View(value.data(), value.rows());
            else
                return View(value.data());
        } else
            return View(value.data(), value.rows(), value.cols());
    }
}

/// @brief selects one batch entry or broadcasts a persistent matrix or scalar operand
template <typename Operand> decltype(auto) batch_operand_value(const Operand& operand, std::size_t i) {
    if constexpr (matrix_batch_expression<Operand>)
        return operand[i];
    else
        return (operand);
}

/// @brief consumes coefficient expressions before their local views expire, materializing other batch nodes
template <typename Operand, typename Function>
auto with_batch_operand(const Operand& operand, std::size_t i, const Function& function) {
    if constexpr (requires { operand.with_coefficients(i, function); }) {
        return operand.with_coefficients(i, function);
    } else {
        const auto& value = batch_operand_value(operand, i);
        const auto view = batch_const_view(value);
        return function(view);
    }
}

/// @brief lifts a native binary matrix operation to corresponding batch entries or a broadcast operand
template <typename Lhs, typename Rhs, typename Operation>
class matrix_batch_binary : public MatrixBatchExpr<matrix_batch_binary<Lhs, Rhs, Operation>> {
   public:
    using LhsType = std::remove_cvref_t<Lhs>;
    using RhsType = std::remove_cvref_t<Rhs>;
    using LhsArgument = decltype(batch_const_view(batch_operand_value(std::declval<const LhsType&>(), 0)));
    using RhsArgument = decltype(batch_const_view(batch_operand_value(std::declval<const RhsType&>(), 0)));
    using Result = std::invoke_result_t<Operation, const LhsArgument&, const RhsArgument&>;
    using MatrixType = typename batch_result_owner<Result>::type;
    static constexpr int NestAsRef = 0;
    static constexpr bool CoefficientWise =
      !std::same_as<Operation, std::multiplies<>> || std::is_arithmetic_v<LhsType> || std::is_arithmetic_v<RhsType>;
    /// @brief nests operands without evaluating entries and rejects unequal batch lengths
    matrix_batch_binary(Lhs lhs, Rhs rhs) : lhs_(std::forward<Lhs>(lhs)), rhs_(std::forward<Rhs>(rhs)) { (void)size(); }
    /// @brief reports the current batch length after checking any paired collection
    std::size_t size() const {
        if constexpr (matrix_batch_expression<LhsType> && matrix_batch_expression<RhsType>)
            fdapde_strong_assert(
              lhs_.size() == rhs_.size(), std::invalid_argument, "MatrixBatch arithmetic requires equal lengths");
        if constexpr (matrix_batch_expression<LhsType>)
            return lhs_.size();
        else
            return rhs_.size();
    }
    /// @brief materializes a native result before local matrix views and temporary owners expire
    MatrixType operator[](std::size_t i) const {
        fdapde_strong_assert(i < size(), std::out_of_range, "MatrixBatch arithmetic: index out of range");
        if constexpr (CoefficientWise) {
            return with_coefficients(i, [](const auto& value) { return MatrixType(value); });
        } else {
            const auto& lhs = batch_operand_value(lhs_, i);
            const auto& rhs = batch_operand_value(rhs_, i);
            const auto left = batch_const_view(lhs);
            const auto right = batch_const_view(rhs);
            (void)shape_(left, right);
            return MatrixType(Operation {}(left, right));
        }
    }
    /// @brief fuses coefficient arithmetic while keeping every operand alive through owning materialization
    template <typename Function>
    auto with_coefficients(std::size_t i, const Function& function) const
        requires CoefficientWise
    {
        return with_batch_operand(lhs_, i, [&](const auto& left) {
            return with_batch_operand(rhs_, i, [&](const auto& right) {
                (void)shape_(left, right);
                const auto value = Operation {}(left, right);
                return function(value);
            });
        });
    }
    /// @brief preserves the result row extent for empty expressions with known operand shapes
    int rows() const { return shape_(lhs_, rhs_).first; }
    /// @brief preserves the result column extent for empty expressions with known operand shapes
    int cols() const { return shape_(lhs_, rhs_).second; }
   private:
    /// @brief derives result shape and checks matrix compatibility even when no entry can be evaluated
    template <typename Left, typename Right> static std::pair<int, int> shape_(const Left& lhs, const Right& rhs) {
        if constexpr (std::is_arithmetic_v<Left>)
            return {rhs.rows(), rhs.cols()};
        else if constexpr (std::is_arithmetic_v<Right>)
            return {lhs.rows(), lhs.cols()};
        else if constexpr (std::same_as<Operation, std::multiplies<>>) {
            fdapde_strong_assert(
              lhs.cols() == rhs.rows(), std::invalid_argument,
              "MatrixBatch product requires compatible inner dimensions");
            return {lhs.rows(), rhs.cols()};
        } else {
            fdapde_strong_assert(
              lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols(), std::invalid_argument,
              "MatrixBatch arithmetic requires matching matrix dimensions");
            return {lhs.rows(), lhs.cols()};
        }
    }
    Lhs lhs_;
    Rhs rhs_;
};

/// @brief rejects compound arithmetic that would implicitly discard a structured destination invariant
template <typename Target, typename Result>
inline constexpr bool batch_compound_compatible_v = [] {
    if constexpr (SPDLike<Target>)
        return true;
    else if constexpr (is_symmetric_matrix_v<Target>)
        return is_symmetric_matrix_v<Result>;
    else if constexpr (is_diagonal_matrix_v<Target>)
        return is_diagonal_matrix_v<Result>;
    else if constexpr (is_skew_symmetric_matrix_v<Target>)
        return is_skew_symmetric_matrix_v<Result>;
    else if constexpr (RotationLike<Target>)
        return RotationLike<Result>;
    else if constexpr (is_orthogonal_matrix_v<Target>)
        return is_orthogonal_matrix_v<Result>;
    else
        return true;
}();

/// @brief applies a stored callable only when a requested batch element is evaluated
template <typename Source, typename Function>
class matrix_batch_map : public MatrixBatchExpr<matrix_batch_map<Source, Function>> {
   public:
    using SourceType = std::remove_cvref_t<Source>;
    using SourceValue = decltype(std::declval<const SourceType&>()[0]);
    using Argument = decltype(batch_const_view(std::declval<const SourceValue&>()));
    using Result = std::invoke_result_t<Function&, const Argument&>;
    using MatrixType = typename batch_result_owner<Result>::type;
    static constexpr int NestAsRef = 0;
    /// @brief retains the callable by value and borrows or owns its source according to nesting
    matrix_batch_map(Source source, Function function) :
        source_(std::forward<Source>(source)), function_(std::move(function)) { }
    /// @brief returns the unchanged source element count without evaluating the callable
    std::size_t size() const { return source_.size(); }
    /// @brief materializes one callable result while the source proxy and returned expression remain alive
    MatrixType operator[](std::size_t i) const {
        fdapde_strong_assert(i < size(), std::out_of_range, "MatrixBatch map: index out of range");
        const auto value = source_[i];
        const auto view = batch_const_view(value);
        return MatrixType(std::invoke(function_, view));
    }
    /// @brief reports a fixed result row count and rejects unknown dynamic empty-result shape
    int rows() const {
        fdapde_strong_assert(
          MatrixType::Rows != Dynamic, std::invalid_argument, "MatrixBatch map: dynamic result shape is unknown");
        return MatrixType::Rows;
    }
    /// @brief reports a fixed result column count and rejects unknown dynamic empty-result shape
    int cols() const {
        fdapde_strong_assert(
          MatrixType::Cols != Dynamic, std::invalid_argument, "MatrixBatch map: dynamic result shape is unknown");
        return MatrixType::Cols;
    }
   private:
    Source source_;
    mutable Function function_;
};

/// @brief infers inline index storage from fixed containers and native vectors
template <typename Indices>
inline constexpr int batch_index_extent = [] {
    using T = std::remove_cvref_t<Indices>;
    if constexpr (requires { std::tuple_size<T>::value; })
        return static_cast<int>(std::tuple_size_v<T>);
    else if constexpr (requires { T::extent; })
        return T::extent == std::dynamic_extent ? Dynamic : static_cast<int>(T::extent);
    else if constexpr (requires {
                           T::Rows;
                           T::Cols;
                       })
        return T::Rows == 1 ? T::Cols : (T::Cols == 1 ? T::Rows : Dynamic);
    else
        return Dynamic;
}();

/// @brief owns ordered node indices while sharing the original matrices and their cache slots
template <typename Source, int Count = Dynamic>
class matrix_batch_selection : public MatrixBatchExpr<matrix_batch_selection<Source, Count>> {
   public:
    using MatrixType = typename std::remove_cvref_t<Source>::MatrixType;
    static constexpr int NestAsRef = 0;
    static constexpr int IndexCount = Count;
    /// @brief copies and validates every index without reordering or evaluating the selected matrices
    template <typename Indices>
    matrix_batch_selection(Source source, const Indices& indices) : source_(std::forward<Source>(source)) {
        if constexpr (Count == Dynamic)
            indices_.resize(indices.size());
        else
            fdapde_assert(
              indices.size() == Count, std::invalid_argument, "MatrixBatch select: static index count mismatch");
        for (std::size_t i = 0; i < static_cast<std::size_t>(indices.size()); ++i) {
            const auto index = indices[i];
            static_assert(std::is_integral_v<std::remove_cvref_t<decltype(index)>>, "batch indices must be integral");
            fdapde_strong_assert(
              std::cmp_less(index, source_.size()), std::out_of_range, "MatrixBatch select: index out of range");
            fdapde_strong_assert(
              std::cmp_greater_equal(index, 0), std::out_of_range, "MatrixBatch select: negative index");
            indices_[i] = static_cast<std::size_t>(index);
        }
    }
    /// @brief returns the number of selected entries including repeated indices
    std::size_t size() const { return indices_.size(); }
    /// @brief preserves the source element row count even for an empty selection
    int rows() const { return source_.rows(); }
    /// @brief preserves the source element column count even for an empty selection
    int cols() const { return source_.cols(); }
    /// @brief returns the original read-only element in the user-supplied order
    auto operator[](std::size_t i) const {
        fdapde_strong_assert(i < size(), std::out_of_range, "MatrixBatch select: index out of range");
        return source_[indices_[i]];
    }
   private:
    Source source_;
    std::conditional_t<
      Count == Dynamic, std::vector<std::size_t>, std::array<std::size_t, Count == Dynamic ? 0 : Count>>
      indices_;
};

}   // namespace internals

/// @brief returns a deferred pairwise or broadcast sum using the native element operators
template <typename Lhs, typename Rhs>
    requires(
      internals::batch_operand<Lhs &&> && internals::batch_operand<Rhs &&> &&
      (internals::matrix_batch_expression<Lhs> || internals::matrix_batch_expression<Rhs>) &&
      !std::is_arithmetic_v<std::remove_cvref_t<Lhs>> && !std::is_arithmetic_v<std::remove_cvref_t<Rhs>>)
auto operator+(Lhs&& lhs, Rhs&& rhs) {
    return internals::matrix_batch_binary<
      internals::batch_operand_nested_t<Lhs&&>, internals::batch_operand_nested_t<Rhs&&>, std::plus<>>(
      std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
}

/// @brief returns a deferred pairwise or broadcast difference preserving operand order
template <typename Lhs, typename Rhs>
    requires(
      internals::batch_operand<Lhs &&> && internals::batch_operand<Rhs &&> &&
      (internals::matrix_batch_expression<Lhs> || internals::matrix_batch_expression<Rhs>) &&
      !std::is_arithmetic_v<std::remove_cvref_t<Lhs>> && !std::is_arithmetic_v<std::remove_cvref_t<Rhs>>)
auto operator-(Lhs&& lhs, Rhs&& rhs) {
    return internals::matrix_batch_binary<
      internals::batch_operand_nested_t<Lhs&&>, internals::batch_operand_nested_t<Rhs&&>, std::minus<>>(
      std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
}

/// @brief returns deferred scalar scaling or per-entry matrix products preserving operand order
template <typename Lhs, typename Rhs>
    requires(
      internals::batch_operand<Lhs &&> && internals::batch_operand<Rhs &&> &&
      (internals::matrix_batch_expression<Lhs> || internals::matrix_batch_expression<Rhs>))
auto operator*(Lhs&& lhs, Rhs&& rhs) {
    return internals::matrix_batch_binary<
      internals::batch_operand_nested_t<Lhs&&>, internals::batch_operand_nested_t<Rhs&&>, std::multiplies<>>(
      std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
}

/// @brief returns deferred division of each matrix by a scalar stored by value
template <typename Lhs, typename Scalar>
    requires(
      internals::matrix_batch_expression<Lhs> && internals::batch_operand<Lhs &&> &&
      std::is_arithmetic_v<std::remove_cvref_t<Scalar>>)
auto operator/(Lhs&& lhs, Scalar rhs) {
    return internals::matrix_batch_binary<
      internals::batch_operand_nested_t<Lhs&&>, std::remove_cvref_t<Scalar>, std::divides<>>(
      std::forward<Lhs>(lhs), rhs);
}

/// @brief negates batch coefficients through deferred native scalar multiplication
template <typename Source>
    requires(internals::matrix_batch_expression<Source> && internals::batch_operand<Source &&>)
auto operator-(Source&& source) {
    return std::forward<Source>(source) * -1;
}

/// @brief composes deferred arithmetic, maps and selections with owning evaluation and sequential reductions
template <typename Derived> class MatrixBatchExpr {
   public:
    /// @brief borrows the complete batch expression without changing its evaluation state
    const Derived& derived() const& { return static_cast<const Derived&>(*this); }
    /// @brief prevents a borrowed expression reference from escaping a temporary batch node
    void derived() const&& = delete;
    /// @brief returns an independent owning batch with inferred element structure and explicit execution policy
    /// @details parallel evaluation joins before returning; mapped callables and shared cache access must be
    /// thread-safe
    template <internals::BatchExecutionPolicy ExecutionPolicy = execution_seq_t>
    auto eval(ExecutionPolicy policy = {}) const {
        return internals::evaluate_matrix_batch<typename Derived::MatrixType>(derived(), policy);
    }
    /// @brief creates a deferred map over a persistent batch or expression
    template <typename Function> auto map(Function function) const& {
        return internals::matrix_batch_map<const Derived&, Function>(derived(), std::move(function));
    }
    /// @brief keeps a temporary expression node alive inside the next deferred map
    template <typename Function>
      auto map(Function function) &&
      requires(Derived::NestAsRef == 0) {
          return internals::matrix_batch_map<Derived, Function>(
            std::move(static_cast<Derived&>(*this)), std::move(function));
      }
      /// @brief rejects borrowing an expiring owning batch
      template <typename Function>
      void map(Function) const&& = delete;
    /// @brief materializes each entry's logical squared Frobenius norm as an owning one-by-one matrix
    template <internals::BatchExecutionPolicy ExecutionPolicy = execution_seq_t>
    auto squared_norm(ExecutionPolicy policy = {}) const {
        return map([](const auto& value) { return value.squared_norm(); }).eval(policy);
    }
    /// @brief folds values in increasing index order without materializing an intermediate batch
    template <typename Accumulator, typename Function> Accumulator redux(Accumulator init, Function function) const {
        for (std::size_t i = 0; i < derived().size(); ++i) {
            const auto value = derived()[i];
            init = std::invoke(function, std::move(init), value);
        }
        return init;
    }
    /// @brief copies indices and borrows a persistent collection
    template <typename Indices> auto select(const Indices& indices) const& {
        return internals::matrix_batch_selection<const Derived&, internals::batch_index_extent<Indices>>(
          derived(), indices);
    }
    /// @brief retains a temporary expression while copying the selection indices
    template <typename Indices>
      auto select(const Indices& indices) &&
      requires(Derived::NestAsRef == 0) {
          return internals::matrix_batch_selection<Derived, internals::batch_index_extent<Indices>>(
            std::move(static_cast<Derived&>(*this)), indices);
      }
      /// @brief rejects selections that would borrow a temporary owning batch
      template <typename Indices>
      void select(const Indices&) const&& = delete;
};

}   // namespace fdapde
#endif   // __FDAPDE_LINALG_MATRIX_BATCH_EXPR_H__
