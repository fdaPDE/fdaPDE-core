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

#ifndef __FDAPDE_LINALG_MATRIX_BATCH_H__
#define __FDAPDE_LINALG_MATRIX_BATCH_H__

#include <span>

#include "header_check.h"
#include "matrix_batch_expr.h"

namespace fdapde {
namespace internals {

/// @brief assigns cache-free state to ordinary matrices
template <typename Matrix, bool = requires { typename Matrix::CachePolicy; }> struct batch_cache_policy {
    using type = Cache::None;
};
/// @brief preserves the selected cache policy of a structured element
template <typename Matrix> struct batch_cache_policy<Matrix, true> {
    using type = typename Matrix::CachePolicy;
};

/// @brief supplies an unused slot type for ordinary matrices without cache metadata
template <typename Matrix, typename = void> struct batch_cache_slot {
    using type = spd_cache_slot<typename Matrix::Scalar, Matrix::Rows, Cache::None>;
};
/// @brief preserves the cache slot representation chosen by a structured owner
template <typename Matrix> struct batch_cache_slot<Matrix, std::void_t<typename Matrix::CacheSlot>> {
    using type = typename Matrix::CacheSlot;
};

/// @brief owns all cache data and slot bindings for a uniformly shaped batch
template <typename Slot> struct batch_cache_storage {
    using Scalar = typename Slot::Scalar;
    std::vector<Scalar> values;
    std::vector<Slot> slots;
    std::vector<const Slot*> pointers;
    /// @brief creates empty aggregate cache storage
    batch_cache_storage() = default;
    /// @brief deep-copies cached quantities and readiness states while rebuilding all slot pointers
    batch_cache_storage(const batch_cache_storage& other) : values(other.values.size()) {
        bind_(other.slots.size(), other.slots.empty() ? 0 : other.slots.front().rows());
        for (std::size_t i = 0; i < slots.size(); ++i) slots[i].copy_from(other.slots[i]);
    }
    /// @brief transfers buffers together so their existing pointers remain associated
    batch_cache_storage(batch_cache_storage&&) = default;
    /// @brief replaces buffers by moving their associated bindings together
    batch_cache_storage& operator=(batch_cache_storage&&) = default;
    /// @brief allocates a fixed number of aggregate buffers independent of the element count
    void allocate(std::size_t count, int order) {
        const auto stride = Slot::scalar_count(order);
        fdapde_strong_assert(
          count <= values.max_size() / stride, std::length_error, "MatrixBatch: cache size overflow");
        values.resize(count * stride);
        bind_(count, order);
    }
   private:
    /// @brief binds contiguous descriptors and the pointer table to corresponding scalar slots
    void bind_(std::size_t count, int order) {
        slots.reserve(count);
        pointers.reserve(count);
        const auto stride = Slot::scalar_count(order);
        for (std::size_t i = 0; i < count; ++i) {
            slots.emplace_back(values.data() + i * stride, order);
            pointers.push_back(&slots.back());
        }
    }
};

}   // namespace internals

/// @brief owns uniformly shaped matrices in contiguous row-major coefficient rows and aggregate optional cache buffers
/// @details replacement, move and swap invalidate views; no resize operation is provided
template <typename MatrixType_, int StorageOrder_>
class MatrixBatch : public MatrixBatchExpr<MatrixBatch<MatrixType_, StorageOrder_>> {
   public:
    using MatrixType = MatrixType_;
    using Scalar = typename MatrixType::Scalar;
    using View = typename MatrixType::View;
    using ConstView = typename MatrixType::ConstView;
    using CachePolicy = typename internals::batch_cache_policy<MatrixType>::type;
    static constexpr int NestAsRef = 1;
    static constexpr int StorageOrder = StorageOrder_;
    static_assert(MatrixType::NestAsRef != 0, "MatrixBatch requires owning matrix elements");
    static_assert(
      StorageOrder == RowMajor && MatrixType::StorageOrder == RowMajor, "MatrixBatch supports row-major storage only");
    static_assert(
      !std::is_const_v<Scalar> && std::is_arithmetic_v<Scalar>,
      "MatrixBatch requires owning arithmetic matrix elements");

    /// @brief constructs fixed-shape elements as identities for SPD and orthogonal matrices or zeros for ordinary
    /// matrices
    explicit MatrixBatch(std::size_t count = 0)
        requires(MatrixType::Rows != Dynamic && MatrixType::Cols != Dynamic)
        : MatrixBatch(count, MatrixType::Rows, MatrixType::Cols) { }
    /// @brief allocates uniformly shaped elements and initializes known identity caches without decompositions
    MatrixBatch(std::size_t count, int rows, int cols) {
        allocate_(count, rows, cols);
        if constexpr (SPDLike<MatrixType> || is_orthogonal_matrix_v<MatrixType>) {
            for (std::size_t i = 0; i < count_; ++i) {
                for (int j = 0; j < rows_; ++j) {
                    const auto index =
                      SPDLike<MatrixType> ? std::size_t(j) * (j + 1) / 2 + j : std::size_t(j) * cols_ + j;
                    data_[i * stride_ + index] = Scalar(1);
                }
                if constexpr (CachePolicy::Flags != 0) cache_.slots[i].set_identity();
            }
        }
    }
    /// @brief deep-copies coefficients and rebuilds independent cache slot bindings
    MatrixBatch(const MatrixBatch&) = default;
    /// @brief transfers coefficient and cache buffers together, leaving an empty source with its element shape
    MatrixBatch(MatrixBatch&& other) noexcept :
        count_(std::exchange(other.count_, 0)),
        rows_(other.rows_),
        cols_(other.cols_),
        stride_(other.stride_),
        data_(std::move(other.data_)),
        cache_(std::move(other.cache_)) {
        other.data_.clear();
        if constexpr (CachePolicy::Flags != 0) {
            other.cache_.pointers.clear();
            other.cache_.slots.clear();
            other.cache_.values.clear();
        }
    }
    /// @brief prepares a complete independent copy before replacing this batch
    MatrixBatch& operator=(const MatrixBatch& other) & {
        if (this != &other) {
            MatrixBatch candidate(other);
            swap(candidate);
        }
        return *this;
    }
    /// @brief transfers a complete batch through a nonthrowing buffer swap
    MatrixBatch& operator=(MatrixBatch&& other) & noexcept {
        if (this != &other) {
            MatrixBatch candidate(std::move(other));
            swap(candidate);
        }
        return *this;
    }
    /// @brief constructs directly from supplied values or a deferred expression, evaluating each element once
    template <typename Source>
        requires requires(const Source& source) {
            source.size();
            source[0].rows();
            source[0].cols();
        }
    explicit MatrixBatch(const Source& source) {
        const auto count = static_cast<std::size_t>(source.size());
        if (count == 0) {
            if constexpr (requires {
                              source.rows();
                              source.cols();
                          })
                allocate_(0, source.rows(), source.cols());
            else
                allocate_(0, MatrixType::Rows, MatrixType::Cols);
        } else {
            const auto first = source[0];
            allocate_(count, first.rows(), first.cols());
            store_(0, first);
            for (std::size_t i = 1; i < count; ++i) {
                const auto value = source[i];
                store_(i, value);
            }
        }
    }
    /// @brief replaces this collection only after all source values have been materialized successfully
    template <typename Source> MatrixBatch& operator=(const MatrixBatchExpr<Source>& source) & {
        return assign(source);
    }
    /// @brief evaluates into the destination element type and cache policy before replacing the entire collection
    template <typename Source, internals::BatchExecutionPolicy ExecutionPolicy = execution_seq_t>
    MatrixBatch& assign(const MatrixBatchExpr<Source>& source, ExecutionPolicy policy = {}) & {
        auto candidate = internals::evaluate_matrix_batch<MatrixType>(source.derived(), policy);
        swap(candidate);
        return *this;
    }
    /// @brief adds compatible batch or broadcast entries with alias-safe sequential replacement
    template <typename Rhs>
        requires requires(MatrixBatch& batch, const Rhs& rhs) {
            batch + rhs;
            requires internals::batch_compound_compatible_v<MatrixType, typename decltype(batch + rhs)::MatrixType>;
        }
    MatrixBatch& operator+=(const Rhs& rhs) & {
        return assign(*this + rhs);
    }
    /// @brief subtracts compatible batch or broadcast entries with alias-safe sequential replacement
    template <typename Rhs>
        requires requires(MatrixBatch& batch, const Rhs& rhs) {
            batch - rhs;
            requires internals::batch_compound_compatible_v<MatrixType, typename decltype(batch - rhs)::MatrixType>;
        }
    MatrixBatch& operator-=(const Rhs& rhs) & {
        return assign(*this - rhs);
    }
    /// @brief scales or multiplies entries when the result retains the destination structure
    template <typename Rhs>
        requires requires(MatrixBatch& batch, const Rhs& rhs) {
            batch * rhs;
            requires internals::batch_compound_compatible_v<MatrixType, typename decltype(batch * rhs)::MatrixType>;
        }
    MatrixBatch& operator*=(const Rhs& rhs) & {
        return assign(*this * rhs);
    }
    /// @brief divides entries by a scalar with alias-safe sequential replacement
    template <typename Scalar>
        requires(
          std::is_arithmetic_v<Scalar> &&
          internals::batch_compound_compatible_v<
            MatrixType, typename decltype(std::declval<MatrixBatch&>() / std::declval<Scalar>())::MatrixType>)
    MatrixBatch& operator/=(Scalar scalar) & {
        return assign(*this / scalar);
    }
    /// @brief exchanges complete buffer ownership and their associated shape metadata
    void swap(MatrixBatch& other) noexcept {
        using std::swap;
        swap(count_, other.count_);
        swap(rows_, other.rows_);
        swap(cols_, other.cols_);
        swap(stride_, other.stride_);
        data_.swap(other.data_);
        if constexpr (CachePolicy::Flags != 0) {
            cache_.values.swap(other.cache_.values);
            cache_.slots.swap(other.cache_.slots);
            cache_.pointers.swap(other.cache_.pointers);
        }
    }
    /// @brief returns the number of uniformly shaped matrix elements
    std::size_t size() const { return count_; }
    /// @brief reports whether the collection contains no matrices
    bool empty() const { return count_ == 0; }
    /// @brief materializes each element's Frobenius norm without preparing an input spectral cache
    template <internals::BatchExecutionPolicy ExecutionPolicy = execution_seq_t>
        requires std::floating_point<Scalar>
    auto norm(ExecutionPolicy policy = {}) const {
        using Result = Matrix<Scalar, 1, 1>;
        return materialize_<Result>(1, 1, policy, [](const auto& point) { return Result(point.norm()); });
    }
    /// @brief materializes each square element's main diagonal as an independent column vector
    template <internals::BatchExecutionPolicy ExecutionPolicy = execution_seq_t>
        requires(MatrixType::Rows == Dynamic || MatrixType::Cols == Dynamic || MatrixType::Rows == MatrixType::Cols)
    auto diagonal(ExecutionPolicy policy = {}) const {
        fdapde_strong_assert(rows_ == cols_, std::invalid_argument, "MatrixBatch diagonal requires square elements");
        using Result = Vector<Scalar, MatrixType::Rows>;
        return materialize_<Result>(rows_, 1, policy, [](const auto& point) { return Result(point.diagonal()); });
    }
    /// @brief materializes one matrix logarithm per SPD element with the requested symmetric output cache
    /// @details defaults to sequential evaluation; parallel evaluation joins all work before returning or rethrowing
    template <typename OutputPolicy = Cache::None, internals::BatchExecutionPolicy ExecutionPolicy = execution_seq_t>
        requires SPDLike<MatrixType>
    auto log(ExecutionPolicy policy = {}) const {
        using Result = SymmetricMatrix<Scalar, MatrixType::Rows, OutputPolicy>;
        return materialize_<Result>(
          rows_, cols_, policy, [](const auto& point) { return point.template log<OutputPolicy>(); });
    }
    /// @brief materializes independent eigenvalue vectors while reusing each symmetric or SPD element's cache
    /// @details preserves element order and does not sort eigenvalues; input coefficients must remain unchanged
    template <internals::BatchExecutionPolicy ExecutionPolicy = execution_seq_t>
        requires(SPDLike<MatrixType> || NativeSymmetricLike<MatrixType>)
    auto eigenvalues(ExecutionPolicy policy = {}) const {
        using Result = Vector<Scalar, MatrixType::Rows>;
        return materialize_<Result>(rows_, 1, policy, [](const auto& point) { return point.eigenvalues(); });
    }
    /// @brief materializes owning orthogonal bases while reusing each symmetric or SPD element's spectral cache
    /// @details columns correspond to eigenvalues of the unchanged element; signs and repeated-eigenvalue bases may
    /// vary
    template <internals::BatchExecutionPolicy ExecutionPolicy = execution_seq_t>
        requires(std::floating_point<Scalar> && (SPDLike<MatrixType> || NativeSymmetricLike<MatrixType>))
    auto eigenvectors(ExecutionPolicy policy = {}) const {
        using Result = OrthogonalMatrix<Scalar, MatrixType::Rows>;
        return materialize_<Result>(rows_, cols_, policy, [](const auto& point) { return point.eigenvectors(); });
    }
    /// @brief materializes checked SPD exponentials from symmetric elements using available input spectral caches
    template <typename OutputPolicy = Cache::None, internals::BatchExecutionPolicy ExecutionPolicy = execution_seq_t>
        requires(std::floating_point<Scalar> && (SPDLike<MatrixType> || NativeSymmetricLike<MatrixType>))
    auto exp(ExecutionPolicy policy = {}) const {
        using Result = SPDMatrix<Scalar, MatrixType::Rows, OutputPolicy>;
        return materialize_<Result>(
          rows_, cols_, policy, [](const auto& point) { return point.template exp<OutputPolicy>(); });
    }
    /// @brief materializes checked SPD square roots using retained square-root factors or input eigenpairs
    template <typename OutputPolicy = Cache::None, internals::BatchExecutionPolicy ExecutionPolicy = execution_seq_t>
        requires SPDLike<MatrixType>
    auto sqrt(ExecutionPolicy policy = {}) const {
        using Result = SPDMatrix<Scalar, MatrixType::Rows, OutputPolicy>;
        return materialize_<Result>(
          rows_, cols_, policy, [](const auto& point) { return point.template sqrt<OutputPolicy>(); });
    }
    /// @brief materializes checked SPD inverse square roots using retained factors or input eigenpairs
    template <typename OutputPolicy = Cache::None, internals::BatchExecutionPolicy ExecutionPolicy = execution_seq_t>
        requires SPDLike<MatrixType>
    auto inv_sqrt(ExecutionPolicy policy = {}) const {
        using Result = SPDMatrix<Scalar, MatrixType::Rows, OutputPolicy>;
        return materialize_<Result>(
          rows_, cols_, policy, [](const auto& point) { return point.template inv_sqrt<OutputPolicy>(); });
    }
    /// @brief materializes structure-preserving inverses using each element's native algorithm and retained factors
    /// @details symmetric and SPD outputs accept an independent cache policy; execution defaults to sequential
    template <typename OutputPolicy = Cache::None, internals::BatchExecutionPolicy ExecutionPolicy = execution_seq_t>
        requires(
          std::floating_point<Scalar> &&
          (MatrixType::Rows == Dynamic || MatrixType::Cols == Dynamic || MatrixType::Rows == MatrixType::Cols) &&
          (SPDLike<MatrixType> || NativeSymmetricLike<MatrixType> || std::same_as<OutputPolicy, Cache::None>))
    auto inv(ExecutionPolicy policy = {}) const {
        fdapde_strong_assert(rows_ == cols_, std::invalid_argument, "MatrixBatch inv requires square elements");
        const auto invert = [](const auto& point) {
            if constexpr (SPDLike<MatrixType> || NativeSymmetricLike<MatrixType>)
                return point.template inv<OutputPolicy>();
            else
                return point.inv();
        };
        using Result = typename internals::batch_result_owner<decltype(invert(std::declval<ConstView>()))>::type;
        return materialize_<Result>(rows_, cols_, policy, [&](const auto& point) { return Result(invert(point)); });
    }
    /// @brief materializes each square element's diagonal sum as an independent one-by-one matrix
    /// @details defaults to sequential evaluation and never prepares an input spectral cache
    template <internals::BatchExecutionPolicy ExecutionPolicy = execution_seq_t>
        requires(MatrixType::Rows == Dynamic || MatrixType::Cols == Dynamic || MatrixType::Rows == MatrixType::Cols)
    auto trace(ExecutionPolicy policy = {}) const {
        fdapde_strong_assert(rows_ == cols_, std::invalid_argument, "MatrixBatch trace requires square elements");
        using Result = Matrix<Scalar, 1, 1>;
        return materialize_<Result>(1, 1, policy, [](const auto& point) { return Result(point.trace()); });
    }
    /// @brief materializes each square element's determinant as an independent one-by-one matrix
    /// @details uses the element's existing determinant kernel and defaults to sequential evaluation
    template <internals::BatchExecutionPolicy ExecutionPolicy = execution_seq_t>
        requires(MatrixType::Rows == Dynamic || MatrixType::Cols == Dynamic || MatrixType::Rows == MatrixType::Cols)
    auto determinant(ExecutionPolicy policy = {}) const {
        fdapde_strong_assert(rows_ == cols_, std::invalid_argument, "MatrixBatch determinant requires square elements");
        using Result = Matrix<Scalar, 1, 1>;
        return materialize_<Result>(1, 1, policy, [](const auto& point) { return Result(point.determinant()); });
    }
    /// @brief returns the row count of each element including an empty batch
    int rows() const { return rows_; }
    /// @brief returns the column count of each element including an empty batch
    int cols() const { return cols_; }
    /// @brief returns the number of physically stored coefficients per matrix
    std::size_t coefficient_stride() const { return stride_; }
    /// @brief borrows read-only contiguous coefficient rows, including an empty span for an empty batch
    std::span<const Scalar> coefficients() const& { return data_; }
    /// @brief exposes writable coefficient rows when no structured validation or cache tracking is required
    std::span<Scalar> coefficients() &
        requires(!SPDLike<MatrixType> && !is_orthogonal_matrix_v<MatrixType> && CachePolicy::Flags == 0)
    {
        return data_;
    }
    /// @brief forbids coefficient spans escaping an expiring owner
    void coefficients() const&& = delete;
    /// @brief borrows the read-only contiguous table of cache slot pointers for layout inspection
    auto cache_pointers() const&
        requires(CachePolicy::Flags != 0)
    {
        using Slot = typename MatrixType::CacheSlot;
        return std::span<const Slot* const>(cache_.pointers);
    }
    /// @brief prevents cache pointers from escaping an expiring owner
    void cache_pointers() const&&
        requires(CachePolicy::Flags != 0)
    = delete;
    /// @brief returns a mutable value view without copying the element or cache
    View operator[](std::size_t i) & {
        check_index_(i);
        return view_<View>(i);
    }
    /// @brief returns a read-only value view without copying the element or cache
    ConstView operator[](std::size_t i) const& {
        check_index_(i);
        return view_<ConstView>(i);
    }
    /// @brief prevents an element view from escaping a temporary owner
    void operator[](std::size_t) const&& = delete;
   private:
    template <typename Result, internals::BatchExecutionPolicy ExecutionPolicy, typename Function>
    friend MatrixBatch<Result> internals::generate_matrix_batch(
      std::size_t count, int rows, int cols, ExecutionPolicy policy, const Function& function);

    /// @brief allocates private result buffers for complete slot initialization without intermediate identities
    MatrixBatch(std::size_t count, int rows, int cols, internals::unchecked_t) { allocate_(count, rows, cols); }

    /// @brief fills disjoint result slots with the requested output shape, including empty batches
    template <typename Result, internals::BatchExecutionPolicy ExecutionPolicy, typename Function>
    auto materialize_(int rows, int cols, ExecutionPolicy policy, const Function& function) const {
        return internals::generate_matrix_batch<Result>(
          count_, rows, cols, policy, [&](std::size_t i) { return function((*this)[i]); });
    }

    using CacheStorage = std::conditional_t<
      CachePolicy::Flags == 0, internals::empty_spd_cache<2>,
      internals::batch_cache_storage<typename internals::batch_cache_slot<MatrixType>::type>>;
    /// @brief validates dimensions and bounded allocation sizes before allocating aggregate storage
    void allocate_(std::size_t count, int rows, int cols) {
        fdapde_strong_assert(
          rows > 0 && cols > 0 && (MatrixType::Rows == Dynamic || rows == MatrixType::Rows) &&
            (MatrixType::Cols == Dynamic || cols == MatrixType::Cols),
          std::invalid_argument, "MatrixBatch: incompatible element dimensions");
        fdapde_strong_assert(
          std::int64_t(rows) * cols <= std::numeric_limits<int>::max(), std::length_error,
          "MatrixBatch: element size exceeds supported range");
        if constexpr (
          is_symmetric_matrix_v<MatrixType> || is_diagonal_matrix_v<MatrixType> ||
          is_skew_symmetric_matrix_v<MatrixType> || is_orthogonal_matrix_v<MatrixType>)
            fdapde_strong_assert(
              rows == cols, std::invalid_argument, "MatrixBatch: structured elements must be square");
        rows_ = rows;
        cols_ = cols;
        count_ = count;
        if constexpr (is_diagonal_matrix_v<MatrixType>)
            stride_ = rows;
        else if constexpr (is_symmetric_matrix_v<MatrixType>)
            stride_ = std::size_t(rows) * (rows + 1) / 2;
        else if constexpr (is_skew_symmetric_matrix_v<MatrixType>)
            stride_ = std::size_t(rows) * (rows - 1) / 2;
        else
            stride_ = std::size_t(rows) * cols;
        fdapde_strong_assert(
          stride_ == 0 || count <= data_.max_size() / stride_, std::length_error,
          "MatrixBatch: coefficient size overflow");
        data_.resize(count * stride_);
        if constexpr (CachePolicy::Flags != 0) cache_.allocate(count, rows);
    }
    /// @brief rejects out-of-range element access before forming any coefficient pointer
    void check_index_(std::size_t i) const {
        fdapde_strong_assert(i < count_, std::out_of_range, "MatrixBatch: index out of range");
    }
    /// @brief creates an element view using its native structure-specific constructor
    template <typename Target> Target view_(std::size_t i) const {
        using Pointer = std::conditional_t<std::same_as<Target, ConstView>, const Scalar*, Scalar*>;
        Pointer data = const_cast<Pointer>(data_.data());
        if (stride_ != 0) data += i * stride_;
        if constexpr (SPDLike<MatrixType> || RotationLike<MatrixType> || NativeSymmetricLike<MatrixType>) {
            if constexpr (CachePolicy::Flags == 0) {
                if constexpr (NativeSymmetricLike<MatrixType>)
                    return Target(data, rows_, static_cast<typename MatrixType::CacheSlot*>(nullptr));
                else
                    return Target(data, rows_, {});
            } else
                return Target(data, rows_, const_cast<typename MatrixType::CacheSlot*>(cache_.pointers[i]));
        } else if constexpr (is_orthogonal_matrix_v<MatrixType>)
            return Target(data, rows_, cols_, typename Target::verified_t {});
        else if constexpr (is_diagonal_matrix_v<MatrixType>) {
            if constexpr (MatrixType::Rows == Dynamic)
                return Target(data, rows_);
            else
                return Target(data);
        } else
            return Target(data, rows_, cols_);
    }
    /// @brief initializes a coefficient row directly from a validated candidate without an identity intermediate
    template <typename Value> void store_(std::size_t i, const Value& value) {
        fdapde_strong_assert(
          value.rows() == rows_ && value.cols() == cols_, std::invalid_argument,
          "MatrixBatch: nonuniform element shape");
        if constexpr (SPDLike<MatrixType> || RotationLike<MatrixType> || NativeSymmetricLike<MatrixType>) {
            const MatrixType candidate(value);
            std::copy_n(candidate.data(), stride_, data_.data() + i * stride_);
            if constexpr (CachePolicy::Flags != 0) cache_.slots[i].copy_from(candidate.cache());
        } else if constexpr (is_orthogonal_matrix_v<MatrixType>) {
            const MatrixType candidate(value);
            std::copy_n(candidate.data().data(), stride_, data_.data() + i * stride_);
        } else {
            if constexpr (is_skew_symmetric_matrix_v<MatrixType>)
                static_assert(
                  is_skew_symmetric_matrix_v<Value>, "MatrixBatch skew elements require skew-symmetric expressions");
            view_<View>(i) = value;
        }
    }
    std::size_t count_ = 0;
    int rows_ = MatrixType::Rows;
    int cols_ = MatrixType::Cols;
    std::size_t stride_ = 0;
    std::vector<Scalar> data_;
    [[no_unique_address]] CacheStorage cache_;
};

namespace internals {
/// @brief allocates unexposed result slots and publishes them only after every successful evaluation has completed
/// @details parallel callbacks must safely share their captures; exceptions are rethrown after submitted work joins
template <typename Result, BatchExecutionPolicy ExecutionPolicy, typename Function>
MatrixBatch<Result>
generate_matrix_batch(std::size_t count, int rows, int cols, ExecutionPolicy, const Function& function) {
    if constexpr (std::same_as<ExecutionPolicy, execution_par_t>) {
        fdapde_strong_assert(
          count <= static_cast<std::size_t>(std::numeric_limits<int>::max()), std::length_error,
          "MatrixBatch: parallel element count exceeds the execution index range");
    }
    MatrixBatch<Result> result(count, rows, cols, unchecked);
    if (count == 0) return result;
    const auto evaluate = [&](std::size_t i) { result.store_(i, function(i)); };
    if constexpr (std::same_as<ExecutionPolicy, execution_par_t>)
        parallel_for(0, static_cast<int>(count), [&](int i) { evaluate(static_cast<std::size_t>(i)); });
    else
        for (std::size_t i = 0; i < count; ++i) evaluate(i);
    return result;
}

/// @brief evaluates the first entry once to infer shape and publishes only a completely initialized result
template <typename Result, BatchExecutionPolicy ExecutionPolicy, typename Source>
MatrixBatch<Result> evaluate_matrix_batch(const Source& source, ExecutionPolicy policy) {
    const auto count = source.size();
    if (count == 0)
        return generate_matrix_batch<Result>(
          0, source.rows(), source.cols(), policy, [&](std::size_t i) { return Result(source[i]); });
    const auto first = source[0];
    return generate_matrix_batch<Result>(count, first.rows(), first.cols(), policy, [&](std::size_t i) {
        if (i == 0) return Result(first);
        return Result(source[i]);
    });
}
}   // namespace internals

}   // namespace fdapde
#endif   // __FDAPDE_LINALG_MATRIX_BATCH_H__
