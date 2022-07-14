#ifndef __NURBS_H__
#define __NURBS_H__

#include "header_check.h"

namespace fdapde {

template<int M>
inline double multicontract(const MdArray<double, full_dynamic_extent_t<M>>& weights,
                            const std::array<std::vector<double>, M>& parts) {
    double result = 0.0;
    std::array<int, M> sizes;
    std::size_t total = 1;

    for (int i = 0; i < M; ++i) {
        sizes[i] = static_cast<int>(parts[i].size());
        total *= sizes[i];
    }

    for (std::size_t idx = 0; idx < total; ++idx) {
        std::array<int, M> multi_idx;
        std::size_t temp = idx;
        for (int i = M - 1; i >= 0; --i) {
            multi_idx[i] = temp % sizes[i];
            temp /= sizes[i];
        }

        double val = weights(multi_idx);
        for (int i = 0; i < M; ++i)
            val *= parts[i][multi_idx[i]];

        result += val;
    }

    return result;
}

template<int M>
class Nurbs {
public:
    using Scalar     = double;
    using VectorType = Eigen::Matrix<Scalar, M, 1>;
    using MatrixType = Eigen::Matrix<Scalar, M, M>;

private:
    std::array<std::shared_ptr<BSplineBasis>, M> spline_basis_;
    MdArray<double, full_dynamic_extent_t<M>> weights_;
    std::array<int, M> index_, degree_, extents_, periodicity_;
    std::array<std::size_t, M> minIdx_;
    double num0_ = 0.0;

public:
    Nurbs() = default;

    // General constructor
    template<typename KnotsVectorType>
    Nurbs(std::array<KnotsVectorType, M> knots,
          MdArray<double, full_dynamic_extent_t<M>>& weights,
          std::array<int, M> index,
          std::array<int, M> degree,
          std::array<int, M> periodicity = {})
        : index_(std::move(index)), degree_(std::move(degree)), periodicity_(std::move(periodicity)) {

        for (int i = 0; i < M; ++i) {
            std::vector<double> padded_knots = pad_knots(knots[i], degree_[i]);
            spline_basis_[i] = std::make_shared<BSplineBasis>(padded_knots, degree_[i], periodicity_[i]);
        }

        initialize_weights_block(weights);
    }

    // 1D shortcut
    template<typename KnotVec>
    Nurbs(KnotVec& knots,
          MdArray<double, full_dynamic_extent_t<M>>& weights,
          int index, int degree, bool periodic = false)
        requires(M == 1)
        : Nurbs(std::array<KnotVec, 1>{knots}, weights,
                std::array<int, 1>{index},
                std::array<int, 1>{degree},
                std::array<int, 1>{periodic ? 1 : 0}) {}

    // Shared basis constructor
    Nurbs(std::array<std::shared_ptr<BSplineBasis>, M> basis,
          MdArray<double, full_dynamic_extent_t<M>>& weights,
          std::array<int, M> index)
        : spline_basis_(std::move(basis)), index_(std::move(index)) {
        for (int i = 0; i < M; ++i) {
            degree_[i]     = spline_basis_[i]->degree();
            periodicity_[i] = spline_basis_[i]->periodicity();
        }
        initialize_weights_block(weights);
    }

    // Operator(): evaluate the scalar NURBS basis function at point p
    Scalar operator()(const VectorType& p) const {
        std::array<std::vector<double>, M> B;
        Scalar num = num0_;

        for (int i = 0; i < M; ++i) {
            B[i] = spline_basis_[i]->evaluate_basis(p(i));
        }

        std::array<std::vector<double>, M> parts;
        for (int i = 0; i < M; ++i) {
            parts[i].resize(extents_[i]);
            for (int j = 0; j < extents_[i]; ++j)
                parts[i][j] = B[i][minIdx_[i] + j];
            num *= parts[i][index_[i] - minIdx_[i]];
        }

        Scalar den = multicontract<M>(weights_, parts);
        return (den == 0.0) ? 0.0 : num / den;
    }

    // Gradient (on demand)
    VectorType gradient(const VectorType& p) const {
        VectorType grad;
        std::array<std::vector<double>, M> B, dB;
        for (int i = 0; i < M; ++i) {
            auto eval = spline_basis_[i]->evaluate_der_basis(p(i), 1);
            B[i]  = eval[0];
            dB[i] = eval[1];
            //grad(i) = 0.0;
        }

        std::array<std::vector<double>, M> parts;
        Scalar num = num0_;
        for (int i = 0; i < M; ++i) {
            parts[i].resize(extents_[i]);
            for (int j = 0; j < extents_[i]; ++j)
                parts[i][j] = B[i][minIdx_[i] + j];
            num *= parts[i][index_[i] - minIdx_[i]];
        }

        Scalar den = multicontract<M>(weights_, parts);

        for (int i = 0; i < M; ++i) {
            Scalar num_i = num / parts[i][index_[i] - minIdx_[i]] * dB[i][index_[i]];
            auto parts_d = parts;
            for (int j = 0; j < extents_[i]; ++j)
                parts_d[i][j] = dB[i][minIdx_[i] + j];
            Scalar den_i = multicontract<M>(weights_, parts_d);

            grad(i) = (num_i * den - num * den_i) / (den * den);
        }
        

        return grad;
    }

    MatrixType hessian(const VectorType& p) const {
        MatrixType H;
        H.setZero();
    
        std::array<std::vector<double>, M> B, dB, ddB;
        for (int i = 0; i < M; ++i) {
            auto eval = spline_basis_[i]->evaluate_der_basis(p(i), 2);
            B[i]   = eval[0];  // basis values
            dB[i]  = eval[1];  // first derivatives
            ddB[i] = eval[2];  // second derivatives
        }
    
        // Evaluate parts for the value computation
        std::array<std::vector<double>, M> parts;
        Scalar num = num0_;
        for (int i = 0; i < M; ++i) {
            parts[i].resize(extents_[i]);
            for (int j = 0; j < extents_[i]; ++j)
                parts[i][j] = B[i][minIdx_[i] + j];
            num *= B[i][index_[i]];
        }
    
        Scalar den = multicontract<M>(weights_, parts);
    
        // Compute first derivatives
        VectorType num_grad, den_grad;
        for (int i = 0; i < M; ++i) {
            Scalar partial = dB[i][index_[i]];
            Scalar basis_val = B[i][index_[i]];
            num_grad(i) = num / basis_val * partial;
    
            auto parts_d = parts;
            for (int j = 0; j < extents_[i]; ++j)
                parts_d[i][j] = dB[i][minIdx_[i] + j];
    
            den_grad(i) = multicontract<M>(weights_, parts_d);
        }
    
        // Compute Hessian
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < M; ++j) {
                Scalar num_ij;
                if (i == j) {
                    Scalar second = ddB[i][index_[i]];
                    Scalar basis_val = B[i][index_[i]];
                    num_ij = num / basis_val * second;
                } else {
                    Scalar bi = B[i][index_[i]];
                    Scalar bj = B[j][index_[j]];
                    Scalar dpi = dB[i][index_[i]];
                    Scalar dpj = dB[j][index_[j]];
                    num_ij = num / (bi * bj) * dpi * dpj;
                }
    
                // Build parts_dd for ∂²den/∂xi∂xj
                auto parts_dd = parts;
    
                if (i == j) {
                    for (int k = 0; k < extents_[i]; ++k)
                        parts_dd[i][k] = ddB[i][minIdx_[i] + k];
                } else {
                    for (int k = 0; k < extents_[i]; ++k)
                        parts_dd[i][k] = dB[i][minIdx_[i] + k];
                    for (int k = 0; k < extents_[j]; ++k)
                        parts_dd[j][k] = dB[j][minIdx_[j] + k];
                }
    
                Scalar den_ij = multicontract<M>(weights_, parts_dd);
    
                Scalar term1 = num_ij * den;
                Scalar term2 = num_grad(i) * den_grad(j);
                Scalar term3 = num_grad(j) * den_grad(i);
                Scalar term4 = num * den_ij;
                Scalar term5 = 2.0 * den_grad(i) * den_grad(j) * num;
    
                H(i, j) = (term1 * den - term2 - term3 - term4 + term5) / (den * den * den);
            }
        }
    
        return H;
    }

    constexpr const std::array<int, M>& degree() const { return degree_; }
    constexpr const std::array<int, M>& index() const { return index_; }
    constexpr const std::array<int, M>& periodicity() const { return periodicity_; }
    constexpr const MdArray<double, full_dynamic_extent_t<M>>& weights() const { return weights_; }
    constexpr const std::array<std::shared_ptr<BSplineBasis>, M>& spline_basis() const { return spline_basis_; }

private:
    void initialize_weights_block(const MdArray<double, full_dynamic_extent_t<M>>& full_weights) {
        std::array<std::size_t, M> maxIdx;
        for (int i = 0; i < M; ++i) {
            int deg = degree_[i];
            minIdx_[i] = (index_[i] >= deg) ? (index_[i] - deg) : 0;
            extents_[i] = std::min(index_[i] + deg + 1, static_cast<int>(full_weights.extent(i))) - int(minIdx_[i]);
            maxIdx[i] = minIdx_[i] + extents_[i] - 1;
        }
        weights_.resize(extents_);
        weights_ = full_weights.block(minIdx_, maxIdx);
        num0_ = full_weights(index_);
    }
};

} // namespace fdapde

#endif // __NURBS_H__