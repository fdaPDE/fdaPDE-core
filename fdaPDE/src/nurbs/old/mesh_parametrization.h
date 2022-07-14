#ifndef __MESH_PARAMETRIZATION_H__
#define __MESH_PARAMETRIZATION_H__

#include "nurbs_basis.h"

namespace fdapde {

template<int M>
class MeshParametrizationBase {
protected:
    NurbsBasis<M> nurbs_basis_;
    MdArray<double, full_dynamic_extent_t<M>> control_points_;

    MeshParametrizationBase() = default;

    MeshParametrizationBase(std::array<std::vector<double>, M>& knots, MdArray<double, full_dynamic_extent_t<M>>& weights,
                            MdArray<double, full_dynamic_extent_t<M>>& control_points, int order)
        : nurbs_basis_(knots, weights, order), control_points_(control_points) {}

    MeshParametrizationBase(std::array<std::vector<double>, M>& knots, MdArray<double, full_dynamic_extent_t<M+1>>& weights,
                            MdArray<double, full_dynamic_extent_t<M>>& control_points, int order, size_t i)
        : MeshParametrizationBase(knots, weights, control_points.template slice<M>(i), order) {}
};

template<int M> //removed the template <int N> from here
class MeshParametrization : public MatrixBase<M, MeshParametrization<M>>, public MeshParametrizationBase<M> {
public:
    using MeshParametrizationBase<M>::MeshParametrizationBase; // Inherit constructors

    inline double operator()(const std::array<double, M>& u) const {
        double x = 0.0;
        for (const auto& nurb : this->nurbs_basis_) {
            x += nurb(x) * this->control_points_(nurb.index());
        }
        return x;
    }
};

template<int M>
class MeshParametrizationDerivative : public MatrixBase<M, MeshParametrizationDerivative<M>>, public MeshParametrizationBase<M> {
private:
    std::size_t derivative_index_;

public:
    MeshParametrizationDerivative(std::array<std::vector<double>, M>& knots, MdArray<double, full_dynamic_extent_t<M>>& weights,
                                  MdArray<double, full_dynamic_extent_t<M>>& control_points, int order, std::size_t derivative_index)
        : MeshParametrizationBase<M>(knots, weights, control_points, order), derivative_index_(derivative_index) {}

    MeshParametrizationDerivative(std::array<std::vector<double>, M>& knots, MdArray<double, full_dynamic_extent_t<M+1>>& weights,
                                  MdArray<double, full_dynamic_extent_t<M>>& control_points, int order, std::size_t derivative_index, size_t i)
        : MeshParametrizationBase<M>(knots, weights, control_points, order, i), derivative_index_(derivative_index) {}

    inline double operator()(const std::array<double, M>& u) const {
        double x = 0.0;
        for (const auto& nurb : this->nurbs_basis_) {
            x += nurb.derive(derivative_index_)(x) * this->control_points_(nurb.index());
        }
        return x;
    }
};

}; // namespace fdapde

#endif // __MESH_PARAMETRIZATION_H__