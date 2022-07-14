#ifndef __ELEMENT_IGA_H__
#define __ELEMENT_IGA_H__

#include "mesh_parametrization.h"

namespace fdapde {

template<int M, int N>
class ElementIGA : { //in the parametric domain

    using ParametrizationType = VectorField<M, N, MeshParametrization<M>>;
    using GradientType = MatrixField<M, N, M, ParametrizationDerivative<M>>;

    std::vector<int> non_vanishing_functions_; // indexes of non-vanishing basis functions

    std::shared_ptr<ParametrizationType> parametrization_; // parametrization of the element
    std::shared_ptr<GradientType> gradient_; // gradient of the parametrization

    std::size_t ID; // element ID

    std::array<int, M> left_coords_; // coordinates of the left corner of the element
    std::array<int, M> right_coords_; // coordinates of the right corner of the element

    double measure_; // measure of the element
    double integral_measure_; // measure of the element divided by 2^M

public:

    ElementIga() = default;

    ElementIga(const std::vector<std::size_t> & non_vanishing_functions, std::size_t ID, 
        const ParametrizationType & parametrization, const GradientType & gradient, 
        const std::array<int,M> & left_coords, const std::array<int,M> & right_coords) 
    : non_vanishing_functions_(non_vanishing_functions), ID_(ID), parametrization_(std::make_shared<ParametrizationType>(parametrization)),
    gradient_(std::make_shared<GradientType>(gradient)), left_coords_(left_coords), right_coords_(right_coords)  {

        measure_ = 1;
        for(std::size_t i = 0; i < M; ++i){
            measure_ *= right_coords_[i] - left_coords_[i];
        }
        integral_measure_ = measure_ / (1<<M);

    }

    // some getters

    double measure() const { return measure_; }
    double integral_measure() const { return integral_measure_; }
    std::size_t ID() const { return ID_; }
    const std::array<int,M> & left_coords() const { return left_coords_; }
    const std::array<int,M> & right_coords() const { return right_coords_; }

    const ParametrizationType& parametrization() const { return *parametrization_; }
    const GradientType& gradient() const { return *gradient_; };

    std::size_t num_functions() const { return non_vanishing_functions_.size(); }
    const std::vector<int> & non_vanishing_functions() const { return non_vanishing_functions_; }

};

};

#endif // __ELEMENT_IGA_H__