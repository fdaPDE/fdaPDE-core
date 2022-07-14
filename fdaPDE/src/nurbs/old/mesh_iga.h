#ifndef __MESH_IGA_H__
#define __MESH_IGA_H__

#include "element_iga.h"
#include "mesh_parametrization.h"

namespace fdapde {

template<int M, int N>
class MeshIGA {

    protected:

    typedef VectorField<M, N, MeshParametrization<M>> ParametrizationType;
    typedef MatrixField<M,N,M,ParametrizationDerivative<M>> DerivativeType;

    template<typename T>
        using DMatrix = MdArray<T, Dynamic, Dynamic>;

    std::array<std::vector<double>,M> knots_; // knots in each direction
    MdArray<double,full_dynamic_extent_t<M>> weights_; // weights in each direction
    MdArray<double,full_dynamic_extent_t<M+1>> control_points_; // control points in each direction

    ParametrizationType parametrization_; // parametrization of the mesh
    DerivativeType gradient_; // gradient of the parametrization
    NurbsBasis<M> basis_; // basis of the mesh


    std::vector<ElementIGA<M,N>> elements_cache_; // elements of the mesh

    DMatrix<double> nodes_ {}; // nodes of the mesh, size: r by M, where r is the product of the numbers of unique knots along each dimension
    std::vector<std::size_t> boundary_ {}; // : r by 1 contains a boolean value for each node, true if the node is on the boundary of Ω
    DMatrix<std::size_t> elements_ {}; // elements of the mesh
    DMatrix<std::size_t> boundary_dof_ {}; // dofs of boundary of the mesh


    public:

    MeshIGA() = default;

    MeshIGA(const std::array<std::vector<double>,M> & knots, const MdArray<double,full_dynamic_extent_t<M>> & weights, 
        const MdArray<double,full_dynamic_extent_t<M+1>> & control_points);

    // some getters

    const ParametrizationType& parametrization() const { return parametrization_; }
    const DerivativeType& gradient() const { return gradient_; }
    const NurbsBasis<M>& basis() const { return basis_; }
    const MdArray<double, full_dynamic_extent_t<M+1>>& control_points() const { return control_points_; }

    const ElementIGA<M,N>& element(std::size_t ID) const { return elements_cache_[ID]; }
    ElementIGA<M,N>& element(std::size_t ID) { return elements_cache_[ID]; }

    std::size_t num_elements() const { return elements_cache_.size(); }

    bool is_on_boundary(std::size_t ID) const;

    std::size_t num_nodes() const { return nodes_.size(); }
    std::size_t num_boundary() const { return boundary_.size(); }
    std::size_t num_elements() const { return elements_.size(); }
    std::size_t num_boundary_dof() const { return boundary_dof_.size(); }


    const DMatrix<double>& nodes() const { return nodes_; }
    const DMatrix<std::size_t>& boundary() const { return boundary_; }
    const DMatrix<std::size_t>& elements() const { return elements_; }
    const DMatrix<std::size_t>& boundary_dof() const { return boundary_dof_; }


    struct iterator {

    };

    struct boundary_iterator {
    };

    template<int M, int N>
    MeshIGA<M,N>::MeshIGA(const std::array<std::vector<double>,M> & knots, const MdArray<double,full_dynamic_extent_t<M>> & weights, 
        const MdArray<double,full_dynamic_extent_t<M+1>> & control_points): knots_(knots), weights_(weights), control_points_(control_points) {

            std::array<Parametrization_type,N> parametrizations;
            std::array<std::array<Derivative_type,M>,N> gradients;

            // reserve space for parametrizations and gradients
            parametrizations.reserve(N);

            // define ith element of the parametrizations
            // i,j element of the gradient of parametrizations is the derivative of the ith parametrization with respect to the jth variable

            for(std::size_t i = 0; i < N; ++i){
                parametrizations[i] = MeshParametrization<M>(knots, weights, control_points,i);
                for(std::size_t j = 0; j < M; ++j){
                    gradients[i][j] = ParametrizationDerivative<M>(knots, weights, control_points, i, j);
                }
            }

            // write the parametrizations and gradients to the mesh
            parametrization_ = ParametrizationType(parametrizations);
            gradient_ = DerivativeType(gradients);

            // compute the number of unique knots along each dimension, to set the size of the nodes matrix
            std::array<std::size_t,M> strides;
            std::array<std::size_t,M> element_strides;

            int nodes_size = 1;
            int elements_size = 1;
            int tmp = 1;
            int tmp_el = 1;

            for(std::size_t i = 0; i < M; ++i){
                nodes_size *= knots[i].size();  //std::unique(knots_[i].begin(), knots_[i].end()) - knots_[i].begin();
                strides[i] = tmp;
                element_strides[i] = tmp_el;
                tmp*=knots[i].size();
                tmp_el*=knots[i].size()-1;
            }

            // set the size of the nodes matrix
            nodes_.resize(nodes_size,M);

            // set the size of the boundary matrix
            boundary_.resize(nodes_size);

            // set the size of the elements matrix
            elements_.resize(elements_size,1<<M);


            // fill the nodes matrix and the boundary matrix
            boundary_.fill(0);

            for (std::size_t i = 0; i < M; ++i) {
                std::size_t knot_size = knots[i].size();
                std::size_t repeat_count = rows / (knot_size * strides[i]);

                for (std::size_t j = 0; j < repeat_count; ++j) {
                    for (std::size_t k = 0; k < knot_size; ++k) {
                        std::size_t start_idx = j * knot_size * strides[i] + k * strides[i];
                        for (std::size_t l = 0; l < strides[i]; ++l) {
                            std::size_t node_idx = start_idx + l;
                            nodes_(node_idx, i) = knots[i][k];

                            // Check and mark boundary points
                            if (k == 0) {
                                boundary_(node_idx) |= 1; // First point of knot vector
                            } else if (k == knot_size - 1) {
                                boundary_(node_idx) |= 3; // Last point of knot vector
                            }
                        }
                    }
                }
            }

            // fill the elements matrix: each row contains the indices of the nodes of an element
            // Da ripensare eventualmente

            std::size_t element_idx = 0;

            for(std::size_t i=0;i<rows;++i){
                if((boundary_[i]&2)!=0){
                    boundary_[i] = 1;
                }
                else{
                    // we have 2^M directions to go
                    for(std::size_t j=0;j<(1<<M);++j){
                        std::size_t element_node_idx = i;
                        for(std::size_t k=0;k<M;++k){
                            // if the kth bit of j is 1, then we move in the kth direction
                            if((j&(1<<k))!=0){
                                element_node_idx += element_strides[k];
                            }
                        }
                        elements_(element_idx,j) = element_node_idx;
                        
                    }
                    ++element_node_idx;
                }
            }


            // neighbors TODO

            // populate the elements cache

            elements_cache_.reserve(elements_size);

            std::array<size::t,M> eMultiIndex;
            eMultiIndex.fill(0); 

            auto order = knots[0].get_order();

            std::size_t fnSize = pow(order+1,M);

            std::vector<std::size_t> non_vanishing_functions;
            non_vanishing_functions.reserve(fnSize);

            for(std::size_t i = 0;i<elements_size;++i){

                std::array<std::site_t,M> fnMultiIndex(eMultiIndex);

                for(std::size_t j = 0;j<M;++j){
                    non_vanishing_functions[j] = basis_.index(eMultiIndex);

                    // increment the multi-index and perform the carry-on operation
                    std::size_t k=0;
                    ++fnMultiIndex[k];

                    while(k<M-1 && fnMultiIndex[k]>elMultiIndex[k]+order){
                        fnMultiIndex[k] = elMultiIndex[k];
                        ++fnMultiIndex[++k];
                    }
                }

                elements_cache[i] = ElementIGA<M,N>(non_vanishing_functions,i,parametrization_,gradient_,nodes_[elements_(i,0)],
                    nodes_[elements_(i,1<<M-1)]);

                // increment the multi-index and perform the carry-on operation
                std::size_t k=0;
                ++eMultiIndex[k];
                // comparison with the last element of the knot vector
                while(k<M-1 && eMultiIndex[k]>knots_[k].size()-1){
                    eMultiIndex[k] = 0;
                    ++eMultiIndex[++k];
                }
            }

            // information of dofs on the boundary, functions that are not zero on the boundary

            std::size_t num_boundary_dofs = 0;
            std::size_t temp = 0;

            for(std::size_t i = 1; i < M; ++i){
                temp*=weights_.extents(i);

            }
            num_boundary_dofs += temp;

            






        };





    };
}; // namespace fdapde

#endif // __MESH_IGA_H__