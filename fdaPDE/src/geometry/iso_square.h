#ifndef __FDAPDE_ISO_SQUARE_H__
#define __FDAPDE_ISO_SQUARE_H__

#include "header_check.h"

namespace fdapde {

/**
 * @brief 2D parametric square element embedded in physical space.
 * Specialization of IsoCell for squares (rectangles)
 * 
 * @tparam MeshType Parent mesh type (must have local_dim = 1)
 */
template <typename MeshType> class IsoSquare: public IsoCell<MeshType::local_dim, MeshType::embed_dim>{
    fdapde_static_assert(MeshType::local_dim == 2, THIS_CLASS_IS_FOR_2D_MESHES_ONLY);
    using Base = IsoCell<MeshType::local_dim, MeshType::embed_dim>;
    public:
    // === Constructors === //
    IsoSquare() = default;
    IsoSquare(int id, const MeshType* mesh) : 
        id_(id), mesh_(mesh), boundary_(false)  {
        boundary_ = mesh_->is_cell_on_boundary(id_);
        this->left_coords_ = mesh_->compute_lr_vertices(id_).first;
        this->right_coords_ = mesh_->compute_lr_vertices(id_).second;

    }

    // === Edge Type === //
    class EdgeType : public IsoCell<MeshType::local_dim, MeshType::embed_dim>::BoundaryCellType{
        private:
        int edge_id_;
        const MeshType* mesh_;
        double const_coord_ = 0.0; // constant coordinate along the edge
        bool x_axis_ = false; // true if edge is aligned with x-axis, false if aligned with y-axis
        public:
        EdgeType() = default;
        EdgeType(int edge_id, const MeshType* mesh) : edge_id_(edge_id), mesh_(mesh) {
            auto node1 = mesh_->edges()(edge_id, 0);
            auto node2 = mesh_->edges()(edge_id, 1);

            auto nodes = mesh_->parametric_nodes();
            auto coord1 = nodes.row(node1);
            auto coord2 = nodes.row(node2);

            if (coord1(0) == coord2(0)) {
                // Edge is vertical (aligned with y-axis)
                this->left_coords_(0) = std::min(coord1(1), coord2(1));
                this->right_coords_(0) = std::max(coord1(1), coord2(1));
                const_coord_ = coord1(0);  // x is constant
                x_axis_ = false;
            } else if (coord1(1) == coord2(1)) {
                // Edge is horizontal (aligned with x-axis)
                this->left_coords_(0) = std::min(coord1(0), coord2(0));
                this->right_coords_(0) = std::max(coord1(0), coord2(0));
                const_coord_ = coord1(1);  // y is constant
                x_axis_ = true;
            } else {
                throw std::runtime_error("Edge does not align with axes.");
            }
        }

        bool on_boundary() const { return mesh_->is_edge_on_boundary(edge_id_);}
        Eigen::Matrix<int, Dynamic, 1> node_ids() const { return mesh_->edges().row(edge_id_); }
        int id() const { return edge_id_; }
        Eigen::Matrix<int, Dynamic, 1> adjacent_cells() const { return mesh_->edge_to_cells().row(edge_id_); }
        int marker() const {   // mesh edge's marker
            return mesh_->edges_markers().size() > edge_id_ ? mesh_->edges_markers()[edge_id_] : Unmarked;
        }

        double const_coord() const { return const_coord_; }
        bool x_axis() const { return x_axis_; }

        std::array<double, 2> node(int i) const {
            fdapde_assert(i == 0 || i == 1);  // only two nodes per edge

            double coord = (i == 0) ? this->left_coords_(0) : this->right_coords_(0);
            if (x_axis_) {
                return {coord, const_coord_};  // horizontal edge
            } else {
                return {const_coord_, coord};  // vertical edge
            }
        }


        Eigen::Matrix<double, MeshType::local_dim,1> param_point(Eigen::Matrix<double, MeshType::local_dim - 1,1> val) const {
            Eigen::Matrix<double, MeshType::local_dim,1> p_param;
            if(x_axis_) {
                p_param(0) = val(0);  // x-coordinate is constant
                p_param(1) = const_coord_;           // y-coordinate varies
            } else {
                p_param(0) = const_coord_;           // x-coordinate varies
                p_param(1) = val(0);  // y-coordinate is constant
            }
            return p_param;
        }

        /**
         * @brief Evaluate the n linspaced physical points for the edge. Only for plotting purposes.
         * 
         * @param n Number of points to evaluate
         * @return Physical coordinates in embedding space
         */
        Eigen::Matrix<double, Eigen::Dynamic, MeshType::embed_dim> evaluation(int n) const {
            Eigen::Matrix<double, Eigen::Dynamic, MeshType::embed_dim> res(n, MeshType::embed_dim);
            auto nodes = node_ids(); // Expected to be Eigen::Matrix<int, 2, 1>
            Eigen::Matrix<double, Eigen::Dynamic, MeshType::local_dim> parametric_nodes = mesh_->parametric_nodes();
            Eigen::Matrix<double, 1, MeshType::local_dim> n1 = parametric_nodes.row(nodes(0));
            Eigen::Matrix<double, 1, MeshType::local_dim> n2 = parametric_nodes.row(nodes(1));

            Eigen::Matrix<double, Eigen::Dynamic, MeshType::local_dim> interpolated_points(n, MeshType::local_dim);
            for (int i = 0; i < n; ++i) {
                double t = static_cast<double>(i) / (n - 1);  // Normalized parameter (0 to 1)
                interpolated_points.row(i) = (1 - t) * n1 + t * n2;  // Linear interpolation
            }
            // Evaluate mapped coordinates in the embedded space
            for (int i = 0; i < n; ++i) {
                Eigen::Matrix<double,  MeshType::local_dim,1> p;
                for (int j = 0; j < MeshType::local_dim; ++j) {
                    p(j) = interpolated_points(i, j);
                }
                res.row(i) = mesh_->eval_param(p);
            }
            return res;
        }
        
        Eigen::Matrix<double, MeshType::embed_dim, 1> parametrization(const Eigen::Matrix<double, MeshType::local_dim - 1 ,1>& p, bool param = false) const {
            auto nodes = node_ids(); // Expected to be Eigen::Matrix<int, 2, 1>
            Eigen::Matrix<double, Eigen::Dynamic, MeshType::local_dim> parametric_nodes = mesh_->parametric_nodes();
            Eigen::Matrix<double, 1, MeshType::local_dim> vertex1 = parametric_nodes.row(nodes(0));
            Eigen::Matrix<double, 1, MeshType::local_dim> vertex2 = parametric_nodes.row(nodes(1));

            // find the coordinate that remains constant
            double const_val;
            int const_index = -1;  // index of the constant coordinate
            if(vertex1(0) == vertex2(0)) {
                const_val = vertex1(0);  // x-coordinate is constant
                const_index = 0;
            } else if(vertex1(1) == vertex2(1)) {
                const_val = vertex1(1);  // y-coordinate is constant
                const_index = 1;
            } else {
                throw std::runtime_error("Edge does not align with axes.");
            }

            Eigen::Matrix<double, MeshType::local_dim, 1> point;

            if(const_index == 0) {
                point(0) = const_val;  // x-coordinate is constant
                point(1) = p(0);       // y-coordinate varies
            } else {
                point(0) = p(0);       // x-coordinate varies
                point(1) = const_val;  // y-coordinate is constant
            }
            if (param)
                return mesh_->eval_param(point);
            else
                return mesh_->eval_param(point);
                //return mesh_->eval_param(this->affine_map(point));
    }

        double metric_determinant(const Eigen::Matrix<double, MeshType::local_dim - 1,1>& p, bool param=false) const {
                        auto nodes = node_ids(); // Expected to be Eigen::Matrix<int, 2, 1>
            Eigen::Matrix<double, Eigen::Dynamic, MeshType::local_dim> parametric_nodes = mesh_->parametric_nodes();
            Eigen::Matrix<double, 1, MeshType::local_dim> vertex1 = parametric_nodes.row(nodes(0));
            Eigen::Matrix<double, 1, MeshType::local_dim> vertex2 = parametric_nodes.row(nodes(1));

            // find the coordinate that remains constant
            double const_val;
            int const_index = -1;  // index of the constant coordinate
            if(vertex1(0) == vertex2(0)) {
                const_val = vertex1(0);  // x-coordinate is constant
                const_index = 0;
            } else if(vertex1(1) == vertex2(1)) {
                const_val = vertex1(1);  // y-coordinate is constant
                const_index = 1;
            } else {
                throw std::runtime_error("Edge does not align with axes.");
            }

            Eigen::Matrix<double, MeshType::local_dim, 1> point;

            if(const_index == 0) {
                point(0) = const_val;  // x-coordinate is constant
                point(1) = p(0);       // y-coordinate varies
            } else {
                point(0) = p(0);       // x-coordinate varies
                point(1) = const_val;  // y-coordinate is constant
            }

            Eigen::Matrix<double, MeshType::embed_dim, MeshType::local_dim> F;

            if(param)
                F =  mesh_->eval_param_derivatives(point,false).first_derivative;
            else
                F =  mesh_->eval_param_derivatives(point,false).first_derivative;
                //F = mesh_->eval_param_derivatives(this->affine_map(point),false).first_derivative;

            // remove the constant coordinate from the jacobian, i.e. the const_index column
            Eigen::Matrix<double, MeshType::embed_dim, MeshType::local_dim - 1> F_reduced;
            //std::cout<<"Entro qua"<<std::endl;
            if(const_index == 0) {
                F_reduced = F.block(0, 1, MeshType::embed_dim, MeshType::local_dim - 1);
            } else {
                F_reduced = F.block(0, 0, MeshType::embed_dim, MeshType::local_dim - 1);
            }
            // compute the metric tensor as Fᵀ·F
            Eigen::Matrix<double, MeshType::local_dim - 1, MeshType::local_dim - 1> metric_tensor = F_reduced.transpose() * F_reduced;
            // return the square root of the determinant    
            return std::sqrt(metric_tensor.determinant()); 
        }

    };

    // === Public Member Functions === //

    /**
     * @brief Evaluate the physical point corresponding to a reference coordinate.
     * 
     * @param p Point in reference domain [-1, 1]
     * @return Physical coordinate in embedding space
     */
    Eigen::Matrix<double, MeshType::embed_dim, 1> parametrization(const Eigen::Matrix<double, MeshType::local_dim,1>& p, bool param = false) const {
        if (param)
            return mesh_->eval_param(p);
        else
            return mesh_->eval_param(this->affine_map(p));
    }

    /**
     * @brief Evaluate the Jacobian of the mapping at a reference point.
     * 
     * @param p Point in reference domain [-1, 1]
     * @return First derivative (Jacobian matrix)
     */
    Eigen::Matrix<double, MeshType::embed_dim, MeshType::local_dim> parametrization_gradient(const Eigen::Matrix<double, MeshType::local_dim,1>& p, bool param=false) const {
        //std::cout<<"Param"<<param<<std::endl;
        if(param){
            return mesh_->eval_param_derivatives(p,false).first_derivative;
        }
        else
            return mesh_->eval_param_derivatives(this->affine_map(p),false).first_derivative;
    }


    MdArray<double, MdExtents<MeshType::embed_dim, MeshType::local_dim, MeshType::local_dim>> parametrization_hessian(const Eigen::Matrix<double, MeshType::local_dim,1>& p, bool param=false) const {
        //std::cout<<"Param"<<param<<std::endl;
        if(param){
            return *(mesh_->eval_param_derivatives(p,true).second_derivative);
        }
        else
            return *(mesh_->eval_param_derivatives(this->affine_map(p),true).second_derivative);
    }

    /**
     * @brief Compute the metric tensor at a reference point. Computes Fᵀ·F where F is the Jacobian of the mapping.
     * 
     * @param p Point in reference domain [-1, 1]
     * @return Symmetric metric tensor matrix
     */
    Eigen::Matrix<double, MeshType::local_dim, MeshType::local_dim> metric_tensor(const Eigen::Matrix<double, MeshType::local_dim,1>& p, bool param=false) const {
        auto F = parametrization_gradient(p,param);
        return F.transpose() * F; 
    }

    /**
     * @brief Compute the square root of the determinant of the metric tensor.
     * 
     * @param p Point in reference domain [-1, 1]
     * @return Determinant of the metric tensor (metric scaling factor)
     */
    double metric_determinant(const Eigen::Matrix<double, MeshType::local_dim,1>& p, bool param=false) const {
        return std::sqrt(metric_tensor(p,param).determinant()); 
    }

    /**
     * @brief Evaluate a regular grid of physical points over the square element. Only for plotting purposes.
     * 
     * Performs a 2D tensor-product linear interpolation in parametric space,
     * maps each point to physical space, and stores the result.
     * 
     * @param n Number of evaluation points per parametric direction (produces n x n grid)
     * @return 3D array (n x n x embed_dim) of physical coordinates
     */
    MdArray<double, full_dynamic_extent_t<MeshType::local_dim + 1>> linspace_evaluation(int n, MdArray<double, full_dynamic_extent_t<MeshType::local_dim + 1>>& parametric_points ) const {
        MdArray<double, full_dynamic_extent_t<MeshType::local_dim + 1>> res(n, n, MeshType::embed_dim);
        auto param_nodes = mesh_->parametric_nodes();
        auto left_coords = this->left_coords_;
        auto right_coords = this->right_coords_;

        // store the parametric points
        parametric_points.resize(n, n, MeshType::local_dim);

        // compute the step
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                Eigen::Matrix<double, MeshType::local_dim, 1> p;
                auto t1 = static_cast<double>(i) / (n - 1);
                auto t2 = static_cast<double>(j) / (n - 1);

                p(0) = (1 - t1) * left_coords(0) + t1 * right_coords(0);  // Linear interpolation
                p(1) = (1 - t2) * left_coords(1) + t2 * right_coords(1);  // Linear interpolation

                for(int k = 0; k < MeshType::local_dim; k++){
                    parametric_points(i, j, k) = p(k);
                }

                auto param = mesh_->eval_param(p);
                for(int k = 0; k < MeshType::embed_dim; k++){
                    res(i, j, k) = param(k);
                }
            }
        }
        return res;

    }


    // === Getters === // 
    int id() const { return id_; }
    Eigen::Matrix<int, 1, 2 * MeshType::local_dim> neighbors() const { return mesh_->neighbors().row(id_); }
    Eigen::Matrix<int, 1, MeshType::local_dim> node_ids() const { return mesh_->cells().row(id_); }
    bool on_boundary() const { return boundary_; }
    operator bool() const { return mesh_ != nullptr; }
    EdgeType edge(int n){
        fdapde_assert(n<this->n_edges);
        return EdgeType(mesh_->cell_to_edes()(id_,n), mesh_);
    }
    int marker() const {return mesh_->cell_markers().size() ? mesh_->cells_markers()[id_] : Unmarked; }

    // === Edge Iterators === //
    class edge_iterator: public internals::index_iterator<edge_iterator, EdgeType>{
        using Base = internals::index_iterator<edge_iterator, EdgeType>;
        using Base::index_;
        friend Base;
        const IsoSquare* sq_;
        // access to i-th square edge
        edge_iterator& operator() (int i) {
            Base::val_ = sq_->edge(i);
            return *this;
        }
        public:
        edge_iterator(int index, const IsoSquare* sq){
            if(index_ < sq_->n_edges) operator()(index_);
        }

    };

    edge_iterator edges_begin() const {return edge_iterator(0, this);}
    edge_iterator edges_end() const {return edge_iterator(this->n_edges, this);}

    protected:
    int id_ = 0;                     ///< id of the square element
    const MeshType* mesh_ = nullptr; ///< pointer to the parent mesh
    bool boundary_ = false;          ///< true if square element is on the boundary
};
    
    
}; // namespace fdapde

#endif // __FDAPDE_ISO_SQUARE_H__