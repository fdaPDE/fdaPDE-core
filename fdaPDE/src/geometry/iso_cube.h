#ifndef __FDAPDE_ISO_CUBE_H__
#define __FDAPDE_ISO_CUBE_H__

#include "header_check.h"


namespace fdapde {

/**
 * @brief 3D parametric cubic element embedded in physical space.
 * Specialization of IsoCell for cubes (hexahedra)
 * 
 * @tparam MeshType Parent mesh type (must have local_dim = 1)
 */
template <typename MeshType> class IsoCube: public IsoCell<MeshType::local_dim, MeshType::embed_dim>{
    fdapde_static_assert(MeshType::local_dim == 3, THIS_CLASS_IS_FOR_3D_MESHES_ONLY);
    using Base = IsoCell<MeshType::local_dim, MeshType::embed_dim>;
    public:
    // === Constructors === //
    IsoCube() = default;
    IsoCube(int id, const MeshType* mesh) : id_(id), mesh_(mesh), boundary_(false) {
        boundary_ = mesh_->is_cell_on_boundary(id_);
        std::tie(this->left_coords_, this->right_coords_) = mesh_->compute_lr_vertices(id_);
        // compute edge indices
        for (int i = 0; i < 12; ++i) {
            edge_ids_[i] = mesh_->cell_to_edges()(id_, i);
        }
    }

    
    // === Edge Type === //
    class EdgeType : public IsoCell<1, MeshType::embed_dim> {
        using Base = IsoCell<1, MeshType::embed_dim>;
        int edge_id_;
        const MeshType* mesh_;
        public:
        using CoordsType = Eigen::Matrix<double, MeshType::embed_dim, MeshType::local_dim>; 
        EdgeType() = default; //// da capire come adattare!!!
        EdgeType(int edge_id, const MeshType* mesh) : edge_id_(edge_id), mesh_(mesh) {
            for (int i = 0; i < this->n_nodes; ++i) { 
                this->left_coords_(0) =  mesh_->parametric_nodes()(mesh_->edges()(edge_id,0), 0);
                this->right_coords_(0) = mesh_->parametric_nodes()(mesh_->edges()(edge_id,0), 1);
            }
        }
        bool on_boundary() const { return mesh_->is_edge_on_boundary(edge_id_); }
        Eigen::Matrix<int, Dynamic, 1> node_ids() const { return mesh_->edges().row(edge_id_); }
        int id() const { return edge_id_; }
        const std::unordered_set<int>& adjacent_cells() const { return mesh_->edge_to_cells().at(edge_id_); }
        int marker() const {   // mesh edge's marker
            return mesh_->edges_markers().size() > edge_id_ ? mesh_->edges_markers()[edge_id_] : Unmarked;
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
                Eigen::Matrix<double,1, MeshType::local_dim> p;
                for (int j = 0; j < MeshType::local_dim; ++j) {
                    p(j) = interpolated_points(i, j);
                }
                res.row(i) = mesh_->eval_param(p);
            }
            return res;
        }
    };

    // === Face Type === //
    class FaceType : public IsoCell<2,MeshType::embed_dim>{
        using Base = IsoCell<2,MeshType::embed_dim>;
        int face_id_;
        const MeshType* mesh_;
        public:
        using CoordsType = Eigen::Matrix<double, MeshType::embed_dim, 4>;
        FaceType() = default;
        FaceType(int face_id, const MeshType* mesh): face_id_(face_id), mesh_(mesh) {
            this->left_coords_[0] =  mesh_->parametric_nodes()(mesh_->faces()(face_id_,0), 0);
            this->right_coords_[0] = mesh_->parametric_nodes()(mesh_->faces()(face_id_,0), 1);
            this->left_coords_[1] =  mesh_->parametric_nodes()(mesh_->faces()(face_id_,1), 0);
            this->right_coords_[1] = mesh_->parametric_nodes()(mesh_->faces()(face_id_,1), 1);
            // initialize
        }
        bool on_boundary() const { return mesh_->is_face_on_boundary(face_id_); }
        Eigen::Matrix<int, Dynamic, 1> node_ids() const { return mesh_->faces().row(face_id_); }
        Eigen::Matrix<int, Dynamic, 1> edge_ids() const { return mesh_->face_to_edges().row(face_id_); }
        int id() const { return face_id_; }
        EdgeType edge(int n) const { return EdgeType(mesh_->face_to_edges()(face_id_, n), mesh_); }
        Eigen::Matrix<int, Dynamic, 1> adjacent_cells() const { return mesh_->face_to_cells().row(face_id_); }
        int marker() const {   // mesh face's marker
            return mesh_->faces_markers().size() > face_id_ ? mesh_->faces_markers()[face_id_] : Unmarked;
        }
    };
    
    // === Public Member Functions === //

    /**
     * @brief Evaluate the physical point corresponding to a reference coordinate.
     * 
     * @param p Point in reference domain [-1, 1]
     * @return Physical coordinate in embedding space
     */
    Eigen::Matrix<double, MeshType::embed_dim, 1> parametrization(const Eigen::Matrix<double, MeshType::local_dim,1>& p) const {
        return mesh_->eval_param(this->affine_map(p));
    }
    
    /**
     * @brief Evaluate the Jacobian of the mapping at a reference point.
     * 
     * @param p Point in reference domain [-1, 1]
     * @return First derivative (Jacobian matrix)
     */
    Eigen::Matrix<double, MeshType::embed_dim, MeshType::local_dim, Eigen::RowMajor> parametrization_gradient(const Eigen::Matrix<double, MeshType::local_dim,1>& p) const {
        return mesh_->eval_param_derivatives(this->affine_map(p),false).first_derivative;
    }

    /**
     * @brief Compute the metric tensor at a reference point. Computes Fᵀ·F where F is the Jacobian of the mapping.
     * 
     * @param p Point in reference domain [-1, 1]
     * @return Symmetric metric tensor matrix
     */
    Eigen::Matrix<double, MeshType::local_dim, MeshType::local_dim, Eigen::RowMajor> metric_tensor(const Eigen::Matrix<double, MeshType::local_dim,1>& p) const {
        auto F = parametrization_gradient(this->affine_map(p));
        return F.transpose() * F; 
    }

    /**
     * @brief Compute the square root of the determinant of the metric tensor.
     * 
     * @param p Point in reference domain [-1, 1]
     * @return Determinant of the metric tensor (metric scaling factor)
     */
    double metric_determinant(const Eigen::Matrix<double, MeshType::local_dim,1>& p) const {
        return std::sqrt(metric_tensor(this->affine_map(p)).determinant()); 
    }


    // === Getters === //
    int id() const { return id_; }
    Eigen::Matrix<int, 1, 2 * MeshType::local_dim> neighbors() const { return mesh_->neighbors().row(id_); }
    Eigen::Matrix<int, 1, MeshType::local_dim> node_ids() const { return mesh_->cells().row(id_); }
    bool on_boundary() const { return boundary_; }
    operator bool() const { return mesh_ != nullptr; }
    EdgeType edge(int n) const { return EdgeType(edge_ids_[n], mesh_); }
    FaceType face(int n) const { return FaceType(mesh_->cell_to_faces()(id_, n), mesh_); }
    // cell marker
    int marker() const { return mesh_->cells_markers().size() > id_ ? mesh_->cells_markers()[id_] : Unmarked; }

    // === Edge Iterators === //
    class edge_iterator : public internals::index_iterator<edge_iterator, EdgeType> {
        using Base = internals::index_iterator<edge_iterator, EdgeType>;
        using Base::index_;
        friend Base;
        const IsoCube* c_;
        // access to i-th edge
        edge_iterator& operator()(int i) {
            Base::val_ = c_->edge(i);
            return *this;
        }
       public:
        edge_iterator(int index, const IsoCube* c) : Base(index, 0, c->n_edges), c_(c) {
            if (index_ < c_->n_edges) operator()(index_);
        }
    };
    edge_iterator edges_begin() const { return edge_iterator(0, this); }
    edge_iterator edges_end() const { return edge_iterator(this->n_edges, this); }

    // iterator over tetrahedron faces
    class face_iterator : public internals::index_iterator<face_iterator, FaceType> {
        using Base = internals::index_iterator<face_iterator, FaceType>;
        using Base::index_;
        friend Base;
        const IsoCube* c_;
        // access to i-th face
        face_iterator& operator()(int i) {
            Base::val_ = c_->face(i);
            return *this;
        }
       public:
        face_iterator(int index, const IsoCube* c) : Base(index, 0, c->n_faces), c_(c) {
            if (index_ < c_->n_faces) operator()(index_);
        }
    };
    face_iterator faces_begin() const { return face_iterator(0, this); }
    face_iterator faces_end() const { return face_iterator(this->n_faces, this); }


    protected:
    int id_ = 0;                   ///< id of the cube element
    std::array<int,12> edge_ids_;  ///< for each edge, the id of the edge in the mesh
    const MeshType* mesh_ = nullptr; ///< pointer to the parent mesh
    bool boundary_ = false;        ///< true if cube element is on the boundary
};
    
    
}; // namespace fdapde

#endif // __FDAPDE_ISO_SQUARE_H__