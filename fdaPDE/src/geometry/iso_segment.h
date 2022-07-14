#ifndef __FDAPDE_ISO_SEGMENT_H__
#define __FDAPDE_ISO_SEGMENT_H__

#include "header_check.h"

namespace fdapde {

/**
 * @brief 1D parametric segment element embedded in physical space.
 * Specialization of IsoCell for segments (intervals)
 * 
 * @tparam MeshType Parent mesh type (must have local_dim = 1)
 */
template <typename MeshType> class IsoSegment: public IsoCell<MeshType::local_dim, MeshType::embed_dim>{
    fdapde_static_assert(MeshType::local_dim == 1, THIS_CLASS_IS_FOR_INTERVAL_MESHES_ONLY);
    public:
    // === Constructors === //
    IsoSegment() = default;
    IsoSegment(int id, const MeshType* mesh) : id_(id), mesh_(mesh), boundary_(false) {
        boundary_ = mesh_->is_cell_on_boundary(id_);
        std::tie(this->left_coords_, this->right_coords_) = mesh_->compute_lr_vertices(id_);
    }

    // === Public Member Functions === //

    /**
     * @brief Evaluate the physical point corresponding to a reference coordinate.
     * 
     * @param p Point in reference domain [-1, 1]
     * @return Physical coordinate in embedding space
     */
    Eigen::Matrix<double, MeshType::embed_dim, 1> parametrization(const Eigen::Matrix<double, MeshType::local_dim,1>& p) const {
        return mesh_->eval_param(affine_map(p));
    }

    /**
     * @brief Evaluate the Jacobian of the mapping at a reference point.
     * 
     * @param p Point in reference domain [-1, 1]
     * @return First derivative (Jacobian matrix)
     */
    Eigen::Matrix<double, MeshType::embed_dim, MeshType::local_dim, Eigen::RowMajor> parametrization_gradient(const Eigen::Matrix<double, MeshType::local_dim,1>& p) const {
        return mesh_->eval_param_derivatives(affine_map(p),false).first_derivative;
    }

    /**
     * @brief Compute the metric tensor at a reference point. Computes Fᵀ·F where F is the Jacobian of the mapping.
     * 
     * @param p Point in reference domain [-1, 1]
     * @return Symmetric metric tensor matrix
     */
    Eigen::Matrix<double, MeshType::local_dim, MeshType::local_dim, Eigen::RowMajor> metric_tensor(const Eigen::Matrix<double, MeshType::local_dim,1>& p) const {
        auto F = parametrization_gradient(affine_map(p));
        return F.transpose() * F; 
    }

    /**
     * @brief Compute the square root of the determinant of the metric tensor.
     * 
     * @param p Point in reference domain [-1, 1]
     * @return Determinant of the metric tensor (metric scaling factor)
     */
    double metric_determinant(const Eigen::Matrix<double, MeshType::local_dim,1>& p) const {
        return std::sqrt(metric_tensor(affine_map(p)).determinant()); 
    }

    // === Getters === //

    int id() const { return id_; }
    Eigen::Matrix<int, 1, 2 * MeshType::local_dim> neighbors() const { return mesh_->neighbors().row(id_); }
    Eigen::Matrix<int, 1, MeshType::local_dim> node_ids() const { return mesh_->cells().row(id_); }
    bool on_boundary() const { return boundary_; }
    operator bool() const { return mesh_ != nullptr; }

    protected:
    int id_ = 0;   ///< Cell ID
    const MeshType* mesh_ = nullptr; ///< Pointer to the parent mesh
    bool boundary_ = false;   ///< True if cell is on the boundary
};
    
    
}; // namespace fdapde

#endif // __FDAPDE_ISO_SEGMENT_H__