#ifndef __FDAPDE_ISO_CELL_H__
#define __FDAPDE_ISO_CELL_H__

#include "header_check.h"


namespace fdapde {

/**
 * @brief Parametric hypercube cell in a NURBS mesh.
 * 
 * Represents an axis-aligned element in the parametric space with 
 * dimension `LocalDim_` embedded in `EmbedDim_`-dimensional space.
 * 
 * @tparam LocalDim_ Parametric (local) dimension (0 to 3)
 * @tparam EmbedDim_ Embedding dimension
 */
template<int LocalDim_, int EmbedDim_> class IsoCell{
    static_assert(LocalDim_ >= 0 && LocalDim_ <= 3);
    public:
    static constexpr int local_dim = LocalDim_;
    static constexpr int embed_dim = EmbedDim_;
    static constexpr int n_nodes = 1 << LocalDim_;
    static constexpr int n_edges = LocalDim_ * (1 << (LocalDim_ - 1));
    static constexpr int n_faces = LocalDim_ * (LocalDim_ - 1) / 2 * (1 << (LocalDim_ - 2));
    static constexpr int n_nodes_per_face = 1 << (LocalDim_ - 1);
    using BoundaryCellType = std::conditional_t<LocalDim_ == 0, IsoCell<0, EmbedDim_>, IsoCell<LocalDim_ - 1, EmbedDim_>>;
    using NodeType = Eigen::Matrix<double, embed_dim, 1>;

    IsoCell() = default;

    /**
     * @brief Construct a cell from lower and upper parametric bounds.
     * 
     * @param left_coords Coordinates of the lower corner (parametric space)
     * @param right_coords Coordinates of the upper corner (parametric space)
     */
    IsoCell(const Eigen::Matrix<double, local_dim, 1> left_coords, const Eigen::Matrix<double, local_dim, 1> right_coords ): 
        left_coords_(left_coords), right_coords_(right_coords) { } 

    /**
     * @brief Map a point from reference element [-1,1]^d to the parametric cell.
     * 
     * Applies an affine transformation from the reference hypercube to the
     * actual parametric element.
     * 
     * @param p Point in the reference domain
     * @return Corresponding point in the parametric element
     */
    Eigen::Matrix<double, local_dim,1> affine_map(const Eigen::Matrix<double, local_dim,1> & p) const {
        Eigen::Matrix<double, local_dim,1> x;
            for(std::size_t i = 0; i < LocalDim_; ++i){
                x(i) = 0.5*(right_coords_(i) + left_coords_(i) + (right_coords_(i) - left_coords_(i)) * p(i));
            }
            return x;
        }


    /**
     * @brief Map a point from the parametric cell to the reference element [-1,1]^d.
     * 
     * Applies an affine transformation from the parametric element to the
     * reference hypercube.
     * 
     * @param p Point in the parametric element
     * @return Corresponding point in the reference domain
     */
    Eigen::Matrix<double, local_dim,1> inverse_affine_map(const Eigen::Matrix<double, local_dim,1> & p) const {
        Eigen::Matrix<double, local_dim,1> x;
            for(std::size_t i = 0; i < LocalDim_; ++i){
                x(i) = 2*(p(i) - left_coords_(i))/(right_coords_(i) - left_coords_(i)) - 1;
            }
            return x;
        }
    
    /**
     * @brief Compute the measure (volume, area, length) of the parametric cell.
     * 
     * Returns the size of the element in parametric space, normalized by the
     * reference element volume.
     * 
     * @return Parametric measure of the cell
     */
    double parametric_measure() const {
        double measure = 1.0;
        for(std::size_t i = 0; i < LocalDim_; i++){
            measure *= right_coords_(i) - left_coords_(i);
        }
        return measure/(1<<LocalDim_);
    }


    // === Getters === //
    Eigen::Matrix<double, local_dim, 1> left_coords() const { return left_coords_; }
    Eigen::Matrix<double, local_dim, 1> right_coords() const { return right_coords_; }

    protected:

    Eigen::Matrix<double, local_dim, 1> left_coords_ {} ;  ///< coordinates of the left corner of the element
    Eigen::Matrix<double, local_dim, 1> right_coords_ {} ; ///< coordinates of the right corner of the element
};


};

#endif // __FDAPDE_ISO_CELL_H__