#ifndef __ISO_MESH_H__
#define __ISO_MESH_H__

#include "header_check.h"
#include <chrono>


namespace fdapde {

template <int LocalDim, int EmbedDim> class IsoMesh;

/**
 * @brief Base class for isogeometric mesh representations in arbitrary dimensions.
 * 
 * @tparam LocalDim Dimension of the parametric (reference) space.
 * @tparam EmbedDim Dimension of the physical (embedded) space.
 * @tparam Derived CRTP derived mesh type.
 */
template <int LocalDim, int EmbedDim, typename Derived> class IsoMeshBase{ 
    public:
    static constexpr int local_dim = LocalDim;              ///< Parametric dimension
    static constexpr int embed_dim = EmbedDim;              ///< Physical embedding dimension
    static constexpr int n_nodes_per_cell = 1<<LocalDim;    ///< Number of nodes per cell
    static constexpr int n_neighbors_per_cell = 2*LocalDim; ///< Number of neighbors per cell
    static constexpr bool is_manifold = !(local_dim == embed_dim); ///< True if mesh represents a lower-dimensional manifold

    using CellType = std::conditional_t<local_dim == 1, IsoSegment<Derived>, 
                     std::conditional_t<local_dim == 2, IsoSquare<Derived>, IsoCube<Derived>>>;
    using MeshType = Derived;

    ///Structure to hold first and optional second derivatives of the parametric mapping.
    struct MeshParamDerivatives {
        Eigen::Matrix<double, EmbedDim, LocalDim> first_derivative; ///< Jacobian matrix
        std::optional<MdArray<double, MdExtents<EmbedDim, LocalDim, LocalDim>>> second_derivative; ///< Optional second derivative tensor
    };
    
    ///Lightweight wrapper for mesh nodes (read-only access).
    class NodeType {
        int id_;
        const MeshType* mesh_;
        public:
        NodeType() = default;
        /// Construct a node with its ID and parent mesh pointer.
        NodeType(int id, const MeshType* mesh) : id_(id), mesh_(mesh) { } 

        /// Get the node index
        int id() const { return id_; }
        /// Get the physical coordinates of the node
        Eigen::Matrix<double, embed_dim, 1> coords() const { return mesh_->phys_node(id_); }
        
        //std::vector<int> patch() const { return mesh_->node_patch(id_); }         // cells having this node as vertex
        //std::vector<int> one_ring() const { return mesh_->node_one_ring(id_); }   // directly connected nodes
    };
    
    /// Default constructor for the mesh.
    IsoMeshBase() = default;

    /**
     * @brief Construct and initialize the mesh from NURBS data.
     * 
     * @param knots Knot vectors for each parametric direction
     * @param weights NURBS weights
     * @param control_points Control points of the mesh
     * @param degree Polynomial degree (per direction)
     * @param flags Optional behavior flags
     */
    IsoMeshBase(std::array<std::vector<double>,LocalDim> & knots,MdArray<double,full_dynamic_extent_t<LocalDim>> & weights, 
         MdArray<double,full_dynamic_extent_t<LocalDim+1>> & control_points, std::array<int,LocalDim> degree, int flags=0) {
            initialize(knots, weights, control_points, degree, flags);
        };
    
    /// Initialize the mesh with the same parameters as the constructor of IsoMeshBase, overwriting any previous data.
    void initialize(std::array<std::vector<double>, LocalDim> & knots,
            MdArray<double, full_dynamic_extent_t<LocalDim>> & weights,
            MdArray<double, full_dynamic_extent_t<LocalDim + 1>> & control_points,
            std::array<int, LocalDim>& degree,
            int flags = 0) {
                // Assign to internal data members
                flags_          = flags;
                degree_          = degree;
                control_points_ = control_points;
                weights_        = weights;

                // Pad and store the knots
                for(int i = 0; i < LocalDim; i++){
                    int n = knots[i].size();
                    knots_[i].resize(n);
                    std::copy(knots[i].begin(), knots[i].end(), knots_[i].begin());
                }

                // Compute the basis
                basis_ = NurbsBasis<LocalDim>(knots_, weights, degree);

                // Compute the parametric nodes
                n_cells_ = 1; 
                n_nodes_ = 1;
                for(int i = 0; i < LocalDim; i++){
                    // Make a local copy, then remove duplicates
                    std::vector<double> unique_knots = knots[i];
                    unique_knots.erase(
                        std::unique(unique_knots.begin(), unique_knots.end()),
                        unique_knots.end()
                    );

                    param_nodes_[i] = unique_knots;
                    // Update counters
                    n_cells_ *= (unique_knots.size() - 1);
                    n_nodes_ *= unique_knots.size();
                }

                detect_periodicity_();
                compute_span_aabbs_();
            }

    // === Getters === //

    const NurbsBasis<LocalDim>& basis() const { return basis_; }
    const MdArray<double, full_dynamic_extent_t<LocalDim+1>>& control_points() const { return control_points_; }
    const std::array<std::vector<double>,LocalDim>& knots() const { return knots_; } // this contains also the repetitions (if any)
    const std::array<std::vector<double>,LocalDim>& param_nodes() const { return param_nodes_; } // (only unique knots)
    const MdArray<double, full_dynamic_extent_t<LocalDim>>& weights() const { return weights_; }
    const std::array<int,LocalDim>& degree() const { return degree_; }
    int n_cells() const { return n_cells_; }
    int n_nodes() const { return n_nodes_; }
    std::array<int, LocalDim> n_control_points() const {
        std::array<int, LocalDim> n_cp;
        for (int i = 0; i < LocalDim; ++i)
            n_cp[i] = control_points_.extent(i);
        return n_cp;
    }
    const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>& cells() const { return cells_; }
    IsoMeshData<LocalDim> data() const {
        return IsoMeshData<LocalDim>{knots_, weights_, control_points_, degree_};
    }  
    std::vector<int> nodes_markers() const { return nodes_markers_; }
    std::vector<int> cells_markers() const { return cells_markers_; }

    // periodic
    std::array<bool,LocalDim> is_periodic() const { return periodic_dims_; }
    bool is_periodic(int dir) const { return periodic_dims_[dir]; }
    // === Core Evaluation Functions === //
    
    /**
     * @brief Evaluate the physical coordinates at parametric location `u`
     * 
     * Implements Algorithm A4.3 from *The NURBS Book* pag 134.
     *
     * @param u Parametric coordinate (LocalDim-vector)
     * @return Physical coordinate in embedding space
     */
    Eigen::Matrix<double, EmbedDim, 1> eval_param(const Eigen::Matrix<double, LocalDim,1>& u) const {
        for(int i = 0; i < LocalDim; i++) fdapde_assert(u(i) >= knots_[i].front() - 1e-9 && u(i) <= knots_[i].back() + 1e-9);
        std::array<std::vector<double>,LocalDim> basis_eval;
        std::array<int,LocalDim> spans= {0};
        auto degree = this->basis_.degree();
        auto& nurb = this->basis_[0];
        double total_weight = 0.0;
        
        for(int i = 0; i < LocalDim; i++){
            auto basis = nurb.spline_basis()[i];
            basis_eval[i] = basis->evaluate_basis(u(i), false); // evaluate basis functions, padding = false
            spans[i] = basis->find_span(u(i)); // find the span of the knot vector
        }
        Eigen::Matrix<double, EmbedDim, 1> Sw = Eigen::Matrix<double, EmbedDim, 1>::Zero();

        std::vector<int> index(LocalDim,0);
        bool done =  false;
        while(!done){
            double eval = 1.0;
            std::array<int,LocalDim> full_indices;
            for(int i = 0; i < LocalDim; i++){
                eval *= basis_eval[i][index[i]];
                full_indices[i] = spans[i] - degree[i] + index[i];
            }

            Eigen::Matrix<double, EmbedDim, 1> cp;
            for(int i = 0; i < EmbedDim; i++){
                const auto cp_slice = this->control_points_.template slice<LocalDim>(i);
                cp(i) = cp_slice(full_indices);
            }


            double w = weights_(full_indices);  // Retrieve weight from weight array
            cp *= w;  // Apply weight to control point
            Sw += eval * cp; // Accumulate weighted control point           
            total_weight += eval * w ;  // Accumulate total weight

            for (int d = LocalDim - 1; d >= 0; d--) {
            if (++index[d] > degree[d]) {
                index[d] = 0;
                if (d == 0) done = true;
            } else 
                break;
            }
        }
            
        return Sw/total_weight;
    }


    /**
     * @brief Evaluate the first and (if needed) second degree derivatives of the NURBS mapping at `u`
     * 
     * Based on Algorithm A4.3 from *The NURBS Book* pag 134.
     * 
     * @param u Parametric coordinate (LocalDim-vector)
     * @param compute_second Whether to compute second derivatives
     * @return MeshParamDerivatives Struct with first and optionally second derivatives
     * 
     * 
     */
    MeshParamDerivatives eval_param_derivatives(const Eigen::Matrix<double, LocalDim, 1>& u, bool compute_second = false) const {
        
        for (int i = 0; i < LocalDim; i++)
            fdapde_assert(u(i) >= knots_[i].front() -1e-9 && u(i) <= knots_[i].back() +1e-9);
    
        std::array<std::vector<double>,LocalDim> basis_eval;
        std::array<std::vector<double>,LocalDim> basis_deriv_eval;
        std::array<std::vector<double>,LocalDim> basis_second_deriv_eval;
        std::array<int, LocalDim> spans = {0};
        auto degree = this->basis_.degree();
        auto nurb = this->basis_[0];
        double total_weight = 0.0;
    
        for (int i = 0; i < LocalDim; i++) {
            auto basis = nurb.spline_basis()[i];
            auto eval = basis->evaluate_der_basis(u(i), 1, false);
            basis_eval[i] = eval[0];
            basis_deriv_eval[i] = eval[1];
            if (compute_second) {
                basis_second_deriv_eval[i] = basis->evaluate_der_basis(u(i), 2, false)[2];
            }
            spans[i] = basis->find_span(u(i));
        }
    
        Eigen::Matrix<double, EmbedDim, LocalDim> dSw = Eigen::Matrix<double, EmbedDim, LocalDim>::Zero();
        Eigen::Matrix<double, EmbedDim, 1> Sw = Eigen::Matrix<double, EmbedDim, 1>::Zero();
        Eigen::Matrix<double, LocalDim, 1> dW = Eigen::Matrix<double, LocalDim, 1>::Zero();
        Eigen::Matrix<double, LocalDim, LocalDim> d2W;
    
        std::optional<MdArray<double, MdExtents<EmbedDim, LocalDim, LocalDim>>> d2Sw;
        if (compute_second) {
            d2Sw.emplace();
            for (auto& val : *d2Sw) val = 0.0;
            d2W.setZero();
        }
    
        std::vector<int> index(LocalDim, 0);
        bool done = false;
    
        while (!done) {
            double eval = 1.0;
            std::array<double, LocalDim> eval_der = {0.0};
            std::array<double, LocalDim> eval_sec_der = {0.0};
            std::array<int, LocalDim> full_indices;
    
            for (int i = 0; i < LocalDim; i++) {
                eval *= basis_eval[i][index[i]];
                eval_der[i] = basis_deriv_eval[i][index[i]];
                if (compute_second) {
                    eval_sec_der[i] = basis_second_deriv_eval[i][index[i]];
                }
                full_indices[i] = spans[i] - degree[i] + index[i];
            }
    
            Eigen::Matrix<double, EmbedDim, 1> cp;
            for (int i = 0; i < EmbedDim; i++) {
                const auto cp_slice = this->control_points_.template slice<LocalDim>(i);
                cp(i) = cp_slice(full_indices);
            }

    
            double w = weights_(full_indices);

            Sw += eval * w * cp;
            total_weight += eval * w;
    
            for (int j = 0; j < LocalDim; j++) {
                double w_temp = w * eval_der[j];
                Eigen::Matrix<double, EmbedDim, 1> cp_temp = eval_der[j] * w * cp;
                for (int i = 0; i < LocalDim; i++) {
                    if (i != j) {
                        w_temp *= basis_eval[i][index[i]];
                        cp_temp *= basis_eval[i][index[i]];
                    }
                }
                dW(j) += w_temp;
                dSw.col(j) += cp_temp;
    
                if (compute_second) {
                    double w_temp_2 = w * eval_sec_der[j];
                    Eigen::Matrix<double, EmbedDim, 1> cp_temp_2 = eval_sec_der[j] * cp * w;
    
                    for (int i = 0; i < LocalDim; i++) {
                        if (i != j) {
                            w_temp_2 *= basis_eval[i][index[i]];
                            cp_temp_2 *= basis_eval[i][index[i]];
                        }
                    }
    
                    d2W(j, j) += w_temp_2;
                    for (int h = 0; h < EmbedDim; h++)
                        (*d2Sw)(h, j, j) += cp_temp_2(h);
    
                    for (int k = 0; k < LocalDim; k++) {
                        if (j != k) {
                            double w_mixed = w * eval_der[j] * eval_der[k];
                            Eigen::Matrix<double, EmbedDim, 1> cp_mixed = eval_der[j] * eval_der[k] * cp * w;
    
                            for (int i = 0; i < LocalDim; i++) {
                                if (i != j && i != k) {
                                    w_mixed *= basis_eval[i][index[i]];
                                    cp_mixed *= basis_eval[i][index[i]];
                                }
                            }
    
                            d2W(j, k) += w_mixed;
                            for (int h = 0; h < EmbedDim; h++)
                                (*d2Sw)(h, j, k) += cp_mixed(h);
                        }
                    }
                }
            }
    
            for (int d = LocalDim - 1; d >= 0; d--) {
                if (++index[d] > degree[d]) {
                    index[d] = 0;
                    if (d == 0) done = true;
                } else {
                    break;
                }
            }
        }
        auto dS = dSw; // backup BEFORE scaling
        for (int j = 0; j < LocalDim; j++) {
            dS.col(j) = (dSw.col(j) - (Sw * dW(j) / total_weight)) / total_weight;
        }
    
        if (compute_second) {
            for (int j = 0; j < LocalDim; j++) {
                for (int k = 0; k < LocalDim; k++) {
                    for (int h = 0; h < EmbedDim; h++) {
                        (*d2Sw)(h, j, k) = ((*d2Sw)(h, j, k) 
                        - (dSw(h, j) * dW(k) + dSw(h, k) * dW(j) + Sw(h) * d2W(j, k)) / total_weight
                        + 2 * Sw(h) * dW(j) * dW(k) / (total_weight * total_weight)) / total_weight;
                    }
                }
            }
        }
            
    
        return {dS, compute_second ? std::move(d2Sw) : std::nullopt};
        
    } 

    // === Utilities === // 

    // other functionalities can be added here :)

    /**
     * @brief Perform inplace knot refinement by inserting additional knots in each parametric direction.
     * 
     * Based on Algorithm A5.5 from *myThe NURBS Book* pag 127. 
     * 
     * @param density Number of midpoint splits per span (per direction)
     * @param add_knot_list Additional user-defined knots to insert
     */
    void refine_knots(const std::array<int, LocalDim>& density = std::array<int, LocalDim>{{1}}, 
        std::array<std::vector<double>, LocalDim> add_knot_list = {}){

        std::array<std::vector<double>,LocalDim> refinement_knots {};

        // Step 1: Generate new knots for refinement
        for (int j = 0; j < LocalDim; j++) {
            std::vector<double> refined_knots, knot_list(param_nodes_[j].begin(), param_nodes_[j].end());
        
            // Insert midpoint knots iteratively for the required density
            for (int d = 0; d < density[j]; d++) {
                std::vector<double> rknots;
                for (size_t i = 0; i < knot_list.size() - 1; i++) {
                    double midpoint = (knot_list[i] + knot_list[i + 1]) / 2.0;
                    rknots.push_back(knot_list[i]);
                    rknots.push_back(midpoint);
                }
                rknots.push_back(knot_list.back());
                knot_list = rknots;  
            }
        
            // Compute valid knot insertions
            std::vector<double> valid_knots;
            for (double mk : knot_list) {
                int s = std::count(knots_[j].begin(), knots_[j].end(), mk);  
                int r = degree_[j] - s;  
                if (s == 0) valid_knots.push_back(mk);
                //for (int _ = 0; _ < r; _++) valid_knots.push_back(mk);
            }

            refinement_knots[j].insert(refinement_knots[j].end(), valid_knots.begin(), valid_knots.end());
        }

        // Step 2: Add user-specified knots
        for(int j = 0; j < LocalDim; j++){
            for (size_t i = 0; i < add_knot_list[j].size(); i++) {
                int s = std::count(knots_[j].begin(), knots_[j].end(), add_knot_list[j][i]);  
                if (s == 0) refinement_knots[j].push_back(add_knot_list[j][i]);
            }
        }


        std::array<int,LocalDim> weights_dims;
        std::array<int,LocalDim+1> cp_dims;
        std::array<std::vector<double>,LocalDim> updated_knots {};

        for(int j = 0; j < LocalDim; j++){
            std::sort(refinement_knots[j].begin(), refinement_knots[j].end());
            weights_dims[j] = (refinement_knots[j].size() + weights_.extent(j));
            cp_dims[j] = weights_dims[j];
        }
        cp_dims[LocalDim] = EmbedDim;

        // Step 4: Resize control points and weights
        MdArray<double,full_dynamic_extent_t<LocalDim>> refined_weights;
        MdArray<double,full_dynamic_extent_t<LocalDim+1>> refined_cp;
        refined_weights.resize(weights_dims);
        refined_cp.resize(cp_dims);
        
        MdArray<double,full_dynamic_extent_t<LocalDim>> previous_weights = weights_;
        MdArray<double,full_dynamic_extent_t<LocalDim+1>> previous_cp = control_points_;

        // Step 5: Apply Knot Refinement for each dimension
        for(int k = 0; k < LocalDim; k++){ 
            if(refinement_knots[k].empty()){
                updated_knots[k] = knots_[k];
                continue;
            } 
            
            updated_knots[k].resize(knots_[k].size() + refinement_knots[k].size());

            std::array<int, LocalDim+1> temp_cp_dims;
            std::array<int, LocalDim> temp_weights_dims;
            for(int j = 0; j < LocalDim; j++){
                if(j <= k){
                    temp_cp_dims[j] = cp_dims[j];
                    temp_weights_dims[j] = weights_dims[j];
                } else{
                    temp_cp_dims[j] = control_points_.extent(j);
                    temp_weights_dims[j] = weights_.extent(j);
                }
            }
            temp_cp_dims[LocalDim] = EmbedDim;

            refined_cp.resize(temp_cp_dims);
            refined_weights.resize(temp_weights_dims);

            // Compute how many 1D knots to refine
            int ref_size = 1;
            for(int j = 0; j < LocalDim; j++) {
                if (j != k) ref_size *= previous_weights.extent(j);
            }

            std::array<int, LocalDim> index = {0};
            for(int i = 0; i < ref_size; i++){

                MdArray<double, MdExtents<Dynamic,Dynamic>> old_cp;
                MdArray<double, MdExtents<Dynamic>> old_w;

                old_cp.resize(weights_.extent(k), EmbedDim);
                old_w.resize(weights_.extent(k));

                // Get the old control points and weights
                for(int m=0;m<weights_.extent(k);m++){
                    std::array<int, LocalDim> current_index = index;
                    current_index[k] = m ;
                    for(int n=0;n<EmbedDim;n++){
                        std::array<int, LocalDim+1> current_index_cp;
                        for(int l=0;l<LocalDim;l++){
                            current_index_cp[l] = current_index[l];
                        }
                        current_index_cp[LocalDim] = n;
                        old_cp(m,n) = previous_cp(current_index_cp);
                    }
                    old_w(m) = previous_weights(current_index);
                }

                IsoMeshData<1> mesh_data(knots_[k], old_w, old_cp, degree_[k], flags_);
                auto refined_mesh = iso_algorithms::knots_refinement(mesh_data,refinement_knots[k]);
                
                updated_knots[k] = refined_mesh.knots[0];
                auto new_w = refined_mesh.weights;
                auto new_cp = refined_mesh.control_points;

                // Put the weights and cp in the total tensors
                for(int m=0;m<new_w.extent(0);m++){
                    std::array<int, LocalDim> current_index = index;
                    current_index[k] = m;
                    refined_weights(current_index) = new_w(m); 
                    for(int n=0;n<EmbedDim;n++){
                        std::array<int, LocalDim+1> current_index_cp;
                        for(int l=0;l<LocalDim;l++) current_index_cp[l] = current_index[l];
                        current_index_cp[LocalDim] = n;
                        refined_cp(current_index_cp) = new_cp(m,n);
                    }
                }

                // Increment indices except for the fixed dimension k
                for (int j = LocalDim - 1; j >= 0; j--) {
                    if (j == k) continue;  // Skip fixed dimension
                    index[j]++;
                    if (index[j] < previous_weights.extent(j)) break;  // No carry-over needed
                    else index[j] = 0;  // Reset and carry over to next dimension
                    
                }
            }
            // Copy refined control points before refining in a new direction
            previous_cp = refined_cp;
            previous_weights = refined_weights;

        }
        // Update the mesh
        initialize(updated_knots, refined_weights, refined_cp, degree_, flags_);

    }

    /**
     * @brief Perform point inversion from physical space (p) to parametric space (u).
     * 
     * Based on the implementation at page 230 of *The NURBS Book*.
     * 
     * @param p Physical coordinate
     * @param t1 Output time (microseconds) for span search, only for debugging
     * @param t2 Output time (microseconds) for Newton iteration, only for debugging
     * @param n Sampling resolution per span during initialization
     * @param tol1 Tolerance on residual norm
     * @param tol2 Tolerance on Newton step size
     * @param max_iters Max Newton iterations
     * @return Parametric coordinate `u` such that F(u) ≈ p
     */
    Eigen::Matrix<double, local_dim,1> invert_point(const Eigen::Matrix<double, embed_dim, 1>& p,
        int n = 2, double tol1=1e-8, double tol2=1e-8, int max_iters = 1000 ) const {
        
        const double eps = 1e-8; // to relax the AABB condition
        Eigen::Matrix<double, local_dim,1> u_old, u ;
        u.setZero();
        u_old.setZero();

        auto Cp = this->control_points_;

        std::array<decltype(Cp.template slice<local_dim>(0)), embed_dim> cp_slices;
        for (int i = 0; i < embed_dim; ++i)
            cp_slices[i] = Cp.template slice<local_dim>(i);
        std::array<int, local_dim> index = this->degree_;
        std::vector<std::array<int, local_dim>> valid_spans;

        // loop over each span and check if the point ins in the AABB box of the span
        bool done = false;
        do {

            auto it = span_aabbs_.find(index);
            if (it != span_aabbs_.end()) {
                const auto& [P_min, P_max] = it->second;
                bool inside = true;
                for (int i = 0; i < embed_dim && inside; ++i) {
                    if (p(i) < P_min(i) - eps || p(i) > P_max(i) + eps)
                        inside = false;
                }
                if (inside) valid_spans.push_back(index);
            }


            for (int d = local_dim - 1; d >= 0; d--) {
                if (++index[d] > this->weights_.extent(d) - 1) {
                    index[d] = this->degree_[d];
                    if (d == 0) done = true;
                } else break;
            }

        } while (!done);

        // initialize u and u_old using a grid search over the valid spans
        double min_dist = std::numeric_limits<double>::max();
        for(const auto& span : valid_spans){
            Eigen::Matrix<double, local_dim,1> u_start;
            Eigen::Matrix<double, embed_dim, 1> steps;
            for(int j = 0; j < local_dim; j++){
                u_start(j) = this->knots_[j][span[j]];
                steps(j) = (this->knots_[j][span[j]+1] - this->knots_[j][span[j]])/n;
            }

            Eigen::Matrix<double, local_dim,1> u_add ;
            u_add.setZero();
            bool done = false;

            while(!done){
                auto S = this->eval_param(u_start + u_add);
                double dist = (S - p).norm();
                if(dist < min_dist){
                    min_dist = dist;
                    u = u_start + u_add;
                }

                // use carry over to update u_add, steps(i) is the step in the i-th direction
                for(int k = local_dim - 1; k >= 0; k--){
                    if(u_add(k) + steps(k) >= n*steps(k)){
                        u_add(k) = 0 ;
                        if(k == 0) done = true;
                    } else{
                        u_add(k) += steps(k);
                        break;
                    }
                }
            }
        }

        int counter = 0;
        u_old = u;
        bool conv1 = false;
        bool conv2 = false;

        while(counter < max_iters && !conv1 && !conv2){
            
            auto S = this->eval_param(u_old);
            Eigen::Matrix<double, embed_dim, 1> r = S - p;

            if(r.norm() < tol1) conv1 = true;

            auto derivatives = eval_param_derivatives(u_old, true);

            auto S_deriv = derivatives.first_derivative;
            auto S_sec_deriv = *(derivatives.second_derivative); 

            Eigen::Matrix<double, local_dim, 1> kappa;
            Eigen::Matrix<double, local_dim, local_dim> J;

            // Compute Jacobian and kappa vector for general local_dim
            for (int i = 0; i < local_dim; ++i) {
                Eigen::Matrix<double, embed_dim, 1> S_i = S_deriv.col(i);
                kappa(i) = - r.dot(S_deriv.col(i));
                for (int j = 0; j < local_dim; ++j) {
                    Eigen::Matrix<double, embed_dim, 1> S_ij;
                    for (int k = 0; k < embed_dim; ++k) {
                        S_ij(k) = S_sec_deriv(k, i, j);
                    }
                    J(i, j) = S_deriv.col(i).dot(S_deriv.col(j)) + r.dot(S_ij);
                }
            }


            auto delta = J.lu().solve(kappa);

            u = u_old + delta;

            // enforce parametric bounds 
            for (int k = 0; k < local_dim; k++) {
                if(periodic_dims_[k]){
                    while(u(k) < this->param_nodes_[k].front() || u(k) > this->param_nodes_[k].back()){
                        if (u(k) < this->param_nodes_[k].front() ){
                            u(k) = this->param_nodes_[k].back() - (this->param_nodes_[k].front() - u(k));
                        } else if (u(k) > this->param_nodes_[k].back()){
                            u(k) = this->param_nodes_[k].front() + (u(k) - this->param_nodes_[k].back());
                        }
                    }
                } else {
                    u(k) = std::clamp(u(k), this->param_nodes_[k].front(), this->param_nodes_[k].back());
                    }
            }

            if (delta.norm() < tol2)
                conv2 = true;

            u_old = u;

            ++counter;
        }

        if(counter == max_iters){
            std::cout<<"Max iterations reached: try to increase the numbers of evaluations."<<std::endl;
        }

        return u;
    }

    // === Node / Cell Helpers === //

    /**
     * @brief Compute the LR (lower-right) parametric vertex bounds of a given cell.
     * @param id Cell ID
     * @return Pair of parametric coordinates (lower-left, upper-right)
     */
    std::pair<Eigen::Matrix<double, LocalDim, 1>, Eigen::Matrix<double, LocalDim, 1>> 
       compute_lr_vertices(const int id) const {
        auto multi_index = compute_multi_index_(id);
        Eigen::Matrix<double, LocalDim, 1> v1, v2;
        for(int i = 0; i < LocalDim; ++i){
            v1(i) = param_nodes_[i][multi_index[i]];
            v2(i) = param_nodes_[i][multi_index[i] + 1];
        }
        return {v1, v2};
    }
    
    /**
     * @brief Get the physical coordinates of a node by its ID.
     * @param id Node ID
     * @return Physical coordinates
     */
    Eigen::Matrix<double, EmbedDim, 1> phys_node(const int id) const {
        auto multi_index = compute_multi_index_(id, false);
        Eigen::Matrix<double, LocalDim,1> u;
        for (int i = 0; i < LocalDim; ++i) {
            u(i) = param_nodes_[i][multi_index[i]];
        }
        //std::cout << "Evaluating physical node for ID " << id << " at parametric coordinates: " << u.transpose() << std::endl;
        //std::cout<<"Evaluated physical node: "<< eval_param(u).transpose() << std::endl;
        return eval_param(u);
    }
    
    /**
     * @brief Get all parametric node coordinates as a matrix
     * @return Matrix of shape (n_nodes x LocalDim)
     */
    Eigen::Matrix<double, Dynamic, LocalDim> parametric_nodes() const {
        Eigen::Matrix<double, Dynamic, LocalDim> nodes;
        nodes.resize(n_nodes_, LocalDim);
        for (int i = 0; i < n_nodes_; ++i) {
            auto multi_index = compute_multi_index_(i, false);
            for (int j = 0; j < LocalDim; ++j) {
                nodes(i, j) = param_nodes_[j][multi_index[j]];
            }
        }
        return nodes;
    }
  
    /**
     * @brief Check if a node lies on the domain boundary
     * @param id Node ID
     * @return True if node is on the boundary
     */
    bool is_node_on_boundary(const int id) const {
        auto multi_index = compute_multi_index_(id,false);
        for(int i = 0; i < LocalDim; ++i){
            if(multi_index[i] == 0 || multi_index[i] == compute_stride_(i,false) - 1){
                return true;
            }
        }
        return false;
    }

    /**
     * @brief Check if a cell lies on the domain boundary
     * @param id Cell ID
     * @return True if cell is on the boundary
     */
    bool is_cell_on_boundary(const int id) const {
        auto multi_index = compute_multi_index_(id);
        for(int i = 0; i < LocalDim; ++i){
            if((multi_index[i] == 0 || multi_index[i] == compute_stride_(i,true) - 1) && !periodic_dims_[i]){
                return true;
            }
        }
        return false;
    }

    //nodes method build a n_nodes x Embeddim matrix with all the nodes
    /**
     * @brief Get the physical coordinates of all nodes in the mesh.
     * 
     * @return Matrix of shape (n_nodes x EmbedDim)
     */
    Eigen::Matrix<double, Dynamic, EmbedDim> nodes() const {
        Eigen::Matrix<double, Dynamic, EmbedDim> nodes;
        nodes.resize(n_nodes_, EmbedDim);
        for (int i = 0; i < n_nodes_; ++i) {
            // use teh phys_node method
            nodes.row(i) = phys_node(i).transpose();
        }
        return nodes;
    }

    /**
     * @brief Get the ID of the cell containing the parametric coordinate `u`.
     * 
     * @param u Parametric coordinate (LocalDim-vector)
     * @return Cell ID
     */
    int locate_param(const Eigen::Matrix<double, LocalDim, 1>& u) const {
        std::array<int, LocalDim> multi_index;
        for (int i = 0; i < LocalDim; ++i) {
            //std::cout << "u(" << i << ") = " << u(i) << std::endl;
            if (u(i) < param_nodes_[i].front() || u(i) > param_nodes_[i].back()) return -1;

            for(int j = 0; j < param_nodes_[i].size() - 1; ++j) {
                if (u(i) >= param_nodes_[i][j] && u(i) <= param_nodes_[i][j + 1]) {
                    multi_index[i] = j;
                    break;
                }
            }
            /*
            // print the param_nodes
            std::cout << "param_nodes[" << i << "] = ";
            for (const auto& val : param_nodes_[i]) {
                std::cout << val << " ";
            }
            std::cout << std::endl;

            // print the multi_index
            std::cout << "multi_index[" << i << "] = " << multi_index[i] << std::endl;
            */
        }
        // reverse the multi_index
        std::reverse(multi_index.begin(), multi_index.end());
        

        return compute_id_(multi_index);
    }
    
    /**
     * @brief Computes the neighboring cell IDs for each cell in the mesh.
     * 
     * Output matrix has shape (n_cells × 2 * LocalDim). For each cell:
     *   - Column `2*j`     → neighbor in the negative direction of dimension `j`
     *   - Column `2*j + 1` → neighbor in the positive direction of dimension `j`
     *   - If a neighbor does not exist (e.g. boundary), the value is -1
     * 
     * @return Matrix of neighbor IDs
     */
    Eigen::Matrix<int, Dynamic, 2 * LocalDim, Eigen::RowMajor> neighbors() const {
        Eigen::Matrix<int, Eigen::Dynamic, 2 * LocalDim, Eigen::RowMajor> neighbors;
        neighbors.resize(n_cells_, 2 * LocalDim);
        for (int i = 0; i < n_cells_; ++i) {
            auto multi_index = compute_multi_index_(i);
            for (int j = 0; j < LocalDim; ++j) {
                if (multi_index[j] > 0) {
                    auto updated_index = multi_index;
                    updated_index[j] -= 1; // Update the index in the negative direction
                    neighbors(i, 2 * j) = compute_id_(updated_index);
                } else {
                    neighbors(i, 2 * j) = -1; // No neighbor in the negative direction
                }
                if (multi_index[j] < compute_stride_(j, true) - 1) {
                    auto updated_index = multi_index;
                    updated_index[j] += 1; // Update the index in the positive direction
                    neighbors(i, 2 * j + 1) = compute_id_(updated_index);
                } else {
                    neighbors(i, 2 * j + 1) = -1; // No neighbor in the positive direction
                }
            }
        }

        return neighbors;
    }
    
    // cells having this node as vertex, to implement if needed
    /*
    std::vector<int> node_patch(int id){
        std::vector<int> patch;
        auto multi_index = compute_multi_index_(id,false);
        for(int i = 0; i < n_cells_; ++i){

        }
        return patch;
    }
    */

    /**
     * @brief Compute the maximum cell diameter (h_max) across the mesh.
     * 
     * The diameter is the Euclidean distance between diagonally opposite parametric nodes of each cell,
     * mapped to physical space.
     * 
     * @return h_max value (maximum cell size in physical space)
     */
    double h_max() const {
        double max_diameter = 0.0;

        for (int cid = 0; cid < n_cells_; ++cid) {
            auto [u0, u1] = compute_lr_vertices(cid);             // Parametric bounding corners
            Eigen::Matrix<double, EmbedDim, 1> x0 = eval_param(u0); // Map to physical space
            Eigen::Matrix<double, EmbedDim, 1> x1 = eval_param(u1); // Map to physical space

            double diameter = (x1 - x0).norm();                   // Euclidean distance
            if (diameter > max_diameter) {
                max_diameter = diameter;
            }
        }

        return max_diameter;
    }


    /// wrapper for the compute multiindex for a cell
    std::array<int, LocalDim> cell_multi_index(int id) const {
        return compute_multi_index_(id);
    }

    // === Cell Iteration & Marking === //
    
    /// Iterator for cells in the mesh
    class cell_iterator: public internals::filtering_iterator<cell_iterator,const CellType*>   { 
    private:
        using Base = internals::filtering_iterator<cell_iterator, const CellType*>;
        using Base::index_;
        friend Base;
        const Derived* mesh_ = nullptr;
        int marker_ ; 
        std::shared_ptr<CellType> cell_ptr_;  // Shared pointer to the cell object

        cell_iterator& operator()(int i) {
            *cell_ptr_ = mesh_->cell(i);
            Base::val_ = cell_ptr_.get();
            return *this;
        }

    public:
        using MeshType = Derived;
        cell_iterator() = default;

        cell_iterator(int index, const Derived* mesh, const BinaryVector<Dynamic>& filter, int marker):
            Base(index, 0, mesh->n_cells_, filter), mesh_(mesh), marker_(marker), cell_ptr_(std::make_shared<CellType>()) {
            for (; index_ < Base::end_ && !filter[index_]; ++index_);
            if (index_ != Base::end_) { operator()(index_); }
        }

        cell_iterator(int index, const Derived* mesh, int marker):
            cell_iterator(index, mesh, marker == TriangulationAll ? BinaryVector<Dynamic>::Ones(mesh->n_cells_) :
                make_binary_vector(mesh->cells_markers_.begin(), mesh->cells_markers_.end(), marker), marker) { }

        int marker() const { return marker_; }

    };

    CellIterator<Derived> cells_begin(int marker = TriangulationAll) const {
        fdapde_assert(marker == TriangulationAll || (marker >= 0 && cells_markers_.size() != 0));
        return CellIterator<Derived>(0, static_cast<const Derived*>(this), marker);
    }
    CellIterator<Derived> cells_end(int marker = TriangulationAll) const {
        fdapde_assert(marker == TriangulationAll || (marker >= 0 && cells_markers_.size() != 0));
        return CellIterator<Derived>(n_cells_, static_cast<const Derived*>(this), marker);
    }

    // === Marker Utilities === //

    /**
     * @brief Mark cells using a lambda that returns true or false
     * 
     * @tparam Lambda Predicate that takes a Cell and returns a bool
     * @param marker Marker value to assign
     * @param lambda Filtering lambda function
     */
    template <typename Lambda> void mark_cells(int marker, Lambda&& lambda)
        requires(requires(Lambda lambda, CellType c) {
            { lambda(c) } -> std::same_as<bool>;
        }) {
        fdapde_assert(marker >= 0);
        cells_markers_.resize(n_cells_);
        for (cell_iterator it = cells_begin(); it != cells_end(); ++it) {
            cells_markers_[it->id()] = lambda(*it) ? marker : Unmarked;
        }
    }

    /// Mark cells with a binary mask (1D Eigen or BinaryVector)
    template <int Rows, typename XprType> void mark_cells(const BinMtxBase<Rows, 1, XprType>& mask) {
        fdapde_assert(mask.rows() == n_cells_);
        cells_markers_.resize(n_cells_);
        for (cell_iterator it = cells_begin(); it != cells_end(); ++it) {
            cells_markers_[it->id()] = mask[it->id()] ? 1 : 0;
        }
    }
    
    /// Mark cells from an iterator range (e.g., vector<int>)
    template <typename Iterator> void mark_cells(Iterator first, Iterator last) {
        fdapde_static_assert(
          std::is_convertible_v<typename Iterator::value_type FDAPDE_COMMA int>, INVALID_ITERATOR_RANGE);
        int n_markers = std::distance(first, last);
        bool all_markers_positive = std::all_of(first, last, [](auto marker) { return marker >= 0; });
        fdapde_assert(n_markers == n_cells_ && all_markers_positive);
        cells_markers_.resize(n_cells_, Unmarked);
        for (int i = 0; i < n_cells_; ++i) { cells_markers_[i] = *(first + i); }
    }
    /// Mark all cells with the same value
    void mark_cells(int marker) {   
        fdapde_assert(marker >= 0);
        cells_markers_.resize(n_cells_);
	std::for_each(cells_markers_.begin(), cells_markers_.end(), [marker](int& marker_) { marker_ = marker; });
    }
    /// Clear all cell markers
    void clear_cell_markers() {
        std::for_each(cells_markers_.begin(), cells_markers_.end(), [](int& marker) { marker = Unmarked; });
    }

    protected:

    // === Protected Utilities === //

    /**
     * @brief Computes the linear stride for flattening multi-indices. It counts only the geometric knots, no repetitions.
     * 
     * @param dim Target dimension (0-based)
     * @param is_cell True for cell-based stride, false for node-based
     * @return Stride for dimension `dim`
     */
    int compute_stride_(int dim, bool is_cell) const {
        int stride = 1;
        for (int d = 0; d < dim; ++d) {
            stride *= is_cell ? (this->param_nodes_[dim - d - 1].size() - 1) : this->param_nodes_[dim - d - 1].size();
        }
        return stride;
    }
    
    /// Convert a multi-index to a flat ID for a cell (or a node if is_cell is false)
    int compute_id_(const std::array<int, LocalDim>& multi_index, bool is_cell=true) const {
        int id = multi_index[0]; 
        for (int i = 1; i < LocalDim; ++i) { 
            id = id * compute_stride_(i, is_cell) + multi_index[i];
        }
        return id;
    }
    /// Convert a flat ID to a multi-index for a cell (or a node if is_cell is false)
    std::array<int, LocalDim> compute_multi_index_(int id, bool is_cell=true) const {
        std::array<int, LocalDim> multi_index;
        for (int i = LocalDim - 1; i >= 0; --i) {  
            int stride = (i == 0) ? 1 : compute_stride_(i , is_cell); 
            multi_index[i] = id / stride;  
            id %= stride; 
        }
        return multi_index;
    }
    
    /**
     * @brief Checks if each parametric direction is periodic.
     * 
     * Compares start and end coordinates across the domain in each direction.
     * Updates `periodic_dims_` accordingly.
     */
    void detect_periodicity_(){
        std::array<int, LocalDim> index = {0};
        for(int k=0;k<LocalDim; k++){
            bool periodic = true;
            do{
                Eigen::Matrix<double, LocalDim, 1> u_start, u_end;
                for(int i = 0; i < LocalDim; i++){ 
                    if(i == k){
                        u_start(i) = param_nodes_[i][0];
                        u_end(i) = param_nodes_[i][param_nodes_[i].size()-1];
                    } else 
                        u_start(i) = u_end(i) = param_nodes_[i][index[i]];
                }
                Eigen::Matrix<double, EmbedDim, 1> P_start = eval_param(u_start);
                Eigen::Matrix<double, EmbedDim, 1> P_end = eval_param(u_end);

                if ((P_start - P_end).norm() > 1e-9) {
                    periodic = false;
                    break;
                }

                // Increment indices except for the fixed dimension k
                for (int j = LocalDim - 1; j >= 0; j--) {
                    if (j == k) continue;  // Skip fixed dimension
                    index[j]++;
                    if (index[j] < param_nodes_[j].size()) break;  // No carry-over needed
                    else index[j] = 0;  // Reset and carry over to next dimension
                    
                }
            } while(index != std::array<int, LocalDim>{0});

            if(periodic) periodic_dims_[k] = true;
        }

    }

    /**
     * @brief Computes AABB (axis-aligned bounding boxes) for each parametric span.
     * 
     * Used to accelerate point inversion and spatial queries.
     * Stores the result in `span_aabbs_`, mapping multi-indices (e.g., {i, j}) to (P_min, P_max)
     */
    void compute_span_aabbs_() {
        //span_aabbs_.clear();
        //std::cout<<"Computing AABBs for spans..."<<std::endl;
        auto Cp = this->control_points_;
        std::array<decltype(Cp.template slice<LocalDim>(0)), EmbedDim> cp_slices;
        for (int i = 0; i < EmbedDim; ++i)
            cp_slices[i] = Cp.template slice<LocalDim>(i);
    
        std::array<int, LocalDim> index = this->degree_;
        bool done = false;
        //std::cout<<"Computing AABBs for each span..."<<std::endl;
        do {
            std::array<int, LocalDim> new_index;
            for (int i = 0; i < LocalDim; ++i)
                new_index[i] = index[i] - this->degree_[i];
    
            Eigen::Matrix<double, EmbedDim, 1> P_min, P_max;
            P_min.setConstant(std::numeric_limits<double>::max());
            P_max.setConstant(std::numeric_limits<double>::lowest());
    
            bool span_done = false;
            do {
                Eigen::Matrix<double, EmbedDim, 1> cp;
                for (int i = 0; i < EmbedDim; ++i)
                    cp(i) = cp_slices[i](new_index);
    
                P_min = P_min.cwiseMin(cp);
                P_max = P_max.cwiseMax(cp);
    
                for (int d = LocalDim - 1; d >= 0; --d) {
                    if (++new_index[d] > index[d]) {
                        new_index[d] = index[d] - this->degree_[d];
                        if (d == 0) span_done = true;
                    } else break;
                }
            } while (!span_done);
            //std::cout<<"Span AABB for index {";
    
            span_aabbs_[index] = std::make_pair(P_min, P_max);
    
            for (int d = LocalDim - 1; d >= 0; --d) {
                if (++index[d] > this->weights_.extent(d) - 1) {
                    index[d] = this->degree_[d];
                    if (d == 0) done = true;
                } else break;
            }
        } while (!done);
    }

    // === Member Variables === //
    
    std::array<std::vector<double>,LocalDim> knots_;            ///< Knot vectors in each direction
    std::array<int,LocalDim> degree_ {};                         ///< Polynomial degree in each direction
    std::array<std::vector<double>,LocalDim> param_nodes_;      ///< Unique parametric node positions in each direction
    MdArray<double,full_dynamic_extent_t<LocalDim>> weights_;   ///< NURBS weights in each direction
    MdArray<double,full_dynamic_extent_t<LocalDim+1>> control_points_; ///< Control points in each direction
    NurbsBasis<LocalDim> basis_;                                ///< NURBS basis functions
    
    Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>  cells_ {};  ///< Connectivity: cells x node IDs
    std::array<bool, LocalDim> periodic_dims_ = {false};        ///< Periodicity flags for each dimension

    int n_nodes_ = 0, n_cells_ = 0;
    int flags_ = 0;

    //BinaryVector<Dynamic> boundary_markers_ {};               ///< Boundary markers for each cell
    std::vector<int> cells_markers_ {};                         ///< Marker for each cell       
    std::vector<int> nodes_markers_ {};                         ///< Marker for each node
    
    std::map<std::array<int, LocalDim>, std::pair<
        Eigen::Matrix<double, EmbedDim, 1>, 
        Eigen::Matrix<double, EmbedDim, 1>>> span_aabbs_;       ///< AABB boxes per span
        
};

/**
 * @brief Specialization of IsoMesh for 2D parametric meshes in 2D or 3D space.
 * @see IsoMeshBase
 */
template <int N> class IsoMesh<2, N>: public IsoMeshBase<2, N, IsoMesh<2, N>> {
    fdapde_static_assert(N == 2 || N == 3, THIS_CLASS_IS_FOR_2D_OR_3D_MESHES_ONLY);
    public:
    using Base = IsoMeshBase<2, N, IsoMesh<2, N>>;
    static constexpr int n_nodes_per_edge = 2;
    static constexpr int n_edges_per_cell = 4;
    static constexpr int n_faces_per_edge = 2;

    using EdgeType = typename Base::CellType::EdgeType; 
    using Base::embed_dim;
    using Base::local_dim;
    using Base::n_cells_;
    using Base::n_nodes_per_cell;
    static constexpr std::array<std::array<int, 2>, 4> edge_pattern = {{
        {0, 1},  // Left edge
        {1, 2},  // Top edge
        {2, 3},  // Right edge
        {3, 0}   // Bottom edge
    }};

    IsoMesh() = default;
    
    /**
     * @brief Construct a 2D IsoMesh from NURBS data.
     * 
     * @see IsoMeshBase::IsoMeshBase
     */
    IsoMesh(std::array<std::vector<double>, 2>& knots, MdArray<double, MdExtents<Dynamic, Dynamic>>& weights,
         MdArray<double, MdExtents<Dynamic,Dynamic,Dynamic>>& control_points, std::array<int,2> degree, int flags = 0) :
        Base(knots, weights, control_points, degree, flags) {  
            compute_cells_();  
        }

    protected:

    // === Protected Member Functions === //

    /** 
    * @brief Compute the cells and edges of the mesh.
    * 
    * This function computes the cells and edges of the mesh based on the
    * parametric nodes and control points. It also detects periodicity in the
    * mesh and computes AABB boxes for each span.
    */
    void compute_cells_(){

        edges_ = {};
        edge_to_cells_ = {};
        boundary_edges_ = {};
        cell_to_edges_ = {};

        if(Base::flags_ & cache_cells){ // & cache_cells
            cell_cache_.reserve(n_cells_);
            for(int i = 0; i < n_cells_; ++i){ cell_cache_.emplace_back(i, this);}
        }
        //compute cells
        this->cells_.resize(n_cells_, n_nodes_per_cell);
        for(int i=0;i<n_cells_; i++){
            auto multi_index = this->compute_multi_index_(i);
            int i_x = multi_index[0]; // Extract X index
            int i_y = multi_index[1]; // Extract Y index
            // Convert multi-index to actual node indices
            int n_x = this->param_nodes_[0].size();  // Number of nodes in X direction
            int n_y = this->param_nodes_[1].size();  // Number of nodes in Y direction
            // Store quadrilateral connectivity
            this->cells_(i, 0) = i_y * n_x + i_x; // bottom left
            this->cells_(i, 1) = i_y * n_x + (i_x + 1); // bottom right
            this->cells_(i, 2) = (i_y + 1) * n_x + (i_x + 1); // top right
            this->cells_(i, 3) = (i_y + 1) * n_x + i_x; // top left

        }
        using edge_t = std::array<int, n_nodes_per_edge>;
        using hash_t = internals::std_array_hash<int, n_nodes_per_edge>;
        struct edge_info {
            int edge_id, face_id;   // for each edge, its ID and the ID of one of the cells insisting on it
        };
        std::unordered_map<edge_t, edge_info, hash_t> edges_map;
        std::vector<bool> boundary_edges;
        edge_t edge;
        cell_to_edges_.resize(n_cells_, n_edges_per_cell);

        // Edge assignmen process
        int edge_id = 0;
        for(int i = 0; i < n_cells_; ++i){
            for(int j= 0; j<n_edges_per_cell; ++j){ // 4 edges per quadriteral
                // extract and normalize the edge
                for(int k = 0; k<n_nodes_per_edge; k++) { // 2 edges per node
                    edge[k] = this->cells_(i,edge_pattern[j][k]);
                }
                std::sort(edge.begin(), edge.end()); // ensure unique representation

                auto it = edges_map.find(edge);
                if(it == edges_map.end()){
                    // New edge: Assign a new ID and mark as boundary
                    edges_.insert(edges_.end(),edge.begin(),edge.end());
                    edge_to_cells_.insert(edge_to_cells_.end(), {i,-1});
                    boundary_edges.push_back(true);
                    edges_map.emplace(edge,edge_info{edge_id, i});
                    cell_to_edges_(i,j) = edge_id;
                    edge_id++;
                } else{
                    // Edge already exists: Cells i and neighbor share this edge
                    int existing_edge_id = it->second.edge_id;
                    cell_to_edges_(i,j) = existing_edge_id;
                    boundary_edges[existing_edge_id] = false; // Mark internal edge
                    edge_to_cells_[2*existing_edge_id + 1] = i;
                    edges_map.erase(it);
                } 
            }
        }

        n_edges_ = edges_.size() / 2;

        // **Step 2: Adjust boundary edges based on periodicity**
        std::vector<std::pair<Eigen::Matrix<double, embed_dim, 1>, Eigen::Matrix<double, embed_dim, 1>>> boundary_edge_list;
        std::vector<int> boundary_edge_indices; // Store indices of boundary edges

        // **Step 2.1: Collect boundary edges and their physical node positions**
        for (int i = 0; i < n_edges_; ++i) {
            if (!boundary_edges[i]) continue; // Skip non-boundary edges

            // Extract physical node coordinates
            int node0 = edges_[2 * i];   // First node of the edge
            int node1 = edges_[2 * i + 1]; // Second node of the edge

            Eigen::Matrix<double, embed_dim, 1> p0 = this->phys_node(node0);
            Eigen::Matrix<double, embed_dim, 1> p1 = this->phys_node(node1);

            // Ensure degreeing is consistent to avoid duplicate mismatches
            if (p0.norm() > p1.norm()) std::swap(p0, p1);

            boundary_edge_list.emplace_back(p0, p1);
            boundary_edge_indices.push_back(i);
        }

        // **Step 2.2: Check for duplicate edges**
        for (size_t i = 0; i < boundary_edge_list.size(); i++) {
            for (size_t j = i + 1; j < boundary_edge_list.size(); j++) {
                if ((boundary_edge_list[i].first - boundary_edge_list[j].first).norm() < 1e-8 &&
                    (boundary_edge_list[i].second - boundary_edge_list[j].second).norm() < 1e-8) {
                    // Mark both edges as NOT on the boundary
                    boundary_edges[boundary_edge_indices[i]] = false;
                    boundary_edges[boundary_edge_indices[j]] = false;
                }
            }
        }

        // **Step 2.3: Convert back to `BinaryVector`**
        boundary_edges_ = BinaryVector<Dynamic>(boundary_edges.begin(), boundary_edges.end(), n_edges_);
    }

    public:
    
    /**
     * @brief Refine the mesh by inserting new knots. Specializes the base class method.
     * 
     * @see IsoMeshBase::refine_knots
     */
    void refine_knots(const std::array<int, 2>& density = std::array<int, 2>{{1, 1}}, std::array<std::vector<double>, 2> add_knot_list = {}) {
        Base::refine_knots(density, add_knot_list);
        compute_cells_();
    }

    /**
     * @brief Create a square IsoMesh over the domain [0,L] x [0,L]
     *
     * @param L Length of the square side (default = 1.0)
     * @return IsoMesh<2,N> representing the square
     */
    static IsoMesh<2, N> square(double L = 1.0) {
        static_assert(N == 2 || N == 3, "This method is only valid for N=2 or N=3");
    
        // Quadratic open uniform knot vector: degree 2 => need 3 repeated knots at each end
        std::array<std::vector<double>, 2> knots = {
            std::vector<double>{0.0, 0.0, 0.0, 1.0, 1.0, 1.0},
            std::vector<double>{0.0, 0.0, 0.0, 1.0, 1.0, 1.0}
        };
    
        std::array<int, 2> degree = {2, 2};
    
        // 3 control points per direction (degree + 1 for open knot vector)
        MdArray<double, MdExtents<Dynamic, Dynamic>> weights(3, 3);
        MdArray<double, MdExtents<Dynamic, Dynamic, Dynamic>> control_points(3, 3, N);
    
        for (int i = 0; i < 3; ++i) {
            double u = i * L / 2.0;  // Since domain goes from 0 to L with 3 points
            for (int j = 0; j < 3; ++j) {
                double v = j * L / 2.0;
    
                weights(i, j) = 1.0;
    
                control_points(i, j, 0) = u;
                control_points(i, j, 1) = v;
    
                if constexpr (N == 3)
                    control_points(i, j, 2) = 0.0;
            }
        }
    
        return IsoMesh<2, N>(knots, weights, control_points, degree);
    }

    /**
     * @brief Create a 2D NURBS mesh of a sphere surface by revolving a semicircle. 
     * 
     * @param r Radius of the sphere
     * @return IsoMesh<2, 3> mesh representing a sphere
     */
    static IsoMesh<2,N> sphere(double r = 1.0) {
        fdapde_static_assert(N == 3, THIS_METHOD_IS_ONLY_FOR_3D_MANIFOLDS);
        fdapde_assert(r > 0);

        // Create a semicircle as a 1D Mesh embdedded in 3D
        std::array<std::vector<double>, 1> start_knots = { std::vector<double>{0, 0, 0, 0.5, 1, 1, 1} };
        std::array<int,1> start_degree = {2};
        int num_ctrl_points = start_knots[0].size() - start_degree[0] - 1;
        std::vector<double> wj = { 1, 1 / 2.0, 1 / 2.0, 1 };

        std::vector<std::vector<double>> Pj = {
            {0, 0, r},  {0, r, r},  {0, r, -r}, {0, 0, -r}
        };

        // Initialize `MdArray`
        MdArray<double, MdExtents<Dynamic>> start_weights(num_ctrl_points);
        MdArray<double, MdExtents<Dynamic, Dynamic>> start_cp(num_ctrl_points, 3);

        // Fill `start_weights` with values from `wj`
        for (int i = 0; i < num_ctrl_points; i++) {
            start_weights(i) = wj[i];
        }

        // Fill `start_cp` with control points `Pj`
        for (int i = 0; i < num_ctrl_points; i++) {
            for (int j = 0; j < 3; j++) {
                start_cp(i, j) = Pj[i][j];
            }
        }

        // Create a IsoMeshData object
        IsoMeshData<1> semicircle(start_knots, start_weights, start_cp, start_degree);

        // Create a 2D mesh by rotating the 1D mesh around the z-axis
        IsoMeshData<2> sphere = iso_algorithms::create_revolved_ISO_surface(semicircle, 2 * M_PI, Eigen::Matrix<double,3,1>(0,0,1));

        // Create the IsoMesh object

        IsoMesh<2, N> mesh(sphere.knots, sphere.weights, sphere.control_points, sphere.degree);
        return mesh;

    }

    /**
     * @brief Create a 2D NURBS mesh of a torus surface by revolving a circle.
     * 
     * @param R Major radius of the torus
     * @param r Minor radius of the torus
     * @return IsoMesh<2, 3> mesh representing a torus
     */
    static IsoMesh<2,N> torus(double R = 2. , double r = 1.) {
        fdapde_static_assert(N == 3, THIS_METHOD_IS_ONLY_FOR_3D_MANIFOLDS);
        fdapde_assert(R > 0 && r > 0 && R - r > 0 && R + r > 0);

        std::array<std::vector<double>, 1> start_knots = {
            std::vector<double>{0,0,0,0.25,0.25,0.5,0.5,0.75,0.75,1,1,1}
        };
        std::array<int,1> start_degree = {2}; // Degree 2 (quadratic)
        int num_ctrl_points = 9;
        
        std::vector<double> wj = {
            1.0, std::sqrt(2.0)/2.0, 1.0,
            std::sqrt(2.0)/2.0, 1.0, std::sqrt(2.0)/2.0,
            1.0, std::sqrt(2.0)/2.0, 1.0
        };
        
        std::vector<std::vector<double>> Pj = {
            { 0, R + r, 0 },
            { 0, R + r, r },
            { 0, R,     r },
            { 0, R - r, r },
            { 0, R - r, 0 },
            { 0, R - r, -r },
            { 0, R,     -r },
            { 0, R + r, -r },
            { 0, R + r, 0 }
        };

        // Initialize `MdArray`
        MdArray<double, MdExtents<Dynamic>> start_weights(num_ctrl_points);
        MdArray<double, MdExtents<Dynamic, Dynamic>> start_cp(num_ctrl_points, 3);

        // Fill `start_weights` with values from `wj`
        for (int i = 0; i < num_ctrl_points; i++) {
            start_weights(i) = wj[i];
        }

        // Fill `start_cp` with control points `Pj`
        for (int i = 0; i < num_ctrl_points; i++) {
            for (int j = 0; j < 3; j++) {
                start_cp(i, j) = Pj[i][j];
            }
        }


        // Create a IsoMeshData object
        IsoMeshData<1> circle(start_knots, start_weights, start_cp, start_degree);

        // Create a 2D mesh by rotating the 1D mesh around the z-axis
        IsoMeshData<2> torus = iso_algorithms::create_revolved_ISO_surface(circle, 2 * M_PI, Eigen::Matrix<double,3,1>(0,0,1));

        // Create the IsoMesh object

        IsoMesh<2, N> mesh(torus.knots, torus.weights, torus.control_points, torus.degree);
        return mesh;

        
    }

    // quarter of a ring, degree (1,2)

    static IsoMesh<2,N> quarter_ring(double R = 2. , double r = 1.) {
        //fdapde_static_assert(N == 3, THIS_METHOD_IS_ONLY_FOR_3D_MANIFOLDS);
        fdapde_assert(R > 0 && r > 0 && R - r > 0 && R + r > 0);

        std::array<std::vector<double>, 1> start_knots = {
            std::vector<double>{0,0,0,1,1,1}
        };
        std::array<int,1> start_degree = {2}; // Degree 1 
        int num_ctrl_points = 3;
        
        std::vector<double> wj = {
            1.0, 1.0, 1.0
        };
        
        std::vector<std::vector<double>> Pj = {
            { r, 0, 0 },
            {(r + R) / 2.0, 0, 0},
            { R, 0, 0 }
        };

        // Initialize `MdArray`
        MdArray<double, MdExtents<Dynamic>> start_weights(num_ctrl_points);
        MdArray<double, MdExtents<Dynamic, Dynamic>> start_cp(num_ctrl_points, 3);

        // Fill `start_weights` with values from `wj`
        for (int i = 0; i < num_ctrl_points; i++) {
            start_weights(i) = wj[i];
        }

        // Fill `start_cp` with control points `Pj`
        for (int i = 0; i < num_ctrl_points; i++) {
            for (int j = 0; j < 3; j++) {
                start_cp(i, j) = Pj[i][j];
            }
        }


        // Create a IsoMeshData object
        IsoMeshData<1> line(start_knots, start_weights, start_cp, start_degree);

        // Create a 2D mesh by rotating the 1D mesh around the z-axis
        IsoMeshData<2> quarter_ring = iso_algorithms::create_revolved_ISO_surface(line, M_PI/2, Eigen::Matrix<double,3,1>(0,0,1));

        // Create the IsoMesh object

        MdArray<double, full_dynamic_extent_t<local_dim+1>> new_cp;

        if constexpr(N== 2){
            new_cp.resize(quarter_ring.control_points.extent(0), quarter_ring.control_points.extent(1), 2);
            for(int i = 0; i < quarter_ring.control_points.extent(0); i++){
                for(int j = 0; j < quarter_ring.control_points.extent(1); j++){
                    new_cp(i,j,0) = quarter_ring.control_points(i,j,0);
                    new_cp(i,j,1) = quarter_ring.control_points(i,j,1);
                }
            }
            
        } else {
            new_cp = quarter_ring.control_points;
        }

        IsoMesh<2, N> mesh(quarter_ring.knots, quarter_ring.weights, new_cp, quarter_ring.degree);
        return mesh;

        
    }
    
    
    // === Getters === //

    const typename Base::CellType& cell(int id) const {
        if (Base::flags_) {   // cell caching enabled
            return cell_cache_[id];
        } else {
            cell_ = typename Base::CellType(id, this);
            return cell_;
        }
    }

    bool is_edge_on_boundary(int id) const { return boundary_edges_[id]; }

    Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>> edges() const {
        return Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>>(
          edges_.data(), n_edges_, n_nodes_per_edge);
    }
    Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>> edge_to_cells() const {
        return Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>>(
          edge_to_cells_.data(), n_edges_, 2);
    }
    const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>& cell_to_edges() const { return cell_to_edges_; }
    const BinaryVector<Dynamic>& boundary_edges() const { return boundary_edges_; }
    int n_edges() const { return n_edges_; }
    int n_boundary_edges() const { return boundary_edges_.count(); }


    // === Iterators === //

    class edge_iterator : public internals::filtering_iterator<edge_iterator, EdgeType> {
        protected:
        using Base =  internals::filtering_iterator<edge_iterator, EdgeType>;
        using Base::index_;
        friend Base;
        const IsoMesh* mesh_;
        int marker_;

        edge_iterator& operator() (int i){
            Base::val_ = EdgeType(i, mesh_);
            return *this;
        }
        public:
        using MeshType = IsoMesh<2, N>;
        edge_iterator(int index, const MeshType* mesh, const BinaryVector<Dynamic>& filter, int marker) :
            Base(index,0,mesh->n_edges_,filter), mesh_(mesh), marker_(marker) {
                for(; index_<Base::end_ && !filter[index_]; ++index_);
                if(index_ != Base::end_){ operator() (index_);}
        }
        edge_iterator(int index, const MeshType* mesh): //apply no filter
            edge_iterator(index, mesh, BinaryVector<Dynamic>::Ones(mesh->n_edges_), Unmarked){}
        edge_iterator(int index, const MeshType* mesh, int marker) : 
            Base(index, 0, mesh->n_edges_), marker_(marker) { }
        int marker() const { return marker_;}
    };

    edge_iterator edges_begin() const {return edge_iterator(0,this);}
    edge_iterator edges_end() const {return edge_iterator(n_edges_,this,Unmarked);}
    // iterator over boundary edges
    struct boundary_edge_iterator : public edge_iterator {
        using MeshType = IsoMesh<2, N>;
        boundary_edge_iterator(int index, const MeshType* mesh) :
            edge_iterator(index, mesh, mesh->boundary_edges_, BoundaryAll) {}
        boundary_edge_iterator(int index, const MeshType* mesh, int marker) :
            edge_iterator(
                index, mesh, 
                marker == BoundaryAll ? 
                mesh->boundary_edges_ : 
                mesh->boundary_edges_ & 
                 make_binary_vector(mesh->edges_markers_.begin(), mesh->edges_markers_.end(),marker),
                marker) { 
                    
                }
    };
    boundary_edge_iterator boundary_edges_begin() const {return boundary_edge_iterator(0, this);}
    boundary_edge_iterator boundary_edges_end() const {return boundary_edge_iterator(n_edges_, this);}
    using boundary_iterator = boundary_edge_iterator; // public view of 2d boundary
    BoundaryIterator<IsoMesh<2,N>> boundary_begin(int marker = BoundaryAll) const {
        return BoundaryIterator<IsoMesh<2,N>>(0,this, marker);
    }
    BoundaryIterator<IsoMesh<2,N>> boundary_end(int marker = BoundaryAll) const {
        return BoundaryIterator<IsoMesh<2,N>>(n_edges_,this, marker);
    }
    std::pair<BoundaryIterator<IsoMesh<2,N>>, BoundaryIterator<IsoMesh<2,N>>>
    boundary(int marker = BoundaryAll) const {
        return std::make_pair(boundary_begin(marker), boundary_end(marker));
    }
    const std::vector<int>& edges_markers() const {return edges_markers_;}

    // === Marker Utilities === //

    template<typename Lambda> void mark_boundary(int marker, Lambda&& lambda)
        requires(requires(Lambda lambda, EdgeType e){
            {lambda(e)} -> std::same_as<bool>;
        }) {
            fdapde_assert(marker >= 0);
            edges_markers_.resize(n_edges_);
            for(boundary_edge_iterator it = boundary_edges_begin(); it!=boundary_edges_end();++it){
                if (lambda(*it)) {
                    edges_markers_[it->id()] = marker;
                }
            }
    }
    template <int Rows, typename XprType> 
    void mark_boundary(const BinMtxBase<Rows,1,XprType>& mask){
        fdapde_assert(mask.rows() == n_edges_);
        edges_markers_.resize(n_edges_, 0);
        for (boundary_edge_iterator it = boundary_edges_begin(); it != boundary_edges_end(); ++it) {
            if(mask[it->id()]){
                edges_markers_[it->id()] = 1;
            }  
        }
    }
    template <typename Iterator> void mark_boundary(Iterator first, Iterator last){
        fdapde_static_assert(std::is_convertible_v<typename Iterator::value_type FDAPDE_COMMA int>, INVALID_ITERATOR_RANGE);
        int n_markers = std::distance(first, last);
        bool all_markers_positive = std::all_of(first, last, [](auto marker) { return marker >= 0; });
        fdapde_assert(n_markers == n_edges() && all_markers_positive);
        edges_markers_.resize(n_edges_, Unmarked);
        for (int i = 0; i < n_edges_; ++i) { edges_markers_[i] = *(first + i); }
        return;
    }
    void mark_boundary(int marker) {
        fdapde_assert(marker>=0);
        edges_markers_.resize(n_edges_, Unmarked);
        std::for_each(edges_markers_.begin(), edges_markers_.end(),[marker](int& marker_) { marker_ = marker; } );
    }
    void clear_boundary_markers() {
        std::for_each(edges_markers_.begin(), edges_markers_.end(), [](int& marker){marker = Unmarked; });

    }

    protected:
    std::vector<int> edges_ {};                        ///< for each edge, the ids of its nodes
    std::vector<int> edge_to_cells_ {};                ///< for each edge, the ids of the cells insisting on it
    Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor> cell_to_edges_ {};   ///< for each cell, the ids of its edges
    BinaryVector<Dynamic> boundary_edges_ {};          ///< for each edge, true if it is on the boundary
    std::vector<int> edges_markers_ {};                ///< for each edge, the marker of the cell it belongs to
    int n_edges_ = 0;                                  ///< number of edges in the mesh
    // cell caching
    std::vector<typename Base::CellType> cell_cache_;  ///< cache of cells
    mutable typename Base::CellType cell_;             ///< temporary cell object for non-cached access

};


/**
 * @brief Specialization of IsoMesh for 3D parametric meshes in 3D space.
 * @see IsoMeshBase
 */
template<> class IsoMesh<3,3>: public IsoMeshBase<3,3,IsoMesh<3,3>>{
    public:
    using Base = IsoMeshBase<3,3,IsoMesh<3,3>>;
    static constexpr int n_nodes_per_face = 4;
    static constexpr int n_nodes_per_edge = 2;
    static constexpr int n_edges_per_face = 4;
    static constexpr int n_faces_per_cell = 6;
    static constexpr int n_edges_per_cell = 12;
    using FaceType = typename Base::CellType::FaceType;
    using EdgeType = typename Base::CellType::EdgeType;
    using Base::embed_dim;
    using Base::local_dim;
    using Base::n_nodes_per_cell;
    static constexpr std::array<std::array<int, n_nodes_per_edge>, n_edges_per_cell> edge_pattern = {{
        {0, 1}, {1, 2}, {2, 3}, {3, 0},  // Front face edges
        {4, 5}, {5, 6}, {6, 7}, {7, 4},  // Back face edges
        {0, 4}, {1, 5}, {2, 6}, {3, 7}   // Vertical edges
        }};
    static constexpr std::array<std::array<int, n_nodes_per_face>, n_faces_per_cell> face_pattern = {{
        {0, 1, 2, 3},  // Front face
        {4, 5, 6, 7},  // Back face
        {0, 1, 5, 4},  // Left face
        {2, 3, 7, 6},  // Right face
        {0, 3, 7, 4},  // Bottom face
        {1, 2, 6, 5}   // Top face
        }};
    
    IsoMesh() =  default;

    /**
     * @brief Construct a 3D IsoMesh from NURBS data.
     * @see IsoMeshBase::IsoMeshBase
     */
    IsoMesh(std::array<std::vector<double>, 3>& knots,MdArray<double, MdExtents<Dynamic, Dynamic, Dynamic>>& weights,
         MdArray<double, MdExtents<Dynamic,Dynamic,Dynamic,Dynamic>>& control_points, std::array<int,3> degree, int flags = 0):
         Base(knots, weights, control_points, degree, flags) {
            compute_cells_();
    }

    protected:

    void compute_cells_(){
        edges_ = {};
        edge_to_cells_ = {};
        boundary_edges_ = {};
        cell_to_edges_ = {};
        faces_ = {};
        face_to_cells_ = {};
        boundary_faces_ = {};
        cell_to_faces_ = {};
        
        using face_t = std::array<int, n_nodes_per_face>;
        using edge_t = std::array<int, n_nodes_per_edge>;

        struct face_info {
            int face_id, cell_id;
        };

        using edge_info = int;

        std::unordered_map<edge_t, edge_info, internals::std_array_hash<int, n_nodes_per_edge>> edges_map;
        std::unordered_map<face_t, face_info, internals::std_array_hash<int, n_nodes_per_face>> faces_map;
        std::vector<bool> boundary_faces, boundary_edges;
        face_t face;
        edge_t edge;

        cell_to_faces_.resize(n_cells_, n_faces_per_cell);
        cell_to_edges_.resize(n_cells_, n_edges_per_cell);

        // Compute cell connectivity
        this->cells_.resize(n_cells_, n_nodes_per_cell);
        for (int i = 0; i < n_cells_; ++i) {
            auto multi_index = this->compute_multi_index_(i);
            int i_x = multi_index[0];
            int i_y = multi_index[1];
            int i_z = multi_index[2];

            int n_x = this->param_nodes_[0].size();
            int n_y = this->param_nodes_[1].size();
            int n_z = this->param_nodes_[2].size();

            // Assign cube vertex indices
            this->cells_(i, 0) = i_z * n_x * n_y + i_y * n_x + i_x; // Bottom-left-front
            this->cells_(i, 1) = i_z * n_x * n_y + i_y * n_x + (i_x + 1); // Bottom-right-front
            this->cells_(i, 2) = i_z * n_x * n_y + (i_y + 1) * n_x + (i_x + 1); // Bottom-right-back
            this->cells_(i, 3) = i_z * n_x * n_y + (i_y + 1) * n_x + i_x; // Bottom-left-back
            this->cells_(i, 4) = (i_z + 1) * n_x * n_y + i_y * n_x + i_x; // Top-left-front
            this->cells_(i, 5) = (i_z + 1) * n_x * n_y + i_y * n_x + (i_x + 1); // Top-right-front
            this->cells_(i, 6) = (i_z + 1) * n_x * n_y + (i_y + 1) * n_x + (i_x + 1); // Top-right-back
            this->cells_(i, 7) = (i_z + 1) * n_x * n_y + (i_y + 1) * n_x + i_x; // Top-left-back

        }

        // Edge assignment process
        int edge_id = 0;
        for (int i = 0; i < n_cells_; ++i) {
            for (int j = 0; j < n_edges_per_cell; ++j) {
                for (int k = 0; k < n_nodes_per_edge; k++) {
                    edge[k] = this->cells_(i, edge_pattern[j][k]);
                }
                std::sort(edge.begin(), edge.end());

                auto it = edges_map.find(edge);
                if (it == edges_map.end()) {
                    edges_.insert(edges_.end(), edge.begin(), edge.end());
                    edge_to_cells_.insert({edge_id, std::unordered_set<int>{i}}); // Corrected insertion
                    boundary_edges.push_back(true);
                    edges_map.emplace(edge, edge_id);
                    cell_to_edges_(i, j) = edge_id;
                    edge_id++;
                } else {
                    int existing_edge_id = it->second;
                    cell_to_edges_(i, j) = existing_edge_id;
                    boundary_edges[existing_edge_id] = false;
                    edge_to_cells_[existing_edge_id].insert(i); // Corrected assignment
                }
                

            }
        }

        n_edges_ = edges_.size() / 2;
        boundary_edges_ = BinaryVector<Dynamic>(boundary_edges.begin(), boundary_edges.end(), n_edges_);

        // Face assignment process
        int face_id = 0;
        for (int i = 0; i < n_cells_; ++i) {
            for (int j = 0; j < n_faces_per_cell; ++j) {
                for (int k = 0; k < n_nodes_per_face; k++) {
                    face[k] = this->cells_(i, face_pattern[j][k]);
                }
                std::sort(face.begin(), face.end());

                auto it = faces_map.find(face);
                if (it == faces_map.end()) {
                    faces_.insert(faces_.end(), face.begin(), face.end());
                    face_to_cells_.insert(face_to_cells_.end(), {i, -1});
                    boundary_faces.push_back(true);
                    faces_map.emplace(face, face_info{face_id, i});
                    cell_to_faces_(i, j) = face_id;
                    face_id++;
                } else {
                    int existing_face_id = it->second.face_id;
                    cell_to_faces_(i, j) = existing_face_id;
                    boundary_faces[existing_face_id] = false;
                    face_to_cells_[existing_face_id] = i; // Fixed assignment
                }

                // Compute `face_to_edges_`
                for (int e = 0; e < n_edges_per_face; ++e) {
                    int edge_id = cell_to_edges_(i, edge_pattern[e][0]);
                    face_to_edges_.push_back(edge_id);
                }
            }
        }

        n_faces_ = faces_.size() / n_nodes_per_face;

        // **Step 3: Adjust boundary faces based on periodicity**
        std::vector<std::pair<std::array<Eigen::Matrix<double, embed_dim, 1>, n_nodes_per_face>, int>> boundary_face_list;
        std::vector<int> boundary_face_indices; // Store indices of boundary faces

        // **Step 3.1: Collect boundary faces and their physical node positions**
        for (int i = 0; i < n_faces_; ++i) {
            if (!boundary_faces[i]) continue; // Skip non-boundary faces

            std::array<Eigen::Matrix<double, embed_dim, 1>, n_nodes_per_face> face_nodes;
            for (int j = 0; j < n_nodes_per_face; ++j) {
                int node_idx = faces_[n_nodes_per_face * i + j];
                face_nodes[j] = this->phys_node(node_idx);
            }

            boundary_face_list.emplace_back(face_nodes, i);
            boundary_face_indices.push_back(i);
        }

        // **Step 3.2: Check for duplicate faces**
        for (size_t i = 0; i < boundary_face_list.size(); i++) {
            for (size_t j = i + 1; j < boundary_face_list.size(); j++) {
                bool match = true;
                for (int k = 0; k < n_nodes_per_face; k++) {
                    if ((boundary_face_list[i].first[k] - boundary_face_list[j].first[k]).norm() > 1e-8) {
                        match = false;
                        break;
                    }
                }
                if (match) {
                    // Mark both faces as NOT on the boundary
                    boundary_faces[boundary_face_indices[i]] = false;
                    boundary_faces[boundary_face_indices[j]] = false;
                }
            }
        }

        // **Step 3.3: Convert back to `BinaryVector`**
        boundary_faces_ = BinaryVector<Dynamic>(boundary_faces.begin(), boundary_faces.end(), n_faces_);

    }


    public:
    /**
     * @brief Refine the mesh by inserting new knots. Specializes the base class method.
     * @see IsoMeshBase::refine_knots
     */
    void refine_knots(const std::array<int, 3>& density = std::array<int, 3>{{1, 1, 1}}, std::array<std::vector<double>, 3> add_knot_list = {}) {
        Base::refine_knots(density, add_knot_list);
        compute_cells_();
    }

    // === Getters === //

    const typename Base::CellType& cell(int id) const {
        if (Base::flags_) {   // cell caching enabled
            return cell_cache_[id];
        } else {
            cell_ = typename Base::CellType(id, this);
            return cell_;
        }
    }

    bool is_face_on_boundary(int id) const { return boundary_faces_[id]; }
    bool is_edge_on_boundary(int id) const { return boundary_edges_[id]; }

    Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>> faces() const {
        return Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>>(
          faces_.data(), n_faces_, n_nodes_per_face);
    }
    Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>> edges() const {
        return Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>>(
          edges_.data(), n_edges_, n_nodes_per_edge);
    }
    const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>& cell_to_faces() const { return cell_to_faces_; }
    Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>> face_to_edges() const {
        return Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>>(
          face_to_edges_.data(), n_faces_, n_edges_per_face);
    }
    Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>> face_to_cells() const {
        return Eigen::Map<const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>>(
          face_to_cells_.data(), n_faces_, 2);
    }
    const std::unordered_map<int, std::unordered_set<int>>& edge_to_cells() const { return edge_to_cells_; }
    const BinaryVector<Dynamic>& boundary_faces() const { return boundary_faces_; }
    int n_faces() const { return n_faces_; }
    int n_edges() const { return n_edges_; }
    int n_boundary_faces() const { return boundary_faces_.count(); }
    int n_boundary_edges() const { return boundary_edges_.count(); }

    const Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor>& cell_to_edges() const { return cell_to_edges_; }

    // === Iterators === //

    /// Iterator over edges
    class edge_iterator: public internals::filtering_iterator<edge_iterator, EdgeType> {
        protected:
        using Base = internals::filtering_iterator<edge_iterator, EdgeType>;
        using Base::index_;
        friend Base;
        const IsoMesh* mesh_;
        int marker_;

        edge_iterator& operator()(int i){
            Base::val_ = EdgeType(i, mesh_);
            return *this;
        }
        public:
        using MeshType = IsoMesh<3,3>;
        edge_iterator(int index, const MeshType* mesh, const BinaryVector<Dynamic>& filter, int marker):
            Base(index, 0, mesh->n_edges_, filter), mesh_(mesh), marker_(marker) {
                for(; index_<Base::end_ && !filter[index_]; ++index_);
                if(index_ != Base::end_){ operator()(index_);}
            }
        edge_iterator(int index, const MeshType* mesh):
            edge_iterator(index, mesh, BinaryVector<Dynamic>::Ones(mesh->n_edges_), Unmarked){}
        edge_iterator(int index, const MeshType* mesh, int marker):
            Base(index, 0, mesh->n_edges_), marker_(marker) { }
        int marker() const { return marker_; }
    };
    edge_iterator edges_begin() const { return edge_iterator(0, this); }
    edge_iterator edges_end() const { return edge_iterator(n_edges_, this); }

    /// Iterator over faces
    class face_iterator: public internals::filtering_iterator<face_iterator, FaceType> {
        protected:
        using Base = internals::filtering_iterator<face_iterator, FaceType>;
        using Base::index_;
        friend Base;
        const IsoMesh* mesh_;
        int marker_;

        face_iterator& operator()(int i){
            Base::val_ = FaceType(i, mesh_);
            return *this;
        }
        public:
        using MeshType = IsoMesh<3,3>;
        face_iterator(int index, const MeshType* mesh, const BinaryVector<Dynamic>& filter, int marker):
            Base(index, 0, mesh->n_faces_, filter), mesh_(mesh), marker_(marker) {
                for(; index_<Base::end_ && !filter[index_]; ++index_);
                if(index_ != Base::end_){ operator()(index_);}
            }
        face_iterator(int index, const MeshType* mesh):
            face_iterator(index, mesh, BinaryVector<Dynamic>::Ones(mesh->n_faces_), Unmarked){}
        face_iterator(int index, const MeshType* mesh, int marker):
            Base(index, 0, mesh->n_faces_), marker_(marker) { }
        int marker() const { return marker_; }
    };
    face_iterator faces_begin() const { return face_iterator(0, this); }
    face_iterator faces_end() const { return face_iterator(n_faces_, this); }

    /// Iterator over boundary faces
    struct boundary_face_iterator : public face_iterator {
        using MeshType = IsoMesh<3,3>;
        boundary_face_iterator(int index, const MeshType* mesh) :
            face_iterator(index, mesh, mesh->boundary_faces_, BoundaryAll) {}
        boundary_face_iterator(int index, const MeshType* mesh, int marker) :
            face_iterator(
                index, mesh, 
                marker == BoundaryAll ? 
                mesh->boundary_faces_ : 
                mesh->boundary_faces_ & 
                 make_binary_vector(mesh->faces_markers_.begin(), mesh->faces_markers_.end(),marker),
                marker) { }
    };
    boundary_face_iterator boundary_faces_begin() const {return boundary_face_iterator(0, this);}
    boundary_face_iterator boundary_faces_end() const {return boundary_face_iterator(n_faces_, this);}
    using boundary_iterator = boundary_face_iterator; // public view of 3d boundary
    BoundaryIterator<IsoMesh<3,3>> boundary_begin(int marker = BoundaryAll) const {
        return BoundaryIterator<IsoMesh<3,3>>(0,this, marker);
    }
    BoundaryIterator<IsoMesh<3,3>> boundary_end(int marker = BoundaryAll) const {
        return BoundaryIterator<IsoMesh<3,3>>(n_faces_,this, marker);
    }

    std::pair<BoundaryIterator<IsoMesh<3,3>>, BoundaryIterator<IsoMesh<3,3>>>
    boundary(int marker = BoundaryAll) const {
        return std::make_pair(boundary_begin(marker), boundary_end(marker));
    }

    const std::vector<int>& faces_markers() const { return faces_markers_; }
    const std::vector<int>& edges_markers() const { return edges_markers_; }

    // === Marker Utilities === //

    // Set faces markers
    template <typename Lambda> void mark_faces(int marker, Lambda&& lambda)
        requires(requires(Lambda lambda, FaceType f) {
            { lambda(f) } -> std::same_as<bool>;
        }) {
        fdapde_assert(marker >= 0);
        faces_markers_.resize(n_faces_);
        for (face_iterator it = faces_begin(); it != faces_end(); ++it) {
            faces_markers_[it->id()] = lambda(*it) ? marker : Unmarked;
        }
    }

    template <int Rows, typename XprType> void mark_faces(const BinMtxBase<Rows, 1, XprType>& mask) {
        fdapde_assert(mask.rows() == n_faces_);
        faces_markers_.resize(n_faces_);
        for (face_iterator it = faces_begin(); it != faces_end(); ++it) {
            faces_markers_[it->id()] = mask[it->id()] ? 1 : 0;
        }
    }

    template <typename Iterator> void mark_faces(Iterator first, Iterator last) {
        fdapde_static_assert(
          std::is_convertible_v<typename Iterator::value_type FDAPDE_COMMA int>, INVALID_ITERATOR_RANGE);
        int n_markers = std::distance(first, last);
        bool all_markers_positive = std::all_of(first, last, [](auto marker) { return marker >= 0; });
        fdapde_assert(n_markers == n_faces_ && all_markers_positive);
        faces_markers_.resize(n_faces_, Unmarked);
        for (int i = 0; i < n_faces_; ++i) { faces_markers_[i] = *(first + i); }
    }

    void mark_faces(int marker) {   // marks all faces with m
        fdapde_assert(marker >= 0);
        faces_markers_.resize(n_faces_);
        std::for_each(faces_markers_.begin(), faces_markers_.end(), [marker](int& marker_) { marker_ = marker; });
    }

    void clear_face_markers() {
        std::for_each(faces_markers_.begin(), faces_markers_.end(), [](int& marker) { marker = Unmarked; });
    }

    protected:
    std::vector<int> faces_, edges_;   ///< for each face, the ids of its nodes
    std::vector<int> face_to_cells_;   ///< for each face, the ids of the cells insisting on it
    std::unordered_map<int, std::unordered_set<int>> edge_to_cells_;    ///< for each edge, the ids of the cells insisting on it
    Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor> cell_to_faces_ {};   ///< for each cell, the ids of its faces
    Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor> cell_to_edges_ {};   ///< for each cell, the ids of its edges
    std::vector<int> face_to_edges_;                                           ///< for each face, the ids of its edges
    BinaryVector<Dynamic> boundary_faces_ {};           ///< boundary faces
    BinaryVector<Dynamic> boundary_edges_ {};           ///< boundary edges
    std::vector<int> faces_markers_;                    ///< marker for each face
    std::vector<int> edges_markers_;                    ///< marker for each egde
    int n_faces_ = 0, n_edges_ = 0;                     ///< number of faces and edges in the mesh  
    std::vector<typename Base::CellType> cell_cache_;   ///< cache of cells
    mutable typename Base::CellType cell_;              ///< temporary cell object for non-cached access

};

}; // namespace fdapde

#endif // __ISO_MESH_H__