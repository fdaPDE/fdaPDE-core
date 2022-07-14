#ifndef __FDAPDE_NURBS_DOF_HANDLER_H__
#define __FDAPDE_NURBS_DOF_HANDLER_H__


#include "header_check.h"

namespace fdapde{

template <int LocalDim, int EmbedDim, typename DiscretizationCategory> class DofHandler;

template<int N> class DofHandler<2, N, iso_tag> {

    

    public:
    using MeshType = IsoMesh<2, N>;
    static constexpr int local_dim = MeshType::local_dim;
    static constexpr int embed_dim = MeshType::embed_dim;
    using BasisType = NurbsBasis<local_dim>;

    public:
    int flatten(const std::array<int, local_dim>& multi_idx) const {
        int id = 0;
        int stride = 1;
        for (int d = 0; d < local_dim ; d++) {
            id += multi_idx[d] * stride;
            stride *= dims_[d];
        }
        return id;
    }
    std::array<int,local_dim> unflatten(int id) const {
        std::array<int,local_dim> multi_idx;
        for (int d = 0; d < local_dim ; d++)  {
            multi_idx[d] = id % dims_[d];
            id /= dims_[d];
        }
        return multi_idx;
    }
    

    public:
    // a geometrical segment with attached dofs
    struct CellType : public IsoSquare<MeshType> {
        using Base = IsoSquare<MeshType>;
        const DofHandler* dof_handler_;
        public:
        static constexpr int local_dim = 2;
        static constexpr int embed_dim = N;

        CellType() : dof_handler_(nullptr) { }
        CellType(int cell_id, const DofHandler* dof_handler):
            Base(cell_id, dof_handler->mesh()), dof_handler_(dof_handler) { } 
            std::vector<int> dofs() const {
                //return dof_handler_->active_dofs_(Base::id());
                return dof_handler_->get_dofs(Base::id());
            }
            // markers to identify cells to be added :)
            BinaryVector<Dynamic> boundary_dofs() const {
                std::vector<int> dofs_ = dofs();
                BinaryVector<Dynamic> boundary(dofs_.size());
                int i = 0;
                for (int dof : dofs_) {
                    if (dof_handler_->is_dof_on_boundary(dof)) boundary.set(i);
                    ++i;
                }
                return boundary;
            }
    };

    // constructor
    DofHandler() = default;
    DofHandler(const MeshType& mesh,const BasisType& basis) : mesh_(std::addressof(mesh)), dof_constraints_(*this), basis_pde_(basis) {
        degree_ = basis_pde_.degree();
        n_dofs_per_cell_ = 1;
        for(int i = 0; i < local_dim; i++) n_dofs_per_cell_ *= degree_[i] + 1;
        int n_cells = mesh_->n_cells();
        n_dofs_ = (basis_pde_).size();
        for (int d = 0; d < local_dim; ++d) {
            dims_[d] = basis_pde_[0].spline_basis()[d]->n_knots() - degree_[d] - 1;
        }
        

        dofs_.resize(n_cells, n_dofs_per_cell_);

        for (int cell_id = 0; cell_id < n_cells; ++cell_id) {
            auto local_dof_multi_indices = active_dofs_(cell_id);  // list of [i,j]
            for (int k = 0; k < local_dof_multi_indices.size(); ++k) {
                dofs_(cell_id, k) = local_dof_multi_indices[k];  // flatten [i,j] → scalar
            }
        }

        // Initialize boundary dofs
        boundary_dofs_.resize(n_dofs_);
        adj_boundary_dofs_.resize(n_dofs_);

        for(int id = 0; id < n_dofs_; id++) {
            auto multi_index = unflatten(id);
            for(int d = 0; d < local_dim; d++) {
                if((multi_index[d] == 0 || multi_index[d] == dims_[d] - 1) && (!mesh_->is_periodic(d))) {
                    boundary_dofs_.set(id);
                }
                if((multi_index[d] == 1 || multi_index[d] == dims_[d] - 2) && (!mesh_->is_periodic(d))) {
                    adj_boundary_dofs_.set(id);
                }
            }
        }

        // for the moment unmarked dofs
        dofs_markers_ = std::vector<int>(n_dofs_, Unmarked);
        // Mark boundary dofs
        for (typename MeshType::boundary_edge_iterator it = mesh_->boundary_edges_begin();
                 it != mesh_->boundary_edges_end(); ++it) {
                int marker = it->marker();

                for(int i = 0; i<1; i++){
                    int cell_id = mesh_->edge_to_cells()(it->id(), i);
                    if (cell_id < 0 ) continue; // skip if no cell is associated

                    auto dofs = get_dofs(cell_id);

                    bool x_aligned = it->x_axis();
                    int dir;
                    if (x_aligned) dir = 0; // x-axis aligned edge
                        else dir = 1; // y-axis aligned edge

                    for(int j = 0; j<dofs.size(); j++) {
                        auto dof_multi = unflatten(dofs[j]);
                        // check if the dof is on the edge
                        for(int d = 0; d < local_dim; d++) {
                            if((dof_multi[d] == 0 || dof_multi[d] == dims_[d] - 1) && (!mesh_->is_periodic(d))) {
                                if (d != dir) {
                                    if(marker > dofs_markers_[dofs[j]]) {
                                    dofs_markers_[dofs[j]] = marker; // mark dof on the edge;
                                    }
                                }
                            }
                        }
                    }

                }
        }

        dof_map_.resize(n_dofs_);
        for (int i = 0; i < n_dofs_; ++i) {
            dof_map_[i] = i;
        }

        build_periodic_dof_map();

        int counter = 0;
        for (int i = 0; i < dof_map_.size(); ++i) {
            if (dof_map_[i] == i) {
                reduced_indices_[i] = counter++;
            }
        }


    }
    DofHandler(const MeshType& mesh) : DofHandler(mesh, mesh.basis()) { }
    

     template <typename SystemMatrix, typename SystemRhs>
     void enforce_constraints(SystemMatrix&& A, SystemRhs&& b) const {
         dof_constraints_.enforce_constraints(std::forward<SystemMatrix>(A), std::forward<SystemRhs>(b));
     }

     void enforce_constraints(Eigen::SparseMatrix<double>& A) const {
        dof_constraints_.enforce_constraints(A);
    }

    void enforce_constraints(Eigen::Matrix<double, Dynamic, 1>& b) const {
        dof_constraints_.enforce_constraints(b);
    }

    void set_hom_dirichlet_constraint(int marker = BoundaryAll) {
        for(int i = 0; i < n_dofs_; ++i) {
            if (is_dof_on_boundary(i) && (dofs_markers_[i] == marker || marker == BoundaryAll)) {
                dirichlet_dofs_.push_back(i);
                dirichlet_vals_.push_back(0.0); // default value for homogeneous Dirichlet condition
            }
        }
        dof_constraints_.set_hom_dirichlet_constraint(marker);
        
    }

    void set_clamped_hom_constraint(){
        dof_constraints_.set_clamped_hom_constraint();
    }

    // Overload for matrix only
    void enforce_periodic_constraints(Eigen::SparseMatrix<double>& A) {
        enforce_periodic_constraints_(&A, nullptr);
    }

    // Overload for vector only
    void enforce_periodic_constraints(Eigen::VectorXd& b) {
        enforce_periodic_constraints_(nullptr, &b);
    }

    // Overload for both matrix and vector
    void enforce_periodic_constraints(Eigen::SparseMatrix<double>& A, Eigen::VectorXd& b) {
        enforce_periodic_constraints_(&A, &b);
    }


    // Expand a reduced solution vector to the full DOF vector (for periodicity)
    Eigen::VectorXd expand_solution(const Eigen::VectorXd& reduced_sol) const {
        Eigen::VectorXd full_sol(dof_map_.size());
        for (int i = 0; i < dof_map_.size(); ++i) {
            int mapped = dof_map_[i];
            full_sol[i] = reduced_sol[reduced_indices_.at(mapped)];
        }
        return full_sol;
    }


     // dimension n_dofs_ x local_dim: parametric coordinates of each dof
     Eigen::Matrix<double, Dynamic, local_dim> dof_coords() const{
        Eigen::Matrix<double, Dynamic, local_dim> coords(n_dofs_, local_dim);
        auto nurb = basis_pde_[0]; // take a nurb
        std::array<std::vector<double>, local_dim> knot_coords;

        // Extract 1D knot positions for each parametric direction
        for (int i = 0; i < local_dim; i++) {
            auto basis = nurb.spline_basis()[i];
            for (const auto& b : basis) {
                knot_coords[i].push_back(b.knot());
            }
        }

        // Carry-on logic: Cartesian product of knot coordinates
        std::vector<int> idx(local_dim, 0);
        while (true) {
            Eigen::Matrix<double, local_dim, 1> coord;
            for (int d = 0; d < local_dim; ++d) {
                coord(d) = knot_coords[d][idx[d]];
            }
            coords.row(flatten(idx)) = coord;

            // Increment multi-index
            int d = local_dim - 1;
            while (d >= 0) {
                idx[d]++;
                if (idx[d] < static_cast<int>(knot_coords[d].size())) break;
                idx[d] = 0;
                --d;
            }
            if (d < 0) break;
        }

     }
     
 
    void build_periodic_dof_map() {
        dof_map_.resize(n_dofs_);
        std::iota(dof_map_.begin(), dof_map_.end(), 0);
    
        // Step 1: assign direct periodic connections (corner and edge)
        auto assign_slave = [&](int slave, int master) {
            // Find canonical master
            while (dof_map_[master] != master) {
                master = dof_map_[master];
            }
            // Assign
            dof_map_[slave] = master;
        };
    
        // Corner collapse (both directions periodic)
        if (mesh_->is_periodic(0) && mesh_->is_periodic(1)) {
            int p0 = degree_[0], p1 = degree_[1];
            int n0 = dims_[0], n1 = dims_[1];
    
            for (int i = 0; i < p0; ++i) {
                for (int j = 0; j < p1; ++j) {
                    int master = flatten({i, j});
                    int slave  = flatten({i + n0 - p0, j + n1 - p1});
                    if (slave != master)
                        assign_slave(slave, master);
                }
            }
        }
    
        // Edge wrap per direction
        for (int d = 0; d < local_dim; ++d) {
            if (!mesh_->is_periodic(d)) continue;
            int n = dims_[d];
    
            for (int i = 0; i < n_dofs_; ++i) {
                auto multi = unflatten(i);
                if (multi[d] < degree_[d]) {
                    auto mapped = multi;
                    mapped[d] += n - degree_[d];
                    int target = flatten(mapped);
                    if (target != i)
                        assign_slave(i, target);
                }
            }
        }
    
        // Step 2: Flatten all dof_map_ so every entry directly points to its root
        for (int i = 0; i < n_dofs_; ++i) {
            int root = i;
            while (dof_map_[root] != root) {
                root = dof_map_[root];
            }
            dof_map_[i] = root;
        }   
        // Step 3: Count unique
        std::unordered_set<int> unique;
        for (int i = 0; i < n_dofs_; ++i)
            unique.insert(dof_map_[i]);
        n_mapped_dofs_ = unique.size();
    }
    
    
    // getters
    const MeshType* mesh() const {return mesh_;}
    CellType cell(int id) const { return CellType(id, this); }
    int n_dofs() const { return n_dofs_; }
    int n_dofs_per_cell() const { return n_dofs_per_cell_; }
    bool is_dof_on_boundary(int i) const { return boundary_dofs_[i]; }
    bool is_dof_on_adjacent_boundary(int i) const { return adj_boundary_dofs_[i]; }
    std::vector<int> dirichlet_dofs() const { return dirichlet_dofs_; }
    std::vector<double> dirichlet_values() const { return dirichlet_vals_; }
    const std::vector<int>& dofs_markers() const { return dofs_markers_; }
    int dof_marker(int dof) const { return dofs_markers_[dof]; }
    int n_boundary_dofs() const { return boundary_dofs_.count(); }
    int n_boundary_dofs(int marker) const {
        int i = 0, sum = 0;
        for (int dof_marker : dofs_markers_) { sum += (dof_marker == marker && boundary_dofs_[i++]) ? 1 : 0; }
        return sum;
    }
    std::vector<int> filter_dofs_by_marker(int marker) const {
        std::vector<int> result;
        for (int i = 0; i < n_dofs_; ++i) {
            if (dofs_markers_[i] == marker) result.push_back(i);
        }
        return result;
    }
    std::vector<int> dof_map() const { return dof_map_; }
    std::array<int, local_dim> dims() const { return dims_; }
    std::vector<int> get_dofs(int id) const{
        std::vector<int> dofs;
        for (int i = 0; i < n_dofs_per_cell_; ++i) {
            dofs.push_back(dofs_(id, i));
        }
        return dofs;
    }

    int n_mapped_dofs() const { return n_mapped_dofs_; }
    // iterate over geometric cells coupled with dofs, possibly filtered by marker
    class cell_iterator :  public internals::filtering_iterator<cell_iterator, CellType> {
        using Base = internals::filtering_iterator<cell_iterator, CellType>;
        using Base::index_;
        friend Base;
        const DofHandler* dof_handler_;
        int marker_;

        cell_iterator& operator()(int i){
            Base::val_ = dof_handler_->cell(i);
            return *this;
        }

        public:
        cell_iterator() = default;
        cell_iterator(int index, const DofHandler* dof_handler, const BinaryVector<Dynamic>& filter, int marker):
            Base(index, 0, dof_handler->mesh()->n_cells(), filter),
            dof_handler_(dof_handler), 
            marker_(marker){
                for (; index_ < Base::end_ && !filter[index_]; ++index_);
                if (index_ != Base::end_) { operator()(index_); }
            }
            cell_iterator(int index, const DofHandler* dof_handler, int marker) :
            cell_iterator(
              index, dof_handler,
              marker == TriangulationAll ?
                BinaryVector<Dynamic>::Ones(dof_handler->mesh()->n_cells()) :   // apply no filter
                make_binary_vector(
                  dof_handler->mesh()->cells_markers().begin(),
                  dof_handler->mesh()->cells_markers().end(), marker),
              marker) { }
        int marker() const { return marker_; }

        
    }; 

    cell_iterator cells_begin(int marker = TriangulationAll) const {
        const std::vector<int>& cells_markers = mesh_->cells_markers();
        fdapde_assert(marker == TriangulationAll || (marker >= 0 && cells_markers.size() != 0));
        return cell_iterator(0, this, marker);
    }
    cell_iterator cells_end(int marker = TriangulationAll) const {
        fdapde_assert(marker == TriangulationAll || (marker >= 0 && mesh_->cells_markers().size() != 0));
        return cell_iterator(mesh_->n_cells(), this, marker);
    }

    template <typename EdgeType>
    class DofEdgeWrapper : public EdgeType {
        const DofHandler* dof_handler_;

    public:
        DofEdgeWrapper() : EdgeType(), dof_handler_(nullptr) {}
        DofEdgeWrapper(const EdgeType& edge, const DofHandler* dof_handler)
            : EdgeType(edge), dof_handler_(dof_handler) {}

        std::vector<int> dofs() const {
            auto cell_id = dof_handler_->mesh()->edge_to_cells()(this->id(), 0);
            return dof_handler_->get_dofs(cell_id);
        }

        const DofEdgeWrapper* operator->() const { return this; }
        const DofEdgeWrapper& operator*() const { return *this; }
    };

using EdgeType = typename CellType::EdgeType;
using DofEdge = DofEdgeWrapper<EdgeType>;

class edge_iterator : public internals::filtering_iterator<edge_iterator, DofEdge> {
protected:
    using Base = internals::filtering_iterator<edge_iterator, DofEdge>;
    using Base::index_;
    friend Base;

    const DofHandler* dof_handler_;

    edge_iterator& operator()(int i) {
        Base::val_ = DofEdge(EdgeType(i, dof_handler_->mesh()), dof_handler_);
        return *this;
    }

public:
    edge_iterator(int index, const DofHandler* dof_handler, const BinaryVector<Dynamic>& filter)
        : Base(index, 0, dof_handler->mesh()->n_edges(), filter), dof_handler_(dof_handler) {
        for (; index_ < Base::end_ && !Base::filter_[index_]; ++index_);
        if (index_ != Base::end_) { operator()(index_); }
    }

    edge_iterator(int index, const DofHandler* dof_handler)
        : edge_iterator(index, dof_handler, BinaryVector<Dynamic>::Ones(dof_handler->mesh()->n_edges())) {}

    const DofEdge& operator*() const { return Base::val_; }
    const DofEdge* operator->() const { return &this->operator*(); }
};
    edge_iterator edges_begin() const { return edge_iterator(0, this); }
    edge_iterator edges_end() const { return edge_iterator(mesh_->n_edges(), this); }


    
    struct boundary_edge_iterator : public edge_iterator {
       private:
        int marker_;
       public:
        boundary_edge_iterator(int index, const DofHandler* dof_handler) :
            edge_iterator(index, dof_handler, dof_handler->mesh()->boundary_edges()), marker_(BoundaryAll) { }
        boundary_edge_iterator(
          int index, const DofHandler* dof_handler, int marker) :   // filter boundary edges by marker
            edge_iterator(
              index, dof_handler,
              marker == BoundaryAll ? 
              dof_handler->mesh()->boundary_edges() :
                                      dof_handler->mesh()->boundary_edges() &
                                        make_binary_vector(
                                          dof_handler->mesh()->edges_markers().begin(),
                                          dof_handler->mesh()->edges_markers().end(), 
                                        marker))
                                        { }
        int marker() const { return marker_; }
    };
    boundary_edge_iterator boundary_edges_begin() const { return boundary_edge_iterator(0, this); }
    boundary_edge_iterator boundary_edges_end() const {
        return boundary_edge_iterator(mesh_->n_edges(), this);
    }
    using boundary_iterator = boundary_edge_iterator;



    class BoundaryDofType {
        int id_; // id of the dof in the global dof vector
        const DofHandler* dof_handler_;
       public:
        BoundaryDofType() = default;
        BoundaryDofType(int id, const DofHandler* dof_handler) : id_(id), dof_handler_(dof_handler) { 
        }
        int id() const { return id_; }
        int marker() const { return dof_handler_->dofs_markers_[id_]; }
    };

    class boundary_dofs_iterator : public internals::filtering_iterator<boundary_dofs_iterator, BoundaryDofType> {
        using Base = internals::filtering_iterator<boundary_dofs_iterator, BoundaryDofType>;
        using Base::index_;
        friend Base;
        const DofHandler* dof_handler_;
        int marker_;
        boundary_dofs_iterator& operator()(int i) {
            Base::val_ = BoundaryDofType(i, dof_handler_);
            return *this;
        }
       public:
        boundary_dofs_iterator(
          int index, const DofHandler* dof_handler, const BinaryVector<Dynamic>& filter, int marker) :
          Base(index, 0, dof_handler->n_dofs(), filter), dof_handler_(dof_handler), marker_(marker)
             {
            for (; index_ < Base::end_ && !filter[index_]; ++index_);
            if (index_ != Base::end_) { operator()(index_); }
        }
        // filter boundary dofs by marker
        boundary_dofs_iterator(int index, const DofHandler* dof_handler, int marker) :
            boundary_dofs_iterator(
              index, dof_handler,
                marker == BoundaryAll ? dof_handler->boundary_dofs_ :
                            dof_handler->boundary_dofs_ &
                            make_binary_vector(
                                dof_handler->dofs_markers_.begin(), dof_handler->dofs_markers_.end(), marker),
              marker) { 
              }
        int marker() const { return marker_; }
    };
    boundary_dofs_iterator boundary_dofs_begin(int marker = BoundaryAll) const {
        return boundary_dofs_iterator(0, this, marker);
    }
    boundary_dofs_iterator boundary_dofs_end(int marker = BoundaryAll) const {
        return boundary_dofs_iterator(n_dofs_, this, marker);
    }





    private:

        // In any given knot span [u_i, u_{i+1}) at most p+1 basis functions are non zero, namely N_{i-p,p}, ..., N_{i,p}
    // (property P2.2, pag 55, Piegl, L., & Tiller, W. (2012). The NURBS book. Springer Science & Business Media.)
    // Evaluation of the non zero basis functions in the a given knot span
    // voglio gli ID, non i punti std::vector<int>
    std::vector<int> active_dofs_(int id) const { // id is the cell id
        std::vector<int> dofs;
        auto multi_index = mesh_->cell_multi_index(id);
        std::array<std::vector<double>,local_dim> param_nodes = mesh_->param_nodes();
        Eigen::Matrix<double,local_dim,1> u;
        for(int i = 0; i < local_dim; i++) u(i) = param_nodes[i][multi_index[i]];
        auto nurb = basis_pde_[0];
        auto spline_basis = nurb.spline_basis();

        std::vector<std::vector<int>> span_indices(local_dim);

        for(int i = 0; i < local_dim; i++) {
            auto basis = spline_basis[i];
            int span = basis->find_span(u(i));
            int p = degree_[i];
            for(int j = 0; j <= p; j++) {
                span_indices[i].push_back(span - p + j); 
            }
        }
        // Carry-on logic: Cartesian product of all local spans
        std::vector<int> idx(local_dim, 0);
        while (true) {
            std::array<int,local_dim> dof_index;
            for (int d = 0; d < local_dim; ++d) {
                dof_index[d] = span_indices[d][idx[d]];
            }
            
            dofs.push_back(flatten(dof_index));

            // Increment multi-index
            int d = local_dim - 1;
            while (d >= 0) {
                idx[d]++;
                if (idx[d] < span_indices[d].size()) break;
                idx[d] = 0;
                --d;
            }
            if (d < 0) break;
        }

        return dofs;
    }

    // Enforce periodic constraints by reducing the system to the set of unique DOFs (for periodic BCs)
    void enforce_periodic_constraints_(Eigen::SparseMatrix<double>* A = nullptr,
        Eigen::VectorXd* b = nullptr) {

        const auto& map = dof_map_;
        auto reduce_vector = [&](Eigen::VectorXd* vec) {
            if (!vec) return Eigen::VectorXd{};
            Eigen::VectorXd reduced = Eigen::VectorXd::Zero(reduced_indices_.size());
            for (int i = 0; i < vec->size(); ++i) {
                int mapped = map[i];
                if (reduced_indices_.count(mapped)) {
                    reduced[reduced_indices_[mapped]] += (*vec)[i];
                }
            }
            *vec = reduced;
            return reduced;
        };

        if (b) {
            *b = reduce_vector(b);
        }

        if (A) {
            Eigen::SparseMatrix<double> A_reduced(reduced_indices_.size(), reduced_indices_.size());
            for (int k = 0; k < A->outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(*A, k); it; ++it) {
                    int i = map[it.row()];
                    int j = map[it.col()];
                    if (reduced_indices_.count(i) && reduced_indices_.count(j)) {
                        A_reduced.coeffRef(reduced_indices_[i], reduced_indices_[j]) += it.value();
                    }
                }
            }
            *A = A_reduced;
        }
    }




    private:
    // Mesh and basis context
    const MeshType* mesh_;
    BasisType basis_pde_;
    std::array<int,local_dim> dims_; // number of basis functions in each direction
    std::array<int,local_dim> degree_;
    int n_dofs_per_cell_ = 0, n_dofs_ = 0;

    // DOF layout
    Eigen::Matrix<int, Dynamic, Dynamic, Eigen::RowMajor> dofs_; // [cell_id][local_dof]
    BinaryVector<Dynamic> boundary_dofs_; // boundary dofs
    BinaryVector<Dynamic> adj_boundary_dofs_; // nonvanishing first der boundary dofs
    std::vector<int> dirichlet_dofs_; // dirichlet dofs
    std::vector<double> dirichlet_vals_; // dirichlet values


    // DOF mapping and reduction (for periodicity)
    std::vector<int> dof_map_;  // dof_map_[original_dof] = reduced_dof
    std::unordered_map<int, int> reduced_indices_; // reduced_indices_[original_dof] = compressed index
    int n_mapped_dofs_ = 0; // number of unique dofs after periodicity

    // Constraints
    std::vector<int> dofs_markers_; // dofs markers
    DofConstraints<DofHandler,iso_tag> dof_constraints_; // strong constraints on DOFs

};

}


#endif // __FDAPDE_NURBS_DOF_HANDLER_H__