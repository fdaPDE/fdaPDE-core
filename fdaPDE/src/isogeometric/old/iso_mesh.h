#ifndef __ISO_MESH_H__
#define __ISO_MESH_H__


#include "nurbs.h"
#include "iso_square.h"


namespace fdapde {

template<int M, int N>
class IsoMesh {

    friend class IsoSquare<M, N>;

    private:

    static constexpr int local_dim = M;
    static constexpr int embed_dim = N;
    static constexpr int n_nodes_per_cell = 1<<M;
    static constexpr int n_neighbors_per_cell = 2*M;
    static constexpr bool is_manifold = !(local_dim == embed_dim);

    using CellType = IsoSquare<M,N>;


    std::array<std::vector<double>,M> knots_; // knots in each direction
    MdArray<double,full_dynamic_extent_t<M>> weights_; // weights in each direction
    MdArray<double,full_dynamic_extent_t<M+1>> control_points_; // control points in each direction

    int n_cells_; // number of cells
    int n_nodes_; // number of nodes

    std::vector<int> cells_markers_;   // marker associated to i-th cell
    std::vector<int> nodes_markers_;   // marker associated to i-th node
    
    //membri privati nodi e celle

    NurbsBasis<M> basis_; // basis of the mesh

    //DMatrix<std::size_t> boundary_dof_ {}; // dofs of boundary of the mesh ( DofHandler.h will be implemented later)
    
    //se una funzione o un dato è privato si mette compute_stride_ invece di compute_stride_

    // Compute stride based on dimension and type (is_cell or node)
    std::size_t compute_stride_(std::size_t dim, bool is_cell) const {
        return is_cell ? this->knots_[dim].size() - 1 : this->knots_[dim].size();   
    }   
    //is_cell chiamare cell 

    // Algo A4.3 from NURBS book pag. 103
    Eigen::Matrix<double, N, 1> eval_param_(const std::array<double, M>& u) const {

        std::vector<std::vector<double>> basis_eval(M);
        std::array<int,M> spans= {0};
        auto order= this->basis_[0].order();
        auto nurb = this->basis_[0];
        double total_weight = 0.0;
        
        for(int i = 0; i < M; i++){
            basis_eval[i] = nurb.spline_basis()[i]->evaluate_basis(u[i], false); // evaluate basis functions, padding = false
            spans[i] = std::distance(knots_[i].begin(), std::upper_bound(knots_[i].begin(), knots_[i].end(), u[i])) - 1 + order;
        }

        Eigen::Matrix<double, N, 1> Sw = Eigen::Matrix<double, N, 1>::Zero();

        std::vector<int> index(M,0);
        bool done =  false;
        while(!done){
            double eval = 1.0;

            std::array<int,M> full_indices;
            for(int i = 0; i < M; i++){
                eval *= basis_eval[i][index[i]];
                full_indices[i] = spans[i] - order + index[i];
            }

            Eigen::Matrix<double, N, 1> cp;
            for(int i = 0; i < N; i++){
                const auto cp_slice = this->control_points_.template slice<M>(i);
                cp(i) = cp_slice(full_indices);
            }

            double w = weights_(index);  // Retrieve weight from weight array
            cp *= w;  // Apply weight to control point
            Sw += eval * cp; // Accumulate weighted control point
            
            total_weight += eval * w ;  // Accumulate total weight

            for (int d = M - 1; d >= 0; d--) {
            if (++index[d] > order) {
                index[d] = 0;
                if (d == 0) done = true;
            } else 
                break;
            }
        }
        return Sw/total_weight;
    }

    // Inspired by algo A4.3 from NURBS book pag. 103
    // evaluate the derivative of the physical coordinates of a point given its parametric coordinates
    Eigen::Matrix<double, N, M, Eigen::RowMajor> eval_param_derivative_(const std::array<double, M>& u) const {
        std::vector<std::vector<double>> basis_eval(M);
        std::vector<std::vector<double>> basis_deriv_eval(M);
        std::array<int, M> spans = {0};
        auto order = this->basis_[0].order();
        auto nurb = this->basis_[0];
        
        for (int i = 0; i < M; i++) {
            basis_eval[i] = nurb.spline_basis()[i]->evaluate_basis(u[i], false);
            basis_deriv_eval[i] = nurb.spline_basis()[i]->evaluate_der_basis(u[i],1,false);
            spans[i] = std::distance(knots_[i].begin(), std::upper_bound(knots_[i].begin(), knots_[i].end(), u[i])) - 1 + order;
        }
        
        Eigen::Matrix<double, N, M, Eigen::RowMajor> dSw = Eigen::Matrix<double, N, M>::Zero();
        Eigen::Matrix<double, N, 1> Sw = Eigen::Matrix<double, N, 1>::Zero();
        Eigen::Matrix<double, M, 1> dW = Eigen::Matrix<double, M, 1>::Zero();
        double total_weight = 0.0;
        
        std::vector<int> index(M, 0);
        bool done = false;
        while (!done) {
            double eval = 1.0;
            std::array<double, M> eval_der = {0.0};        
            std::array<int, M> full_indices;

            for (int i = 0; i < M; i++) {
                eval *= basis_eval[i][index[i]];
                eval_der[i] = basis_deriv_eval[i][index[i]];
                full_indices[i] = spans[i] - order + index[i];
            }

            
            Eigen::Matrix<double, N, 1> cp;
            for (int i = 0; i < N; i++) {
                const auto cp_slice = this->control_points_.template slice<M>(i);
                cp(i) = cp_slice(full_indices);
            }
            
            double w = weights_(index);
            cp *= w;
            Sw += eval * cp;
            total_weight += eval * w;
            
            for (int j = 0; j < M; j++) {
                double w_temp = w*eval_der[j];
                Eigen::Matrix<double, N, 1> cp_temp = eval_der[j] *cp;
                for(int i = 0; i < M; i++){
                    if(i != j){
                        w_temp *= basis_eval[i][index[i]];
                        cp_temp *= basis_eval[i][index[i]];
                    }
                }
                dW(j) += w_temp;
                dSw.col(j) += cp_temp;
            }
            
            for (int d = M - 1; d >= 0; d--) {
                if (++index[d] > order) {
                    index[d] = 0;
                    if (d == 0) done = true;
                } else 
                    break;
            }
        }
        
        // Apply derivative of the ratio d/du (Sw / total_weight)
        for (int j = 0; j < M; j++) {
            dSw.col(j) = (dSw.col(j) - (Sw * dW(j) / total_weight)) / total_weight;
        }
        
        return dSw;
    }


    // fanne due separate per is_cell e node, falla privata, compute_cell_ID, compute_node_ID
    // puoi spostarla in iso_square se non usata qui
    // Compute the id of a cell (or a node if is_cell is false) from the multi-index
    std::size_t compute_id_(const std::array<std::size_t, M>& multi_index, bool is_cell=true) const {
        std::size_t id = multi_index[M - 1]; // Start with the last index
        for (std::size_t i = M - 1; i > 0; --i) { // Iterate backward
            id = id * compute_stride_(i-1,is_cell) + multi_index[i - 1];
        }
        return id;
    }

    // Compute the multi-index of a cell (or a node if is_cell is false) from the id
    // falla privata, tieni il bool , ma non ci deve essere nell'interfaccia pubblica
    //cambia bool is_cell in is_cell, compute_multi_index_
    std::array<std::size_t, M> compute_multi_index_(const std::size_t& id, bool is_cell=true) const {
        auto tempID = id;
        std::array<std::size_t, M> multi_index;
        for (std::size_t i = M; i > 0; --i) {
            if (i == 1) {
                multi_index[M-1] = tempID; // The last index is the remaining id
            } else {
                std::size_t stride = compute_stride_(i - 2, is_cell); // Use (i-2) to match dimensions
                multi_index[M - i] = tempID % stride; // Extract the current index
                tempID /= stride; // Update id for the next dimension
            }
        }
        return multi_index;
    }

    // Compute the vertices of a square (cell) given its id, fai MdArray , sono metodi pirvati
    std::array<std::array<int, M>,2> compute_lr_vertices_(const std::size_t& id)const {
        auto multi_index = compute_multi_index_(id);
        std::array<std::array<int, M>,2> vertices;
        for(std::size_t i = 0; i < M; ++i){
            vertices[0][i] = multi_index[i];
            vertices[1][i] = multi_index[i] + 1;
        }
        return vertices;
    }

    

    public:

    // Constructor, metti l'ordine alla fine
    IsoMesh(std::array<std::vector<double>,M> & knots, MdArray<double,full_dynamic_extent_t<M>> & weights, 
         MdArray<double,full_dynamic_extent_t<M+1>> & control_points, int order): knots_(knots), weights_(weights), control_points_(control_points){

            basis_ = NurbsBasis<M>(knots, weights, order);

            // compute number of cells and nodes
            n_cells_ = 1; // is_cell cambialo in cells, ovunque
            n_nodes_ = 1; //
            for(std::size_t i=0; i < M; ++i){
                n_cells_ *= knots_[i].size() - 1;
                n_nodes_ *= knots_[i].size();
            }

        };

    // Compute all nodes of the mesh (dimension #nodes x M ), usa Eigen::Matrix<double, M, Dynamic>, fai Dynamic
    //distingui tra fisiche e parametriche, physical_nodes, parametric_nodes, senza compute

    // pyhsical_coords() che fa lo stesso per il dominio fisico,  
    Eigen::Matrix<double, Dynamic, M, Eigen::RowMajor> parametric_nodes() const {
        Eigen::Matrix<double, Dynamic, M> nodes;
        nodes.resize(n_nodes_, M);
        for (std::size_t i = 0; i < n_nodes_; ++i) {
            auto multi_index = compute_multi_index_(i, false);
            for (std::size_t j = 0; j < M; ++j) {
                nodes(i, j) = knots_[j][multi_index[j]];
            }
        }
        return nodes;
    }


    // Compute all cells of the mesh (dimension #cells x M ) // is_cell diventa cells, Eigen::Matrix<int, M, Dynamic>
    Eigen::Matrix<int, Dynamic, M, Eigen::RowMajor> cells() const {
        Eigen::Matrix<int, Dynamic, M> cells;
        cells.resize(n_cells_,M);
        for(std::size_t i = 0; i < n_cells_; ++i){
            cells[i] = compute_multi_index_(i);
        }
        return cells;
    }
    
    

    // check if a cell/node is on the boundary, usa is_node_on_boundary, is_cell_on_boundary
    bool is_node_on_boundary(const std::size_t& id) const {
        auto multi_index = compute_multi_index_(id,false);
        for(std::size_t i = 0; i < M; ++i){
            if(multi_index[i] == 0 || multi_index[i] == compute_stride_(i,false) - 1){
                return true;
            }
        }
        return false;
    }

    bool is_cell_on_boundary(const std::size_t& id) const {
        auto multi_index = compute_multi_index_(id);
        for(std::size_t i = 0; i < M; ++i){
            if(multi_index[i] == 0 || multi_index[i] == compute_stride_(i,true) - 1){
                return true;
            }
        }
        return false;
    }

    //guarda triangulation.h per vedere come fare, riga 83
    
    // Compute the id neighbors of a cell given its id (dimension #cells x 2M(at most))
    //   - Colonne `2*j` : Voisin dans la direction négative pour la dimension `j`.
    //   - Colonne `2*j + 1` : Voisin dans la direction positive pour la dimension `j`.
    // - Si un voisin n'existe pas dans une direction, la valeur est -1.
    Eigen::Matrix<int, Eigen::Dynamic, 2 * M, Eigen::RowMajor> neighbors() const {
        // Initialize the neighbors matrix with dynamic rows and 2 * M columns
        Eigen::Matrix<int, Eigen::Dynamic, 2 * M, Eigen::RowMajor> neighbors;
        neighbors.resize(n_cells_, 2 * M);

        // Loop through each cell
        for (int i = 0; i < n_cells_; ++i) {
            // Compute the multi-index for the current cell
            auto multi_index = compute_multi_index_(i);

            // Loop through each dimension
            for (std::size_t j = 0; j < M; ++j) {
                // Handle the neighbor in the negative direction
                if (multi_index[j] > 0) {
                    auto updated_index = multi_index;
                    updated_index[j] -= 1; // Update the index in the negative direction
                    neighbors(i, 2 * j) = compute_id_(updated_index);
                } else {
                    neighbors(i, 2 * j) = -1; // No neighbor in the negative direction
                }

                // Handle the neighbor in the positive direction
                if (multi_index[j] < compute_stride_(j, true) - 1) {
                    auto updated_index = multi_index;
                    updated_index[j] += 1; // Update the index in the positive direction
                    neighbors(i, 2 * j + 1) = compute_id_(updated_index);
                } else {
                    neighbors(i, 2 * j + 1) = -1; // No neighbor in the positive direction
                }
            }
        }
        // order the neighbors rows in increasing order, but put the -1 at the end
        // ONLY FOR THE TESTS
        for(int i = 0; i < n_cells_; ++i){
            std::vector<int> row(neighbors.row(i).data(), neighbors.row(i).data() + neighbors.cols());
            std::sort(row.begin(), row.end(), [](int a, int b) { return a == -1 ? false : b == -1 ? true : a < b; });
            std::copy(row.begin(), row.end(), neighbors.row(i).data());
        }

        return neighbors;
    }

    // Compute the coordinates of the physical coords using eval_param method of the mesh
    // point in Parametric Space (M dim) -> point in Physical Space (N dim)
    // physical_coords, Eigen::Matrix<double, N, 1>
    // already implemented in the private section
    /*
    std::array<double,N> physical_coords(const std::array<double, M> & p) const {
        std::array<double, N> x;
        // note that the control points are stored in a tensor of dimension M+1, we need to slice it
        for(int i = 0; i < N; ++i){
            x[i] = eval_param(p,i);
        }
        return x;
    }
    */

    // Access individual cells, Create a cell object given its id
    // cell(id) -> IsoSquare
    IsoSquare<M,N> cell(const std::size_t& id) const {
        return IsoSquare<M,N>(id, this);
    }


    class cell_iterator: public internals::filtering_iterator<cell_iterator,const CellType*>   { 
    // cell_iterator, guarda triangulation.h al rigo 96, uniforma a quella implementazione
    // public internals::filtering_iterator<cell_iterator, const CellType*>
    // call operator a riga 102
    private:
        using Base = internals::filtering_iterator<cell_iterator, const CellType*>;
        using Base::index_;
        friend Base;
        const IsoMesh* mesh_;
        int marker_;

        mutable CellType cell_;  // Store the actual cell (mutable for const correctness)

        cell_iterator& operator()(int i) {
            cell_ = mesh_->cell(i);  // Store the value in a variable
            Base::val_ = &cell_;  // Take the address of the variable
            return *this;
        }

    public:
        cell_iterator() = default;

        cell_iterator(int index, const IsoMesh* mesh, const BinaryVector<Dynamic>& filter, int marker):
            Base(index, 0, mesh->n_cells_, filter), mesh_(mesh), marker_(marker) {
            for (; index_ < Base::end_ && !filter[index_]; ++index_);
            if (index_ != Base::end_) { operator()(index_); }
        }

        cell_iterator(int index, const IsoMesh* mesh, int marker):
            cell_iterator(index, mesh, marker == TriangulationAll ? BinaryVector<Dynamic>::Ones(mesh->n_cells_) :
                make_binary_vector(mesh->cells_markers_.begin(), mesh->cells_markers_.end(), marker), marker) { }

        int marker() const { return marker_; }

    };

    // Metodi per ottenere gli iteratori
    cell_iterator cells_begin(int marker = TriangulationAll) const { 
        fdapde_assert(marker == TriangulationAll || (marker >= 0 && cells_markers_.size() != 0));
        return cell_iterator(0,this,marker);}
    cell_iterator cells_end(int marker = TriangulationAll) const { 
        fdapde_assert(marker == TriangulationAll || (marker >= 0 && cells_markers_.size() != 0));
        return cell_iterator(n_cells_,this,marker);}
    

    class BoundaryIterator {
        private:
            const IsoMesh* mesh_;
            std::size_t currentIndex_;
            IsoSquare<M,N> currentCell_;
        public:
            BoundaryIterator(const IsoMesh* mesh, std::size_t index)
                : mesh_(mesh), currentIndex_(index), currentCell_(mesh->cell(index)) {
                while (currentIndex_ < mesh_->n_cells_ && !mesh_->is_boundary(currentIndex_)) {
                    currentIndex_++;
                }
            }
            IsoSquare<M,N> operator*() const { return currentCell_; }

            // Pointer-like access to IsoSquare
            IsoSquare<M, N>* operator->() { return &currentCell_; }
            const IsoSquare<M, N>* operator->() const { return &currentCell_; }
            
            BoundaryIterator& operator++() {
                do {
                    currentIndex_++;
                } while (currentIndex_ < mesh_->n_cells_ && !mesh_->is_boundary(currentIndex_));
                if (currentIndex_ < mesh_->n_cells_) {
                    currentCell_ = mesh_->cell(currentIndex_);
                }
                return *this;
            }
            bool operator==(const BoundaryIterator& other) const { return currentIndex_ == other.currentIndex_; }
            bool operator!=(const BoundaryIterator& other) const { return !(*this == other); }
        };

    BoundaryIterator beginBoundaryCells() const { return BoundaryIterator(this, 0);}
    BoundaryIterator endBoundaryCells() const { return BoundaryIterator(this, n_cells_); }


    // some getters
    const NurbsBasis<M>& basis() const { return basis_; }
    const MdArray<double, full_dynamic_extent_t<M+1>>& control_points() const { return control_points_; }
    const std::array<std::vector<double>,M>& knots() const { return knots_; }

    // for the test to access private members, put a public version evl_param and eval_param_derivative
    Eigen::Matrix<double, N, 1> eval_param(const std::array<double, M>& u) const { return eval_param_(u); }
    Eigen::Matrix<double, N, M, Eigen::RowMajor> eval_param_derivative(const std::array<double, M>& u) const { return eval_param_derivative_(u); }


    std::size_t n_cells() const { return n_cells_; }
    std::size_t n_nodes() const { return n_nodes_; }
};
}; // namespace fdapde

#endif // __ISO_MESH_H__