    // evaluate the derivative of the physical coordinates of a point given its parametric coordinates with index sliceIdx and derivative_index as the derivative index
    // puoi ritornare Eigen::Matrix<double, N, M>, direttamente lo jacobiano
    Eigen::Matrix<double, N, M, Eigen::RowMajor> eval_param_derivative_(const std::array<double, M>& u) const {
        Eigen::Matrix<double, N, M, Eigen::RowMajor> x = Eigen::Matrix<double, N, M>::Zero();
        for(int i = 0;i<N;i++){
            for(int j = 0; j<M; j++){
                const auto cp_slice = this->control_points_.template slice<M>(i);
                for (const auto& nurb : this->basis_) {
                    x(i,j) += nurb.derive(j)(u) * cp_slice(nurb.index());
                }
            }
        }
        return x;
    }


    // mettili privati e basta, puoi outputtare  Eigen::Matrix<double, N, 1> per fare i calcoli
    // Evaluate the physical coordinates of a point u given its parametric coordinates with index sliceIdx
    // falla u che diventa in [-1,1]^M, portando la affine_map in questa classe
    Eigen::Matrix<double, N, 1> eval_param_(const std::array<double, M>& u) const {
        Eigen::Matrix<double, N, 1> x = Eigen::Matrix<double, N, 1>::Zero();
        x.setZero();
        for(int i = 0;i<N;i++){
            const auto cp_slice = this->control_points_.template slice<M>(i); // cp_slice , slice_idx
            for (const auto& nurb : this->basis_) {
                x(i) += nurb(u) * cp_slice(nurb.index());  
            }
        }
        return x;
    }

    Eigen::Matrix<double, EmbedDim, LocalDim, Eigen::RowMajor> eval_param_derivative_2(const Eigen::Matrix<double, LocalDim, 1>& u) const {
        Eigen::Matrix<double, EmbedDim, LocalDim, Eigen::RowMajor> x = Eigen::Matrix<double, EmbedDim, LocalDim>::Zero();
        for(int i = 0;i<EmbedDim;i++){
            for(int j = 0; j<LocalDim; j++){
                const auto cp_slice = this->control_points_.template slice<LocalDim>(i);
                for (const auto& nurb : this->basis_) {
                    x(i,j) += nurb.derive(j)(u) * cp_slice(nurb.index());
                }
            }
        }
        return x;
    }
    

    void set_periodic_constraint() { // non so se funziona, only C0 continuity
        std::set<int> constrained_dofs;
        std::set<int> master_set;
        for (int d = 0; d < local_dim; ++d) {
            if (!mesh_->is_periodic(d)) continue;
            std::cout << "Periodic constraint in dimension " << d << std::endl;
        
            int p = order_[d]; //order_[p];      // spline degree
            int n = dims_[d];       // number of DOFs in this direction
        
            std::array<int, local_dim> idx_min, idx_max;
            
            // Loop over all combinations in other dimensions
            std::vector<int> dofs_min, dofs_max;
            std::array<int, local_dim> multi;
            int master_north_pole = -1;
            int master_south_pole = -1;
            for (int i = 0; i < n_dofs_; ++i) {
                // For a sphere: v = 0 is south pole, v = dims_[1]-1 is north pole
                if (unflatten(i)[1] == dims_[1] - 1 && master_north_pole == -1) {
                    master_north_pole = i;
                    master_set.insert(i);
                    continue;  // Skip this DoF from periodic constraint setup
                }
                if (unflatten(i)[1] == 0 && master_south_pole == -1) {
                    master_south_pole = i;
                    master_set.insert(i);
                    continue;  // Same here
                }

                multi = unflatten(i);
                if (multi[d] < order_[d]) {
                    auto mapped = multi;
                    mapped[d] += n - order_[d];
                    int mapped_id = flatten(mapped);
                    dofs_min.push_back(i);
                    dofs_max.push_back(mapped_id);
                    master_set.insert(i);
                }
                // TEMPORARY: only for sphere: constraint on the dofs at the poles, put the other dofs equal to the first one
            }

            // print dofs_min and dofs_max
            for(int i = 0; i < dofs_min.size(); ++i) {
                std::cout << "dofs_min: " << dofs_min[i] << " dofs_max: " << dofs_max[i] << std::endl;
            }
            
            
            std::cout<<"Master north pole: "<<master_north_pole<<std::endl;
            std::cout<<"Master south pole: "<<master_south_pole<<std::endl;

            for(int i = 0; i < n_dofs_; ++i) {
                // v = dims_[1] - 1 is north pole, v = 0 is south pole
                if(unflatten(i)[1] == dims_[1] - 1 && i != master_north_pole) { // north pole
                    dof_constraints_.set_master_slave_constraint(master_north_pole, i);
                    constrained_dofs.insert(i);
                }
                if(unflatten(i)[1] == 0 && i != master_south_pole) { // south pole
                    dof_constraints_.set_master_slave_constraint(master_south_pole, i);
                    constrained_dofs.insert(i);
                }
            }
                
                
            //dof_constraints_.set_master_slave_constraint(master_south_pole, master_north_pole);

            std::cout << "number of Periodic constraint: " << dofs_min.size() << " dofs" << std::endl;
        
            // Enforce the periodic constraint: max ↔ min
            for (int i = 0; i < dofs_min.size(); ++i) {
                int master = dofs_min[i];
                int slave  = dofs_max[i];
                std::cout << "Periodic constraint: " << master << " ↔ " << slave << std::endl;
                for(int j = 0; j < local_dim; ++j) {
                    std::cout << "multi: " << unflatten(master)[j] << " ↔ " << unflatten(slave)[j] << std::endl;
                }
                if(constrained_dofs.find(slave) != constrained_dofs.end()) {
                    std::cout << "Skipped constraint: " << slave << " already constrained.\n";
                }
                if(constrained_dofs.find(slave) == constrained_dofs.end()) {
                    dof_constraints_.set_master_slave_constraint(master, slave);
                    constrained_dofs.insert(slave);
                }
                //dof_constraints_.set_master_slave_constraint(master, slave); // enforce slave = master
            }


            
        }
    }     void build_periodic_dof_map() {
        dof_map_.resize(n_dofs_);
        std::iota(dof_map_.begin(), dof_map_.end(), 0);  // default identity map
    
        std::unordered_set<int> unique_dofs;
    
        for (int d = 0; d < local_dim; ++d) {
            if (!mesh_->is_periodic(d)) continue;
    
            int p = order_[d];
            int n = dims_[d];
    
            for (int i = 0; i < n_dofs_; ++i) {
                auto idx = unflatten(i);
                if (idx[d] >= n - p) {
                    auto wrapped = idx;
                    wrapped[d] -= (n - p);
                    int collapsed = flatten(wrapped);
                    dof_map_[i] = collapsed;
                }
            }
        }
    
        for (int i = 0; i < n_dofs_; ++i)
            unique_dofs.insert(dof_map_[i]);
    
        n_mapped_dofs_ = unique_dofs.size();
    }



    void build_periodic_dof_map() {
        // Initialize all entries with identity mapping
        dof_map_.resize(n_dofs_);
        std::iota(dof_map_.begin(), dof_map_.end(), 0);  // identity map

        for (int d = 0; d < local_dim; ++d) {
            if (!mesh_->is_periodic(d)) continue;

            int n = dims_[d];

            for (int i = 0; i < n_dofs_; ++i) {
                auto idx = unflatten(i);
                if (idx[d] == n - 1) {
                    auto wrapped = idx;
                    wrapped[d] = 0;
                    int target = flatten(wrapped);
                    dof_map_[i] = target;
                }
            }
        }
        

        // Handle poles
        /*
        int master_south = -1, master_north = -1;
        for (int i = 0; i < n_dofs_; ++i) {
            auto idx = unflatten(i);
            if (idx[1] == 0) {
                if (master_south == -1) master_south = i;
                dof_map_[i] = master_south;
            } else if (idx[1] == dims_[1] - 1) {
                if (master_north == -1) master_north = i;
                dof_map_[i] = master_north;
            }
        }
            */
            

        // print the dof_map
        std::cout << "DOF map: " << std::endl;
        for (int i = 0; i < n_dofs_; ++i) {
            std::cout << "DOF " << i << " maps to " << dof_map_[i] << std::endl;
        }

        std::unordered_set<int> unique_dofs;
        for (int i = 0; i < n_dofs_; ++i)
            unique_dofs.insert(dof_map_[i]);

        n_mapped_dofs_ = unique_dofs.size();
    }
    
    
    void build_periodic_dof_map2old() {
        dof_map_.resize(n_dofs_);
        std::iota(dof_map_.begin(), dof_map_.end(), 0);

        std::set<int> constrained_dofs;
        std::set<int> master_set;
        
        if (mesh_->is_periodic(0) && mesh_->is_periodic(1)) {
            int p0 = order_[0];
            int p1 = order_[1];
            int n0 = dims_[0];
            int n1 = dims_[1];
            
            for (int i = 0; i < p0; ++i) {
                for (int j = 0; j < p1; ++j) {
                    int master = flatten({i, j});
                    int slave  = flatten({i + n0 - p0, j + n1 - p1});
                    dof_map_[slave] = master;
                    constrained_dofs.insert(slave);
                    master_set.insert(master);
                }
            }
        }
            
        
        for (int d = 0; d < local_dim; ++d) {
            if (!mesh_->is_periodic(d)) continue;
            int n = dims_[d];

            for (int i = 0; i < n_dofs_; ++i) {
                auto multi = unflatten(i);
                if (multi[d] < order_[d]) {
                    auto mapped = multi;
                    mapped[d] += n - order_[d];
                    dof_map_[i] = flatten(mapped);
                    
                    //dof_map_[i] = flatten(mapped);
                }
            }
        }

        
        // Handle poles (collapse DOFs at the north and south poles)
        /*
        int master_south = -1, master_north = -1;
        for (int i = 0; i < n_dofs_; ++i) {
            auto idx = unflatten(i);
            if (idx[1] == 0) {
                if (master_south == -1) master_south = i;
                dof_map_[i] = master_south;
            } else if (idx[1] == dims_[1] - 1) {
                if (master_north == -1) master_north = i;
                dof_map_[i] = master_north;
            }
        }
        */     
            
            

        std::unordered_set<int> unique_dofs;
        for (int i = 0; i < n_dofs_; ++i)
            unique_dofs.insert(dof_map_[i]);
        n_mapped_dofs_ = unique_dofs.size();



    }
   

    void set_periodic_constraint() { // non so se funziona, only C0 continuity
        std::set<int> constrained_dofs;
        std::set<int> master_set;
        for (int d = 0; d < local_dim; ++d) {
            if (!mesh_->is_periodic(d)) continue;
            std::cout << "Periodic constraint in dimension " << d << std::endl;
        
            int p = order_[d]; //order_[p];      // spline degree
            int n = dims_[d];       // number of DOFs in this direction
        
            std::array<int, local_dim> idx_min, idx_max;
            
            // Loop over all combinations in other dimensions
            std::vector<int> dofs_min, dofs_max;
            std::array<int, local_dim> multi;
            int master_north_pole = -1;
            int master_south_pole = -1;
            for (int i = 0; i < n_dofs_; ++i) {
                
                // For a sphere: v = 0 is south pole, v = dims_[1]-1 is north pole
                if (unflatten(i)[1] == dims_[1] - 1 && master_north_pole == -1) {
                    master_north_pole = i;
                    master_set.insert(i);
                    continue;  // Skip this DoF from periodic constraint setup
                }
                if (unflatten(i)[1] == 0 && master_south_pole == -1) {
                    master_south_pole = i;
                    master_set.insert(i);
                    continue;  // Same here
                }
                    

                multi = unflatten(i);
                if (multi[d] < order_[d]) {
                    auto mapped = multi;
                    mapped[d] += n - order_[d];
                    int mapped_id = flatten(mapped);
                    dofs_min.push_back(i);
                    dofs_max.push_back(mapped_id);
                    master_set.insert(i);
                }
                // TEMPORARY: only for sphere: constraint on the dofs at the poles, put the other dofs equal to the first one
            }

            // print dofs_min and dofs_max
            for(int i = 0; i < dofs_min.size(); ++i) {
                std::cout << "dofs_min: " << dofs_min[i] << " dofs_max: " << dofs_max[i] << std::endl;
            }
            
            
            std::cout<<"Master north pole: "<<master_north_pole<<std::endl;
            std::cout<<"Master south pole: "<<master_south_pole<<std::endl;
            
            
            for(int i = 0; i < n_dofs_; ++i) {
                // v = dims_[1] - 1 is north pole, v = 0 is south pole
                if(unflatten(i)[1] == dims_[1] - 1 && i != master_north_pole) { // north pole
                    dof_constraints_.set_master_slave_constraint(master_north_pole, i);
                    constrained_dofs.insert(i);
                }
                if(unflatten(i)[1] == 0 && i != master_south_pole) { // south pole
                    dof_constraints_.set_master_slave_constraint(master_south_pole, i);
                    constrained_dofs.insert(i);
                }
            }
                
                
                
                
            //dof_constraints_.set_master_slave_constraint(master_south_pole, master_north_pole);

            std::cout << "number of Periodic constraint: " << dofs_min.size() << " dofs" << std::endl;
        
            // Enforce the periodic constraint: max ↔ min
            for (int i = 0; i < dofs_min.size(); ++i) {
                int master = dofs_min[i];
                int slave  = dofs_max[i];
                std::cout << "Periodic constraint: " << master << " ↔ " << slave << std::endl;
                for(int j = 0; j < local_dim; ++j) {
                    std::cout << "multi: " << unflatten(master)[j] << " ↔ " << unflatten(slave)[j] << std::endl;
                }
                if(constrained_dofs.find(slave) != constrained_dofs.end()) {
                    std::cout << "Skipped constraint: " << slave << " already constrained.\n";
                }
                if(constrained_dofs.find(slave) == constrained_dofs.end()) {
                    dof_constraints_.set_master_slave_constraint(master, slave);
                    constrained_dofs.insert(slave);
                }
                //dof_constraints_.set_master_slave_constraint(master, slave); // enforce slave = master
            }


            
        }
    }
    void set_periodic_constraint2() { // C0 periodic continuity
        std::set<int> constrained_dofs;

        // Handle corner wrap (only once, when all dimensions are periodic)
        bool all_periodic = true;
        for (int d = 0; d < local_dim; ++d) {
            if (!mesh_->is_periodic(d)) {
                all_periodic = false;
                break;
            }
        }

        if (all_periodic) {
            int p0 = order_[0];
            int p1 = order_[1];
            int n0 = dims_[0];
            int n1 = dims_[1];
            int master = flatten({0, 0});
            for (int i = 0; i < 1; ++i) {
                for (int j = 0; j < 1; ++j) {
                    if (i == 0 && j == 0) continue;
                    int slave = flatten({i + n0 - p0, j + n1 - p1});
                    if (constrained_dofs.find(slave) == constrained_dofs.end()) {
                        dof_constraints_.set_master_slave_constraint(master, slave);
                        constrained_dofs.insert(slave);
                    }
                }
            }

            // Edge wrap along top (in j)
            for (int j = 0; j < p1; ++j) {
                for (int i = 0; i < n0; ++i) {
                    int master = flatten({i, j});
                    int slave = flatten({i, j + n1 - p1});
                    if (constrained_dofs.find(slave) == constrained_dofs.end()) {
                        dof_constraints_.set_master_slave_constraint(master, slave);
                        constrained_dofs.insert(slave);
                    }
                }
            }

            // Edge wrap along right (in i)
            for (int i = 0; i < p0; ++i) {
                for (int j = 0; j < n1; ++j) {
                    int master = flatten({i, j});
                    int slave = flatten({i + n0 - p0, j});
                    if (constrained_dofs.find(slave) == constrained_dofs.end()) {
                        dof_constraints_.set_master_slave_constraint(master, slave);
                        constrained_dofs.insert(slave);
                    }
                }
            }
        }

        for (int d = 0; d < local_dim; ++d) {
            if (!mesh_->is_periodic(d)) continue;
            std::cout << "Periodic constraint in dimension " << d << std::endl;

            int p = order_[d];
            int n = dims_[d];

            // Edge wrapping (1D per dimension)
            for (int i = 0; i < n_dofs_; ++i) {
                auto idx = unflatten(i);
                if (idx[d] < p) {
                    auto mapped = idx;
                    mapped[d] = idx[d] + (n - p);
                    int master = flatten(idx);
                    int slave = flatten(mapped);
                    if (constrained_dofs.find(slave) != constrained_dofs.end()) {
                        std::cout << "Skipped constraint: " << slave << " already constrained.\n";
                    }
                    if (constrained_dofs.find(slave) == constrained_dofs.end()) {
                        dof_constraints_.set_master_slave_constraint(master, slave);
                        constrained_dofs.insert(slave);
                    }
                }
            }
        }


    }



    void set_periodic_constraint3() {
        std::set<int> constrained_dofs;
        for (int i = 0; i < n_dofs_; ++i) {
            int mapped = dof_map_[i];
            if (mapped != i && constrained_dofs.find(i) == constrained_dofs.end()) {
                std::cout << "Periodic constraint: " << i << " ↔ " << mapped << std::endl;
                dof_constraints_.set_master_slave_constraint(mapped, i);
                constrained_dofs.insert(i);
            }
        }
    }
    