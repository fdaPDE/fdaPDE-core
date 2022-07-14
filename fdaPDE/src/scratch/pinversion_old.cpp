// Point inversion algorithm: takes a point p in the physical domain and returns a point u in the parametric domain
    // Implementation of the mathod explained at page 230 of the NURBS book
    Eigen::Matrix<double, local_dim,1> invert_point(Eigen::Matrix<double, embed_dim, 1>& p, 
        double tol1=1e-8, double tol2=1e-8, int max_iters = 100, int n=5) {
        Eigen::Matrix<double, local_dim,1> u_old, u ;
        u.setZero();

        const auto& cp = this->control_points_;

        std::array<decltype(cp.template slice<local_dim>(0)), embed_dim> cp_slices;
        for (int i = 0; i < embed_dim; ++i)
            cp_slices[i] = cp.template slice<local_dim>(i);

        std::array<int, local_dim> index = this->order_;
        std::vector<std::array<int, local_dim>> valid_spans;

        // loop over each span and check if the point ins in the AABB box of the span
        std::array<int,local_dim> index = this->order_;
        bool done = false;
        std::vector<std::array<int,local_dim>> valid_spans = {};
        while (!done) {
            // print the index
            std::array<int,local_dim> new_index;
            for(int i = 0; i < local_dim; i++) {
                new_index[i] = index[i] -  this->order_[i] ;
            }
            MdArray<double,MdExtents<embed_dim>> P_min, P_max;
            for(int j=0;j<local_dim;j++){
                P_min(j) = this->control_points_.template slice<local_dim>(j) (new_index);
                P_max(j) = this->control_points_.template slice<local_dim>(j) (new_index);
            }
            for (int d = 0; d < local_dim; d++) {
                bool done2 = false;
                while (!done2) {
                    Eigen::Matrix<double, embed_dim, 1> cp;
                    for (int i = 0; i < embed_dim; i++) {
                        const auto cp_slice = this->control_points_.template slice<local_dim>(i);
                        cp(i) = cp_slice(new_index);
                    };
                    for (int i = 0; i < embed_dim; i++) {
                        P_min(i) = std::min(P_min(i), cp(i));
                        P_max(i) = std::max(P_max(i), cp(i));
                    }

                    for (int d = local_dim - 1; d >= 0; d--) {
                        if (++new_index[d] > index[d]) {
                            new_index[d] = index[d] - this->order_[d];
                            if (d == 0) done2 = true;
                        } else 
                            break;

                    }
                }
            }
            bool inside = true;
            for (int i = 0; i < embed_dim; i++) {
                if (p(i) < P_min(i) || p(i) > P_max(i)) {
                    inside = false;
                    break;
                }
            }
            if (inside) {
                valid_spans.push_back(index);
            }

            for (int d = local_dim - 1; d >= 0; d--) {
                if (++index[d] > this->weights_.extent(d) - 1) {
                    index[d] = this->order_[d];
                    if (d == 0) done = true;
                } else 
                    break;
            }

        }

        /*
        // print the valid spans
        for(int i = 0; i < valid_spans.size(); i++){
            std::cout<<"Span "<<i<<": ";
            for(int j = 0; j < local_dim; j++){
                std::cout<<valid_spans[i][j]<<" ";
            }
            std::cout<<std::endl;
        }
        std::cout<<"Number of valid spans: "<<valid_spans.size()<<std::endl;
        */

        // initialize u and u_old using a grid search over the valid spans
        double min_dist = std::numeric_limits<double>::max();
        for(int i = 0; i < valid_spans.size(); i++){
            Eigen::Matrix<double, local_dim,1> u_start;
            Eigen::Matrix<double, embed_dim, 1> steps;
            for(int j = 0; j < local_dim; j++){
                u_start(j) = this->knots_[j][valid_spans[i][j]];
                steps(j) = (this->knots_[j][valid_spans[i][j]+1] - this->knots_[j][valid_spans[i][j]])/n;
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

        u_old = u;

        //auto end = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<double> elapsed_seconds = end-start;
        //std::cout<<"Initialization time: "<<elapsed_seconds.count()<<std::endl;
        u.setZero();

        //start = std::chrono::high_resolution_clock::now();

        auto S = this->eval_param(u_old);

        Eigen::Matrix<double, local_dim,local_dim> J;
        Eigen::Matrix<double, local_dim,1> delta;
        Eigen::Matrix<double, local_dim,1> kappa;

        Eigen::Matrix<double, embed_dim, 1> S_u , S_uu, S_v , S_vv, S_uv ;

        int counter = 0;

        bool conv1 = false;
        bool conv2 = false;
    
        while(!conv1 && !conv2 && counter < max_iters){
            S = this->eval_param(u_old);
            auto S_deriv = this->eval_param_derivative(u_old);
            auto S_second_deriv = this->eval_param_second_derivative(u_old);
            auto r = S - p;

            // fill S_u and S_v
            S_u = S_deriv.col(0);
            S_v = S_deriv.col(1);

            // fill S_uu and S_vv

            // Extract values from MdArray and store in Eigen matrices
            for (int i = 0; i < embed_dim; i++) {
                S_uu(i) = S_second_deriv(i, 0, 0);  // Extract S_uu (second derivative w.r.t u)
                S_uv(i) = S_second_deriv(i, 0, 1);  // Extract S_uv (mixed second derivative)
                S_vv(i)= S_second_deriv(i, 1, 1);  // Extract S_vv (second derivative w.r.t v)
            }


            kappa(0) = - r.dot(S_u);
            kappa(1) = - r.dot(S_v);

            J(0,0) = S_u.dot(S_u) + r.dot(S_uu);
            J(0,1) = S_u.dot(S_v) + r.dot(S_uv);
            J(1,0) = S_u.dot(S_v) + r.dot(S_uv);
            J(1,1) = S_v.dot(S_v) + r.dot(S_vv);

            delta = J.lu().solve(kappa);

            u = u_old + delta;

            // clip u (if u is outside the parametric domain) CYCLIC DOMAINS TO BE HANDLED 
            for(int k = 0; k < local_dim; k++){
                if(u(k) < this->param_nodes_[k][0] ) u(k) = this->param_nodes_[k][0];
                if(u(k) > this->param_nodes_[k][this->param_nodes_[k].size()-1]) u(k) = this->param_nodes_[k][this->param_nodes_[k].size()-1];
            }
            u_old = u;
            //std::cout<<"u= "<<u.transpose()<<std::endl;

            if(r.norm() < tol1 && (delta(0)*S_u + delta(1)*S_v).norm() < tol1) conv1 = true;

            if(std::abs((S_u.dot(r)))/(S_u.norm() * r.norm()) < tol2 && std::abs((S_v.dot(r)))/(S_v.norm() * r.norm())< tol2) conv2 = true;
            counter++;
        }
        //end= std::chrono::high_resolution_clock::now();
        //elapsed_seconds = end-start;
        //std::cout<<"Inversion time: "<<elapsed_seconds.count()<<std::endl;
        //std::cout<<"Number of iterations: "<<counter<<std::endl;

        if(counter == max_iters){
            std::cout<<"Max iterations reached"<<std::endl;
        }

        return u;
    }