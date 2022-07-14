   // Algo A4.3 from NURBS book pag. 103, evaluation of the physical derivative of a NURBS curve 
   Eigen::Matrix<double, EmbedDim, LocalDim> eval_param_derivative(const Eigen::Matrix<double, LocalDim,1>& u) const {
    for(int i = 0; i < LocalDim; i++) fdapde_assert(u(i) >= knots_[i].front() && u(i) <= knots_[i].back());
    std::vector<std::vector<double>> basis_eval(LocalDim);
    std::vector<std::vector<double>> basis_deriv_eval(LocalDim);
    std::array<int,LocalDim> spans= {0};
    auto order = this->basis_.order();
    auto nurb = this->basis_[0];
    double total_weight = 0.0;
    
    for(int i = 0; i < LocalDim; i++){
        auto basis = nurb.spline_basis()[i];
        basis_eval[i] = basis->evaluate_basis(u(i), false); // evaluate basis functions, padding = false
        basis_deriv_eval[i] = basis->evaluate_der_basis(u(i), 1, false);
        spans[i] = basis->find_span(u(i)); // find the span of the knot vector
    }
    
    Eigen::Matrix<double, EmbedDim, LocalDim> dSw = Eigen::Matrix<double, EmbedDim, LocalDim>::Zero();
    Eigen::Matrix<double, EmbedDim, 1> Sw = Eigen::Matrix<double, EmbedDim, 1>::Zero();
    Eigen::Matrix<double, LocalDim, 1> dW = Eigen::Matrix<double, LocalDim, 1>::Zero();
    
    std::vector<int> index(LocalDim, 0);
    bool done = false;
    while (!done) {
        double eval = 1.0;
        std::array<double, LocalDim> eval_der = {0.0};        
        std::array<int, LocalDim> full_indices;

        for (int i = 0; i < LocalDim; i++) {
            eval *= basis_eval[i][index[i]];
            eval_der[i] = basis_deriv_eval[i][index[i]];
            full_indices[i] = spans[i] - order[i] + index[i];
        }

        
        Eigen::Matrix<double, EmbedDim, 1> cp;
        for (int i = 0; i < EmbedDim; i++) {
            const auto cp_slice = this->control_points_.template slice<LocalDim>(i);
            cp(i) = cp_slice(full_indices);
        }
        
        double w = weights_(index);
        cp *= w;
        Sw += eval * cp;
        total_weight += eval * w;
        
        for (int j = 0; j < LocalDim; j++) {
            double w_temp = w*eval_der[j];
            Eigen::Matrix<double, EmbedDim, 1> cp_temp = eval_der[j] *cp;
            for(int i = 0; i < LocalDim; i++){
                if(i != j){
                    w_temp *= basis_eval[i][index[i]];
                    cp_temp *= basis_eval[i][index[i]];
                }
            }
            dW(j) += w_temp;
            dSw.col(j) += cp_temp;
        }
        
        for (int d = LocalDim - 1; d >= 0; d--) {
            if (++index[d] > order[d]) {
                index[d] = 0;
                if (d == 0) done = true;
            } else 
                break;
        }
    }
    
    // Apply derivative of the ratio d/du (Sw / total_weight)
    for (int j = 0; j < LocalDim; j++) {
        dSw.col(j) = (dSw.col(j) - (Sw * dW(j) / total_weight)) / total_weight;
    }

    return dSw;
}  

// Algo A4.3 from NURBS book pag. 103, evaluation of the hessian of a NURBS curve 
MdArray<double, MdExtents<EmbedDim, LocalDim, LocalDim>> eval_param_second_derivative(const Eigen::Matrix<double, LocalDim,1>& u) const {
    for (int i = 0; i < LocalDim; i++) 
        fdapde_assert(u(i) >= knots_[i].front() && u(i) <= knots_[i].back());

    std::vector<std::vector<double>> basis_eval(LocalDim);
    std::vector<std::vector<double>> basis_deriv_eval(LocalDim);
    std::vector<std::vector<double>> basis_second_deriv_eval(LocalDim);
    std::array<int, LocalDim> spans = {0};
    auto order = this->basis_.order();
    auto nurb = this->basis_[0];
    double total_weight = 0.0;

    for (int i = 0; i < LocalDim; i++) {
        auto basis = nurb.spline_basis()[i];
        basis_eval[i] = basis->evaluate_basis(u(i), false); // Basis functions
        basis_deriv_eval[i] = basis->evaluate_der_basis(u(i), 1, false); // First derivative
        basis_second_deriv_eval[i] = basis->evaluate_der_basis(u(i), 2, false); // Second derivative
        spans[i] = basis->find_span(u(i));

    }


    Eigen::Matrix<double, EmbedDim, LocalDim> dSw = Eigen::Matrix<double, EmbedDim, LocalDim>::Zero();
    Eigen::Matrix<double, EmbedDim, 1> Sw = Eigen::Matrix<double, EmbedDim, 1>::Zero();
    Eigen::Matrix<double, LocalDim, 1> dW = Eigen::Matrix<double, LocalDim, 1>::Zero();

    MdArray<double, MdExtents<EmbedDim, LocalDim, LocalDim>> d2Sw;  // Second derivative tensor
    d2Sw.set_constant(0.0);
    Eigen::Matrix<double, LocalDim, LocalDim> d2W = Eigen::Matrix<double, LocalDim, LocalDim>::Zero();

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
            eval_sec_der[i] = basis_second_deriv_eval[i][index[i]];
            full_indices[i] = spans[i] - order[i] + index[i];
        }

        Eigen::Matrix<double, EmbedDim, 1> cp;
        for (int i = 0; i < EmbedDim; i++) {
            const auto cp_slice = this->control_points_.template slice<LocalDim>(i);
            cp(i) = cp_slice(full_indices);
        }

        double w = weights_(index);
        cp *= w;
        Sw += eval * cp;
        total_weight += eval * w;

        for (int j = 0; j < LocalDim; j++) {
            double w_temp = w * eval_der[j];
            Eigen::Matrix<double, EmbedDim, 1> cp_temp = eval_der[j] * cp;
            for (int i = 0; i < LocalDim; i++) {
                if (i != j) {
                    w_temp *= basis_eval[i][index[i]];
                    cp_temp *= basis_eval[i][index[i]];
                }
            }
            dW(j) += w_temp;
            dSw.col(j) += cp_temp;

            // Compute second derivatives
            double w_temp_2 = w * eval_sec_der[j];
            Eigen::Matrix<double, EmbedDim, 1> cp_temp_2 = eval_sec_der[j] * cp;

            for (int i = 0; i < LocalDim; i++) {
                if (i != j) {
                    w_temp_2 *= basis_eval[i][index[i]];
                    cp_temp_2 *= basis_eval[i][index[i]];
                }
            }

            d2W(j, j) += w_temp_2;

            for (int h = 0 ; h < EmbedDim; h++) d2Sw(h, j, j) += cp_temp_2(h);
            

            for (int k = 0; k < LocalDim; k++) {
                if (j != k) {
                    double w_mixed = w * eval_der[j] * eval_der[k];
                    Eigen::Matrix<double, EmbedDim, 1> cp_mixed = eval_der[j] * eval_der[k] * cp;

                    for (int i = 0; i < LocalDim; i++) {
                        if (i != j && i != k) {
                            w_mixed *= basis_eval[i][index[i]];
                            cp_mixed *= basis_eval[i][index[i]];
                        }
                    }

                    d2W(j, k) += w_mixed;
                    for (int h = 0; h < EmbedDim; h++) d2Sw(h, j, k) += cp_mixed(h);
                }
            }
        }

        for (int d = LocalDim - 1; d >= 0; d--) {
            if (++index[d] > order[d]) {
                index[d] = 0;
                if (d == 0) done = true;
            } else {
                break;
            }
        }
    }


    // Apply the quotient rule for second derivatives
    for (int j = 0; j < LocalDim; j++) {
        for (int k = 0; k < LocalDim; k++) {
            for(int h = 0; h < EmbedDim; h++){
                d2Sw(h, j, k) = (d2Sw(h, j, k) 
                - (dSw(h, j) * dW(k) + dSw(h, k) * dW(j) + Sw(h) * d2W(j, k)) / total_weight + 2*Sw(h)*dW(j)*dW(k)/ (total_weight* total_weight) ) / total_weight;
            }
            
        }
    }
    return d2Sw;
}