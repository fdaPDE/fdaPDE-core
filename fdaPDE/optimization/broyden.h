// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __BROYDEN_H__
#define __BROYDEN_H__

#include <exception>

#include "../utils/symbols.h"

#include <unsupported/Eigen/IterativeSolvers>

namespace fdapde {
namespace core {

template <int N>
struct Broyden {
   private:
    typedef typename std::conditional<N == Dynamic, DVector<double>, SVector<N>>::type VectorType;
    typedef typename std::conditional<N == Dynamic, DMatrix<double>, SMatrix<N>>::type MatrixType;
    std::size_t max_iter_ = 500;   // maximum number of iterations before forced stop
    double tol_ = 1e-3;            // tolerance on error before forced stop
    std::size_t nmax_ = max_iter_ + 1;

   public:
    // constructor
    Broyden() = default;
    Broyden(std::size_t max_iter, double tol) : max_iter_(max_iter), tol_(tol) {};
    Broyden(std::size_t max_iter, std::size_t nmax, double tol) : max_iter_(max_iter), nmax_(nmax), tol_(tol) {};

    // solvers
    template <typename F> VectorType solve(F&, VectorType&) const;
    template <typename F> VectorType solveArmijo(F&, VectorType&) const;
    template <typename F> VectorType solve_modified(F&, VectorType&) const;
    template <typename F> VectorType solve_modified_inv(F&, VectorType&) const;
    template <typename F> VectorType solve_trust_region(F&, VectorType&);
};

// implementative details

// solves the system F(x) = 0, where F is a vector field
// Algorithm following the algorithm in Kelley
// this is ok <=> output_dim = input_dim
template <int N>
template <typename F>
typename Broyden<N>::VectorType Broyden<N>::solve(F& f_, VectorType& x) const {
    static_assert(
        std::is_same<decltype(std::declval<F>().operator()(VectorType())), VectorType>::value,
        "cannot find definition for F.operator()(const VectorType&)");

    int n = -1;

    // define s (we'll have s.col(n) = x_n+1 - x_n)
    MatrixType s;
    s.resize(x.size(), nmax_);
    s.col(0) = -f_(x);

    int itc = 0;   // current iteration

    // store x at each step
    std::ofstream file("Broyden_(" + std::to_string(x(0)) + "," + std::to_string(x(1)) + ")");    //it will be exported in the current build directory

    while (itc < max_iter_){
        n += 1;
        itc += 1;

        // update the point x
        x = x + s.col(n);

        //storing x at each step 
        if (file.is_open()){
            file << std::setprecision(15) << x(0) << '\n' << x(1) << '\n';
        } else {
            std::cerr << "Unable to save solution" << std::endl;
        }

        // std::cout << "iter "<< itc <<", x = " << x.norm() << std::endl;

        if (f_(x).norm() < tol_) {return x;}

        if (n < nmax_-1){
            VectorType z(f_(x).size());
            z = -f_(x);
            for (size_t j = 0; j < n; ++j){
                z += s.col(j+1) * s.col(j).transpose() * z / (s.col(j).squaredNorm());
            } // now z is Bn_inv * F(x_n+1)

            // compute s_n+1
            double a = s.col(n).transpose() * z;
            double b = s.col(n).squaredNorm();
            s.col(n+1) = z / (1. - a / b);
        }
        else {
            n = -1;
            s.col(n+1) = -f_(x);
        }
    }
    file.close();
    
    return x;
} // end solve

// 3 points line search for Broyden Armijo
template <typename T = double>
double parab3p(T lambdac, T lambdam, T ff0, T ffc, T ffm) {
    // std::cout << "lambdac, lambdam, ff0, ffc, ffm = " << lambdac << " "<< lambdam << " " << ff0 << " " << ffc << " " << ffm << std::endl;
    double sigma0 = 0.1;
    double sigma1 = 0.5;
    double c2 = lambdam*(ffc-ff0) - lambdac*(ffm-ff0);
    if (c2>=0)
        return sigma1*lambdac;
    double c1 = lambdac*lambdac*(ffm-ff0)-lambdam*lambdam*(ffc-ff0);
    double lambdap = -c1*0.5/c2;
    if (lambdap < sigma0*lambdac)
        return sigma0*lambdac;
    if (lambdap > sigma1*lambdac)
        return sigma1*lambdac;
    return lambdap;
} // end of parab3p

template <int N>
template <typename F>
typename Broyden<N>::VectorType Broyden<N>::solveArmijo(F& f_, VectorType& x) const {
    static_assert(
        std::is_same<decltype(std::declval<F>().operator()(VectorType())), VectorType>::value,
        "cannot find definition for F.operator()(const VectorType&)");
    
    int maxArm = 7;
    double alpha = 1.e-4;

    int dim = x.size(); //dim potrebbe diventare un parametro della classe dim_ ?
    double sqrtdim = std::sqrt(dim);

    VectorType f0 = f_(x);
    VectorType fc; // set later
    double fnrm = f0.norm()/sqrtdim;;
    double fnrmo;
    double lambda;
    int n = -1;
    VectorType xold(dim); //set later
    VectorType lambda_vec = DVector<double>::Ones(nmax_);

    MatrixType s = DMatrix<double>::Zero(dim, nmax_);
    VectorType snrm = DVector<double>::Zero(nmax_);

    s.col(0) = -f0;
    snrm(0) = s.col(0).squaredNorm(); // NON c'è /sqrt(dim)
    
    // store x at each step
    std::ofstream file("BroydenArmijo_(" + std::to_string(x(0)) + "," + std::to_string(x(1)) + ")");    //it will be exported in the current build directory

    for (int itc = 0; itc < max_iter_; itc++){
        n++;

        // std::cout << "iter " << n << ": " << std::endl;

        fnrmo = fnrm;
        xold = x;
        lambda = 1;

        x += s.col(n);
        
        //storing x at each step 
        if (file.is_open()){
            file << std::setprecision(15) << x(0) << '\n' << x(1) << '\n';
        } else {
            std::cerr << "Unable to save solution" << std::endl;
        }

        fc = f_(x);
        fnrm = fc.norm()/sqrtdim;

        double ff0 = fnrmo*fnrmo;
        double ffc = fnrm*fnrm;
        double lambda_c = lambda;

        double lambda_m; //set later
        double ffm; //set later

        // START LINE SEARCH (correct the step)
        for (int j = 0; j < maxArm; j++){
            if ( fnrm < ( 1 - lambda * alpha )*fnrmo ){
                // std::cout << "\tline search converged -> break" << std::endl;
                break;
            }

            if (j==0)
                lambda*=0.5;
            else
                lambda = parab3p(lambda_c, lambda_m, ff0, ffc, ffm);

            // update values
            // std::cout << "\tline search: iter "<< j << ", lambda = " << lambda << std::endl;
            lambda_m = lambda_c;
            ffm = ffc;
            lambda_c = lambda;
            x = xold + lambda*s.col(n);
            fc = f_(x);
            fnrm = fc.norm()/sqrtdim;
            ffc = fnrm*fnrm;
            // if (j >= maxArm - 1) { std::cout<<"\t failure in line search"<<std::endl; }
        }// end of line search, step corrected

        // stopping criterion
        if (fnrm < tol_){
            return x;
        }

        lambda_vec(n) = lambda;

        // correct the step (save it in the history!)
        if (lambda != 1){
            s.col(n) *= lambda;
            snrm(n) *= lambda*lambda;
        }

        // compute the next step IF we have space, otherwise restart
        if (n<nmax_-1){ // nmax + 1 (?)
            VectorType z = -fc;
            double tmp1, tmp2, tmp3;
            for (int k = 0; k < n; k++){
                tmp1 = s.col(k).dot(z) / snrm(k);
                z += tmp1 * (  ( lambda_vec(k)/lambda_vec(k+1) )*s.col(k+1) + lambda_vec(k)*s.col(k) - s.col(k)  );
            }
            tmp1 = s.col(n).dot(z);
            tmp2 = tmp1 * ( 1 - lambda_vec(n) ) / snrm(n);
            tmp3 = 1 - lambda_vec (n) / snrm(n) * tmp1;
            s.col(n+1) = ( z - tmp2 * s.col(n) ) / tmp3;
            snrm(n+1) = s.col(n+1).squaredNorm();

            // std::cout << std::setprecision(16) << "\n\t x = " << x(0) << ", " << x(1) << std::endl;
            // std::cout << std::setprecision(16) << "\n\tsnrm(n+1) = " << snrm(n+1) << std::endl;

        } else {
            // std::cout << "\n\nrestart\n\n" << std::endl;
            s.col(0) = -fc;
            snrm(0) = s.col(0).squaredNorm();
        }

    } // end for itc
    file.close();
    // std::cout << "not converged" << std::endl;
    return x;       // return without convergence

} // end solveArmijo

template <int N>
template <typename F>
typename Broyden<N>::VectorType Broyden<N>::solve_modified(F& f_, VectorType& x) const {
    static_assert(
        std::is_same<decltype(std::declval<F>().operator()(VectorType())), VectorType>::value,
        "cannot find definition for F.operator()(const VectorType&)");

    int dim = x.size(); // = f_(x).size();

    VectorType d_B(dim), d_MB(dim), d(dim);
    VectorType s(dim), y(dim);

    MatrixType B = DMatrix<double>::Identity(dim, dim);

    int itc = 0;   // current iteration

    // parameters
    double tau = 1.;
    double rho = sqrt(0.9);
    double alfa_k = 1.;
    double theta_k = 1.;
    double r = 0.1;
    auto eps = [](int n) -> double {return 1./((n+1)*(n+1));};
    double sigma_1 = 0.1;
    double sigma_2 = 0.01;
    bool step_5 = false;

    double norm_fx = 0;
    VectorType fx(dim);
    double norm_fz = 0;
    VectorType fz(dim);

    while (itc < max_iter_){
        itc += 1;
        step_5 = false;
        fx = f_(x);
        norm_fx = fx.norm();

        // std::cout << "Current iteration: " << itc << std::endl;
        // std::cout << "||f(x)|| = " << norm_fx << std::endl;

        // step 2
        if (norm_fx < tol_) {return x;}

        // step 3
        // compute s_B_n+1
        Eigen::GMRES<DMatrix<double>> solver(B);
        d_B = solver.solve(-fx);   // solve linear system

        auto z = x + d_B;
        fz = f_(z);
        norm_fz = fz.norm();

        if (norm_fz > tau*(norm_fx)){
            d = d_B;
            // and go to step 4
        }
        else {
            d_MB = solver.solve(-fz);
            if (f_(x + d_B + d_MB).norm() <= rho*norm_fx - sigma_1*(d_B + d_MB).squaredNorm()){
                d = d_B + d_MB;
                alfa_k = 1.;
                // and go to step 5
                step_5 = true;
            }
            else d = d_B; // and go to step 4
        }

        // step 4
        if (step_5 == false){
            if (f_(x+d).norm() <= rho*norm_fx - sigma_1*d.squaredNorm())
                alfa_k = 1.;
            else {
                alfa_k = 1.;
                while (f_(x+alfa_k*d).norm() > norm_fx - sigma_2*(alfa_k*d).squaredNorm() + eps(itc)*norm_fx)
                    alfa_k *= r;
                // std::cout << "iteration " << itc << " alfa = " << alfa_k << std::endl;
                // compute alfa_k = max{1, r, r^2, ...} s.t.
                // f_(x+alfa_k*d.col(n)).norm() <= f_(x).norm() - sigma_2*(alfa_k*d.col(n)).norm() + eps(n)*f_(x).norm()
            }
        }

        // step 5
        auto x_new = x + alfa_k*d;

        // step 6
        // compute B_n+1
        s = x_new - x;
        y = f_(x_new) - fx;
        B = B + theta_k * (y - B*s)*s.transpose() / s.squaredNorm();

        x = x_new;
    } 
    
    return x;
} // end solve_modified

template <int N>
template <typename F>
typename Broyden<N>::VectorType Broyden<N>::solve_modified_inv(F& f_, VectorType& x) const {
    static_assert(
        std::is_same<decltype(std::declval<F>().operator()(VectorType())), VectorType>::value,
        "cannot find definition for F.operator()(const VectorType&)");

    int n = -1;
    int dim = x.size(); // = f_(x).size();

    VectorType d_B(dim), d_MB(dim), d(dim);
    VectorType s(dim), s_old(dim), y(dim);

    MatrixType Id = DMatrix<double>::Identity(dim, dim);
    MatrixType inv_B = Id;

    int itc = 0;   // current iteration

    // DMatrix<double>  inv_B(dim, dim);
    // inv_B = inv_B.inverse(); 

    // parameters
    double tau = 1.;
    double rho = sqrt(0.9);
    double alfa_k = 1.;
    double theta_k = 1.;
    double r = 0.1;
    auto eps = [](int n) -> double {return 1./((n+1)*(n+1));};
    double sigma_1 = 0.1;
    double sigma_2 = 0.1;
    bool step_5 = false;

    double norm_fx = 0;
    VectorType fx(dim);
    double norm_fz = 0;
    VectorType fz(dim);

    fx = f_(x);
    norm_fx = fx.norm();
    s = -fx;

    while (itc < max_iter_){
        n += 1;
        itc += 1;
        step_5 = false;

        // std::cout << "Current iteration: " << itc << std::endl;
        // std::cout << "Current n: " << n << std::endl;
        // std::cout << "Current x: \n" << x << std::endl;
        // std::cout << "||f(x)|| = " << norm_fx << std::endl;

        // step 2
        if (norm_fx < tol_) {return x;}

        // step 3
        d_B = - inv_B * f_(x);
        // std::cout << "s_n+1 = \n" << s.col(n+1) << std::endl;
        auto z = x + d_B;
        fz = f_(z);
        norm_fz = fz.norm();

        if (norm_fz > tau*(norm_fx)){
            d = d_B;
            // and go to step 4
        }
        else {
            d_MB = - inv_B * fz;
            // d_MB = solver.solve(-fz);
            if (f_(x + d_B + d_MB).norm() <= rho*norm_fx - sigma_1*(d_B + d_MB).squaredNorm()){
                d = d_B + d_MB;
                alfa_k = 1.;
                // and go to step 5
                step_5 = true;
            }
            else d = d_B; // and go to step 4
        }

        // step 4
        if (step_5 == false){
            if (f_(x+d).norm() <= rho*norm_fx - sigma_1*d.squaredNorm())
                alfa_k = 1.;
            else {
                alfa_k = 1.;
                while (f_(x+alfa_k*d).norm() > norm_fx - sigma_2*(alfa_k*d).squaredNorm() + eps(itc)*norm_fx)
                    alfa_k *= r;
                // compute alfa_k = max{1, r, r^2, ...} s.t.
                // f_(x+alfa_k*d.col(n)).norm() <= f_(x).norm() - sigma_2*(alfa_k*d.col(n)).norm() + eps(n)*f_(x).norm()
            }
        }

        // step 5
        auto x_new = x + alfa_k*d;

        // step 6
        // compute B_n+1
        s_old = s;
        s = x_new - x;
        y = f_(x_new) - fx;

        //compute (B_n+1)^-1
        // inv_B *= (Id + s*s_old.transpose() / s_old.squaredNorm());
        inv_B += (s - inv_B*y)*y.transpose() / y.squaredNorm();
    
        // update variables
        x = x_new;
        fx = f_(x);
        norm_fx = fx.norm();
    } 

    return x;
} // end solve_modified_inv

// function we need for solve_trust_region
template <int N>
double solve_quadratic(double Delta, typename Broyden<N>::VectorType sU, typename Broyden<N>::VectorType sN){
    // Delta^2 = ||sC + (tau-1)(sN-sC)||^2    Nocedal-Wright pag 75
    // if tau = 1 we will go for full gradient descent in the trust region, if tau = 0 -> full newton
    /*
        * solve the quadratic function
        * Delta^2 = [ sC + (tau-1)(sN-sC) ]^T * [ sC + (tau-1)(sN-sC) ]
        * ...
        * */

//                Delta = std::min(sU.norm(), sN.norm()); //così siamo sicuri che una soluzione ci sia

    //define some auxiliaries
    double c1 = sU.dot(sU);
    double c2 = sU.dot(sN);
    double c3 = sN.dot(sU);
    double c4 = sN.dot(sN);

    // now we should solve
    // (c4-c3-c2+c1)y^2 + (c2+c3-2*c1)y + (c1-Delta*Delta) = 0    where (tau-1) = y
    // y_1_2 = ( -B +- sqrt(B^2-4AC) ) / 2A
    double A = c4 - c3 - c2 + c1;
    double B = c2 - 2*c1 + c3;
    double C = c1 - Delta*Delta;

    double D = B*B-4*A*C;
    if (D < 0){
        // std::cout << "error in solving for tau, no roots" << std::endl;
        return 1;
    }

    double y1 = ( -B + std::sqrt(D) ) / (2*A);
    double y2 = ( -B - std::sqrt(D) ) / (2*A);

    double tau1 = y1 + 1;
    double tau2 = y2 + 1;

    if (tau1 >=0 && tau1 <= 2){
        // std::cout << "returning tau1 = " << tau1 << std::endl;
        return tau1;
    }
    if (tau2 >=0 && tau2 <= 2){
        // std::cout << "returning tau2 = " << tau2 << std::endl;
        return tau2;
    }

    // std::cout << "error in solving for tau, roots not in range" << std::endl;
    return 1;
}

// solve with Broyden Trust Region
template <int N>
template <typename F>
typename Broyden<N>::VectorType Broyden<N>::solve_trust_region(F& f_, VectorType& x) {
    static_assert(
        std::is_same<decltype(std::declval<F>().operator()(VectorType())), VectorType>::value,
        "cannot find definition for F.operator()(const VectorType&)");
    
    bool debug = 0;

    VectorType Fk, Fk1, sN, xk, xk1;
    MatrixType Bk;

    // initialize variables and parameters
    xk = x;
    Fk = f_(xk);

    // set parameters for trust region
    double Delta_MAX = 10;  //? booh
    double Delta = 4;       //? booh
    double eta = 0.125;     //? booh, this should be in [0, 1/4)    Nocedal-Wright pag 69

    // initialize Bk for Broyden method
    // Bk = f_.derive()(xk);    //perché vectorField non ha derive !?!?!
    if constexpr (N == Dynamic) {   // inv_hessian approximated with identity matrix
        Bk = MatrixType::Identity(x.rows(), x.rows());
    } else {
        Bk = MatrixType::Identity();
    }
    /*double den = x(0)*x(0)*x(1)*x(1) + 1;
    Bk.coeffRef(0,0) = x(1)/den;
    Bk.coeffRef(0,1) = x(0)/den;
    Bk.coeffRef(1,0) = 1;
    Bk.coeffRef(1,1) = -5;
    std::cout << Bk << std::endl;*/

    std::size_t k = 0;
    while (k < max_iter_ && Fk.norm() > tol_) {
        // std::cout << "\n\nbegin iteration " << k << std::endl;
        // std::cout << "||Fk|| = " << std::setprecision(16) << Fk.norm() << std::endl;
        // std::cout << "Delta = " << Delta << std::endl;

        // assert Bk is not singular, and actually we need the determinant to be larger than a tolerance,
        // so that the determinant of B^T B is larger than machine eps
        //...

        // compute newton direction
        // solve linear system Bk sN = Fk
        // if (debug) std::cout << "B = \n" << Bk << std::endl;
        Eigen::GMRES<MatrixType> solver(Bk);
        sN = solver.solve(-Fk);
        // if (debug) std::cout << "Newton Direction: [" << std::setprecision(8) << sN(0) << ", " << sN(1) << "], ||sN|| = :" << sN.norm() << std::endl;

        // compute the hessian matrix of the associate optimization problem
        MatrixType Hk = Bk.transpose()*Bk;  // this matrix may be ill conditioned, but we need it

        // compute the Cauchy Point direction
        /*
            * Worst case scenario, the trust region method performs a step which corresponds to the steepest
            * descent in the trust region. I compute it only for the case in which $\nabla^2 f$ is symmetric
            * positive definite. See Nocedal-Wriht (4.11) - (4.12)
            * It should correspond to the steepest descent direction of the associated minimization problem
            * with the constraint of the trust region (lambda). Therefore, the cauchy point must be inside the
            * trust region, ||sC||<=Delta
            * */
        double num = (Bk.transpose()*Fk).norm();
        num *= num*num;
        double den = Fk.transpose()*Hk*Hk*Fk;
        den *= Delta;
        double tau_k = std::min(1., num/den);
        tau_k *= -Delta/(Bk.transpose()*Fk).norm();
        VectorType sC = tau_k*Bk.transpose()*Fk;
        // if (debug) std::cout << "Cauchy Direction: [" << std::setprecision(8) << sC(0) << ", " << sC(1) << "], ||sC|| = :" << sC.norm() << std::endl;

        // begin the Trust-Region dogleg method
        bool step_accepted = false;
        std::size_t j = 0;
        std::size_t maxit_dogleg = 4;
        while(!step_accepted && j < maxit_dogleg){
            j++;
            // std::cout << "\nDogleg iteration " << j << "/" << maxit_dogleg << std::endl;
            step_accepted = true;

            //find the dogleg step
            VectorType s_=sN;
            if (s_.norm()>Delta){
                // solve for tau ( should be in the interval [0,2] )
//                            VectorType sU = -(Bk.transpose()*Fk);       // (4.15)
//                            sU *= (Fk.dot(Hk*Fk))/(Fk.dot(Hk*Hk*Fk));   // (4.15)
                /*if ( sU.norm()>Delta ){
                    std::cout << "shrinking sU into trust region" << std::endl;
                    sU = Delta*sU/sU.norm();
                }*/
//                            std::cout << "sU Direction: [" << std::setprecision(8) << sU(0) << ", " << sU(1) << "], ||sU|| = :" << sU.norm() << std::endl;
                //compute full dogleg step
                double tau = solve_quadratic(Delta, sC, sN);
                if (tau <= 1) {
                    s_ = tau * sC;
                } else {
                    s_ = sC + (tau - 1) * (sN - sC);
                }

            } else {
                // std::cout << "trying with newton direction" << std::endl;
            }

            // update with dogleg direction, later we'll decide if we accept it or not.
            xk1 = xk + s_;

            // compute some auxiliaries
            Fk1 = f_(xk1);
            double fk = Fk.dot(Fk);
            double fk1 = Fk1.dot(Fk1);
            double mk1 = fk + Fk.dot(Bk*s_) + 0.5*s_.dot(Hk*s_); // evaluation of the associated quadratic model
            // if (debug) std::cout << "fk = " << fk << "\tfk+1 = " << fk1 << "\tmk+1 = " << mk1 << std::endl;

            // compute the model fit parameter ( Nocedal-Wright 4.4 pag 68 )
            double rho = (fk-fk1)/(fk-mk1);  // should be in the interval [0,1]
            // std::cout << "rho = " << rho << std::endl;

            // decide what to do with the trust region
            if (rho < 0.25 || rho > 2){
                Delta = std::max(Delta*0.5, 0.001);    // quadratic model is not a good fit, shrink trust region
                // std::cout << "shrinking trust region " << std::endl;
            } else if(rho>0.75 && std::abs(s_.norm() - Delta)<1e-10 && rho < 2) {   // I added rho < 3
                Delta = std::min(2*Delta, Delta_MAX);
                // std::cout << "expanding trust region " << std::endl;
            }
            // if the two above conditions are not satisfied, leave the trust region unchanged
            // std::cout << "Delta = " << Delta << std::endl;

            // decide whether to accept xk1 or not
            // if ( rho <= eta )   //eta is a parameter in [0,1/4) ( Nocedal-Wright pag 69 )
            if (fk1>fk) {
                // if nothing happened to the trust region, change it...
                step_accepted = false;  // try again with shrunk trust region
                // std::cout << "step rejected" << std::endl;
            }
        }
        // step accepted !
        // if (debug) std::cout << "xk = [" << std::setprecision(8) << xk(0) << ", " << xk(1) << "]" << std::endl;
        // if (debug) std::cout << "xk+1 = [" << std::setprecision(8) << xk1(0) << ", " << xk1(1) << "]" << std::endl;

        // update Broyden matrix
        VectorType s = xk1 - xk;
        VectorType y = Fk1 - Fk;
        Bk += (y-Bk*s)*s.transpose() / s.dot(s);

        // update matrix of the associated quadratic model
        Hk = Bk.transpose()*Bk;

        // prepare for next iteration
        xk = xk1;
        Fk = Fk1;
        k++;
    } // end of nonlinear loop

    return xk1;
} //end solve with trust region

}   // namespace core
}   // namespace fdapde

#endif   // __BROYDEN_H__
