
#ifndef __NURBS_H__
#define __NURBS_H__

#include "../linear_algebra/constexpr_matrix.h"
#include "../linear_algebra/mdarray.h"
#include "../fields/matrix_field.h"
#include "scalar_field.h"
#include "../splines/spline_basis.h"

namespace fdapde{

    // Fai un check computazionale confrontando le nurbs evaluation con quelle di  DeGaspari

// multi-contract function (iterative version)    
template<int M>
inline double multicontract(const MdArray<double, full_dynamic_extent_t<M>>& weights,
                     const std::array<std::vector<double>, M>& parts) {

    double contracted_value = 0.0;
    std::array<int, M> sizes = {0};
    std::size_t total_size = 1;

    for (int i = 0; i < M; ++i) {
        sizes[i] = parts[i].size();
        total_size *= sizes[i];
    }

    for(int flat_idx = 0; flat_idx < total_size; ++flat_idx) {
        std::array<int, M> multi_idx = {0};
        int temp = flat_idx;
        for (int i = M - 1; i >= 0; --i) {
            multi_idx[i] = temp % sizes[i];
            temp /= sizes[i];
        }
        double sub_value = weights(multi_idx);
        for (int i = 0; i < M; ++i) {
            sub_value *= parts[i][multi_idx[i]];
        }
        contracted_value += sub_value;
    }

    return contracted_value;
}



template<int M>
class Nurbs: public ScalarBase<M,Nurbs<M>> {
        public:

            using Base = ScalarBase<M,Nurbs<M>>;
            static constexpr int StaticInputSize = M;
            static constexpr int NestAsRef = 0;   // avoid nesting as reference, .derive() generates temporaries
            static constexpr int XprBits = 0;
            static constexpr int Order = Dynamic;
            using Scalar = double;
            using InputType = cexpr::Vector<Scalar, StaticInputSize>;

        private:
            std::array<std::vector<double>,M> knots_ ;
            MdArray<double,full_dynamic_extent_t<M>> weights_;
            std::array<int,M> index_ ;
            int order_;

            std::array<std::size_t, M> minIdx_;
            std::array<int, M> extents_; 
            double num0_ = 0.0;


        public:
            Nurbs() = default;

            Nurbs(std::array<std::vector<double>,M>& knots, MdArray<double,full_dynamic_extent_t<M>>& weights, std::array<int,M>& index, int order): 
                knots_(knots), index_(index), order_(order){

                std::array<std::size_t, M> maxIdx;

                for (std::size_t i = 0; i < M; ++i) {

                    minIdx_[i] = (index_[i] >= order)? (index_[i]-order) : 0;
                    extents_[i] = (index_[i] + order < weights.extent(i))? (index_[i]+order+1-minIdx_[i]) : (weights.extent(i)-minIdx_[i]); 
                    maxIdx[i] = (minIdx_[i] + extents_[i]-1);

                    gradient_[i] = FirstDerivative(knots, weights, index, order, i);

                    // hessian da fare

                }

                // allocate space for the weights
                weights_.resize(extents_);
                // fill the block of weights with only the necessary values;
                weights_ = weights.block(minIdx_, maxIdx);
                // compute the starting numerator  
                num0_ = weights(index_); 

                // wrap the gradient components into a vectorfield


                // Costruisci una base spline per ogni dimensione
                // la puoi usare la funzione di valutazione rapida


                // Parte da un vettore di shared pointers di basi spline che costruira la nurbs basis
                // vettore di shared pointer per salvare le basi spline
                // se gli sharedpointer non attivi crei la base spline

                

                //std::cout << "NURBS correctly initialized! " << std::endl; 

            };

            // add a constructor for the 1d knot passed as std::vector<double>, using the previous constructor
            // mettere uno static assert che fa un check che M=1 , 
            Nurbs(std::vector<double>& knots, MdArray<double,full_dynamic_extent_t<M>>& weights, std::array<int,M>& index, int order): 
            Nurbs(std::array<std::vector<double>,M>{knots}, weights, index, order) {
                fdapde_static_assert(M == 1, THIS_METHOD_IS_ONLY_FOR_1D_NURBS);
            };

            
            //evaluates the NURBS at a given point, funziona
            template <typename InputType_>
            requires(fdapde::is_subscriptable<InputType_, int>)
            constexpr Scalar operator()(const InputType_& p_) const {
                
                double num = num0_;
                std::array<std::vector<double>,M> spline_evaluation {};
                double den;

                // Initialize each vector to the appropriate size
                for (std::size_t i = 0; i < M; ++i) {
                    spline_evaluation[i].resize(extents_[i]);
                }

                for(std::size_t i=0;i<M;i++){

                    // pad the knots
                    auto n = knots_[i].size();
                    std::vector<double> padded_knots;
                    padded_knots.resize(n + 2 * order_);
                    for (int j = 0; j < n + 2 * order_; ++j) {
                        if (j < order_) {
                            padded_knots[j] = knots_[i][j];
                        } else {
                            if (j < n + order_) {
                                padded_knots[j] = knots_[i][j - order_];
                            } else {
                                padded_knots[j] = knots_[i][n - 1];
                            }
                        }
                    }
                    
                    auto basis_eval = SplineBasis(padded_knots, order_).evaluate_basis(p_[i]);
                    //std::cout << "basis_eval : "  << std::endl;

                    // print the basis evaluation
                    //for (auto& el : basis_eval)
                    //    std::cout << el << std::endl;
                    
                    // spline evaluation for i-th dimension
                    for(std::size_t j = 0; j<extents_[i]; j++ ){
                        //std::cout<<"MinIdx: "<<minIdx_[i]<<std::endl;
                        //auto spline = Spline(knots_[i], minIdx_[i]+j, order_);
                        //spline_evaluation[i][j] = spline(std::vector<double>{p_[i]}); 
                        //std::cout << "basis_eval: " << basis_eval[minIdx_[i]+j] << std::endl;
                        spline_evaluation[i][j] = basis_eval[minIdx_[i]+j]; // more efficient
                    } 
                    // print spline evalutazion
                    //std::cout << "spline_evaluation : "  << std::endl;
                    //for (auto& el : spline_evaluation[i])
                    //    std::cout << el << std::endl;

                    //numerator update
                    num *= spline_evaluation[i][index_[i] - minIdx_[i]];
                    //spline evaluation for i-th dimension
                }
                // avoid division by 0
                if(num == 0)
                    return 0;
                // compute the sum that appears at the denominator of the formula
                den = multicontract<M>(weights_, spline_evaluation);

                return num/den;
            }; 

        private:
            class FirstDerivative: public MatrixBase<M,FirstDerivative> {
            public:
            
                using Base = MatrixBase<M,FirstDerivative>;
                static constexpr int StaticInputSize = M;
                static constexpr int NestAsRef = 0;   // avoid nesting as reference, .derive() generates temporaries
                static constexpr int XprBits = 0;
                static constexpr int Order = Dynamic;
                using Scalar = double;
                using InputType = cexpr::Vector<Scalar, StaticInputSize>;

            private:
                std::array<std::vector<double>,M> knots_ ;
                MdArray<double,full_dynamic_extent_t<M>> weights_;
                std::array<int,M> index_ ;
                int order_;

                std::array<std::size_t, M> minIdx_;
                std::array<int, M> extents_; // fare il min_idx e max idx ??? al posto di minIdx e extents, usa un int al posto di eigen::index
                double num0_ = 0.0;

                size_t i_ = 0; // index of the derivative, questa i è necessaria ? se mi interessa il gradiente in una direzione si

            public:
                FirstDerivative() = default;

                FirstDerivative(const std::array<std::vector<double>,M>& knots,const MdArray<double,full_dynamic_extent_t<M>>& weights, const std::array<int,M>& index, int order, std::size_t i): 
                    knots_(knots), index_(index), order_(order), i_(i){


                    std::array<std::size_t, M> maxIdx;

                    for (std::size_t i = 0; i < M; ++i) {
                        minIdx_[i] = (index_[i] >= order)? (index_[i]-order) : 0;
                        extents_[i] = (index_[i] + order < weights.extent(i))? (index_[i]+order+1-minIdx_[i]) : (weights.extent(i)-minIdx_[i]); 
                        maxIdx[i] = (minIdx_[i] + extents_[i]-1);

                    }
                    // allocate space for the weights
                    weights_.resize(extents_);
                    // fill the block of weights with only the necessary values;
                    weights_ = weights.block(minIdx_, maxIdx);
                    // compute the starting numerator  
                    num0_ = weights(index_);   

                    //std::cout << "NURBS derivative correctly initialized! " << std::endl;

                };

                // evalutes the first order partial derivative of the NURBS at a given point, funziona
                constexpr Scalar operator()(const InputType& p) const {

                    //std::cout<<"Inizio a calcolare"<<std::endl;

                    double num = num0_;
                    std::array<std::vector<double>,M> spline_evaluation {};
                    double den;

                    double num_derived = 0.0;
                    double den_derived = 0.0;

                    // Initialize each vector to the appropriate size
                    for (std::size_t i = 0; i < M; ++i) {
                        spline_evaluation[i].resize(extents_[i]);
                    }

                    for(std::size_t k=0;k<M;k++){
                        // spline evaluation for i-th dimension
                        auto basis_eval = SplineBasis(knots_[k], order_).evaluate_basis(p[k]);

                        for(std::size_t j = 0; j<extents_[k]; j++ ){
                            // compute a spline basis function
                            spline_evaluation[k][j] = basis_eval[minIdx_[k]+j]; // rivedi 

                            // metti in spline un overload del call operator che accetta un double
                        }
                        if (k!=i_)
                            num*=spline_evaluation[k][index_[k] - minIdx_[k]];
                    
                    }

                    //compute the derivative of the i_th spline
                    num_derived = num * Spline(knots_[i_], index_[i_], order_).gradient(1)(p[i_]);

                    // compute the non derived numerator
                    num*=spline_evaluation[i_][index_[i_] - minIdx_[i_]];

                    if (num== 0 && num_derived == 0)
                        return 0;


                    // compute the sum that appears at the denominator of the formula
                    den = multicontract<M>(weights_, spline_evaluation);

                    // by replacing the i-th evaluations with their derivatives we get the derivative of the NURBS denominator
                    for (std::size_t j = 0; j<extents_[i_]; j++ ){
                        auto spline = Spline(knots_[i_], minIdx_[i_]+j, order_);
                        spline_evaluation[i_][j] = dx(spline)(p[i_]); 
                    }

                    den_derived = multicontract<M>(weights_, spline_evaluation);

                    //  ( N )'      N'D - ND'
                    //  (---)   =  ----------
                    //  ( D )         D^2
                    // where f' = df/dx_i
                    return (num_derived*den - num*den_derived)/(den*den);
                };
            };

            private:

                // gradient and hessian$
                VectorField<M, M, FirstDerivative> gradient_; //gradient
                //MatrixField<M,M,M,NurbsSecondDerivative>> hessian_; //hessian accedi con hessian_(i,j)

            public:
                constexpr FirstDerivative first_derive(int i=0) const { return gradient_[i]; }
                //constexpr SecondDerivarive second_derive(int i, int j) const { return SecondDerivative(knots_, weights_, index_, order_, i, j); }
                
                // overload the call operator for the first derivative for a double
                constexpr Scalar operator()(double p) const { return operator()(InputType{p}); }
    
    };

}// namespace fdapde

#endif


