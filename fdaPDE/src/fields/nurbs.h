#ifndef __NURBS_H__
#define __NURBS_H__

#include "header_check.h"

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


/**
 * @brief NURBS (Non-Uniform Rational B-Spline) scalar field class.
 * 
 * Provides evaluation, derivatives, gradient, and Hessian of a NURBS basis function
 * in arbitrary dimension.
 * 
 * @tparam M The dimension of the NURBS domain.
 */
template<int M>
class Nurbs: public ScalarFieldBase<M,Nurbs<M>> {
    public:
        using Base = ScalarFieldBase<M,Nurbs<M>>;
        static constexpr int StaticInputSize = M;
        static constexpr int NestAsRef = 0;   // avoid nesting as reference, .derive() generates temporaries
        static constexpr int XprBits = 0;
        static constexpr int Order = Dynamic;
        using Scalar = double;
        using InputType = Vector<Scalar, StaticInputSize>;

    private:
        std::array<std::shared_ptr<BSplineBasis>, M> spline_basis_;
        MdArray<double,full_dynamic_extent_t<M>> weights_;
        std::array<int,M> index_ ;
        std::array<int,M> degree_;
        double num0_ = 0.0;
        std::array<std::size_t, M> minIdx_;
        std::array<int, M> extents_;
        std::array<int, M> periodicity_ = {0}; // 0 = non-periodic, 1 = periodic

        /**
         * @brief Initialize basis and weights arrays for efficient evaluation.
         * @param weights The full weights array.
         */
        void initialize_basis_and_weights_(const MdArray<double, full_dynamic_extent_t<M>>& weights) {
            std::array<std::size_t, M> maxIdx;
            for (std::size_t i = 0; i < M; ++i) {
                int deg = degree_[i];
                minIdx_[i] = (index_[i] >= deg) ? (index_[i] - deg) : 0;
                extents_[i] = (index_[i] + deg < weights.extent(i)) ?
                            (index_[i] + deg + 1 - minIdx_[i]) :
                            (weights.extent(i) - minIdx_[i]);
                maxIdx[i] = minIdx_[i] + extents_[i] - 1;
            }
            weights_.resize(extents_);
            weights_ = weights.block(minIdx_, maxIdx);
            num0_ = weights(index_);
        }

    public:
        /**
         * @brief Default constructor. Constructs an empty NURBS object.
         */
        Nurbs() = default;

        /**
         * @brief Construct a NURBS basis function from knot vectors, weights, indices, and degrees.
         *
         * @tparam KnotsVectorType Type of the knot vector (container of doubles).
         * @param knots Array of knot vectors for each dimension.
         * @param weights The weights array (control points).
         * @param index The multi-index for the basis function.
         * @param degree The polynomial degree for each dimension.
         */
        template <typename KnotsVectorType>
        requires(requires(KnotsVectorType knots, int i) {
                { knots[i] } -> std::convertible_to<double>;
                { knots.size() } -> std::convertible_to<std::size_t>;
            })
        Nurbs(std::array<KnotsVectorType,M>&& knots, MdArray<double,full_dynamic_extent_t<M>>& weights, std::array<int,M>&& index, std::array<int,M>& degree, std::array<int,M>& periodicity = {0}):
            index_(std::move(index)), degree_(degree), periodicity_(periodicity) {
            
            for (std::size_t i = 0; i < M; ++i) {
                std::vector<double> knots_ = pad_knots(knots[i], degree_[i]);
                spline_basis_[i] = std::make_shared<BSplineBasis>(knots_, degree_[i], periodicity[i]);
            }
            // initialize the gradient
            for (std::size_t i = 0; i < M; ++i){
                gradient_[i] = FirstDerivative(spline_basis_, weights, index, i);
            }
            // initialize the hessian
            for (std::size_t i = 0; i < M; ++i){
                for (std::size_t j = 0; j < M; ++j){
                    hessian_(i,j) = SecondDerivative(spline_basis_, weights, index, i, j);
                }
            }
            initialize_basis_and_weights_(weights);
        }

        /**
         * @brief Construct a 1D NURBS basis function from a single knot vector, weights, index, and degree.
         * @tparam KnotsVectorType Type of the knot vector.
         * @param knots Knot vector.
         * @param weights The weights array.
         * @param index The basis function index.
         * @param degree The polynomial degree.
         */
        template <typename KnotsVectorType>
        requires(requires(KnotsVectorType knots) {
                { knots.size() } -> std::convertible_to<std::size_t>;
            })
        Nurbs(KnotsVectorType& knots, MdArray<double,full_dynamic_extent_t<M>>& weights, int index, int degree, bool periodicity = false ): 
        Nurbs(std::array<std::vector<double>,M>{std::move(knots)}, weights, std::array<int,M>{index}, std::array<int,M>{degree}, std::array<int,M>{periodicity}) {
            fdapde_static_assert(M == 1, THIS_METHOD_IS_ONLY_FOR_1D_NURBS);
        }

        /**
         * @brief Construct a NURBS basis function from shared B-spline bases, weights, and index.
         *        Used by the NurbsBasis.
         * @param spline_basis Array of shared pointers to B-spline bases.
         * @param weights The weights array.
         * @param index The multi-index for the basis function.
         */
        Nurbs(std::array<std::shared_ptr<BSplineBasis>, M> spline_basis, MdArray<double,full_dynamic_extent_t<M>>& weights,std::array<int,M>& index) : spline_basis_(spline_basis), index_(index) { 
            for (std::size_t i = 0; i < M; ++i) {
                degree_[i] = spline_basis[i]->degree();   
                periodicity_[i] = spline_basis[i]->periodicity();        
            }
            // allocate for the gradient
            for (std::size_t i = 0; i < M; ++i){
                gradient_[i] = FirstDerivative(spline_basis_, weights, index, i);
            }
            // initialize the hessian
            for (std::size_t i = 0; i < M; ++i){
                for (std::size_t j = 0; j < M; ++j){
                    hessian_(i,j) = SecondDerivative(spline_basis_, weights, index, i, j);
                }
            }
            initialize_basis_and_weights_(weights);
            // STAI SALTANDO L'INIZIALIZZAZIONE DELLA BASE SPLINE, QUINDI SE CREO 1000 Nurbs questo è più efficiente
        }

        /**
         * @brief Evaluate the NURBS basis function at a given point.
         * @tparam InputType_ Point type (must be subscriptable by int).
         * @param p_ Point at which to evaluate the NURBS function.
         * @return Value of the NURBS basis function at @p p_.
         */
        template <typename InputType_>
        requires(internals::is_subscriptable<InputType_, int>)
        constexpr Scalar operator()(const InputType_& p_) const {
            double num = num0_;
            std::array<std::vector<double>,M> spline_evaluation {};
            double den;
            for(std::size_t i=0;i<M;i++){
                auto basis_eval = spline_basis_[i]->evaluate_basis(p_(i));
                spline_evaluation[i].resize(extents_[i]);
                for(std::size_t j = 0; j<extents_[i]; j++ ){
                    spline_evaluation[i][j] = basis_eval[minIdx_[i]+j]; 
                }               
                num *= spline_evaluation[i][index_[i] - minIdx_[i]]; 
            }
            if(num == 0) return 0;
            // compute the sum that appears at the denominator of the formula
            den = multicontract<M>(weights_, spline_evaluation);
            return num/den;
        }

        private:
            class FirstDerivative: public MatrixFieldBase<M,FirstDerivative> {
            public:
            
                using Base = MatrixFieldBase<M,FirstDerivative>;
                static constexpr int StaticInputSize = M;
                static constexpr int NestAsRef = 0;   // avoid nesting as reference, .derive() generates temporaries
                static constexpr int XprBits = 0;
                static constexpr int degree = Dynamic;
                using Scalar = double;
                using InputType = Vector<Scalar, StaticInputSize>;

            private:
                std::array<std::shared_ptr<BSplineBasis>, M> spline_basis_;
                MdArray<double,full_dynamic_extent_t<M>> weights_;
                std::array<int,M> index_ ;
                std::array<int,M> degree_;

                std::array<std::size_t, M> minIdx_;
                double num0_ = 0.0;
                std::array<int, M> extents_;

                size_t i_ = 0; // index of the derivative, questa i è necessaria ? se mi interessa il gradiente in una direzione si

            public:
                FirstDerivative() = default;

                FirstDerivative(std::array<std::shared_ptr<BSplineBasis>, M> spline_basis, const MdArray<double,full_dynamic_extent_t<M>>& weights, const std::array<int,M>& index, std::size_t i): 
                     spline_basis_(spline_basis), index_(index), i_(i){

                    std::array<std::size_t, M> maxIdx;
                    for (std::size_t i = 0; i < M; ++i) {
                        degree_[i] = spline_basis[i]->degree();
                        minIdx_[i] = (index_[i] >= degree_[i])? (index_[i]-degree_[i]) : 0;
                        extents_[i] = (index_[i] + degree_[i] < weights.extent(i))? (index_[i]+degree_[i]+1-minIdx_[i]) : (weights.extent(i)-minIdx_[i]);
                        maxIdx[i] = (minIdx_[i] + extents_[i]-1);
                    }


                    weights_.resize(extents_);
                    weights_ = weights.block(minIdx_, maxIdx);

                    num0_ = weights(index_);  

                };

                // evalutes the first degree partial derivative of the NURBS at a given point, funziona
                constexpr Scalar operator()(const Eigen::Matrix<Scalar, StaticInputSize, 1>& p) const { 

                    double num = num0_;
                    std::array<std::vector<double>,M> spline_evaluation {};
                    double den;

                    double num_derived = 0.0;
                    double den_derived = 0.0;


                    for(std::size_t i=0;i<M;i++){
                        auto basis_eval = spline_basis_[i]->evaluate_basis(p(i));
                        spline_evaluation[i].resize(extents_[i]);
                        for(std::size_t j = 0; j<extents_[i]; j++ ){
                        spline_evaluation[i][j] = basis_eval[minIdx_[i]+j]; 
                    }
                        if (i!=i_)
                            num*=spline_evaluation[i][index_[i] - minIdx_[i]];
                    
                    }

                    auto der_eval = spline_basis_[i_]->evaluate_der_basis(p(i_),1)[1];

                    num_derived = num * der_eval[index_[i_]];

                    num*=spline_evaluation[i_][index_[i_] - minIdx_[i_]];

                    if (num== 0 && num_derived == 0) return 0;


                    // compute the sum that appears at the denominator of the formula
                    den = multicontract<M>(weights_, spline_evaluation);
                    
                    for (std::size_t j = 0; j<extents_[i_]; j++ ){
                        spline_evaluation[i_][j] = der_eval[ minIdx_[i_]+j];
                    }

                    den_derived = multicontract<M>(weights_, spline_evaluation);

                    //  ( N )'      N'D - ND'
                    //  (---)   =  ----------
                    //  ( D )         D^2
                    // where f' = df/dx_i
                    
                    return (num_derived*den - num*den_derived)/(den*den);
                };
            };

                // overload the call operator for the first derivative for a double
                //constexpr Scalar operator()(double p) const {
                //    return operator()(InputType{p});
                //}


            class SecondDerivative: public MatrixFieldBase<M,SecondDerivative> {
            public:
            
                using Base = MatrixFieldBase<M,SecondDerivative>;
                static constexpr int StaticInputSize = M;
                static constexpr int NestAsRef = 0;   // avoid nesting as reference, .derive() generates temporaries
                static constexpr int XprBits = 0;
                static constexpr int degree = Dynamic;
                using Scalar = double;
                using InputType = Eigen::Matrix<double,M,1> ; //Vector<Scalar, StaticInputSize>; // da capire

            private:
                std::array<std::shared_ptr<BSplineBasis>, M> spline_basis_;
                MdArray<double,full_dynamic_extent_t<M>> weights_;
                std::array<int,M> index_ ;
                std::array<int,M> degree_ ;

                std::array<std::size_t, M> minIdx_;
                double num0_ = 0.0;
                std::array<int, M> extents_;

                size_t i_ = 0; // first index of the 2nd derivative
                size_t j_ = 0; // second index of the 2nd derivative

            public:
                SecondDerivative() = default;

                SecondDerivative(std::array<std::shared_ptr<BSplineBasis>, M> spline_basis, const MdArray<double,full_dynamic_extent_t<M>>& weights, const std::array<int,M>& index, std::size_t i, std::size_t j): 
                     spline_basis_(spline_basis), index_(index), i_(i), j_(j){


                    std::array<std::size_t, M> maxIdx;
                    for (std::size_t i = 0; i < M; ++i) {
                        degree_[i] = spline_basis[i]->degree();
                        minIdx_[i] = (index_[i] >= degree_[i])? (index_[i]-degree_[i]) : 0;
                        extents_[i] = (index_[i] + degree_[i] < weights.extent(i))? (index_[i]+degree_[i]+1-minIdx_[i]) : (weights.extent(i)-minIdx_[i]);
                        maxIdx[i] = (minIdx_[i] + extents_[i]-1);
                    }

                    weights_.resize(extents_);
                    weights_ = weights.block(minIdx_, maxIdx);
                    num0_ = weights(index_);  

                };


                // evalutes the hessian(i,j) of the NURBS at a given point
                constexpr Scalar operator()(const InputType& p) const {

                    double num = num0_; // numerator of the NURBS formula
                    double num_der_i; // partial derivative of num w.r.t. i-th coordinate
                    double num_der_j; // partial derivative of num w.r.t. j-th coordinate
                    double num_der_ij; // mixed partial derivative of num
                    std::array<std::vector<double>,M> spline_evaluation {}; // pointwise evaluation of all splines along each coordinate
                    double den; // denominator of the NURBS formula
                    double den_der_i; // partial derivative of den w.r.t. i-th coordinate
                    double den_der_j; // partial derivative of den w.r.t. j-th coordinate
                    double den_der_ij; // mixed partial derivative of den

                    for(std::size_t i=0;i<M;i++){
                        // spline evaluation for i-th dimension
                        auto basis_eval = spline_basis_[i]->evaluate_basis(p[i]);
                        spline_evaluation[i].resize(extents_[i]);
                        
                        for(std::size_t j = 0; j<extents_[i]; j++ )
                            spline_evaluation[i][j] = basis_eval[minIdx_[i]+j]; // rivedi 
                        
                        if (i!=i_ && i!=j_)
                            num*=spline_evaluation[i][index_[i] - minIdx_[i]];
                    }

                    if (i_!=j_){
                        auto der_eval_i = spline_basis_[i_]->evaluate_der_basis(p[i_],1)[1];
                        auto der_eval_j = spline_basis_[j_]->evaluate_der_basis(p[j_],1)[1];

                        auto der_i = der_eval_i[index_[i_]];
                        auto der_j = der_eval_j[index_[j_]];

                        num_der_i = num * der_i * spline_evaluation[j_][index_[j_] - minIdx_[j_]];
                        num_der_j = num * der_j * spline_evaluation[i_][index_[i_] - minIdx_[i_]];
                        num_der_ij = num * der_i * der_j;

                        num*=spline_evaluation[i_][index_[i_] - minIdx_[i_]]*spline_evaluation[j_][index_[j_] - minIdx_[j_]];

                        if (num== 0 && num_der_i == 0 && num_der_j == 0 && num_der_ij == 0)
                            return 0;
                        
                        // denominator evaluation
                        den = multicontract<M>(weights_, spline_evaluation);

                        auto spline_eval_temp = spline_evaluation;

                        for (std::size_t j = 0; j<extents_[i_]; j++ ){
                            // extract the knots
                            spline_eval_temp[i_][j] = der_eval_i[ minIdx_[i_]+j];
                        }

                        // compute the derivative of the denominator w.r.t. i-th coordinate
                        den_der_i = multicontract<M>(weights_, spline_eval_temp);

                        spline_eval_temp = spline_evaluation;

                        for (std::size_t j = 0; j<extents_[j_]; j++ ){
                            // extract the knots
                            spline_eval_temp[j_][j] = der_eval_j[ minIdx_[j_]+j];
                        }
                        
                        // compute the derivative of the denominator w.r.t. j-th coordinate
                        den_der_j = multicontract<M>(weights_, spline_eval_temp);


                        for (std::size_t j = 0; j<extents_[i_]; j++ ){
                            // extract the knots
                            spline_eval_temp[i_][j] = der_eval_i[ minIdx_[i_]+j];
                        }

                        // compute the mixed partial derivative of the denominator
                        den_der_ij = multicontract<M>(weights_, spline_eval_temp);
                    }


                    else{
                        auto der_eval_i = spline_basis_[i_]->evaluate_der_basis(p[i_],1)[1];
                        auto der_eval_ij = spline_basis_[i_]->evaluate_der_basis(p[i_],2)[2];

                        num_der_i = num_der_j = num * der_eval_i[index_[i_]];
                        num_der_ij = num * der_eval_ij[index_[i_]];
                        

                        num*=spline_evaluation[i_][index_[i_] - minIdx_[i_]];
                        if (num== 0 && num_der_i == 0  && num_der_ij == 0)
                            return 0;

                        // denominator evaluation    
                        den = multicontract<M>(weights_, spline_evaluation);

                        

                        auto spline_eval_temp = spline_evaluation;

                        for (std::size_t j = 0; j<extents_[i_]; j++ ){
                            // extract the knots
                            spline_eval_temp[i_][j] = der_eval_i[ minIdx_[i_]+j];
                
                        }
                        // compute the derivative of the denominator w.r.t. i-th coordinate
                        den_der_i = den_der_j = multicontract<M>(weights_, spline_eval_temp);

                        

                        for (std::size_t j = 0; j<extents_[i_]; j++ ){
                            // extract the knots
                            spline_eval_temp[i_][j] = der_eval_ij[ minIdx_[i_]+j];
                        }


                        // compute the mixed (2nd derivative on i-th coordinate) partial derivative of the denominator
                        den_der_ij = multicontract<M>(weights_, spline_eval_temp);

                    }

                    //  ( N )'°     D(N'°D - N'D° - N°D' -ND'°) + 2D'D°N
                    //  (---)   =   ------------------------------------
                    //  ( D )                       D^3
                    // where f' = df/dx_i and f° = df/dx_j

                    return (den*(num_der_ij*den - num_der_i*den_der_j - num_der_j*den_der_i - num*den_der_ij) +
                             2*den_der_i*den_der_j*num)/(den*den*den);

                }

            };
            private:

                // gradient and hessian$
                VectorField<M, M, FirstDerivative> gradient_; //gradient
                MatrixField<M,M,M, SecondDerivative> hessian_; //hessian accedi con hessian_(i,j)

    public:
        /**
         * @brief Return the first derivative field in the specified direction.
         * @param i Direction index (default 0).
         * @return FirstDerivative object for direction @p i.
         */
        constexpr FirstDerivative derive(int i=0) const { return gradient_[i]; }
        /**
         * @brief Return the second derivative field in the specified directions.
         * @param i First direction index (default 0).
         * @param j Second direction index (default 0).
         * @return SecondDerivative object for directions @p i and @p j.
         */
        constexpr SecondDerivative deriveTwice(int i=0, int j=0) const { return hessian_(i,j); }

        /**
         * @brief Evaluate the gradient of the NURBS at the given point.
         * @param p Point at which to evaluate the gradient.
         * @return Gradient vector at point @p p.
         */
        Eigen::Matrix<double, M, 1> gradient(const Eigen::Matrix<double, M, 1>& p) const {
            Eigen::Matrix<double, M, 1> grad;
            for (int i = 0; i < M; ++i) {
                grad(i) = gradient_[i](p);
            }
            return grad;
        }

        /**
         * @brief Evaluate the Hessian matrix of the NURBS at the given point.
         * @param p Point at which to evaluate the Hessian.
         * @return Hessian matrix at point @p p.
         */
        Eigen::Matrix<double, M, M> hessian(const Eigen::Matrix<double, M, 1>& p) const {
            Eigen::Matrix<double, M, M> hess;
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < M; ++j) {
                    hess(i, j) = hessian_(i, j)(p);
                }
            }
            return hess;
        }

        constexpr std::array<int,M> degree() const { return degree_; }
        constexpr int size() const { return weights_.size(); }
        constexpr const MdArray<double,full_dynamic_extent_t<M>>& weights() const { return weights_; }
        constexpr const std::array<int,M>& index() const { return index_; }
        constexpr const std::array<std::shared_ptr<BSplineBasis>, M>& spline_basis() const { return spline_basis_; }

        /**
         * @brief Overload for 1D: evaluate the NURBS basis function at a scalar point.
         * @param p Scalar parameter value.
         * @return Value of the NURBS basis function at @p p.
         */
        constexpr Scalar operator()(double p) const { return operator()(std::vector<double>{p}); }
};


}// namespace fdapde

#endif


