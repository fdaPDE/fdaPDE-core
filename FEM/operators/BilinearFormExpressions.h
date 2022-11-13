#ifndef __BILINEAR_FORM_EXPRESSIONS_H__
#define __BILINEAR_FORM_EXPRESSIONS_H__

#include "../../MESH/Element.h"
using fdaPDE::core::MESH::Element;
#include "../../utils/Symbols.h"
#include <tuple>

namespace fdaPDE{
namespace core{
namespace FEM{

  // struct defined to activate some standard behaviour of operators
  struct DefaultOperator {};
  
#define DEF_BILINEAR_FORM_EXPR_OPERATOR(OPERATOR, FUNCTOR)		\
  template <typename E1, typename E2>					\
  BilinearFormBinOp<E1, E2, FUNCTOR >					\
  OPERATOR(const BilinearFormExpr<E1>& op1,				\
	   const BilinearFormExpr<E2>& op2) {				\
    return BilinearFormBinOp<E1, E2, FUNCTOR >				\
      {op1.get(), op2.get(), FUNCTOR()};				\
  }

  template <typename E> struct BilinearFormExpr{
    // call integration method on base type
    template <typename... Args>
    auto integrate(const std::tuple<Args...>& mem_buffer) const {
      return static_cast<const E&>(*this).integrate(mem_buffer);
    }
    // get underyling type composing the expression node
    const E& get() const { return static_cast<const E&>(*this); }
    // returns the set of types associated with this expression
    auto getTypeList() const {
      return static_cast<const E&>(*this).getTypeList();
    }
    // query the bilinear form to check if it has some non-constant coefficients
    static constexpr bool is_space_varying = E::is_space_varying;
  };

  // a generic binary operation node
  template <typename OP1, typename OP2, typename BinaryOperation>
  class BilinearFormBinOp : public BilinearFormExpr<BilinearFormBinOp<OP1, OP2, BinaryOperation>> {
  private:
    typename std::remove_reference<OP1>::type op1_;   // first  operand
    typename std::remove_reference<OP2>::type op2_;   // second operand
    BinaryOperation f_;                               // operation to apply
  public:
    // constructor
    BilinearFormBinOp() = default; // let default constructible to allow assignment
    BilinearFormBinOp(const OP1& op1, const OP2& op2, BinaryOperation f)
      : op1_(op1), op2_(op2), f_(f) { };
    // integrate method. Apply the functor f_ to the result of integrate() applied to both operands.
    template <typename... Args>
    auto integrate(const std::tuple<Args...>& mem_buffer) const {
      return f_(op1_.integrate(mem_buffer), op2_.integrate(mem_buffer));
    }
    auto getTypeList() const { return std::tuple_cat(op1_.getTypeList(), op2_.getTypeList()); }
    static constexpr bool is_space_varying = OP1::is_space_varying || OP2::is_space_varying;
  };
  DEF_BILINEAR_FORM_EXPR_OPERATOR(operator+, std::plus<>)
  DEF_BILINEAR_FORM_EXPR_OPERATOR(operator-, std::minus<>)
  
  // node representing a single scalar in an expression
  class BilinearFormScalar : public BilinearFormExpr<BilinearFormScalar> {
  private:
    double value_;
  public:
    // constructor
    BilinearFormScalar(double value) : value_(value) { }
    // integrate method. Just return the stored value
    template <typename... Args>
    auto integrate(const std::tuple<Args...>& mem_buffer) const {
      return value_;
    }
    std::tuple<BilinearFormScalar> getTypeList() const { return std::make_tuple(*this); }
    static constexpr bool is_space_varying = false; // constants does not affect the space-variability properties of the PDE
  };
  // allow scalar*operator expressions
  template <typename E>                                                 
  BilinearFormBinOp<BilinearFormScalar, E, std::multiplies<> >
  operator*(double op1, const BilinearFormExpr<E> &op2) {
    return BilinearFormBinOp<BilinearFormScalar, E, std::multiplies<> >
      (BilinearFormScalar(op1), op2.get(), std::multiplies<>());
  }  

  // utility macro to import symbols from memory buffer passed to operators
#define IMPORT_MEM_BUFFER_SYMBOLS(mem_buff)	    \
  /* pair of basis functions \psi_i, \psi_j*/	    \
  auto psi_i = std::get<0>(mem_buff);		    \
  auto psi_j = std::get<1>(mem_buff);		    \
  /* gradient of \psi_i, \psi_j */		    \
  auto NablaPsi_i = std::get<2>(mem_buff);	    \
  auto NablaPsi_j = std::get<3>(mem_buff);	    \
  /* affine map to reference element */		    \
  auto invJ = std::get<4>(mem_buff);		    \
  
}}}
#endif // __BILINEAR_FORM_EXPRESSIONS_H__
