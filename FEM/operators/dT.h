#ifndef __DT_H__
#define __DT_H__

#include "BilinearFormExpressions.h"
using fdaPDE::core::FEM::BilinearFormExpr;

namespace fdaPDE{
namespace core{
namespace FEM{

  // class representing the time derivative operator.
  // This operator just acts as a tag in a bilinear form expression to activate the time loop in the assembler phase. See
  // is_parabolic<> trait in BilinearFormTraits.h
  template <unsigned int L = 0> struct dT : public BilinearFormExpr<dT<L>> {
    // constructor
    dT() = default;
    std::tuple<dT<L>> getTypeList() const { return std::make_tuple(*this); }

    template <unsigned int N, int M, unsigned int ORDER, typename B>
    double integrate(const B& basis, const Element<ORDER, N>& e, int i , int j, const SVector<M>& quadrature_point) const{
      return 0; // dirty hack to make dT not contribute in the space discretization of the problem
    }
  };

}}}
#endif // __DT_H__
