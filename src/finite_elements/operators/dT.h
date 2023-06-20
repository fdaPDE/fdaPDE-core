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
  template <typename T = DefaultOperator> struct dT : public BilinearFormExpr<dT<T>> {
    // constructor
    dT() = default;
    std::tuple<dT<T>> getTypeList() const { return std::make_tuple(*this); }
    static constexpr bool is_space_varying = false; // dT() does not affect the space-variability properties of the PDE

    // return zero field
    template <typename... Args>
    auto integrate(const std::tuple<Args...>& mem_buffer) const {
      IMPORT_MEM_BUFFER_SYMBOLS(mem_buffer);
      // recover dimensionality of weak formulation from \psi_i
      return ScalarField<decltype(psi_i)::PtrType::domain>::Zero();
    }
  };

}}}
#endif // __DT_H__
