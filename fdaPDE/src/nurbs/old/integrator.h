#ifndef __INTEGRATOR_H__
#define __INTEGRATOR_H__

#include "iso_mesh.h"
#include "iso_square.h"
#include "integration_tables.h"

namespace fdapde {

// A set of utilities to perform numerical integration
// M: dimension of the domain of integration, K number of quadrature nodes
template<int M, int K> class IsoIntegrator {
    private:
        const IntegratorTableLGL<M, K> integration_table_;

    public:
        IsoIntegrator() : integration_table_() {}

        // integrate a function f over an iso_square (we integrate in the parametric domain)
        template<int N, typename F> double integrate(const IsoSquare<M,N>& square,const F& f) const {
            double integral = 0.0;
            for(std::size_t iq = 0; iq < integration_table_.order; ++iq){
                std::array<double, M> p;
                for(std::size_t i = 0; i < M; ++i){
                    p[i] = integration_table_.nodes(iq,i);
                }
                auto p_param = square.affine_map(p);
                integral += integration_table_.weights(iq,0) * f(p_param);
            }
            // multiply by the jacobian of the affine map

            return integral*square.parametric_measure();
        }

        template <int N, typename F> double integrate(const IsoMesh<M,N>& mesh, const F& f) const {
            double integral = 0.0;
            for(auto it = mesh.beginCells(); it != mesh.endCells(); ++it){
                std::cout<<"Integrating cell "<<(*it).ID()<<std::endl;
                integral += integrate(*it, f);
            }

            return integral;
        }

        // integrate a function f over an element of the physical domain
        // perform integration of \int_e [f] using a basis system defined over the reference element and the change of
        // variables formula: \int_e [f(x)] = \int_{E} [f(J(X)) g(X) ] |detJ|
        // where J is the affine mapping from the reference element E to the physical element e
        // da capire se mettere la phi oppure passare tutto dentro f
        template<int N, typename F> double integrate_physical(const IsoSquare<M,N>& square,const F& f) const {
            double integral = 0.0;
            for(std::size_t iq = 0; iq < integration_table_.order; ++iq){
                std::array<double, M> p;
                for(std::size_t i = 0; i < M; ++i){
                    p[i] = integration_table_.nodes(iq,i);
                }

                std::array<double,N> Jx = square.parametrization(p); // Jx is now on the physical domain
                double g = square.metric_determinant(p);
                integral += integration_table_.weights(iq,0) * f(Jx) * g;
            }
            // multiply by the jacobian of the affine map

            return integral*square.parametric_measure();
        }


};

} // namespace fdapde

#endif // __INTEGRATOR_H__