function lambdap = parab3p ( lambdac, lambdam, ff0, ffc, ffm )

%*****************************************************************************80
%
%% parab3p() applies a three-point parabolic model for a line search.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    14 April 2021
%
%  Author:
%
%    Tim Kelley
%
%  Reference:
%
%    Tim Kelley,
%    Iterative Methods for Linear and Nonlinear Equations,
%    SIAM, 2004,
%    ISBN: 0898713528,
%    LC: QA297.8.K45.
%
%  Input:
%
%    real lambdac: the current steplength
%
%    real lambdam: the previous steplength
%
%    real ff0:  ||F(x_c)||^2
%
%    real ffc: ||F(x_c + lambdac d)||^2
%
%    real ffm: ||F(x_c + lambdam d)||^2
%
%  Output:
%
%    real LAMBDAP: the updated value for lambda.
%
%  Local:
%
%    real sigma0: a linesearch parameter.
% 
%    real sigma1: a linesearch parameter.
%
  sigma0 = 0.1;
  sigma1 = 0.5;
%
%  Compute C1, C2, the coefficients of the interpolation polynomial:
%
%  p(lambda) = ff0 + (c1 lambda + c2 lambda^2)/d1
%
%  d1 = (lambdac - lambdam)*lambdac*lambdam < 0
%
%  if 0 < c2, we have negative curvature and default to
%    lambdap = sigam1 * lambda
%
  c2 = lambdam * ( ffc - ff0 ) - lambdac * ( ffm - ff0 );

  if ( 0.0 <= c2 )
    lambdap = sigma1 * lambdac;
    return
  end

  c1 = lambdac * lambdac * ( ffm - ff0 ) - lambdam * lambdam * ( ffc - ff0 );
  lambdap = - 0.5 * c1 / c2;

  lambdap = max ( lambdap, sigma0 * lambdac );
  lambdap = min ( lambdap, sigma1 * lambdac );

  return
end

