function [ sol, it_hist, ierr ] = broyden_armijo ( x, f, tol, parms )

%*****************************************************************************80
%
%% broyden_armijo() applies a globally convergent version of Broyden's method.
%
%  Discussion:
%
%    Armijo's rule is used.
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
%    real X(N): the initial solution estimate.
%
%    function fx = F ( x ): MATLAB code that evaluates the function.
%
%    real TOL(2): [ atol, rtol ] error tolerances.
%    atol: absolute error tolerance.
%    rtol: relative error tolerance.
%
%    integer PARMS(2): [ MAXIT, MAXDIM ]
%    integer MAXIT: maximum number of nonlinear iterations
%    integer MAXDIM: maximum number of Broyden iterations before restart.,
%
%  Output:
%
%    real SOL(N): the estimated solution.
%
%    integer IT_HIST(MAXIT,3): [ FNRM, FNUM, IARM ] for each iteration.
%    FNRM: scaled l2 residual norm;
%    FNUM: number of function evaluations,
%    IARM: number of steplength reductions.
%
%    integer IERR: termination flag.
%    0: success
%    1: the termination criterion was not satsified after MAXIT iterations.
%    2: the line search failed. 
%
%  Local:
%
%    logical DEBUG: if TRUE, prints iteration statistics.
%
%    real ALPHA: 0.0001, parameter to measure sufficient decrease.
%
%    integer MAXARM: maximum number of steplength reductions allowed.
%
  debug = false;
%
%  Initialize.
%
  ierr = 0;
  maxit = 80;
  maxdim = 80;  
  it_histx = zeros ( maxit, 3 );
  maxarm = 7;

  if ( nargin == 4 )
    maxit = parms(1);
    maxdim = parms(2) - 1; 
  end
%
% file where we save the steps
  fileID = fopen("BroydenArmijo_matlab_(" + x(1) + "0000," + x(2) + "0000).txt",'w');
%
  rtol = 0;
  atol = tol(1);
  n = length ( x );
  fnrm = 1.0;
  itc = 0;
  nbroy = 0;
%
%  Evaluate f at the initial iterate.
%
  f0 = feval ( f, x );
  fc = f0;
  fnrm = norm ( f0 ) / sqrt ( n );
  it_hist(itc+1) = fnrm;
  it_histx(itc+1,1) = fnrm;
  it_histx(itc+1,2) = 0; 
  it_histx(itc+1,3) = 0;
  fnrmo = 1.0;
  outstat(itc+1,:) = [ itc, fnrm, 0, 0 ];
%
%  Compute the stopping tolerance.
%
  stop_tol = atol + rtol * fnrm;
%
%  Terminate on entry?
%
  if ( fnrm < stop_tol )
    sol = x;
    return
  end
%
%  Initialize the iteration history storage matrices
%
  stp = zeros(n,maxdim);
  stp_nrm = zeros(maxdim,1);
  lam_rec = ones(maxdim,1);
%
%  Set the initial step to -F, compute the step norm
%
  lambda = 1.0;
  stp(:,1) = - fc;
  stp_nrm(1) = stp(:,1)' * stp(:,1);
%
%  Iteration.
%
  while ( itc < maxit )

    nbroy = nbroy + 1;
    fnrmo = fnrm;
    itc = itc + 1;
%
%  Compute the new point, test for termination before
%  adding to iteration history
%
    xold = x;
    lambda = 1.0;
    iarm = 0;
    lrat = 0.5;
    alpha = 1.0e-4;
    x = x + stp(:,nbroy);
    fprintf(fileID,'%.15f\n%.15f\n',x(1),x(2));
    fc = feval ( f, x );
    fnrm = norm ( fc ) / sqrt ( n );
    ff0 = fnrmo * fnrmo;
    ffc = fnrm * fnrm;
    lamc = lambda;
%
%  Line search: assume that the Broyden direction is an
%  inexact Newton direction. If the line search fails to
%  find sufficient decrease after maxarm steplength reductions,
%  return with failure. 
%
%  Three-point parabolic line search.
%
    while ( ( 1.0 - lambda * alpha ) * fnrmo <= fnrm && iarm < maxarm )

      if ( iarm == 0 )
        lambda = lambda * lrat;
      else
        lambda = parab3p ( lamc, lamm, ff0, ffc, ffm );
      end

      lamm = lamc;
      ffm = ffc;
      lamc = lambda;
      x = xold + lambda * stp(:,nbroy);
      fc = feval ( f, x );
      fnrm = norm ( fc ) / sqrt ( n );
      ffc = fnrm * fnrm;
      iarm = iarm + 1;

    end
%
%  Set error flag and return on failure of the line search
%
    if ( iarm == maxarm )
      disp('broyden_armijo: Line search failure.')
    end
%
%  How many function evaluations did this iteration require?
%
    it_histx(itc+1,1) = fnrm;
    it_histx(itc+1,2) = it_histx(itc,2) + iarm + 1;
    if ( itc == 1 )
      it_histx(itc+1,2) = it_histx(itc+1,2) + 1;
    end
    it_histx(itc+1,3) = iarm;
%
%  Terminate?
%
    if ( fnrm < stop_tol )
      sol = x;
      ratio = fnrm / fnrmo;
      outstat(itc+1, :) = [ itc, fnrm, iarm, ratio ];
      it_hist = it_histx(1:itc+1,:);
      if ( debug )
        disp(outstat(itc+1,:))
      end
      return
    end
%
%  Modify the step and step norm if needed to reflect the line search.
%
    lam_rec(nbroy) = lambda;
    if ( lambda ~= 1.0 )
      stp(:,nbroy) = lambda * stp(:,nbroy);
      stp_nrm(nbroy) = lambda * lambda * stp_nrm(nbroy);
    end

    ratio = fnrm / fnrmo;
    outstat(itc+1,:) = [ itc, fnrm, iarm, ratio ];
    if ( debug )
      disp(outstat(itc+1,:))
    end
%
%  If there's room, compute the next search direction and step norm and
%  add to the iteration history 
%
    if ( nbroy < maxdim + 1 )

      z = - fc;

      if ( 1 < nbroy )
        for kbr = 1 : nbroy - 1
          ztmp = stp(:,kbr+1) / lam_rec(kbr+1);
          ztmp = ztmp + ( 1.0 - 1.0 / lam_rec(kbr) ) * stp(:,kbr);
          ztmp = ztmp * lam_rec(kbr);
          z = z + ztmp * ( ( stp(:,kbr)' * z ) / stp_nrm(kbr));
        end
      end
%
%  Store the new search direction and its norm.
%
      a2 = - lam_rec(nbroy) / stp_nrm(nbroy);
      a1 = 1.0 - lam_rec(nbroy);
      zz = stp(:,nbroy)' * z;
      a3 = a1 * zz / stp_nrm(nbroy);
      a4 = 1.0 + a2 * zz;
      stp(:,nbroy+1) = ( z - a3 * stp(:,nbroy) ) / a4;
      stp_nrm(nbroy+1) = stp(:,nbroy+1)' * stp(:,nbroy+1);

    else
%
%  Out of room, time to restart
%
      stp(:,1) = - fc;
      stp_nrm(1) = stp(:,1)' * stp(:,1);
      nbroy = 0;

    end

  end
%
%  We've taken the maximum number of iterations and not terminated.
%
  fclose(fileID);
  sol = x;
  it_hist = it_histx(1:itc+1,:);
  ierr = 1;
  if ( debug )
    disp(outstat)
  end

  return
end
