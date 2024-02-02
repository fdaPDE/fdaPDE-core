%f1 = @(x,y) atan(x*y);
%f2 = @(x,y) x-5*y;
%f = @(x,y) [f1; f2];

f = @(x) [atan(x(1)*x(2)); x(1)-5*x(2)];

tol = [1e-6, 1e-6];
params = [80, 80, 1];

% Initial data
n = 10;
x0 = [0.01; 0.01];
S = ones(2, n);
for i = 0:1
    for j = 0:n-1
        S(i+1,j+1) = x0(i+1) + 0.1*i + 0.1*j;
    end
end

for j = 1:n
    brsol(S(:,j), f, tol, params)
    %broyden_armijo(S(:,j), f, tol, params)
end