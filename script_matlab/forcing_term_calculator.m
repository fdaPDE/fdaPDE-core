clear all; close all; clc;

% define parameters
syms x y
syms mu b1 b2 alpha
b = [b1; b2];

% exact solution
% u = 3*sin(x) + 2*y;  % choose
u = 3*x*x + 2*y*y;

% operators
gradu = [diff(u,x); diff(u,y)];
deltau = diff(gradu(1),x) + diff(gradu(2),y);
nonlinear = (1-u)*u;

% define equation
equation = -mu*deltau + alpha*nonlinear;

% compute the forcing term 
f = simplify(expand(equation))

f = subs(f, alpha,1);
f = subs(f,mu,1)