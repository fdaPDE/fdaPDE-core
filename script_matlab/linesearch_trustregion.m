clear all; close all; clc;
% Define the function
f = @(x, y) (x.^2 + y.^2 - 2).^2 + (exp(x-1) + y.^3 - 2).^2;

% Create a grid of values for x and y
plot_region_x = 3;
plot_region_y = 3;
x = linspace(-plot_region_x, plot_region_x, 100);
y = linspace(-1.5, plot_region_y, 100);
[X, Y] = meshgrid(x, y);

% Evaluate the function on the grid
Z = f(X, Y);

contourLevels = linspace(0, 20, 20);  % Adjust the range and number of levels as needed

% Plot the level curves
contour(X, Y, Z, contourLevels) %, 'ShowText', 'on');

% Add a colorbar to show the function values
colorbar;
%%
syms a b
% define function
s = (a.^2 + b.^2 - 2).^2 + (exp(a-1) + b.^3 - 2).^2;
% compute gradient
grad1 = diff(s, a); grad2 = diff(s, b);
% compute hessian
h11 = diff(grad1, a); h12 = diff(grad1, b); h21 = diff(grad2, a); h22 = diff(grad2, b);
% select point
px = 0.5; py = 0.5; % choose the point for the quadratic model
% compute H(p)
H = zeros(2); H(1,1) = double(subs(subs(h11, a, px), b, py)); H(1,2) = double(subs(subs(h12, a, px), b, py)); H(2,1) = double(subs(subs(h21, a, px), b, py)); H(2,2) = double(subs(subs(h22, a, px), b, py));
% compute f(p)
fk = double(subs(subs(s,a,px),b,py));
% compute grad(p)
gradf = [double(subs(subs(grad1,a,px),b,py)), double(subs(subs(grad2,a,px),b,py))];
% turn quadratic model in p
m = fk + gradf*[a-px;b-py] - 0.5*[a-px, b-py]*H*[a-px;b-py];
m_func = matlabFunction(m);

%% plot quadratic model and trust region
hold on;

% plot quadratic model
% Evaluate the function on the grid
Z = m_func(X, Y);
contourLevels = linspace(0, fk, 5)
contour(X, Y, Z, contourLevels, 'red')
scatter(px, py, 'filled')
scatter(1,1,'filled', 'k')

% plot the vertex of the parabula
x_vertex = H\gradf';
scatter(x_vertex(1)+px, x_vertex(2)+py, 'filled')

% plot the newton direction
plot([px, x_vertex(1)+px], [py, x_vertex(2)+py], 'k-')

% plot trustregion
r = 0.5;
half_circle = @(x) sqrt(r.^2-(x-px).^2);
x_trust = linspace(px-r, px+r, 100);
y_trust = half_circle(x_trust);
x_trust = [x_trust, flip(x_trust)];
y_trust = [y_trust+py, flip(-y_trust+py)];
plot(x_trust, y_trust, 'b-')

% plot dogleg direction
x_dog = 0.87; % a caso
y_dog = half_circle(x_dog)+py;
scatter(x_dog, y_dog, 'm', 'filled')
plot([px, x_dog], [py, y_dog], 'm--')

legend('Nonlinear function', 'quadratic model', 'x_C', 'x_{ex}', 'x_N', 's^N','Trust-Region', 'x_{D}', 's^D','Location','northwest')
%% plot 3D 
% figure
% plot_region_x = 3;
% plot_region_y = 2.5;
% x = linspace(0.5, 1.5, 100);
% y = linspace(0.5, 1.5, 100);
% [X, Y] = meshgrid(x, y);
% Z = f(X, Y);
% % Plot the level curves
% surf(X, Y, Z, 'EdgeColor','none')
% hold on
% Z = m_func(X, Y);
% surf(X, Y, Z, 'EdgeColor','none')
% scatter3(px, py, fk, 'filled')