close all
clear all
clc
addpath('../test/build/')
% plot exact solution VS our solution
% unit_square_32
format LONG
n = 32;
[X,Y] = meshgrid(0:(1./n):1);
exact_sol = 3*X.*X + 2*Y.*Y;
%2D plot with colors
surf(X,Y,exact_sol, 'EdgeColor','none')

view(2)
cb=colorbar;
cb.Title.String = "Z";
colormap jet

%3D plot
figure()
surf(X,Y,exact_sol)

% our solution obtained with pde dependent Newton
fileID = fopen("solution_nonlinear_P1.txt", 'r');
our_sol = zeros(33,33);
for i=1:n+1
    for j=1:n+1
        au = fgetl(fileID);
        our_sol(i,j) = sscanf(au, '%f');
    end
end
fclose(fileID);
figure()
%2D plot with colors
surf(X,Y, our_sol, 'EdgeColor','none')
view(2)
cb=colorbar;
cb.Title.String = "Z";
colormap jet
%3D plot
figure()
surf(X,Y,our_sol)

%plot the error
figure()
error = exact_sol - our_sol;
%2D plot with colors
surf(X,Y, error, 'EdgeColor','none')
view(2)
cb=colorbar;
cb.Title.String = "Z";
colormap jet
caxis([0, 1]);

%3D plot
figure()
surf(X,Y,abs(error))
caxis([0,1]);