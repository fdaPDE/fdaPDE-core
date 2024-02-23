clear all; close all; clc; format long;

% read error fixed point 
maxit_fp = 4; % indicate here the number of iterations fixedpoint run
fileID = fopen("convergence_test_fixedpoint.txt", 'r');
error_fp = zeros(maxit_fp-1,1);
for i=1:maxit_fp
    au = fgetl(fileID);
    error_fp(i) = sscanf(au, '%f');
end
fclose(fileID);

% read error newton
maxit_nw = 3; % indicate here the number of iterations fixedpoint run
fileID = fopen("convergence_test_newtonPDE_freefem.txt", 'r');
error_nw = zeros(maxit_nw-1,1);
for i=1:maxit_nw
    au = fgetl(fileID);
    error_nw(i) = sscanf(au, '%f');
end
fclose(fileID);

xx_fp = linspace(1, maxit_fp, maxit_fp)';
xx_nw = linspace(1, maxit_nw, maxit_nw)';

%% compute errors fixedpoint

e2 = error_fp(3:end);   % e_{k+1}
e1 = error_fp(2:end-1); % e_k
e0 = error_fp(1:end-2); % e_{k-1}

% compute order of convergence (convergence order)
q_fp = log(error_fp(3:end)./error_fp(2:end-1))./log(error_fp(2:end-1)./error_fp(1:end-2))
q_fp = mean(q_fp)

% compute the real lambda (convergence rate)
lambda_fp = error_fp(2:end)./(error_fp(1:end-1).^q_fp);

% compute linear reference
q = 1;
linear_ref_fp = 1;
for i = 2:length(xx_fp)
    linear_ref_fp = [linear_ref_fp, lambda_fp(i-1)*linear_ref_fp(i-1)^q];
end

% compute quadratic reference
q = 1.5;
quadratic_ref_fp = 1;
for i = 2:length(xx_fp)
    quadratic_ref_fp = [quadratic_ref_fp, lambda_fp(i-1)*quadratic_ref_fp(i-1)^q];
end

%% compute errors newton

e2 = error_nw(3:end);   % e_{k+1}
e1 = error_nw(2:end-1); % e_k
e0 = error_nw(1:end-2); % e_{k-1}

% compute order of convergence (convergence order)
q_nw = log(error_nw(3:end)./error_nw(2:end-1))./log(error_nw(2:end-1)./error_nw(1:end-2))
q_nw = mean(q_nw)

% compute the real lambda (convergence rate)
lambda_nw = error_nw(2:end)./(error_nw	(1:end-1).^q_nw	);

% compute linear reference
q = 1;
linear_ref_nw = 1;
for i = 2:length(xx_nw)
    linear_ref_nw = [linear_ref_nw, lambda_nw(i-1)*linear_ref_nw(i-1)^q];
end

% compute quadratic reference
q = 2;
quadratic_ref_nw = 0.5;
for i = 2:length(xx_nw)
    quadratic_ref_nw	 = [quadratic_ref_nw, lambda_nw(i-1)*quadratic_ref_nw(i-1)^q];
end

%% plots

%fixedpoint
figure
semilogy(xx_fp, error_fp, 'o-')
hold on
plot(xx_fp, linear_ref_fp, '--', 'Color',[.5,.5,.5])
plot(xx_fp, quadratic_ref_fp, '-.', 'Color',[.5,.5,.5])

legend('fixedpoint error', 'linear decrease', 'qudratic decrease'); %, 'quadratic');
title('fixedpoint convergence')
grid on
xticks(unique(round(xticks)));
xticklabels(cellstr(num2str(round(xticks'))));
xlabel('iterations')
ylabel('error')


% plot newton
figure
semilogy(xx_nw, error_nw, 'o-')
hold on
plot(xx_nw, linear_ref_nw(1:maxit_nw), '--', 'Color',[.5,.5,.5])
plot(xx_nw, quadratic_ref_nw(1:maxit_nw), '-.', 'Color',[.5,.5,.5])
legend('newton error', 'linear decrease', 'quadratic decrease', 'Location','southwest');
title('newton convergence')
grid on
xticks(unique(round(xticks)));
xticklabels(cellstr(num2str(round(xticks'))));
xlabel('iterations')
ylabel('error')

% subplots
figure
% plot fixedpoint
subplot(1,3,1)
semilogy(xx_fp, error_fp, 'o-')
hold on
plot(xx_fp, linear_ref_fp, 'r-')
% plot(xx_fp, quadratic_ref_fp, 'b-')
xlabel('iterations')
ylabel('error')
legend('fixedpoint error', 'linear decrease'); %, 'quadratic');
title('fixedpoint convergence')
grid on

% plot newton
subplot(1,3,2)
semilogy(xx_nw, error_nw, 'o-')
hold on
plot(xx_nw, linear_ref_fp(1:maxit_nw), 'r-')
plot(xx_nw, quadratic_ref_fp(1:maxit_nw), 'b-')
xlabel('iterations')
ylabel('error')
legend('fixedpoint error', 'linear decrease', 'quadratic');
title('fixedpoint convergence')
grid on

% plot all
subplot(1,3,3)
semilogy(xx_fp, error_fp, 'o-')
hold on
semilogy(xx_nw, error_nw, 'o-')
plot(xx_fp, linear_ref_fp, 'r-')
% plot(xx_nw, linear_ref_nw, 'r-')
plot(xx_nw, quadratic_ref_fp(1:maxit_nw), 'b-')
xlabel('iterations')
ylabel('error')
legend('fixedpoint error', 'newton error', 'linear f', 'quadratic');
title('fixedpoint convergence')
grid on