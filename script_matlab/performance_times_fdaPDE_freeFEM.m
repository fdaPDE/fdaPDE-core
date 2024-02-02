%% performance times fdaPDE - freeFEM
clear all; close all; clc;

% data
x1 = {'P1\_16', 'P1\_32', 'P1\_64'};
x2 = {'P2\_16', 'P2\_32', 'P2\_64'};

p1_freeFEM_fixed = [0.142, 0.147, 0.313];
p2_freeFEM_fixed = [0.148, 0.221, 1.462];

p1_fdaPDE_fixed = [0.029, 0.055, 0.181];
p2_fdaPDE_fixed = [0.185, 0.71, 3.07];

p1_freeFEM_newt = [0.14, 0.147, 0.223];
p2_freeFEM_newt = [0.141, 0.217, 1.006];

p1_fdaPDE_newt = [0.27, 0.56, 0.205];
p2_fdaPDE_newt = [0.194, 0.843, 3.441];


%% fixedpoint
figure;

subplot(2,2,1)
bar([p1_freeFEM_fixed', p1_fdaPDE_fixed']);
xlabel('Mesh Size');
ylabel('Time (s)');
title('fixedpoint P1');
xticks(1:length(x1));
xticklabels(x1);
legend('freeFEM', 'femR', 'Location','best');
grid on;
hold off;

subplot(2,2,2)
bar([p2_freeFEM_fixed', p2_fdaPDE_fixed']);
xlabel('Mesh Size');
ylabel('Time (s)');
title('fixedpoint P2');
xticks(1:length(x2));
xticklabels(x2);
legend('freeFEM', 'femR', 'Location','best');
grid on;
hold off;

%% newton with three iterations

subplot(2,2,3)
bar([p1_freeFEM_newt', p1_fdaPDE_newt']);
xlabel('Mesh Size');
ylabel('Time (s)');
title('newton P1');
xticks(1:length(x1));
xticklabels(x1);
legend('freeFEM', 'femR', 'Location','best');
grid on;
hold off;

subplot(2,2,4)
bar([p2_freeFEM_newt', p2_fdaPDE_newt']);
xlabel('Mesh Size');
ylabel('Time (s)');
title('newton P2');
xticks(1:length(x2));
xticklabels(x2);
legend('freeFEM', 'femR', 'Location','best');
grid on;
hold off;