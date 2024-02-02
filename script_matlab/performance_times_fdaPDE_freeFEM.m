%% performance times fdaPDE - freeFEM
clear all; close all; clc;

% data
x1 = {'P1\_16', 'P1\_32', 'P1\_64'};
x2 = {'P2\_16', 'P2\_32', 'P2\_64'};

p1_freeFEM_fixed = [0.3, 1.014, 4.278];
p2_freeFEM_fixed = [1.14, 3.672, 24.084];

p1_fdaPDE_fixed = [0.0540000, 0.1260000, 0.4850000];
p2_fdaPDE_fixed = [0.5380000, 2.3510000, 10.4470000];

p1_freeFEM_newt = [0.1680000, 0.5520000, 2.3520000];
p2_freeFEM_newt = [0.4320000, 2.1000000, 12.1620000];

p1_fdaPDE_newt = [0.0590000, 0.0950000, 0.3440000];
p2_fdaPDE_newt = [0.3390000, 1.3900000, 5.6450000];


%% fixedpoint
figure;

subplot(2,2,1)
bar([p1_freeFEM_fixed', p1_fdaPDE_fixed']);
xlabel('Mesh Size');
ylabel('Solver Time per Processor');
title('fixedpoint P1');
xticks(1:length(x1));
xticklabels(x1);
legend('freeFEM', 'femR', 'Location','best');
grid on;
hold off;

subplot(2,2,2)
bar([p2_freeFEM_fixed', p2_fdaPDE_fixed']);
xlabel('Mesh Size');
ylabel('Solver Time per Processor');
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
ylabel('Solver Time per Processor');
title('newton P1');
xticks(1:length(x1));
xticklabels(x1);
legend('freeFEM', 'femR', 'Location','best');
grid on;
hold off;

subplot(2,2,4)
bar([p2_freeFEM_newt', p2_fdaPDE_newt']);
xlabel('Mesh Size');
ylabel('Solver Time per Processor');
title('newton P2');
xticks(1:length(x2));
xticklabels(x2);
legend('freeFEM', 'femR', 'Location','best');
grid on;
hold off;