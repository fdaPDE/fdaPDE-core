%% performance times fdaPDE - freeFEM
clear all; close all; clc;

% data
x1 = {'h=0.0625 | P1\_16', 'h=0.3125 | P1\_32', 'h=0.15625 | P1\_64'};

desolve = [0.7556348, 0.8060179, 0.764384];
femR = [0.01479602, 0.05429697, 0.2281511];

nodes = [256, 1024, 4096]
dofs = [289, 1089, 4225]

% plots
figure;
subplot(1, 2, 1)
b = bar([desolve', femR']);
b(1).FaceColor = [0.2196, 0.4902, 0.9216];%[0.5098, 0.5686, 0.6824];
b(2).FaceColor = [0.7137    0.3725    0.8118]; %[0.6078, 0.5255, 0.6314];
% colormap jet;
% b(1).CData = 1:size(desolve', 1);
% b(2).CData = 1:size(femR', 1);

xlabel('Step | Mesh Size');
ylabel('Time (s)');
title('Time comparison deSolve / femR');
xticks(1:length(x1));
xticklabels(x1);
legend('deSolve', 'femR', 'Location','northeast');
grid on;
hold off;

subplot(1, 2, 2)
b = bar([nodes', dofs']);
b(1).FaceColor = [0.2196, 0.2196, 0.2196];
b(2).FaceColor = [0.7137    0.7137    0.7137];

xlabel('Step | Mesh Size');
ylabel('Time (s)');
title('Time comparison deSolve / femR');
xticks(1:length(x1));
xticklabels(x1);
legend('deSolve nodes', 'femR dofs', 'Location','northeast');
grid on;
hold off;