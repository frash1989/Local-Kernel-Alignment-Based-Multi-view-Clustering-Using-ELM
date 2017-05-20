% Iterative weighted extreme learning machine for discriminative clustering
% Ref: Discriminative clustering via extreme learning machine, Neural
% Networks
clc
clear all;

addpath('functions');

load ecoli
label = ecoli(:,end);
x = ecoli(:,1:end-1);
clear ecoli


% normalization
x = transpose(mapminmax(x',-1,1));

% paramter setting
para.p= 2^(-1);
para.C= 10^(-2);
para.num_hidden_neurons = 2000;
num_class = length(unique(label));

% clustering
for trial = 1:50
    y_init = kmeans(x,num_class);
    [~, ite(trial), acc_record{trial}, st{trial}, acc(trial)] = iterWELM(x, y_init,label, para);
end

% display results
acc_mean = mean(acc)
acc_std = std(acc)
acc_max = max(acc)

