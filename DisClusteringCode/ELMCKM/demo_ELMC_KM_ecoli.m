% Extreme learning machine for clustering based on kernel k-means
% Ref: Discriminative clustering via extreme learning machine, Neural
% Networks

format compact;
clear;
close all;

addpath(genpath('functions'))


load ecoli
y = ecoli(:,end);
X = ecoli(:,1:end-1);
clear ecoli


% normalization
X = transpose(mapminmax(X',-1,1));

% parameter setting
rho_set=[10.^(-6:1:6)];
paras.lambda=rho_set(9); 

paras.K=length(unique(y));
paras.y=y;
paras.NumHiddenNeuron=2000;

% clustering
for trial =1 :50
    label=elmc_kmeans(X,paras);
    acc(trial)=accuracy(y,label);
end

% display results
acc_mean = mean(acc)
acc_std = std(acc)
acc_max = max(acc)
