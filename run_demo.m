clear;
clc;
stream = RandStream.getGlobalStream;
reset(stream);
addpath(genpath('ClusteringMeasure'));
addpath(genpath('.\SS-US-ELM'));
addpath(genpath('.\RMSC-ELM'));
addpath(genpath('.\MMSC-ELM'));
addpath(genpath('.\MKC-LKAM'));

addpath(genpath('.\DisClusteringCode'));
addpath(genpath('C:\Program Files\Mosek\7\toolbox\r2013a'));
datapath = 'D:\博士阶段\多视图ELM后期融合\无监督聚类\dataset\singleview\';
%load([datapath,'iris'],'data','label');
%load([datapath,'libras'],'data','label');
%load([datapath,'segment'],'data','label');
%load([datapath,'glass'],'data','label');
%load([datapath,'ecoli'],'data','label');
%load([datapath,'diabetes'],'data','label');
%load([datapath,'vehicle'],'data','label');
%load([datapath,'seeds'],'data','label');
%load([datapath,'heart'],'data','label');
%load([datapath,'uspst'],'data','label');
%load([datapath,'yale'],'data','label');
%load([datapath,'yaleb'],'data','label');
%load([datapath,'orl'],'data','label');
%load([datapath,'vowel'],'data','label');
load([datapath,'CNAE'],'data','label');
%load([datapath,'coil20'],'data','label');
%load([datapath,'letterABC'],'data','label');
%load([datapath,'letterAB'],'data','label');
%load([datapath,'balance'],'data','label');
x = transpose(mapminmax(data',-1,1));
num_class = length(unique(label));
%% Kmeans
% fprintf('Result of Kmeans\n');
% [F,P,R,nmi,avgent,AR,ACC,Purity]=MyKmeans(x,num_class,label);
% fprintf('======================================\n');

%% ELM kmeans
% fprintf('Result of ELM Kmenas\n');
% L = 2000;
% [H,OutputWeight]=myelm(x,label,L, 'sigmoid');
% fprintf('Nodes of hidden layers:%d\n',L);
% [F,P,R,nmi,avgent,AR,ACC,Purity]=MyKmeans(H,num_class,label);
% fprintf('======================================\n');

%% US-ELM
% fprintf('Result of US-ELM\n');
% hidden =2000;
% MyUSELM_kmeans(x,num_class,label,hidden);
% fprintf('======================================\n');

%% Iter welm
% fprintf('Result of Iter WELM\n');
% hidden=2000;
% MyIterWELM(x,num_class,label,hidden);
% fprintf('======================================\n');

%% ELMC LDA
%letterAB
% fprintf('Result of ELMC LDA\n');
% hidden=2000;
% My_ELMC_LDA(x,num_class,label,hidden);
% fprintf('======================================\n');

%% ELMC KM
%yaleb
% fprintf('Result of ELMC KM\n');
% hidden =2000;
% My_ELMC_KM(x,num_class,label,hidden);
% fprintf('======================================\n');

%% ELM baseline

% fprintf('Result of ELM baseline\n');
% num_views=20;
% sigma_value=2^(10);
% MyELM_baseline(x,num_class,label,num_views,sigma_value);
% fprintf('======================================\n');

%% ELM Co-Reg Spectral
%yaleb
% fprintf('Result of ELM Co-Reg Spectral\n');
% num_views =20;
% sigma_value=2^(8);
% ELM_CoReg_result = ELM_Co_Reg_spectral(x,num_class,label,num_views,sigma_value);
% fprintf('======================================\n');

%% ELM RMSC
%yaleb
% fprintf('Result of ELM RMSC\n');
% num_views=20;
% sigma_value=2^(10);
% ELM_RMSC_result = MyELM_RMSC(x,num_class,label,num_views,sigma_value);
% fprintf('======================================\n');


%% ELM MMSC

% fprintf('Result of ELM MMSC\n');
% num_views =20;
% MyELM_MMSC(x,num_class,label,num_views);
% fprintf('======================================\n');

%% ELMC MKKM-MR

% fprintf('Result of MMKM-MR\n');
% num_views = 20;
% sigma_value=2^(0);
% MyELM_MMKM_MR(x,num_class,label,num_views,sigma_value);
% fprintf('======================================\n');

%% ELMC MKKM-LKAM
%yale 2^0 1 2 3 4已测
% fprintf('Result of MMV LKAM\n');
% num_views = 20;
% sigma_value=2^(5);
% MyELM_MKC_LKAM(x,num_class,label,num_views,sigma_value);
% fprintf('======================================\n');



%% A-MKKM

% fprintf('Result of A-MKKM\n');
% num_views = 20;
% sigma_value=2^(4);
% mykernelkmeans(x,num_class,label,num_views,sigma_value);
% fprintf('======================================\n');

%% SB-KKM
% fprintf('Result of SB-KKM\n');
% num_views = 20;
% sigma_value=2^(2);
% mySBkernelkmeans(x,num_class,label,num_views,sigma_value);
% fprintf('======================================\n');

%% MKKM
% fprintf('Result of MKKM\n');
% num_views = 20;
% sigma_value=2^(4);
% mkkmeans_train(x,num_class,label,num_views,sigma_value);
% fprintf('======================================\n');

%% LMKKM
% fprintf('Result of LMKKM\n');
% num_views = 20;
% sigma_value=2^(3);
% lmkkmeans_train(x,num_class,label,num_views,sigma_value);
% fprintf('======================================\n');

%% RMKKM
fprintf('Result of LMKKM\n');
num_views = 20;
sigma_value=2^(8);
my_RMKKM(x,num_class,label,num_views,sigma_value);
fprintf('======================================\n');


