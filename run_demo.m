clear;
clc;
stream = RandStream.getGlobalStream;
reset(stream);
addpath(genpath('ClusteringMeasure'));
addpath(genpath('.\SS-US-ELM'));
addpath(genpath('.\RMSC-ELM'));
addpath(genpath('.\MMSC-ELM'));
addpath(genpath('.\MKC-LKAM'));
addpath(genpath('SSC_1.0'));

addpath(genpath('.\DisClusteringCode'));
addpath(genpath('C:\Program Files\Mosek\7\toolbox\r2013a'));
datapath = 'F:\博士阶段\多视图ELM后期融合\无监督聚类\dataset\singleview\';
load([datapath,'iris'],'data','label');
%load([datapath,'heart'],'data','label');
%load([datapath,'libras'],'data','label');
%load([datapath,'balance'],'data','label');
%load([datapath,'diabetes'],'data','label');
%load([datapath,'vehicle'],'data','label');
%load([datapath,'CNAE'],'data','label');
%load([datapath,'segment'],'data','label');
%load([datapath,'yale'],'data','label');
%load([datapath,'orl'],'data','label');

x = transpose(mapminmax(data',-1,1));
num_class = length(unique(label));
%% Kmeans
fprintf('Result of Kmeans\n');
[F,P,R,nmi,avgent,AR,ACC,Purity]=MyKmeans(x,num_class,label);
fprintf('======================================\n');

%% Kernel kmeans without ELM
% fprintf('Result of KKM without ELM\n');
% tic;
% options.KernelType = 'Gaussian';
% % options.t=optSigma(x);%sigma取平均值
% lambda_values = [];
% lambda_ii = [];
% ACC_avg = [];
% ACC_std = [];
% nmi_avg = [];
% nmi_std = [];
% Purity_avg = [];
% Purity_std = [];
% 
% for ii=-10:10
%     lambda_values = [lambda_values,2^ii];
%     lambda_ii = [lambda_ii,ii];
% end
% for i=1:length(lambda_values)
%     options.t=lambda_values(i);
%     myK = constructKernel(x,x,options);
%     K = (myK+myK')/2 + 1e-8*eye(size(myK));
%     opt.disp = 0;
%     [H,~] = eigs(K,num_class,'la',opt);
%     H_normalized = H;
%     [res_kkm] = accuFucV2(H_normalized,label,num_class);
% 
%     ACC_avg =[ACC_avg, res_kkm.Acc_avg];
%     ACC_std = [ACC_std,res_kkm.Acc_std];
%     nmi_avg = [nmi_avg,res_kkm.NMI_avg];
%     nmi_std = [nmi_std,res_kkm.NMI_std];
%     Purity_avg =[Purity_avg, res_kkm.Purity_avg];
%     Purity_std =[Purity_std, res_kkm.Purity_std];
% end
% disp(['SSC运行时间:',num2str(toc),'s']); 
% step_ACC = lambda_ii(find(ACC_avg ==max(ACC_avg)));
% step_NMI = lambda_ii(find(nmi_avg ==max(nmi_avg)));
% step_Purity =lambda_ii(find(Purity_avg ==max(Purity_avg)));
% fprintf('step_ACC=%f, step_NMI=%f, step_Purity=%f\n',step_ACC(1),step_NMI(1),step_Purity(1));
% max_acc_std = ACC_std(find(ACC_avg==max(ACC_avg)));
% max_nmi_std = nmi_std(find(nmi_avg==max(nmi_avg)));
% max_Purity_std = Purity_std(find(Purity_avg==max(Purity_avg)));
% fprintf('ACC=%0.4f(%0.4f),nmi score=%0.4f(%0.4f),Purity=%0.4f(%0.4f)\n',max(ACC_avg),max_acc_std(1),max(nmi_avg),max_nmi_std(1),max(Purity_avg),max_Purity_std(1));
% 
% fprintf('======================================\n');
%% Spectral clustering on kernel without ELM
% fprintf('Result of Spectral clusteirng on kernel withou ELM\n')
% tic;
% options.KernelType = 'Gaussian';
% % options.t=optSigma(x);
% lambda_values = [];
% lambda_ii = [];
% ACC_avg = [];
% ACC_std = [];
% nmi_avg = [];
% nmi_std = [];
% Purity_avg = [];
% Purity_std = [];
% 
% for ii=-10:10
%     lambda_values = [lambda_values,2^ii];
%     lambda_ii = [lambda_ii,ii];
% end
% for i=1:length(lambda_values)
%     options.t=lambda_values(i);
%     K_sok = constructKernel(x,x,options);
%     [~,~,~,~,~,nmi_sok,~,~,~,ACC_sok,Purity_sok] = baseline_spectral_onkernel(K_sok,num_class,label);
%     
%     ACC_avg =[ACC_avg, ACC_sok(1)];
%     ACC_std = [ACC_std,ACC_sok(2)];
%     nmi_avg = [nmi_avg,nmi_sok(1)];
%     nmi_std = [nmi_std,nmi_sok(2)];
%     Purity_avg =[Purity_avg, Purity_sok(1)];
%     Purity_std =[Purity_std, Purity_sok(2)];
% end
% disp(['SSC运行时间:',num2str(toc),'s']); 
% step_ACC = lambda_ii(find(ACC_avg ==max(ACC_avg)));
% step_NMI = lambda_ii(find(nmi_avg ==max(nmi_avg)));
% step_Purity =lambda_ii(find(Purity_avg ==max(Purity_avg)));
% fprintf('step_ACC=%f, step_NMI=%f, step_Purity=%f\n',step_ACC(1),step_NMI(1),step_Purity(1));
% max_acc_std = ACC_std(find(ACC_avg==max(ACC_avg)));
% max_nmi_std = nmi_std(find(nmi_avg==max(nmi_avg)));
% max_Purity_std = Purity_std(find(Purity_avg==max(Purity_avg)));
% fprintf('ACC=%0.4f(%0.4f),nmi score=%0.4f(%0.4f),Purity=%0.4f(%0.4f)\n',max(ACC_avg),max_acc_std(1),max(nmi_avg),max_nmi_std(1),max(Purity_avg),max_Purity_std(1));
% 
% disp(['SCoK运行时间:',num2str(toc),'s']); 
% fprintf('======================================\n');
%% SSC
% fprintf('Result of SSC without ELM\n');
% tic;
% X = x';
% r = 0; %Enter the projection dimension e.g. r = d*n, enter r = 0 to not project
% Cst = 0; %Enter 1 to use the additional affine constraint sum(c) == 1
% OptM = 'Lasso'; %OptM can be {'L1Perfect','L1Noise','Lasso','L1ED'}
% lambda = 0.01; %Regularization parameter in 'Lasso' or the noise level for 'L1Noise'
% K = 0;%max(d1,d2); %Number of top coefficients to build the similarity graph, enter K=0 for using the whole coefficients
% Xp = DataProjection(X,r,'NormalProj');
% CMat = SparseCoefRecovery(Xp,Cst,OptM,lambda);
% 
% options.KernelType = 'Gaussian';
% options.t=optSigma(CMat);
% % options.t=2^(9);
% K_ssc = constructKernel(CMat,CMat,options);
% [~,~,~,~,~,nmi_ssc,~,~,~,ACC_ssc,Purity_ssc] = baseline_spectral_onkernel(K_ssc,num_class,label);
% disp(['SSC运行时间:',num2str(toc),'s']);
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
% fprintf('Result of LMKKM\n');
% num_views = 20;
% sigma_value=2^(8);
% my_RMKKM(x,num_class,label,num_views,sigma_value);
% fprintf('======================================\n');



