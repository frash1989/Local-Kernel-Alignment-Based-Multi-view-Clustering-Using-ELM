clear all
clc
stream = RandStream.getGlobalStream;
reset(stream);
addpath('./code_coregspectral');
projev = 1.5;

%% SensITVehicle dataset
dataset='./data/SenITVehicle_2views_300samples_3clusters.mat';
numClust=3;
num_views=2;
load(dataset);
truth = label;
%% ================data normalization==========
data{1} = Acou;
data{2} = Seis;
d1 = mapstd(data{1}');
d2 = mapstd(data{2}');
clear data;
data{1} = d1';
data{2} = d2';
%%
ACC_avg = [];
ACC_std = [];
nmi_avg = [];
nmi_std = [];
Purity_avg = [];
Purity_std = [];
nurons_i = [];
%% =======================ELM Process============================
% nurons ranges from 100 to 20000, with 29 different values
nurons= 1000;
[H1,~]=myelm(data{1},truth,nurons,'sigmoid');
[H2,~]=myelm(data{2},truth,nurons,'sigmoid');
X1 =H1;
X2 =H2;
clear data;
data{1}=H1;
data{2}=H2;
sigma(1)=optSigma(data{1});
sigma(2)=optSigma(data{2});

%% Construct kernel and transition matrix
K=[];
T=[];
for j=1:num_views
    options.KernelType = 'Gaussian';
    options.t=sigma(j);%same setting as co-regspectral multiview spectral
    K(:,:,j) = constructKernel(data{j},data{j},options);
    D=diag(sum(K(:,:,j),2));
    L_rw=D^-1*K(:,:,j);
    T(:,:,j)=L_rw;
end

%% init evaluation result
best_single_view.nmi=0;
feature_concat.nmi=0;
kernel_addition.nmi=0;
markov_mixture.nmi=0;
co_reg.nmi=0;
markov_ag.nmi=0;
%% single best view
fprintf('======================================\n');
fprintf('Running with Best Single View.\n');
for j=1:num_views
    fprintf('view %d:\n',j);
    [V Eval F P R nmi avgent AR C ACC Purity] = baseline_spectral_onkernel(K(:,:,j),numClust,truth);
    if nmi(1)>best_single_view.nmi
        best_single_view.F=F;
        best_single_view.P=P;
        best_single_view.R=R;
        best_single_view.nmi=nmi;
        best_single_view.avgent=avgent;
        best_single_view.AR=AR;
        best_single_view.ACC = ACC;
        best_single_view.Purity = Purity;
    end
end
%% feature concatenation multiview spectral
fprintf('======================================\n');
fprintf('Running with the Feature Concatenation.\n');
conc_feature=[];
for i=1:num_views
    conc_feature=[conc_feature data{i}];
end
conc_sigma=optSigma(conc_feature);
[V Eval F P R nmi avgent AR ACC Purity] = baseline_spectral(conc_feature,numClust,conc_sigma,truth);
if nmi(1)>feature_concat.nmi
    feature_concat.F=F;
    feature_concat.P=P;
    feature_concat.R=R;
    feature_concat.nmi=nmi;
    feature_concat.avgent=avgent;
    feature_concat.AR=AR;
    feature_concat.ACC=ACC;
    feature_concat.Purity=Purity;
end

%% Kernel Addition:
fprintf('======================================\n');
fprintf('Running with Kernel Addition.\n');
[V Eval F P R nmi avgent AR C ACC Purity] = baseline_spectral_onkernel(sum(K,3),numClust,truth);
if nmi(1)>kernel_addition.nmi
    kernel_addition.F=F;
    kernel_addition.P=P;
    kernel_addition.R=R;
    kernel_addition.nmi=nmi;
    kernel_addition.avgent=avgent;
    kernel_addition.AR=AR;
    best_single_view.ACC = ACC;
    best_single_view.Purity = Purity;
end

%% co-regspectral multiview spectral
numiter = 5;
fprintf('======================================\n');
fprintf('Co-regspectral multiview spectral\n');
co_sigma = sigma;
lambda=0.01;
[F P R nmi std_nmi avgent AR ACC std_ACC Purity std_Purity] = spectral_pairwise_multview(data,num_views,numClust,co_sigma,lambda,truth,numiter);
if max(nmi)>max(co_reg.nmi)
    co_reg.F=F;
    co_reg.P=P;
    co_reg.R=R;
    co_reg.nmi=nmi;
    co_reg.avgent=avgent;
    co_reg.AR=AR;
    co_reg.ACC=ACC;
    co_reg.Purity=Purity;
end

fprintf('nurons=%d\n',nurons);
max_acc =max(ACC);
std_acc=std_ACC(find(ACC==max(ACC)));
max_nmi=max(nmi);
std_nmi =std_nmi(find(nmi==max(nmi)));
max_purity =max(Purity);
std_purity =std_Purity(find(Purity==max(Purity)));
fprintf('ACC=%0.4f(%0.4f),  nmi score=%0.4f(%0.4f), Purity=%0.4f(%0.4f)\n',max_acc(1),std_acc(1),max_nmi(1),std_nmi(1),max_purity(1),std_purity(1));

%% RMSC
fprintf('======================================\n');
fprintf('running RMSC\n');

lambda_values = [];
lambda_ii = [];
for ii=-20:10
    lambda_values = [lambda_values,2^ii];
    lambda_ii = [lambda_ii,ii];
end
PP=[];
NMI_vs_lambda=[];
Fscore_vs_lambda=[];
AR_vs_lambda=[];
for i=1:length(lambda_values)
    lambda=lambda_values(i);
    opts.DEBUG=0;
    
    opts.eps=1e-6;
    opts.max_iter=300;
    P_hat=RMSC(T, lambda, opts);
    [V Eval F P R nmi avgent AR C ACC Purity] = baseline_spectral_onRW(P_hat,numClust,truth,projev);
    
    fprintf('i=%d\n',lambda_ii(i));
    fprintf('lambda=%f, ACC=%0.4f(%0.4f),  nmi score=%0.4f(%0.4f), Purity=%0.4f(%0.4f)\n',lambda,ACC(1),ACC(2),nmi(1),nmi(2),Purity(1),Purity(2));
    
    if nmi(1)>markov_ag.nmi
        markov_ag.F=F;
        markov_ag.P=P;
        markov_ag.R=R;
        markov_ag.nmi=nmi;
        markov_ag.avgent=avgent;
        markov_ag.AR=AR;
        markov_ag.ACC=ACC;
        markov_ag.Purity=Purity;
    end
    NMI_vs_lambda=[NMI_vs_lambda nmi(1)];
    Fscore_vs_lambda=[Fscore_vs_lambda F(1)];
    AR_vs_lambda=[AR_vs_lambda AR(1)];
    
    ACC_avg =[ACC_avg, ACC(1)];
    ACC_std = [ACC_std,ACC(2)];
    nmi_avg = [nmi_avg,nmi(1)];
    nmi_std = [nmi_std,nmi(2)];
    Purity_avg =[Purity_avg, Purity(1)];
    Purity_std =[Purity_std, Purity(2)];
end
step_ACC = lambda_ii(find(ACC_avg ==max(ACC_avg)));
step_NMI = lambda_ii(find(nmi_avg ==max(nmi_avg)));
step_Purity =lambda_ii(find(Purity_avg ==max(Purity_avg)));

max_acc =max(ACC_avg);
std_acc=ACC_std(find(ACC_avg==max(ACC_avg)));
max_nmi=max(nmi_avg);
std_nmi =nmi_std(find(nmi_avg==max(nmi_avg)));
max_purity =max(Purity_avg);
std_purity =Purity_std(find(Purity_avg==max(Purity_avg)));
fprintf('\n');
fprintf('nurons=%d\n',nurons);
fprintf('step_ACC=%f, step_NMI=%f, step_Purity=%f\n',step_ACC(1),step_NMI(1),step_Purity(1));
fprintf('ACC=%0.4f(%0.4f),  nmi score=%0.4f(%0.4f), Purity=%0.4f(%0.4f)\n',max_acc(1),std_acc(1),max_nmi(1),std_nmi(1),max_purity(1),std_purity(1));
