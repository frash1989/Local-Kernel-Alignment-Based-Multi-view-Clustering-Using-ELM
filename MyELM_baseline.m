%% ELM Co-Reg Spectral
function MyELM_baseline(data,num_class,truth,num_views,sigma_value)
%% bbcsport dataset
numClust=num_class;
%=======================ELM Process============================
for j=1:num_views
    L = j*100;
    [H,OutputWeight]=myelm(data,truth,L, 'sigmoid');
    HM{j} = H;
    %sigma(j)=optSigma(H);
end
clear data;
data = HM;
%% Construct kernel and transition matrix
K=[];
T=[];
for j=1:num_views
    options.KernelType = 'Gaussian';
    options.t=sigma_value;%same setting as co-regspectral multiview spectral
    K(:,:,j) = constructKernel(data{j},data{j},options);
    D=diag(sum(K(:,:,j),2));
    L_rw=D^-1*K(:,:,j);
    T(:,:,j)=L_rw;
end

%% init evaluation result
best_single_view.nmi_avg=0;
best_single_view.nmi_std=0;
best_single_view.nmi_L=0;
best_single_view.acc_avg=0;
best_single_view.acc_std=0;
best_single_view.acc_L=0;
best_single_view.purity_avg=0;
best_single_view.purity_std=0;
best_single_view.purity_L=0;

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
    if nmi(1)>best_single_view.nmi_avg
        best_single_view.nmi_avg=nmi(1);
        best_single_view.nmi_std=nmi(2);
        best_single_view.nmi_L=j*100;
    end
    if ACC(1)>best_single_view.acc_avg
        best_single_view.acc_avg=ACC(1);
        best_single_view.acc_std=ACC(2);
        best_single_view.acc_L=j*100;
    end
    if Purity(1)>best_single_view.purity_avg
        best_single_view.purity_avg=Purity(1);
        best_single_view.purity_std=Purity(2);
        best_single_view.purity_L=j*100;
    end
end
fprintf('Best Single ACC %.4f(%.4f), L = %d\n',best_single_view.acc_avg,best_single_view.acc_std,best_single_view.acc_L);
fprintf('Best Single NMI %.4f(%.4f), L = %d\n',best_single_view.nmi_avg,best_single_view.nmi_std,best_single_view.nmi_L);
fprintf('Best Single Purity %.4f(%.4f), L = %d\n',best_single_view.purity_avg,best_single_view.purity_std,best_single_view.purity_L);
%
%% feature concatenation multiview spectral
fprintf('======================================\n');
fprintf('Running with the Feature Concatenation.\n');
conc_feature=[];
for i=1:num_views
    conc_feature=[conc_feature data{i}];
end
conc_sigma=sigma_value;%optSigma(conc_feature);%1;
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

