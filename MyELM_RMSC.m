%% ELM RMSC
function [myresult]=MyELM_RMSC(data,numClust,truth,num_views,sigma_value)
projev = 1.5;

%=======================ELM Process============================
for j=1:num_views
    L = j*100;
    [H,OutputWeight]=myelm(data,truth,L, 'sigmoid');
    HM{j} = H;
    %sigma(j)=optSigma(H);
    %sigma(j)=0.3;
end
clear data;
data = HM;
%%==============================================================
ACC_avg = [];
ACC_std = [];
nmi_avg = [];
nmi_std = [];
Purity_avg = [];
Purity_std = [];

%% Construct kernel and transition matrix
K=[];
T=[];
for j=1:num_views
    options.KernelType = 'Gaussian';
    options.t=sigma_value;%%optSigma(H);%100;%same setting as co-regspectral multiview spectral
    K(:,:,j) = constructKernel(data{j},data{j},options);
    D=diag(sum(K(:,:,j),2));
    L_rw=D^-1*K(:,:,j);
    T(:,:,j)=L_rw;
end
%% RMSC
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
    fprintf('i=%d\n',i-21);
    fprintf('lambda=%f, ACC=%0.4f(%0.4f),  nmi score=%0.4f(%0.4f), Purity=%0.4f(%0.4f)\n',lambda,ACC(1),ACC(2),nmi(1),nmi(2),Purity(1),Purity(2));    
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
fprintf('step_ACC=%f, step_NMI=%f, step_Purity=%f\n',step_ACC(1),step_NMI(1),step_Purity(1));
max_acc_std = ACC_std(find(ACC_avg==max(ACC_avg)));
max_nmi_std = nmi_std(find(nmi_avg==max(nmi_avg)));
max_Purity_std = Purity_std(find(Purity_avg==max(Purity_avg)));
fprintf('ACC=%0.4f(%0.4f),nmi score=%0.4f(%0.4f),Purity=%0.4f(%0.4f)\n',max(ACC_avg),max_acc_std(1),max(nmi_avg),max_nmi_std(1),max(Purity_avg),max_Purity_std(1));
myresult.ACC_avg =max(ACC_avg);
myresult.ACC_std =max_acc_std(1);
myresult.step_ACC = step_ACC(1);
myresult.NMI_avg =max(nmi_avg);
myresult.NMI_std =max_nmi_std(1);
myresult.step_NMI = step_NMI(1);
myresult.Purity_avg =max(Purity_avg);
myresult.Purity_std =max_Purity_std(1);
myresult.step_Purity = step_Purity(1);
