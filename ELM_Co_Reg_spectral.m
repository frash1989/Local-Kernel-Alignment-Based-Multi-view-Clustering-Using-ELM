%% ELM Co-Reg Spectral
function [myresult] = ELM_Co_Reg_spectral(data,num_class,truth,num_views,sigma_value)
%=======================ELM Process============================

for j=1:num_views
    L = j*100;
    [H,OutputWeight]=myelm(data,truth,L, 'sigmoid');
    HM{j} = H;
    %sigma(j)=optSigma(H);
    sigma(j)=sigma_value;
end
%% co-regspectral multiview spectral
co_reg.nmi=0;
numiter = 5;
fprintf('======================================\n');
fprintf('Co-regspectral multiview spectral\n');
%co_sigma=[100 100 100 100 100 100 100 100 100 100];
co_sigma = sigma;
lambda=0.01;
[F P R nmi std_nmi avgent AR ACC std_ACC Purity std_Purity] = spectral_pairwise_multview(HM,num_views,num_class,co_sigma,lambda,truth,numiter);
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

max_acc =max(ACC);
std_acc=std_ACC(find(ACC==max(ACC)));
max_nmi=max(nmi);
std_nmi =std_nmi(find(nmi==max(nmi)));
max_purity =max(Purity);
std_purity =std_Purity(find(Purity==max(Purity)));
fprintf('ACC=%0.4f(%0.4f),  nmi score=%0.4f(%0.4f), Purity=%0.4f(%0.4f)\n',max_acc(1),std_acc(1),max_nmi(1),std_nmi(1),max_purity(1),std_purity(1));

myresult.ACC_avg =max_acc(1);
myresult.ACC_std =std_acc(1);

myresult.NMI_avg =max_nmi(1);
myresult.NMI_std =std_nmi(1);

myresult.Purity_avg =max_purity(1);
myresult.Purity_std =std_purity(1);
