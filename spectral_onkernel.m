function [sok_acc_avg,sok_acc_std,sok_nmi_avg,sok_nmi_std,sok_purity_avg,sok_purity_std]=spectral_onkernel(data,num_class,truth,num_views)
numClust=num_class;
sigma = zeros(num_views,1);
%=======================ELM Process============================
for j=1:num_views
    L = j*100;
    [H,~]=myelm(data,truth,L, 'sigmoid');
    HM{j} = H;
    sigma(j)=optSigma(H);%sigma取平均值
end
clear data;
data = HM;
%% Construct kernel and transition matrix
K=[];
for j=1:num_views
    options.KernelType = 'Gaussian';
    %options.t=sigma_value;%same setting as co-regspectral multiview spectral
    options.t=sigma(j);
    K(:,:,j) = constructKernel(data{j},data{j},options);
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


sok_acc_avg = zeros(num_views,1);
sok_acc_std = zeros(num_views,1);
sok_nmi_avg = zeros(num_views,1);
sok_nmi_std = zeros(num_views,1);
sok_purity_avg = zeros(num_views,1);
sok_purity_std = zeros(num_views,1);

%% single best view
fprintf('======================================\n');
fprintf('Running with Best Single View.\n');
for j=1:num_views
    fprintf('view %d:\n',j);
    [~,~,~,~,~,nmi,~,~,~,ACC,Purity] = baseline_spectral_onkernel(K(:,:,j),numClust,truth);
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
    
    sok_acc_avg(j) = ACC(1);
    sok_acc_std(j) = ACC(2);
    sok_nmi_avg(j) = nmi(1);
    sok_nmi_std(j) = nmi(2);
    sok_purity_avg(j) = Purity(1);
    sok_purity_std(j) = Purity(2);
end
fprintf('Best Single ACC %.4f(%.4f), L = %d\n',best_single_view.acc_avg,best_single_view.acc_std,best_single_view.acc_L);
fprintf('Best Single NMI %.4f(%.4f), L = %d\n',best_single_view.nmi_avg,best_single_view.nmi_std,best_single_view.nmi_L);
fprintf('Best Single Purity %.4f(%.4f), L = %d\n',best_single_view.purity_avg,best_single_view.purity_std,best_single_view.purity_L);





