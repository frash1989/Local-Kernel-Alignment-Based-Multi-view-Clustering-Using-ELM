function  my_RMKKM(x,num_class,label,num_views,sigma_value)
%% ELM Mapping
for j=1:num_views
    L = j*100;
    [H,~]=myelm(x,label,L, 'sigmoid');
    HM{j} = H;
end
data = HM;
%% Construct kernel and transition matrix
KK=[];
for j=1:num_views
    options.KernelType = 'Gaussian';
    options.t=sigma_value;%same setting as co-regspectral multiview spectral
    KK(:,:,j) = constructKernel(data{j},data{j},options);
end
KH=KK;

gammaset9 = [0.1:0.1:0.9];

myRMKKM.nmi_avg=0;
myRMKKM.nmi_std=0;
myRMKKM.nmi_gamma=0;
myRMKKM.acc_avg=0;
myRMKKM.acc_std=0;
myRMKKM.acc_gamma=0;
myRMKKM.purity_avg=0;
myRMKKM.purity_std=0;
myRMKKM.purity_gamma=0;


for il =1:length(gammaset9)
    for ii=1:10
        [indx9] = RMKKM(KH, num_class, 'gamma', gammaset9(il), 'maxiter',100, 'replicates',1);
        C = indx9;
        [~,nmii(ii),~] = compute_nmi(label,C);
        [result,~] = ClusteringMeasure(label, C);
        ACCi(ii) = result(1);
        Purityi(ii)=result(3);
    end
    nmi(1) = mean(nmii); nmi(2) = std(nmii);
    ACC(1) =mean(ACCi);ACC(2) =std(ACCi);
    Purity(1) =mean(Purityi);Purity(2)=std(Purityi);
    %输出聚类结果
    fprintf('gamma = %f\n',gammaset9(il));
    fprintf('ACC: %0.4f(%0.4f)\n', ACC(1), std(ACCi));
    fprintf('nmi: %0.4f(%0.4f)\n', nmi(1), std(nmii));
    fprintf('Purity: %0.4f(%0.4f)\n', Purity(1), std(Purityi));
    
    if nmi(1)>myRMKKM.nmi_avg
        myRMKKM.nmi_avg=nmi(1);
        myRMKKM.nmi_std=nmi(2);
        myRMKKM.nmi_gamma=gammaset9(il);
    end
    if ACC(1)>myRMKKM.acc_avg
        myRMKKM.acc_avg=ACC(1);
        myRMKKM.acc_std=ACC(2);
        myRMKKM.acc_gamma=gammaset9(il);
    end
    if Purity(1)>myRMKKM.purity_avg
        myRMKKM.purity_avg=Purity(1);
        myRMKKM.purity_std=Purity(2);
        myRMKKM.purity_gamma=gammaset9(il);
    end
    
end
fprintf('Best ACC %.4f(%.4f), gamma = %d\n',myRMKKM.acc_avg,myRMKKM.acc_std,myRMKKM.acc_gamma);
fprintf('Best NMI %.4f(%.4f), gamma = %d\n',myRMKKM.nmi_avg,myRMKKM.nmi_std,myRMKKM.nmi_gamma);
fprintf('Best Purity %.4f(%.4f), gamma = %d\n',myRMKKM.purity_avg,myRMKKM.purity_std,myRMKKM.purity_gamma);
