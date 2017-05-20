%% ELM MKC_LKAM
function MyELM_MKC_LKAM(data,numClust,truth,num_views,sigma_value)
Y = truth;

for j=1:num_views
    L = j*100;
    [H,OutputWeight]=myelm(data,truth,L, 'sigmoid');
    HM{j} = H;
end

%Construct kernel and transition matrix
KH=[];
for j=1:num_views
    options.KernelType = 'Gaussian';
    options.t=sigma_value;%100;%same setting as co-regspectral multiview spectral
    KH(:,:,j) = constructKernel(HM{j},HM{j},options);
end
KH = kcenter(KH);
KH = knorm(KH);
num = size(KH,1);
numclass = numClust;
numker = size(KH,3);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
HH = zeros(numker);
for p =1:numker
    for q = p:numker
        HH(p,q) = trace(KH(:,:,p)'*KH(:,:,q));
    end
end
HH = (HH+HH')-diag(diag(HH));
gamma0 = ones(numker,1)/numker;
avgKer  = mycombFun(KH,gamma0);
lambdaset = 2.^[-15:1:15];
%tauset = [0.1:0.1:0.9];
tauset = [0.05:0.1:0.95];
myacc_avg =0;
myacc_std =0;
myacc_lambda =0;
myacc_tau =0;
mynmi_avg =0;
mynmi_std =0;
mynmi_lambda=0;
mynmi_tau =0;
mypurity_avg =0;
mypurity_std =0;
mypurity_lambda=0;
mypurity_tau =0;
for it =1:length(tauset)
    numSel = round(tauset(it)*num);
    A = genarateNeighborhood(avgKer,numSel);
    HE = calHessian(KH,A);
    for il =1:length(lambdaset)
        fprintf('tau = %d , lambda = %d\n',tauset(it),log2(lambdaset(il)));
        [H_normalized,gamma,obj] = mylocalizedregmultikernelclustering(KH,HE,A,numclass,lambdaset(il));
        [res] = accuFucV2(H_normalized,Y,numclass);
        if(res.Acc_avg>myacc_avg)
            myacc_avg =res.Acc_avg;
            myacc_std = res.Acc_std;
            myacc_lambda= lambdaset(il);
            myacc_tau =tauset(it);
        end
        if(res.NMI_avg>mynmi_avg)
            mynmi_avg =res.NMI_avg;
            mynmi_std =res.NMI_std;
            mynmi_lambda= lambdaset(il);
            mynmi_tau =tauset(it);
        end
        if(res.Purity_avg>mypurity_avg)
            mypurity_avg =res.Purity_avg;
            mypurity_std = res.Purity_std;
            mypurity_lambda= lambdaset(il);
            mypurity_tau =tauset(it);
        end
    end
end
fprintf('Selected Best Results!\n');
fprintf('Best ACC: %0.4f(%0.4f) log2(lambda) = %d tau = %.4f \n', myacc_avg, myacc_std,log2(myacc_lambda),myacc_tau);
fprintf('Best NMI: %0.4f(%0.4f) log2(lambda) = %d tau = %.4f \n', mynmi_avg, mynmi_std,log2(mynmi_lambda),mynmi_tau);
fprintf('Best Purity: %0.4f(%0.4f) log2(lambda) = %d tau = %.4f \n', mypurity_avg, mypurity_std,log2(mypurity_lambda),mypurity_tau);
