function mySBkernelkmeans(x,num_class,label,num_views,sigma_value)

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

%% kernelkmeans
myacc_avg =0;
myacc_std =0;
mynmi_avg =0;
mynmi_std =0;
mypurity_avg =0;
mypurity_std =0;
myacc_L =0;
mynmi_L =0;
mypurity_L =0;
for p =1:num_views
    myK = KK(:,:,p);
    K = (myK+myK')/2 + 1e-8*eye(size(myK));
    opt.disp = 0;
    [H,~] = eigs(K,num_class,'la',opt);
    obj = trace(H' * K * H) - trace(K);
    H_normalized = H;
    
    [res] = accuFucV2(H_normalized,label,num_class);
    if(res.Acc_avg>myacc_avg)
        myacc_avg =res.Acc_avg;
        myacc_std = res.Acc_std;
        myacc_L=p*100;
    end
    if(res.NMI_avg>mynmi_avg)
        mynmi_avg =res.NMI_avg;
        mynmi_std =res.NMI_std;
        mynmi_L=p*100;
    end
    if(res.Purity_avg>mypurity_avg)
        mypurity_avg =res.Purity_avg;
        mypurity_std = res.Purity_std;
        mypurity_L=p*100;
    end
    
end
fprintf('Selected Best Results!\n');
fprintf('Best ACC: %0.4f(%0.4f) L=%d\n', myacc_avg, myacc_std,myacc_L);
fprintf('Best NMI: %0.4f(%0.4f) L=%d\n', mynmi_avg, mynmi_std,mynmi_L);
fprintf('Best Purity: %0.4f(%0.4f) L=%d\n', mypurity_avg, mypurity_std,mypurity_L);









% fprintf('Best ACC: %0.4f(%0.4f) \n',res.Acc_avg,res.Acc_std);
% fprintf('Best NMI: %0.4f(%0.4f) \n', res.NMI_avg,res.NMI_std);
% fprintf('Best Purity: %0.4f(%0.4f) \n',res.Purity_avg,res.Purity_std);
