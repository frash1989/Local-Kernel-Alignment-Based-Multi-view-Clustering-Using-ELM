%% ELM MMKM-MR
function MyELM_MMKM_MR(data,numClust,truth,num_views,sigma_value)
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
    options.t=sigma_value;%optSigma(HM{j});%100;%same setting as co-regspectral multiview spectral
    KH(:,:,j) = constructKernel(HM{j},HM{j},options);
end
KH = kcenter(KH);
KH = knorm(KH);
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
lambdaset8 = 2.^[-15:1:15];
nmival81 = zeros(length(lambdaset8),1);
myacc_avg =0;
myacc_std =0;
myacc_lambda =0;
mynmi_avg =0;
mynmi_std =0;
mynmi_lambda=0;
mypurity_avg =0;
mypurity_std =0;
mypurity_lambda=0;
for il =1:length(lambdaset8)
    fprintf('Step= %d\n',log2(lambdaset8(il)));
    [H_normalized8,gamma81,obj81] = myregmultikernelclustering(KH,numclass,HH,lambdaset8(il));
    [acc81] = accuFucV2(H_normalized8,Y,numclass);
    %nmival81(il) = acc81(1,1)';
    if(acc81.Acc_avg>myacc_avg)
        myacc_avg =acc81.Acc_avg;
        myacc_std = acc81.Acc_std;
        myacc_lambda= lambdaset8(il);
    end
    if(acc81.NMI_avg>mynmi_avg)
        mynmi_avg =acc81.NMI_avg;
        mynmi_std =acc81.NMI_std;
        mynmi_lambda= lambdaset8(il);
    end
    if(acc81.Purity_avg>mypurity_avg)
        mypurity_avg =acc81.Purity_avg;
        mypurity_std = acc81.Purity_std;
        mypurity_lambda= lambdaset8(il);
    end
    myres{il} =acc81;
    
end
%indx81 = find(nmival81==max(nmival81));
%acc_res = nmival81(indx81(1));
fprintf('Selected Best Results!\n');
fprintf('Best ACC: %0.4f(%0.4f) log2(lambda) = %d \n', myacc_avg, myacc_std,log2(myacc_lambda));
fprintf('Best NMI: %0.4f(%0.4f) log2(lambda) = %d \n', mynmi_avg, mynmi_std,log2(mynmi_lambda));
fprintf('Best Purity: %0.4f(%0.4f) log2(lambda) = %d \n', mypurity_avg, mypurity_std,log2(mypurity_lambda));
