% Extreme learning machine for clustering based on kernel k-means
% Ref: Discriminative clustering via extreme learning machine, Neural
% Networks
%% ELM iterWELM
function My_ELMC_KM(data,numClust,truth,hidden)
X = data;
% parameter setting
rho_set=[10.^(-6:1:6)];
paras.lambda=rho_set(13);

paras.K=numClust;
paras.y=truth;
paras.NumHiddenNeuron=hidden;
myacc_avg =0;
myacc_std =0;
myacc_rho=0;

mynmi_avg =0;
mynmi_std =0;
mynmi_rho=0;

mypurity_avg =0;
mypurity_std =0;
mypurity_rho=0;
for i=1:length(rho_set)
    paras.lambda=rho_set(i);
    for trial =1 :50
        label=elmc_kmeans(X,paras);
        acc(trial)=accuracy(truth,label);
        [result,~] = ClusteringMeasure(truth, label);
        myacc(trial) = result(1);
        mypurity(trial) = result(3);
        [~,nmi(trial),~] = compute_nmi (truth, label);
    end    
    NMI(1) = mean(nmi); NMI(2) = std(nmi);
    Acc(1) =mean(myacc);Acc(2) =std(myacc);
    Purity(1) =mean(mypurity);Purity(2)=std(mypurity);
    fprintf('parameter is log10(%d)\n',log10(paras.lambda));
    fprintf('ACC: %0.4f(%0.4f)\n', Acc(1), Acc(2));
    fprintf('nmi: %0.4f(%0.4f)\n', NMI(1), NMI(2));
    fprintf('Purity: %0.4f(%0.4f)\n', Purity(1), Purity(2));
    if(Acc(1)>myacc_avg)
        myacc_avg =Acc(1);
        myacc_avg = Acc(1);
        myacc_std = Acc(2);
        myacc_rho = paras.lambda;
    end
    if(NMI(1)>mynmi_avg)
        mynmi_avg =NMI(1);
        mynmi_avg = NMI(1);
        mynmi_std = NMI(2);
        mynmi_rho = paras.lambda;
        
    end
    if(Purity(1)>mypurity_avg)
        mypurity_avg =Purity(1);
        mypurity_avg = Purity(1);
        mypurity_std = Purity(2);
        mypurity_rho = paras.lambda;
    end
end
fprintf('Selected Results!\n');
fprintf('Best ACC: %0.4f(%0.4f) log10(rho) = %d \n', myacc_avg, myacc_std,log10(myacc_rho));
fprintf('Best NMI: %0.4f(%0.4f) log10(rho) = %d \n', mynmi_avg, mynmi_std,log10(mynmi_rho));
fprintf('Best Purity: %0.4f(%0.4f) log10(rho) = %d \n', mypurity_avg, mypurity_std,log10(mypurity_rho));
