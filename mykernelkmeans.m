%function [H_normalized,obj]= mykernelkmeans(K,cluster_count)
function [H_normalized,obj]= mykernelkmeans(x,num_class,label,num_views,sigma_value)
%% ELM Mapping
for j=1:num_views
    L = j*100;
    [H,~]=myelm(x,label,L, 'sigmoid');
    HM{j} = H;
end
data = HM;
%% Construct kernel and transition matrix
K=[];
for j=1:num_views
    options.KernelType = 'Gaussian';
    options.t=sigma_value;%same setting as co-regspectral multiview spectral
    K(:,:,j) = constructKernel(data{j},data{j},options);
end

%% kernelkmeans

gamma0 = ones(num_views,1)/num_views;
avgKer  = mycombFun(K,gamma0);

K = (avgKer+avgKer')/2 + 1e-8*eye(size(avgKer));
opt.disp = 0;
[H,~] = eigs(K,num_class,'la',opt);
obj = trace(H' * K * H) - trace(K);
H_normalized = H;

[res] = accuFucV2(H_normalized,label,num_class);

% fprintf('Best ACC: %0.4f(%0.4f) \n',res.Acc_avg,res.Acc_std);
% fprintf('Best NMI: %0.4f(%0.4f) \n', res.NMI_avg,res.NMI_std);
% fprintf('Best Purity: %0.4f(%0.4f) \n',res.Purity_avg,res.Purity_std);
