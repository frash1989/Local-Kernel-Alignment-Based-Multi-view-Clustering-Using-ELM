%function [H_normalized,theta]= mkkmeans_train(Km, cluster_count)

function  mkkmeans_train(x,num_class,label,num_views,sigma_value)
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
Km=KK;
cluster_count = num_class;


numker = size(Km, 3);
theta = ones(numker,1)/numker;
K_theta = mycombFun(Km, theta.^2);

opt.disp = 0;
iteration_count = 0;
flag =1;
while flag
    iteration_count = iteration_count+1;
    fprintf(1, 'running iteration %d...\n', iteration_count);
    [H, ~] = eigs(K_theta, cluster_count, 'la', opt);
    Q = zeros(numker);
    for m = 1:numker
        Q(m, m) = trace(Km(:, :, m)) - trace(H' * Km(:, :, m) * H);
    end
    res = mskqpopt(Q, zeros(numker, 1), ones(1, numker), 1, 1, zeros(numker, 1), ones(numker, 1), [], 'minimize echo(0)');
    theta = res.sol.itr.xx;
    K_theta = mycombFun(Km, theta.^2);
    
    objective(iteration_count) = trace(H' * K_theta * H) - trace(K_theta);
    if iteration_count>2 && (abs((objective(iteration_count)-objective(iteration_count-1))/(objective(iteration_count)))<1e-3)
        flag =0;
    end
end
H_normalized = H;
%clustering evaluation
[res] = accuFucV2(H_normalized,label,num_class);