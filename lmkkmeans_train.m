%function [H_normalized,objective] = lmkkmeans_train(Km, cluster_count)
function  lmkkmeans_train(x,num_class,label,num_views,sigma_value)

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


N = size(Km, 2);
P = size(Km, 3);
Theta = ones(N, P) / P;
K_Theta = calculate_localized_kernel_theta(Km, Theta);

opt.disp = 0;
iteration_count = 0;
flag =1;
while flag
    iteration_count = iteration_count +1;
    fprintf(1, 'running iteration %d...\n', iteration_count);
    [H, ~] = eigs(K_Theta, cluster_count, 'la', opt);
    HHT = H * H';
    
    Q = zeros(N * P, N * P);
    for m = 1:P
        start_index = (m - 1) * N + 1;
        end_index = m * N;
        Q(start_index:end_index, start_index:end_index) = eye(N, N) .* Km(:, :, m) - HHT .* Km(:, :, m);
    end
    res = mskqpopt(Q, zeros(N * P, 1), repmat(eye(N, N), 1, P), ones(N, 1), ones(N, 1), zeros(N * P, 1), ones(N * P, 1), [],...
        'minimize echo(0)');
    Theta = reshape(res.sol.itr.xx, N, P);
    K_Theta = calculate_localized_kernel_theta(Km, Theta);
    
    objective(iteration_count) = trace(H' * K_Theta * H) - trace(K_Theta);
    
    if iteration_count>2 && (abs((objective(iteration_count)-objective(iteration_count-1))/(objective(iteration_count)))<1e-3)
        flag =0;
    end
end
H_normalized = H ./ repmat(sqrt(sum(H.^2, 2)), 1, cluster_count);


%clustering evaluation
[res] = accuFucV2(H_normalized,label,num_class);
