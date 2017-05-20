function [V E F P R nmi avgent AR ACC Purity] = baseline_spectral(X,numClust,sigma,truth)
% INPUT:
% X: N x P data matrix. Each row is an example
% numClust: desired number of clusters
% truth: N x 1 vector of ground truth clusterings
% OUTPUT:
% C, U, F, P, R: clustering, U matrix, F-score, Precision, Recall

if (min(truth)==0)
    truth = truth + 1;
end
numEV = numClust*1;
N = size(X,1);
options.KernelType = 'Gaussian';
options.t = sigma; % width parameter for Gaussian kernel
fprintf('constructing kernel...\n');
K = constructKernel(X,X,options);
K = K;
D = diag(sum(K,1));
L = sqrt(inv(D))*K*sqrt(inv(D));
L = (L+L')/2;
%convert_libsvm(L,'handwritten_L1.vb');
% now do an eigen-decomposition of L
fprintf('doing eigenvalue decomp...\n');
opts.disp = 0;
[V E] = eigs(L,numEV,'LA',opts);
U = V(:,1:numClust);
%[U E] = eig(L);
%[E1 I] = sort(diag(E));  %sort in increasing order
%U = U(:,I(end-numEV+1:end));
if (1)
    norm_mat = repmat(sqrt(sum(U.*U,2)),1,size(U,2));
    %%avoid divide by zero
    for i=1:size(norm_mat,1)
        if (norm_mat(i,1)==0)
            norm_mat(i,:) = 1;
        end
    end
    U = U./norm_mat;
end
fprintf('running k-means...\n');

for i=1:50
    %C = kmeans(U,numClust,'EmptyAction','drop');
    [C, center] = litekmeans(U,numClust,'MaxIter', 100, 'Replicates',10);
    [A nmii(i) avgenti(i)] = compute_nmi(truth,C);
    [Fi(i),Pi(i),Ri(i)] = compute_f(truth,C);
    [ARi(i),RIi(i),MIi(i),HIi(i)]=RandIndex(truth,C);
    [result,~] = ClusteringMeasure(truth, C);
    ACCi(i) = result(1);
    Purityi(i)=result(3);
end
F(1) = mean(Fi); F(2) = std(Fi);
P(1) = mean(Pi); P(2) = std(Pi);
R(1) = mean(Ri); R(2) = std(Ri);
nmi(1) = mean(nmii); nmi(2) = std(nmii);
avgent(1) = mean(avgenti); avgent(2) = std(avgenti);
AR(1) = mean(ARi); AR(2) = std(ARi);
ACC(1) =mean(ACCi);ACC(2) =std(ACCi);
Purity(1) =mean(Purityi);Purity(2)=std(Purityi);

%fprintf('F: %0.4f(%0.4f)\n', F(1), F(2));
%fprintf('P: %0.4f(%0.4f)\n', P(1), P(2));
%fprintf('R: %0.4f(%0.4f)\n', R(1), R(2));
fprintf('ACC: %0.4f(%0.4f)\n', ACC(1), std(ACCi));
fprintf('nmi: %0.4f(%0.4f)\n', nmi(1), nmi(2));
%fprintf('avgent: %0.4f(%0.4f)\n', avgent(1), avgent(2));
%fprintf('AR: %0.4f(%0.4f)\n', AR(1), AR(2));
fprintf('Purity: %0.4f(%0.4f)\n', Purity(1), std(Purityi));
