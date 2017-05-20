function [ACC,nmi,Purity] = MyKmeans(data,numClust,truth)
% INPUT:
% data: N x P data matrix. Each row is an example
% numClust: desired number of clusters
% truth: N x 1 vector of ground truth clusterings
% OUTPUT:
%F: F-score
%P: Precision
%R: Recall
%nmi:
%avgent:
%AR:
%ACC
%Purity
for i=1:50        
    [C, center] = litekmeans(data,numClust,'MaxIter', 100, 'Replicates',10);
    %C = kmeans(data,numClust,'EmptyAction','drop');
    [Fi(i),Pi(i),Ri(i)] = compute_f(truth,C);
    [A nmii(i) avgenti(i)] = compute_nmi(truth,C);
    if (min(truth)==0)
        [ARi(i),RIi(i),MIi(i),HIi(i)]=RandIndex(truth+1,C);
    else
        [ARi(i),RIi(i),MIi(i),HIi(i)]=RandIndex(truth,C);
    end
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
%输出聚类结果
fprintf('ACC: %0.4f(%0.4f)\n', ACC(1), std(ACCi));
fprintf('nmi: %0.4f(%0.4f)\n', nmi(1), std(nmii));
fprintf('Purity: %0.4f(%0.4f)\n', Purity(1), std(Purityi));

