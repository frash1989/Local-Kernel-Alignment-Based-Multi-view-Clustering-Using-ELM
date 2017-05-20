function [res]= accuFucV2(U,Y,numclass)

stream = RandStream.getGlobalStream;
reset(stream);
U_normalized = U ./ repmat(sqrt(sum(U.^2, 2)), 1,numclass);
maxIter = 50;
tmp1 = zeros(maxIter,1);
tmp2 = zeros(maxIter,1);
for iter = 1:maxIter
    indx = litekmeans(U_normalized,numclass,'MaxIter',100, 'Replicates',10);
%% xinwang liu    
%     indx = indx(:);
%     [newIndx] = bestMap(Y,indx);
%     tmp1(iter) = mean(Y==newIndx);
%     tmp2(iter) = MutualInfo(Y,newIndx);
%% qiang wang 
    C = indx;
    [A nmii(iter) avgenti(iter)] = compute_nmi(Y,C);
    [result,~] = ClusteringMeasure(Y, C);
    ACCi(iter) = result(1);
    Purityi(iter)=result(3);
end
%res = [max(tmp1),max(tmp2)];
%% qiang wang 
nmi(1) = mean(nmii); nmi(2) = std(nmii);
ACC(1) =mean(ACCi);ACC(2) =std(ACCi);
Purity(1) =mean(Purityi);Purity(2)=std(Purityi);
%输出聚类结果
fprintf('ACC: %0.4f(%0.4f)\n', ACC(1), std(ACCi));
fprintf('nmi: %0.4f(%0.4f)\n', nmi(1), std(nmii));
fprintf('Purity: %0.4f(%0.4f)\n', Purity(1), std(Purityi));
res.Acc_avg = ACC(1);
res.Acc_std = ACC(2);
res.NMI_avg = nmi(1);
res.NMI_std = nmi(2);
res.Purity_avg = Purity(1);
res.Purity_std = Purity(2);
