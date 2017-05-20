function [F,P,R,nmi,avgent,AR,ACC,Purity] = MyUSELM_kmeans(x,numClust,truth,hidden)
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
myacc_avg =0;
myacc_std =0;
myacc_p=0;
myacc_n=0;
myacc_lambda=0;
mynmi_avg =0;
mynmi_std =0;
mynmi_p=0;
mynmi_n=0;
mynmi_lambda=0;
mypurity_avg =0;
mypurity_std =0;
mypurity_p=0;
mtpurity_n=0;
mypurity_lambda=0;
p_array =[0.5,1,2];
n_array =[1,5,10];
lambda_array=10.^[-6:1:6];
for pi=1:length(p_array)
    options.LaplacianDegree=p_array(pi);
    for ni=1:length(n_array)
        options.NN=n_array(ni);
        for lambda_i=1:length(lambda_array)
            paras.lambda=lambda_array(lambda_i);
            % hyper-parameter settings for graph
            options.GraphWeights='binary';
            options.GraphDistanceFunction='euclidean';
            options.LaplacianNormalize=0;
            L=laplacian(options,x);
            % hyper-parameter settings for us-elm
            paras.NE=3; % specify dimensions of embedding
            paras.NumHiddenNeuron=hidden;
            paras.NormalizeInput=0;
            paras.NormalizeOutput=0;
            paras.Kernel='sigmoid';
            elmModel=uselm(x,L,paras);
            for i=1:50
                [C, center] = litekmeans(elmModel.Embed,numClust,'MaxIter', 100, 'Replicates',10);
                %C = kmeans(data,numClust,'EmptyAction','drop');
                [A nmii(i) avgenti(i)] = compute_nmi(truth,C);
                [result,~] = ClusteringMeasure(truth, C);
                ACCi(i) = result(1);
                Purityi(i)=result(3);
            end
            nmi(1) = mean(nmii); nmi(2) = std(nmii);
            ACC(1) =mean(ACCi);ACC(2) =std(ACCi);
            Purity(1) =mean(Purityi);Purity(2)=std(Purityi);
            %输出聚类结果
            fprintf('p = %0.4f, NN = %d lambda=%.4f\n',options.LaplacianDegree, options.NN,paras.lambda);
            fprintf('ACC: %0.4f(%0.4f)\n', ACC(1), std(ACCi));
            fprintf('nmi: %0.4f(%0.4f)\n', nmi(1), std(nmii));
            fprintf('Purity: %0.4f(%0.4f)\n', Purity(1), std(Purityi));
            if(ACC(1)>myacc_avg)
                myacc_avg = ACC(1);
                myacc_std = ACC(2);
                myacc_p = options.LaplacianDegree;
                myacc_n= options.NN;
                myacc_lambda=paras.lambda;
            end
            if(nmi(1)>mynmi_avg)
                mynmi_avg = nmi(1);
                mynmi_std = nmi(2);
                mynmi_p = options.LaplacianDegree;
                mynmi_n= options.NN;
                mynmi_lambda=paras.lambda;                
            end
            if(Purity(1)>mypurity_avg)                
                mypurity_avg = Purity(1);
                mypurity_std = Purity(2);
                mypurity_p = options.LaplacianDegree;
                mypurity_n= options.NN;
                mypurity_lambda=paras.lambda;
            end
        end
    end
end
fprintf('Selected Results!\n');
fprintf('Best ACC: %0.4f(%0.4f) log2(p) = %d NN = %d log10(lambda) = %d\n', myacc_avg, myacc_std,log2(myacc_p),myacc_n,log10(myacc_lambda));
fprintf('Best NMI: %0.4f(%0.4f) log2(p) = %d NN = %d log10(lambda) = %d\n', mynmi_avg, mynmi_std,log2(mynmi_p),mynmi_n,log10(mynmi_lambda));
fprintf('Best Purity: %0.4f(%0.4f) log2(p) = %d NN = %d log10(lambda) = %d\n', mypurity_avg, mypurity_std,log2(mypurity_p),mypurity_n,log10(mypurity_lambda));
