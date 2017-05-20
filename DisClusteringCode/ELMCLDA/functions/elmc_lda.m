function [label,i,cost,acc_tmp]=elmc_lda(X,paras)


[N,d]=size(X);
K=paras.K;
l=paras.NumHiddenNeuron;


% Random generate input weights and random bias
elmModel.InputWeight=rand(d,l)*2-1;
elmModel.Bias = rand(1,l);
tempH = X * elmModel.InputWeight;
tempH = bsxfun(@plus,tempH,elmModel.Bias);


% Calculate hidden neuron output matrix
H = 1 ./ (1 + exp(-tempH));

% shift data
H=bsxfun(@minus,H,mean(H,1));
St=H'*H;

[label, center] = litekmeans(H, paras.K, 'MaxIter',200);
acc_tmp=accuracy(paras.y,label);


MaxIter=50;
for i=1:MaxIter
    
    
    for iK=1:K
        nK(iK)=sum(label==iK);
        Mu(iK,:)=mean(H(label==iK,:),1);
    end
    
    Sb=Mu'*bsxfun(@times,Mu,nK');
    Sw=St-Sb;
    
    A=(Sb+Sb')/2+1e-10*eye(l);
    %B=Sw+paras.lambda*eye(l);
    
    B=St+paras.lambda*diag(diag(St))+1e-10*eye(l);
    
    opts.tol = 1e-9;
    opts.issym=1;
    opts.disp = 0;
    [w,v]=eigs(A,B,K-1,'lm',opts);
    Hs=H*w;
    
    cost(i)=trace((w'*B*w)\(w'*A*w));
    
    
    label0=label;
    initcenter=Mu*w;
    [label, ~] = litekmeans(Hs, K, 'MaxIter',200,'Start', initcenter);
    
    acc_tmp(i)=accuracy(paras.y,label);
    if label==label0;
        %disp(['Algorithm converges at iteration: ',num2str(i)]);
        break;
    end
end
