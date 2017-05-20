function label=elmc_kmeans(X,paras)


[N,d]=size(X);
K=paras.K;
L=paras.NumHiddenNeuron;
lambda=paras.lambda;

% Random generate input weights and bias
elmModel.InputWeight=rand(d,L)*2-1;
elmModel.Bias = rand(1,L);
tempH = X * elmModel.InputWeight;
tempH = bsxfun(@plus,tempH,elmModel.Bias);


% Calculate hidden neuron output matrix
H = 1 ./ (1 + exp(-tempH));

% shift data 
H=bsxfun(@minus,H,mean(H,1));

G=H*H';

Ker=eye(N)-inv(eye(N)+1/lambda*G);


[label, energy] = knkmeans(Ker,K);