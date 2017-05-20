function elmModel=uselm(X,L,paras)

[N,elmModel.InputDim]=size(X);

% Normalize the input
elmModel.NormalizeInput=paras.NormalizeInput;
if paras.NormalizeInput
    [X,elmModel.PreProcess]=mapminmax(X,-1,1);
end

% Random generate input weights
elmModel.InputWeight=rand(elmModel.InputDim,paras.NumHiddenNeuron)*2-1;

% Calculate hidden neuron output matrix
elmModel.Kernel=paras.Kernel;
switch paras.Kernel
    case 'sigmoid'
        H=1 ./ (1 + exp(-X*elmModel.InputWeight));
end

% Calculate output weights
opts.tol = 1e-9;
opts.issym=1;
opts.disp = 0;
if  (paras.NumHiddenNeuron<N)
    A=eye(paras.NumHiddenNeuron)+paras.lambda*H'*L*H;
    B=H'*H;
    [E,V] = eigs(A,B,paras.NE+1,'sm',opts);
    [~,idx]=sort(diag(V));
    elmModel.OutputWeight=E(:,idx(2:end));
    norm_term=H*E(:,idx(2:end));
    elmModel.OutputWeight=bsxfun(@times,E(:,idx(2:end)),sqrt(1./sum(norm_term.*norm_term)));
else
    B=H*H';
    A=eye(N)+paras.lambda*L*B;
    [E,V] = eigs(A,B,paras.NE+1,'sm',opts);
    [~,idx]=sort(diag(V));
    norm_term=B*E(:,idx(2:end));
    elmModel.OutputWeight=bsxfun(@times,H'*E(:,idx(2:end)),sqrt(1./sum(norm_term.*norm_term)));
end

Embed=H*elmModel.OutputWeight;

if ~paras.NormalizeOutput
    elmModel.Embed=Embed;
else
    elmModel.Embed=bsxfun(@times,Embed,1./sqrt(sum(Embed.*Embed,2)));
end


