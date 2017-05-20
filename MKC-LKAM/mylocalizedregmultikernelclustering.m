function [H_normalized,gamma,obj] = mylocalizedregmultikernelclustering(K,HE0,A,cluster_count,lambda)

nbkernel = size(K,3);
gamma = ones(nbkernel,1)/nbkernel;
KC  = mycombFun(K,gamma);
flag = 1;
iter = 0;
while flag
    iter = iter + 1;   
    H = mylocalkernelkmeans(KC,A,cluster_count);
    ZH = callZH(H,K,A);
    obj(iter)  = callocalObj(HE0,ZH,gamma,lambda);
    [gamma]= updatelocalkernelweights(HE0,ZH,lambda);
    if (iter>2 && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-6))||(iter>20)
        flag =0;
    end
    KC  = mycombFun(K,gamma);
end
H_normalized = H./ repmat(sqrt(sum(H.^2, 2)), 1,cluster_count);