function [G] = MVSpectralClustering(Ln, cluster_num, r, discrete_model)
% Ln: n*n*k array, including k Laplacian matrices
% cluster_num: cluster number
% r: parameter
% discrete_model: the method that from continuous to discrete solution
% G: n*cluster_num cluster indicator matrix

% Ref:
% Xiao Cai, Feiping Nie, Heng Huang, Farhad Kamangar.
% Heterogeneous image feature integration via multi-modal spectral clustering.
% The 24th IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2011.


n = size(Ln,1);
I = eye(n);

v = size(Ln,3);
LI = zeros(n);
for i = 1:v
    LI = LI + inv(Ln(:,:,i) + r*I);
end;
LI = max(LI, LI');

[v d] = eig(LI);
d = diag(d);
[temp, idx] = sort(d, 'descend');
G0 = v(:,idx(1:cluster_num));

switch discrete_model
    case 'rotation'
        [G] = SpectralRotation(G0);
    case 'kmeans'
        StartInd = randsrc(n,1,1:cluster_num);
        [res] = kmeans_ldj(G0, StartInd);
        G = zeros(n, cluster_num);
        for cn = 1:cluster_num
            G((res==cn),cn) = 1;
        end;
    case 'nmf'
        D = eye(n);
        [Gr] = SpectralRotation(G0);
        [G] = NMFdiscrete(Gr+0.2, D, LI);
end;
    



% Nonnegative relaxation for clustering
% max trace(Q'*A*Q)|Q'*D*Q=I or max trace(Q'*An*Q)|Q'*Q=I
function [Qd obj orobj objhard] = NMFdiscrete(Q, D, A)


ITER = 100;
[total_num, class_num] = size(Q);
obj = zeros(ITER,1);
orobj = zeros(ITER,1);
objhard = zeros(ITER,1);
for iter = 1:ITER

Q = Q*diag(sqrt(1./diag(Q'*D*Q)));
QQ = Q'*D*Q;
Lamda = Q'*A*Q;
Lamda = (Lamda + Lamda')/2;

QQI = QQ - eye(class_num);
obj(iter) = trace(Lamda) - trace(Lamda*(QQI));
orobj(iter) = sqrt(trace(QQI'*QQI)/(class_num*(class_num-1)));

[dumb res] = max(Q,[],2);
Fr = zeros(total_num, class_num);
for cn = 1:class_num
    Fr((res==cn),cn) = 1;
end;
Fr = Fr*diag(sqrt(1./diag(Fr'*Fr)));
objhard(iter) = trace(Fr'*A*Fr);

QA = Q'*A*Q;
S = (A*Q + eps)./(D*Q*QA + eps);
S = S.^(1/2);
Q = Q.*S;
Q = Q*diag(sqrt(1./diag(Q'*D*Q)));

end;

Qd = Q;






function [Ind,sumd,center] = kmeans_ldj(M,StartIndMeanK)
% each row is a data point

[nSample, nFeature] = size(M);
if isscalar(StartIndMeanK)
%     t = randperm(nSample);
%     StartIndMeanK = t(1:StartIndMeanK);
    StartIndMeanK = randsrc(nSample,1,1:StartIndMeanK);
end
if isvector(StartIndMeanK)
    K = length(StartIndMeanK);
    if K == nSample
        K = max(StartIndMeanK);
        means = zeros(K,nFeature);
        for ii=1:K
            means(ii,:) = mean(M(find(StartIndMeanK==ii),:),1);
        end
    else
        means = zeros(K,nFeature);
        for ii=1:K
            means(ii,:) = M(StartIndMeanK(ii),:);
        end
    end
else
    K = size(StartIndMeanK,1);
    means = StartIndMeanK;
end
center = means;
M2 = sum(M.*M, 2)';
twoMp = 2*M';
M2b = repmat(M2,[K,1]);
Center2 = sum(center.*center,2);Center2a = repmat(Center2,[1,nSample]);[xx, Ind] = min(abs(M2b + Center2a - center*twoMp));
Ind2 = Ind;
it = 1;
%while true
while it < 200
    for j = 1:K
        dex = find(Ind == j);
        l = length(dex);
        if l > 1;                 center(j,:) = mean(M(dex,:));
        elseif l == 1;            center(j,:) = M(dex,:);
        else                      t = randperm(nSample);center(j,:) = M(t(1),:);
        end;
    end;
    Center2 = sum(center.*center,2);Center2a = repmat(Center2,[1,nSample]);[dist, Ind] = min(abs(M2b + Center2a - center*twoMp));
    if Ind2==Ind;       break;    end
    Ind2 = Ind;
    it = it+1;
end
sumd = zeros(K,1);
for ii=1:K
    idx = find(Ind==ii);
    dist2 = dist(idx);
    sumd(ii) = sum(dist2);
end






function [Fr obj Q] = SpectralRotation(F)

[n,c] = size(F);

F(sum(abs(F),2) <= 10^-18,:) = 1;
F = diag(diag(F*F').^(-1/2)) * F;

con_flag = 0;
Q = orth(rand(c));  
obj_old = 10^10;
for iter = 1:30
    M = F*Q;
    G = binarizeM(M, 'max');
    
    aa = M - G; obj = trace(aa'*aa);
    if (obj_old - obj)/obj < 0.000001
        con_flag = 1;
        break;
    end;
    obj_old = obj;
    
    [U, d, V] = svd(F'*G);
    Q = U*V'; 
end;
Fr = G;

if con_flag == 0
    warning('does not converge');
end;





function B = binarizeM(M, type)
% binarize matrix M to 0 or 1

[n,c] = size(M);

B = zeros(n,c);

if strcmp(type, 'median')
    B(find(M > 0.5)) = 1;
else
    
if strcmp(type, 'min')
    [temp idx] = min(M,[],2);
elseif strcmp(type, 'max')
    [temp idx] = max(M,[],2);
end;

for i = 1:n
    B(i,idx(i)) = 1;
end;

end;


    







