function [label, kw, center, bCon, sumD, D, objHistory] = RMKKM(Ks, k, varargin)
%   [1] Liang Du, Peng Zhou, Lei Shi, Hanmo Wang, Mingyu Fan, Wenjian Wang, Yi-Dong Shen,
%   Robust Multiple Kernel Kmeans, IJCAI 2015
%
%**************************************************
%     Author: Liang Du <csliangdu@gmail.com>
%     Version: 1.0
%     Last modified: 2015-04-29 04:37:36
%**************************************************
m = size(Ks,3);
n = size(Ks,1);
assert(n == size(Ks,2), 'The input kernel matrix should be squared');

if ~(isscalar(k) && isnumeric(k) && isreal(k) && k > 0 && (round(k)==k))
    error('RMKKM:Invalid k', 'k must be a positive integer value.');%#ok
elseif n < k
    error('RMKKM:TooManyClusters', ...
        'K must have more rows than the number of clusters.');
end

pnames = {'gamma', 'start'   'maxiter'  'replicates' 'onlinephase' };
dflts =  {'0.7', 'sample'       []        []        'off'                };
[eid,errmsg, gamma, start, maxiter, replicates] = getargs(pnames, dflts, varargin{:});
if ~isempty(eid)
    error(sprintf('RMKKM:%s',eid),errmsg);
end

center = [];
if ischar(start)
    startNames = {'sample'};
    j = find(strncmpi(start,startNames,length(start)));
    if length(j) > 1
        error(message('RMKKM:AmbiguousStart',start));
    elseif isempty(j)
        error(message('RMKKM:UnknownStart', start));
    elseif isempty(k)
        error('RMKKM:MissingK', 'You must specify the number of clusters, K.');
    end
    if j == 2
        if floor(.1*n) < 5*k
            j = 1;
        end
    end
    start = startNames{j};
end

% The maximum iteration number is default 100
if isempty(maxiter)
    maxiter = 100;
end

% Assume one replicate
if isempty(replicates)
    replicates = 1;
end

bestlabel = [];
bestkw = [];
sumD = zeros(1,k);
bCon = false;
bestObjHistory = [];

for t=1:replicates
    objHistory  = [];
    % cluster center initialization, each cluster is represented a linear combination of the data points
    switch start
        case 'sample'
            seed = randsample(n,k);
            center = zeros(n, k);
            for i = 1:k
                center(seed(i),i) = 1;
            end
            center = bsxfun(@rdivide, center, max(sum(center), 1e-10)); % weighted indicator
        case 'numeric'
            center = rand(n, k);
            center = bsxfun(@rdivide, center, max(sum(center), 1e-10)); % weighted indicator
    end
    last = 0;label=1;
    
    % kernel weight initialization
    kw = ones(m,1) / m;
    
    % kernel k-means with L21-norm
    iter = 0;
    while any(label ~= last) && iter < maxiter
        % update Kernel
        Ka = zeros(n);
        for i = 1:m
            Ka = kw(i) * Ks(:,:,i) + Ka;
        end
        
        if 1
            [label, center] = L21KKM(Ka, k, 'maxiter', 30, 'Replicates', 10);
        else
            [label, center] = L21KKM_single(Ka, center, label);
        end
        % update kernel weight
        Z = full(sparse(1:n,label,ones(n,1),n,k,n)); % indicator matrix
        A = zeros(n, m);
        for i = 1:m
            bb = sum((Ks(:,:,i) * center) .* center);
            ab = Ks(:,:,i) * center;
            D = bsxfun(@plus, -2*ab, bb);
            D = bsxfun(@plus, D, diag(Ks(:,:,i)));
            A(:, i) = sum(Z .* D, 2);
        end
        A = max(A, eps);
        Aw = bsxfun(@times, A, kw');
        D = 2 * sqrt(sum(Aw,2));
        D = 1 ./ D;
        A = bsxfun(@times, A, D);
        h = sum(A);
        kw = lp_simplex_proj(h, gamma)';
        
        obj = rmkkm_obj(Ks, kw, label, center);
        objHistory = [objHistory; obj];%#ok<AGROW>
        iter = iter + 1;
    end
    
    if iter<maxiter
        bCon = true;
    end
    if isempty(bestlabel)
        bestlabel = label;
        bestkw = kw;
        bestcenter = center;
        bestObjHistory = objHistory;
        if replicates>1
            Ka = zeros(n);
            for i = 1:m
                Ka = kw(i) * Ks(:,:,i) + Ka;
            end
            aa = full(diag(Ka));
            bb = sum((Ka * center) .* center);
            ab = Ka * center;
            D = bsxfun(@plus, aa, bb) - 2*ab;
            D(D<0) = 0;
            D = sqrt(D);
            for j = 1:k
                sumD(j) = sum(D(label==j,j));
            end
            bestsumD = sumD;
            bestD = D;
        end
    else
        Ka = zeros(n);
        for i = 1:m
            Ka = kw(i) * Ks(:,:,i) + Ka;
        end
        aa = full(diag(Ka));
        bb = sum((Ka * center) .* center);
        ab = Ka * center;
        D = bsxfun(@plus, aa, bb) - 2*ab;
        D(D<0) = 0;
        D = sqrt(D);
        for j = 1:k
            sumD(j) = sum(D(label==j,j));
        end
        if sum(sumD) < sum(bestsumD)
            bestlabel = label;
            bestkw = kw;
            bestcenter = center;
            bestsumD = sumD;
            bestD = D;
            bestObjHistory = objHistory;
        end
    end
end

label = bestlabel;
kw = bestkw;
center = bestcenter;
objHistory = bestObjHistory;
if replicates>1
    sumD = bestsumD;
    D = bestD;
elseif nargout > 3
    Ka = zeros(n);
    for i = 1:m
        Ka = kw(i) * Ks(:,:,i) + Ka;
    end
    aa = full(diag(Ka));
    bb = sum((Ka * center) .* center);
    ab = Ka * center;
    D = bsxfun(@plus, aa, bb) - 2*ab;
    D(D<0) = 0;
    D = sqrt(D);
    for j = 1:k
        sumD(j) = sum(D(label==j,j));
    end
end