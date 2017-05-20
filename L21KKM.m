function [label, center, bCon, sumD, D] = L21KKM(K, k, varargin)
%   L21 Kernel K-means clustering, accelerated by matlab matrix operations.
%
%   label = KernelKmeans(X, K) partitions the points in the N-by-P data matrix
%   X into K clusters.  This partition minimizes the sum, over all
%   clusters, of the within-cluster sums of point-to-cluster-centroid
%   distances.  Rows of X correspond to points, columns correspond to
%   variables.  KMEANS returns an N-by-1 vector label containing the
%   cluster indices of each point.
%
%   [label, center] = KernelKmeans(X, K) returns the K cluster centroid
%   locations in the K-by-P matrix center.
%
%   [label, center, bCon] = KernelKmeans(X, K) returns the bool value bCon to
%   indicate whether the iteration is converged.
%
%   [label, center, bCon, SUMD] = KernelKmeans(X, K) returns the
%   within-cluster sums of point-to-centroid distances in the 1-by-K vector
%   sumD.
%
%   [label, center, bCon, SUMD, D] = KernelKmeans(X, K) returns
%   distances from each point to every centroid in the N-by-K matrix D.
%
%   [ ... ] = KernelKmeans(..., 'PARAM1',val1, 'PARAM2',val2,
...) specifies
    %   optional parameter name/value pairs to control the iterative algorithm
%   used by KMEANS.  Parameters are:
%
%     'cosine'       - One minus the cosine of the included angle
%              between points (treated as vectors). Each
%              row of X SHOULD be normalized to unit. If
%              the intial center matrix is provided, it
%              SHOULD also be normalized.
%
%   'Start' - Method used to choose initial cluster centroid positions,
%      sometimes known as "seeds".  Choices are:
%         {'sample'}  - Select K observations from X at random (the default)
%          'cluster' - Perform preliminary clustering phase on random 10%
%              subsample of X.  This preliminary phase is itself
%              initialized using 'sample'. An additional parameter
%              clusterMaxIter can be used to control the maximum
%              number of iterations in each preliminary clustering
%              problem.
%           matrix   - A K-by-P matrix of starting locations; or a K-by-1
%              indicate vector indicating which K points in X
%              should be used as the initial center. In this case,
%              you can pass in [] for K, and KMEANS infers K from
%              the first dimension of the matrix.
%
%   'MaxIter'    - Maximum number of iterations allowed.  Default is 100.
%
%   'Replicates' - Number of times to repeat the clustering, each with a
%          new set of initial centroids. Default is 1. If the %          initial centroids are provided, the replicate will be
%          automatically set to be 1.
%
% 'clusterMaxIter' - Only useful when 'Start' is 'cluster'. Maximum number
%            of iterations of the preliminary clustering phase.
%            Default is 10.
%
%
%    Examples:
%
%       fea = rand(500,10);
%       [label, center] = KernelKmeans(fea, 5, 'MaxIter', 50);
%
%       fea = rand(500,10);
%       [label, center] = KernelKmeans(fea, 5, 'MaxIter', 50, 'Replicates', 10);
%
%       fea = rand(500,10);
%       [label, center, bCon, sumD, D] = KernelKmeans(fea, 5, 'MaxIter', 50);
%       TSD = sum(sumD);
%
%       fea = rand(500,10);
%       initcenter = rand(5,10);
%       [label, center] = KernelKmeans(fea, 5, 'MaxIter', 50, 'Start', initcenter);
%
%       fea = rand(500,10);
%       idx=randperm(500);
%       [label, center] = KernelKmeans(fea, 5, 'MaxIter', 50, 'Start', idx(1:5));
%
%
%   See also KMEANS
%
%    [Cite] Deng Cai, "KernelKmeans: the fastest matlab implementation of
%           kmeans," Available at:
%           http://www.zjucadcg.cn/dengcai/Data/Clustering.html, 2011.
%
%   version 2.0 --December/2011
%   version 1.0 --November/2011
%
%   Written by Deng Cai (dengcai AT gmail.com)
%
%   [1] Liang Du, Peng Zhou, Lei Shi, Hanmo Wang, Mingyu Fan, Wenjian Wang, Yi-Dong Shen, 
%   Robust Multiple Kernel Kmeans, IJCAI 2015
%
%**************************************************
%     Author: Liang Du <csliangdu@gmail.com>
%     Version: 1.0
%     Last modified: 2015-04-29 04:37:36
%**************************************************



% assert(nargin < 2, 'KernelKmeans:TooFewInputs','At least two input arguments required.');

n = size(K, 1);
assert(n == size(K, 2), 'The input kernel matrix should be squared');

if ~(isscalar(k) && isnumeric(k) && isreal(k) && k > 0 && (round(k)==k))
    error('KernelKmeans:Invalid k', 'k must be a positive integer value.');
elseif n < k
    error('KernelKmeans:TooManyClusters', ...
        'K must have more rows than the number of clusters.');
end


pnames = { 'start'   'maxiter'  'replicates' 'onlinephase' 'clustermaxiter'};
dflts =  { 'sample'       []        []        'off'              []        };
[eid,errmsg,start,maxit,reps,clustermaxit] = getargs(pnames, dflts, varargin{:});
if ~isempty(eid)
    error(sprintf('KernelKmeans:%s',eid),errmsg);
end

center = [];
if ischar(start)
    startNames = {'sample','cluster'};
    j = find(strncmpi(start,startNames,length(start)));
    if length(j) > 1
        error(message('KernelKmeans:AmbiguousStart',start));
    elseif isempty(j)
        error(message('KernelKmeans:UnknownStart', start));
    elseif isempty(k)
        error('KernelKmeans:MissingK', 'You must specify the number of clusters, K.');
    end
    if j == 2
        if floor(.1*n) < 5*k
            j = 1;
        end
    end
    start = startNames{j};
elseif isnumeric(start)
    if size(start,2) == p
        center = start;
    elseif (size(start,2) == 1 || size(start,1) == 1)
        center = K(start,:);
    else
        error('KernelKmeans:MisshapedStart', 'The ''Start'' matrix must have the same number of columns as K.');
    end
    if isempty(k)
        k = size(center,1);
    elseif (k ~= size(center,1))
        error('KernelKmeans:MisshapedStart', 'The ''Start'' matrix must have K rows.');
    end
    start = 'numeric';
else
    error('KernelKmeans:InvalidStart', 'The ''Start'' parameter value must be a string or a numeric matrix or array.');
end

% The maximum iteration number is default 100
if isempty(maxit)
    maxit = 100;
end

% The maximum iteration number for preliminary clustering phase on random
% 10% subsamples is default 10
if isempty(clustermaxit)
    clustermaxit = 10;
end

% Assume one replicate
if isempty(reps)
    reps = 1;
end

bestlabel = [];
sumD = zeros(1,k);
bCon = false;

aa = full(diag(K));

for t=1:reps
    switch start
        case 'sample'
            seed = randsample(n,k);
            center = zeros(n, k);
            for i = 1:k
                center(seed(i),i) = 1;
            end
			center = bsxfun(@rdivide, center, max(sum(center), 1e-10)); % weighted indicator
        case 'cluster'
            seed = randsample(n,floor(.1*n));
            Ksubset = K(seed,:);
            Ksubset = Ksubset(:, seed');
            label_s = KernelKmeans(Ksubset, k, varargin{:}, 'start','sample', 'replicates',1 ,'MaxIter', clustermaxit);
            center = zeros(n, k);
            for i = 1:k
                center(seed(i),label_s(i)) = 1;
            end
            center = bsxfun(@rdivide, center, max(sum(center), 1e-10)); % weighted indicator
        case 'numeric'
    end
    
    last = 0;
    label=1;
    it=0;
    
    while any(label ~= last) && it<maxit
        last = label;
        
        bb = sum((K * center) .* center);
        ab = K * center;
        D = bsxfun(@plus, -2*ab, bb);
        
        [val,label] = min(D,[],2); % assign samples to the nearest centers
        ll = unique(label);
        if length(ll) < k
            %disp([num2str(k-length(ll)),' clusters dropped at iter ',num2str(it)]);
            missCluster = 1:k;
            missCluster(ll) = [];
            missNum = length(missCluster);
            
            val = aa + val;
            [~,idx] = sort(val,1,'descend');
            label(idx(1:missNum)) = missCluster;
        end
		minDist = max(aa + val, eps);
        idx = minDist < 1e-10;
        sw = .5 ./ sqrt(minDist);
        if sum(~idx) > 0
            sw(idx) = mean(sw(~idx));% without this setting, the weight of data point close to cluster center will be infinity!
        end
        sw = sw/max(sw);
		
        center = full(sparse(1:n,label,sw,n,k,n)); % indicator matrix
        center = bsxfun(@rdivide, center, max(sum(center), 1e-10)); % weighted indicator
        it=it+1;
    end
    if it<maxit
        bCon = true;
    end
    if isempty(bestlabel)
        bestlabel = label;
        bestcenter = center;
        if reps>1
            if it>=maxit
                bb = sum((K * center) .* center);
                ab = K * center;
                D = bsxfun(@plus, aa, bb) - 2*ab;
                D(D<0) = 0;
            else
                D = aa(:,ones(1,k)) + D;
                D(D<0) = 0;
            end
            D = sqrt(D);
            for j = 1:k
                sumD(j) = sum(D(label==j,j));
            end
            bestsumD = sumD;
            bestD = D;
        end
    else
        if it>=maxit
            bb = sum((K * center) .* center);
            ab = K * center;
            D = bsxfun(@plus, aa, bb) - 2*ab;
            D(D<0) = 0;
        else
            D = aa(:,ones(1,k)) + D;
            D(D<0) = 0;
        end
        D = sqrt(D);
        for j = 1:k
            sumD(j) = sum(D(label==j,j));
        end
        if sum(sumD) < sum(bestsumD)
            bestlabel = label;
            bestcenter = center;
            bestsumD = sumD;
            bestD = D;
        end
    end
end


label = bestlabel;
center = bestcenter;
if reps>1
    sumD = bestsumD;
    D = bestD;
elseif nargout > 3
    if it>=maxit
        bb = sum((K * center) .* center);
        ab = K * center;
        D = bsxfun(@plus, aa, bb) - 2*ab;
        D(D<0) = 0;
    else
        D = aa(:,ones(1,k)) + D;
        D(D<0) = 0;
    end
    D = sqrt(D);
    for j = 1:k
        sumD(j) = sum(D(label==j,j));
    end
end