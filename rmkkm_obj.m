function obj = rmkkm_obj(Ks, kw, label, center)
[n, k] = size(center);
m = size(Ks,3);
Ka = zeros(n);
for i = 1:m
    Ka = kw(i) * Ks(:,:,i) + Ka;
end
bb = sum((Ka * center) .* center);
ab = Ka * center;
D = bsxfun(@plus, -2*ab, bb);
Z = full(sparse(1:n,label,ones(n,1),n,k,n)); % indicator matrix
dist = sum(Z.*D, 2) + diag(Ka);
dist = sqrt(max(dist, eps));
obj = sum(dist);