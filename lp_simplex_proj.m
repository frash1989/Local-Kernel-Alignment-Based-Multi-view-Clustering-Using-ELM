function a = lp_simplex_proj(h, alpha)
% This function solve the following problem
%
%   \min_{a} \quad \sum_i a_i h_i = a^T h
%    s.t.    a_i >=0, \sum_i a_i^alpha = 1
%
% [1]Weighted Feature Subset Non-negative Matrix Factorization and Its Applications to Document Understanding. ICDM 2010
%

assert( (alpha > 0) && (alpha < 1), 'alpha should be (0, 1)');
t1 = 1 / alpha;
t2 = 1 / (alpha - 1);
t3 = alpha / (alpha - 1);

t4 = sum(h .^ t3);
t5 = (1 / t4)^t1;
t6 = h .^ t2;
a = t5 * t6;
end