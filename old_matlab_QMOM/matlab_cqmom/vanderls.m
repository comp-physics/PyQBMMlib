function [w] = vanderls(x,q,n)
% [w] = vanderls(x,q,n) solves Vandermonde linear system x^(k-1) w = q
%
% Input:
%   x = vector of Vandermonde abscissas
%   q = vector on right-hand side
%   n = size of system
%
% Output:
%   w = solution to linear system
%
c = zeros(n,1) ;
w = zeros(n,1) ;
if n == 1
    w(1) = q(1) ;
else
    c(n) = -x(1) ;
    for i = 2:n
        xx = -x(i) ;
        for j = n+1-i:n-1
            c(j) = c(j) + xx*c(j+1) ;
        end
        c(n) = c(n) + xx ;
    end
    for i = 1:n
        xx = x(i) ;
        t = 1 ;
        b = 1 ;
        s = q(n) ;
        for k = n:-1:2
            b = c(k) + xx*b ;
            s = s + q(k-1)*b ;
            t = xx*t + b ;
        end
        w(i) = s/t ;
    end
end
end