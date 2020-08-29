function [ N, U, V ] = four_node_7mom_hycqmom_2D( M )
% [ N, U, V ] = nine_node_10mom_hycqmom_2D( M ) 
%   2-D four-node HyCQMOM with realizability checking
%   
% input: 10 bivariate velocity moments (in this order)
%   M = [ m00, m10, m01, m20, m11, m02 ]    
%
% output:
%   N = [ n1, n2, n3, n4] weights
%   U = [ u1, u2, u3, u4] u abscissas
%   V = [ v1, v2, v3, v4] v abscissas
%
m00 = M(1); m10 = M(2); m01 = M(3); m20 = M(4); m11 = M(5); m02 = M(6); 
%
if isnan(m00) == 1
    display(M)
    error('corrupted moments in nine_node_10mom_hycqmom_2D')
end
% 
N = zeros(4,1) ; U = N; V = N ;
%
csmall = 1.d-10 ;  % smallest nonzero variance
verysmall = 1.d-14 ; % smallest nonzero mass
%
% mean velocities
bu = m10/m00 ;
bv = m01/m00 ;
% normalized moments
d20 = m20/m00 ;
d11 = m11/m00 ;
d02 = m02/m00 ;
% central moments
%c00 = 1 ;
%c10 = 0 ;
%c01 = 0 ;
c20 = d20 - bu^2 ;
c11 = d11 - bu*bv ;
c02 = d02 - bv^2 ;
%
% 2 node HyQMOM with realizability checking
M1 = [1 0 c20 ] ;
[ rho, up ] = two_node_hyqmom( M1 ) ;
%
% HyQMOM - condition on u direction with c20 > 0
Vf = c11*up/c20 ;
%
% find conditional variance
mu2avg = c02 - sum(rho.*Vf.^2) ; % must be nonnegative
mu2avg = max(mu2avg,0) ;
%
mu2 = mu2avg ;
% 2-node HyQMOM
M2 = [1 0 mu2 ] ;
[ rh2, up2 ] = two_node_hyqmom( M2 ) ;
vp21 = up2(1) ;
vp22 = up2(2) ;
rho21 = rh2(1) ;
rho22 = rh2(2) ;
    
%
N(1) = rho(1)*rho21; 
N(2) = rho(1)*rho22; 
N(3) = rho(2)*rho21; 
N(4) = rho(2)*rho22; 
N = m00*N ;
%
U(1) = up(1); 
U(2) = up(1); 
U(3) = up(2);
U(4) = up(2); 
U = bu + U ;
%
V(1) = Vf(1) + vp21; 
V(2) = Vf(1) + vp22; 
V(3) = Vf(2) + vp21; 
V(4) = Vf(2) + vp22; 
V = bv + V ;
%
if max(isnan(N)) == 1 || max(isnan(U)) == 1 || max(isnan(V)) == 1
    display(N)
    display(U)
    display(V)
    error('corrupted moments in nine_node_10mom_hycqmom_2D')
end
end

