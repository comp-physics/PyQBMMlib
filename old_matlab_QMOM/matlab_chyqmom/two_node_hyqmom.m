function [ N, U ] = two_node_hyqmom( M )
% [ N, U ] = two_node_hyqmom( M ) 
%   two-node HyQMOM with realizability checking 
%   
% input: 3 velocity moments (in this order)
%   M = [ m0, m1, m2]
%       
% output:
%   N = [ n1, n2] weights
%   U = [ u1, u2] abscissas
%
m0 = M(1); m1 = M(2); m2 = M(3); 
%
%
if isnan(m0) == 1
    display(M)
    error('corrupted moments in three_node_hyqmom')
end

verysmall = 1.d-14 ; % smallest nonzero mass

N = zeros(2,1) ; U = N ; 

% mean velocities
bu = m1/m0 ;
% normalized moments
d2 = m2/m0 ;
% central moments
% SHB: eq. (13) or after eq. (10)
c2 = d2 - bu^2 ;
% HyQMOM parameters (scaled)
% SHB: see text following (13) for 3 node or (9) for 2 node
rho(1) = 0.5;
rho(2) = 0.5;

scale = sqrt(c2) ;
up(1) =  scale ;
up(2) =  -scale ; 
%
N(1) = rho(1) ;
N(2) = rho(2) ;
N = N*m0
%
U(1) = up(1);
U(2) = up(2);
U = bu + U ;
%
% check moment error
m0o = sum(N) ;
m1o = sum(N.*U) ;
m2o = sum(N.*U.^2) ;
Mout = [ m0o m1o m2o ] ; 
err = Mout - M ;

if err(3) > 1.d-6 && m0 > verysmall
    display(err)
    display(M)
end
%
end