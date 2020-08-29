function [ N, U ] = three_node_hyqmom( M )
% [ N, U ] = three_node_hyqmom( M ) 
%   three-node HyQMOM with realizability checking 
%   
% input: 5 velocity moments (in this order)
%   M = [ m0, m1, m2, m3, m4]
%       
% output:
%   N = [ n1, n2, n3] weights
%   U = [ u1, u2, u3] abscissas
%
m0 = M(1); m1 = M(2); m2 = M(3); m3 = M(4); m4 = M(5); 
%
%
if isnan(m0) == 1
    display(M)
    error('corrupted moments in three_node_hyqmom')
end
%
etasmall = 1.d-10 ;  % smallest variance for computing eta
verysmall = 1.d-14 ; % smallest nonzero mass
realsmall = 1.d-14 ; % smallest nonzero rho2
qmax = 30 ; % maximum normalized skewness
% check for zero density
N = zeros(3,1) ; U = N ; 
if m0 <= verysmall
    N(2) = m0 ;
    return
end
% mean velocities
bu = m1/m0 ;
% normalized moments
d2 = m2/m0 ;
d3 = m3/m0 ;
d4 = m4/m0 ;
% central moments
% SHB: eq. (13)
c2 = d2 - bu^2 ;
c3 = d3 - 3*bu*d2 + 2*bu^3 ;
c4 = d4 - 4*bu*d3 + 6*bu^2*d2 - 3*bu^4 ;
%
% realizability check
realizable = c2*c4 - c2^3 - c3^2 ;
if c2 < 0
    if c2 < - verysmall
        warning('c2 negative in three_node_hyqmom')
        display(c2)
    end
    c2 = 0 ;
    c3 = 0 ;
    c4 = 0 ;
elseif realizable < 0
    if c2 >= etasmall
        q = c3/sqrt(c2)/c2 ; eta = c4/c2/c2 ;
        if abs(q) > verysmall
            slope = (eta - 3)/q ;
            det = 8 + slope^2 ;
            qp = 0.5*( slope + sqrt(det) ) ;
            qm = 0.5*( slope - sqrt(det) ) ;
            if sign(q) == 1
                q = qp ;
            else
                q = qm ;
            end
        else
            q = 0 ;
        end
        eta = q^2 + 1 ;
        c3 = q*sqrt(c2)*c2 ;
        c4 = eta*c2^2 ;
        if realizable < - 1.d-6
            warning('c4 too small in three_node_hyqmom')
            display(realizable)
            display(M)
        end
    else
        c3 = 0 ;
        c4 = c2^2 ;
    end
end
%
% HyQMOM parameters (scaled)
% SHB: see text following (13) for 3 node or (9) for 2 node
scale = sqrt(c2) ;
if c2 >= etasmall     
    q = c3/sqrt(c2)/c2 ; eta = c4/c2/c2 ;
else
    q = 0 ; eta = 1 ;
end
% bound skewness < qmax
if q^2 > qmax^2
    slope = (eta - 3)/q ; % move towards Gaussian moments
    q = qmax*sign(q) ;
    eta = 3 + slope*q ;
    realizable = eta - 1 - q^2 ;
    if realizable < 0
        eta = 1 + q^2 ;
    end
end

%SHB: eq. (16)
ups(1) = (q - sqrt(4*eta - 3*q^2))/2 ;
ups(2) = 0 ;
ups(3) = (q + sqrt(4*eta - 3*q^2))/2 ;

%SHB: eq. (18/19)
dem = 1/sqrt(4*eta - 3*q^2) ;
prod = - ups(1)*ups(3) ;
prod = max(prod,1+realsmall) ; % control round-off error
rho(1) = - dem/ups(1) ;
rho(2) = 1 - 1/prod ;
rho(3) =   dem/ups(3) ;
%
% return exact moment 0, 1, 2
srho = sum(rho) ;
rho = rho/srho ;
scales = sum(rho.*ups.^2)/sum(rho) ;
up = ups*scale/sqrt(scales) ;
%
% error checking
if max(isnan(rho)) == 1 || max(isnan(up)) == 1
    display(rho)
    display(up)
    error('corrupted moments in HyQMOM')
end
if min(rho) < 0
    format long
    display(rho)
    display(prod)
    display(c2)
    display(q)
    display(eta)
    warning('negative weight in HyQMOM')
    format short
end
%
N(1) = rho(1) ;
N(2) = rho(2) ;
N(3) = rho(3) ;
N = m0*N ;
%
U(1) = up(1);
U(2) = up(2);
U(3) = up(3);
U = bu + U ;
%
% check moment error
m0o = sum(N) ;
m1o = sum(N.*U) ;
m2o = sum(N.*U.^2) ;
m3o = sum(N.*U.^3) ;
m4o = sum(N.*U.^4) ;
Mout = [ m0o m1o m2o m3o m4o] ; 
err = Mout - M ;
%Merr = norm(err) ;
if err(3) > 1.d-6 && m0 > verysmall
    display(err)
    display(M)
end
%
end