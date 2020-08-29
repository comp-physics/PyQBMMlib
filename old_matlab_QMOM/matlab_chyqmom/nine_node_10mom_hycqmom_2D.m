function [ N, U, V ] = nine_node_10mom_hycqmom_2D( M )
% [ N, U, V ] = nine_node_10mom_hycqmom_2D( M ) 
%   2-D nine-node HyCQMOM with realizability checking
%   
% input: 10 bivariate velocity moments (in this order)
%   M = [ m00, m10, m01, m20, m11, m02, m30, m03, m40, m04 ]    
%
% output:
%   N = [ n1, n2, n3, n4, n5, n6, n7, n8, n9] weights
%   U = [ u1, u2, u3, u4, u5, u6, u7, u8, u9] u abscissas
%   V = [ v1, v2, v3, v4, v5, v6, v7, v8, v9] v abscissas
%
m00 = M(1); m10 = M(2); m01 = M(3); m20 = M(4); m11 = M(5); m02 = M(6); 
m30 = M(7); m03 = M(8); m40 = M(9); m04 = M(10); 
%
if isnan(m00) == 1
    display(M)
    error('corrupted moments in nine_node_10mom_hycqmom_2D')
end
% 
N = zeros(9,1) ; U = N; V = N ;
%
csmall = 1.d-10 ;  % smallest nonzero variance
verysmall = 1.d-14 ; % smallest nonzero mass
%
if m00 < verysmall
    N(4) = m00 ;
    return
end
% mean velocities
bu = m10/m00 ;
bv = m01/m00 ;
% normalized moments
d20 = m20/m00 ;
d11 = m11/m00 ;
d02 = m02/m00 ;
d30 = m30/m00 ;
d03 = m03/m00 ;
d40 = m40/m00 ;
d04 = m04/m00 ; 
% central moments
%c00 = 1 ;
%c10 = 0 ;
%c01 = 0 ;
c20 = d20 - bu^2 ;
c11 = d11 - bu*bv ;
c02 = d02 - bv^2 ;
c30 = d30 - 3*bu*d20 + 2*bu^3 ;
c03 = d03 - 3*bv*d02 + 2*bv^3 ;
c40 = d40 - 4*bu*d30 + 6*bu^2*d20 - 3*bu^4 ;
c04 = d04 - 4*bv*d03 + 6*bv^2*d02 - 3*bv^4 ;
%
% 3 node HyQMOM with realizability checking
M1 = [1 0 c20 c30 c40] ;
[ rho, up ] = three_node_hyqmom( M1 ) ;
%
% check c20 = 0 for degenerate case with 1 node in u direction
if c20 <= csmall  % rho(1) = rho(3) = 1/2 from three_node_HyQMOM
    rho(1) = 0 ;
    rho(2) = 1 ;
    rho(3) = 0 ;
    Vf = 0*up ;
    M2 = [1 0 c02 c03 c04] ;
    [ rho2, up2 ] = three_node_hyqmom( M2 ) ;
    vp21 = up2(1) ;
    vp22 = up2(2) ;
    vp23 = up2(3) ;
    rho21 = rho2(1) ;
    rho22 = rho2(2) ;
    rho23 = rho2(3) ;
else % HyQMOM - condition on u direction with c20 > 0
    Vf = c11*up/c20 ;
    %
    % find conditional variance
    mu2avg = c02 - sum(rho.*Vf.^2) ; % must be nonnegative
    mu2avg = max(mu2avg,0) ;
    %
    mu2 = mu2avg ; mu3 = 0*mu2 ; mu4 = mu2^2 ;
    %
    if mu2 > csmall
        % 3rd order moment 
        q = ( c03 - sum(rho.*Vf.^3) )/mu2^(3/2) ;
        %
        % 4th order moment with realizability check
        eta = ( c04 - sum(rho.*Vf.^4) - 6*sum(rho.*Vf.^2)*mu2 )/mu2^2 ;
        % revert to QMOM if needed
        if eta < q^2 + 1
            %display(eta)
            %display(q)
            if abs(q) > verysmall
                slope = (eta - 3)/q ;
                det = 8 + slope^2 ;
                qp = 0.5*( slope + sqrt(det) ) ;
                qm = 0.5*( slope - sqrt(det) ) ;
                %display(qp)
                %display(qm)
                if sign(q) == 1
                    q = qp ;
                else
                    q = qm ;
                end
            else
                q = 0 ;
            end
            eta = q^2 + 1 ;
            %display(eta)
            %display(q)
        end 
        mu3 = q*mu2^(3/2) ;
        mu4 = eta*mu2^2 ;
    end
    % 
    % 3-node HyQMOM
    M3 = [1 0 mu2 mu3 mu4] ;
    [ rh3, up3 ] = three_node_hyqmom( M3 ) ;
    vp21 = up3(1) ;
    vp22 = up3(2) ;
    vp23 = up3(3) ;
    rho21 = rh3(1) ;
    rho22 = rh3(2) ;
    rho23 = rh3(3) ;
end
%
N(1) = rho(1)*rho21; 
N(2) = rho(1)*rho22; 
N(3) = rho(1)*rho23;
N(4) = rho(2)*rho21; 
N(5) = rho(2)*rho22; 
N(6) = rho(2)*rho23;
N(7) = rho(3)*rho21; 
N(8) = rho(3)*rho22;
N(9) = rho(3)*rho23;
N = m00*N ;
%
U(1) = up(1); 
U(2) = up(1); 
U(3) = up(1);
U(4) = up(2); 
U(5) = up(2); 
U(6) = up(2);
U(7) = up(3); 
U(8) = up(3);
U(9) = up(3);
U = bu + U ;
%
V(1) = Vf(1) + vp21; 
V(2) = Vf(1) + vp22; 
V(3) = Vf(1) + vp23; 
V(4) = Vf(2) + vp21; 
V(5) = Vf(2) + vp22; 
V(6) = Vf(2) + vp23;
V(7) = Vf(3) + vp21; 
V(8) = Vf(3) + vp22;
V(9) = Vf(3) + vp23;
V = bv + V ;
%
if max(isnan(N)) == 1 || max(isnan(U)) == 1 || max(isnan(V)) == 1
    display(N)
    display(U)
    display(V)
    error('corrupted moments in nine_node_10mom_hycqmom_2D')
end
end

