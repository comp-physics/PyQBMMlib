function [ N, U, V, W ] = twentyseven_node_16mom_hycqmom_3D( M )
% [ N, U, V, W ] = twentyseven_node_16mom_hycqmom( M ) 
%   3-D twentyseven-node HyCQMOM with realizability checking 
%   
%   
% input: 16 trivariate velocity moments (in this order)
%   M = [ m000, m100, m010, m001, m200, m110, m101, m020, m011, m002, ...
%         m300, m030, m003, m400, m040, m004]
%       
% output:
%   N = [ n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, ..., n27] weights
%   U = [ u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, ..., u27] u abscissas
%   V = [ v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, ..., v27] v abscissas
%   W = [ w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, ..., w27] w abscissas
%
m000 = M(1); m100 = M(2); m010 = M(3); m001 = M(4); 
m200 = M(5); m110 = M(6); m101 = M(7); m020 = M(8); m011 = M(9); m002 = M(10); 
m300 = M(11); m030 = M(12); m003 = M(13); m400 = M(14); m040 = M(15); m004 = M(16);
%
if isnan(m000) == 1
    display(M)
    error('corrupted moments in twentyseven_node_16mom_hycqmom_3D')
end
%
small = 1.d-10 ; % smallest correlated moments
isosmall = 1.d-14 ; % smallest anisotropic moments
csmall = 1.d-10 ;  % smallest nonzero variance
verysmall = 1.d-14 ; % smallest nonzero mass
wsmall = 1.d-4 ; % smallest nonzero correlation value
% check for zero density
N = zeros(27,1) ; U = N ; V = N ; W = N ;
if m000 <= verysmall
    N(14) = m000 ;
    return
end
% mean velocities
bu = m100/m000 ;
bv = m010/m000 ;
bw = m001/m000 ;
if m000 <= isosmall % if density <= small, central moments are isotropic
    d200 = m200/m000 ;
    d020 = m020/m000 ;
    d002 = m002/m000 ;
    d300 = m300/m000 ;
    d030 = m030/m000 ;
    d003 = m003/m000 ;
    d400 = m400/m000 ;
    d040 = m040/m000 ;
    d004 = m004/m000 ;
    %
%     c000 = 1 ;
%     c100 = 0 ;
%     c010 = 0 ;
%     c001 = 0 ;
    c200 = d200 - bu^2 ;
    c020 = d020 - bv^2 ;
    c002 = d002 - bw^2 ;
    c300 = d300 - 3*bu*d200 + 2*bu^3 ;
    c030 = d030 - 3*bv*d020 + 2*bv^3 ;
    c003 = d003 - 3*bw*d002 + 2*bw^3 ;
    c400 = d400 - 4*bu*d300 + 6*bu^2*d200 - 3*bu^4 ;
    c040 = d040 - 4*bv*d030 + 6*bv^2*d020 - 3*bv^4 ;
    c004 = d004 - 4*bw*d003 + 6*bw^2*d002 - 3*bw^4 ;
    % isotropic
    c110 = 0; c101 = 0; c011 = 0; 
else
    % normalized moments
    d200 = m200/m000 ;
    d110 = m110/m000 ;
    d101 = m101/m000 ;
    d020 = m020/m000 ;
    d011 = m011/m000 ;
    d002 = m002/m000 ;
    d300 = m300/m000 ;
    d030 = m030/m000 ;
    d003 = m003/m000 ;
    d400 = m400/m000 ;
    d040 = m040/m000 ;
    d004 = m004/m000 ;
    % central moments
%     c000 = 1 ;
%     c100 = 0 ;
%     c010 = 0 ;
%     c001 = 0 ;
    c200 = d200 - bu^2 ;
    c110 = d110 - bu*bv ;
    c101 = d101 - bu*bw ;
    c020 = d020 - bv^2 ;
    c011 = d011 - bv*bw ;
    c002 = d002 - bw^2 ;
    c300 = d300 - 3*bu*d200 + 2*bu^3 ;
    c030 = d030 - 3*bv*d020 + 2*bv^3 ;
    c003 = d003 - 3*bw*d002 + 2*bw^3 ;
    c400 = d400 - 4*bu*d300 + 6*bu^2*d200 - 3*bu^4 ;
    c040 = d040 - 4*bv*d030 + 6*bv^2*d020 - 3*bv^4 ;
    c004 = d004 - 4*bw*d003 + 6*bw^2*d002 - 3*bw^4 ;
end
% check realizability of univariate moments
if c200 <= 0
    c200 = 0 ;
    c300 = 0 ;
    c400 = 0 ;
end
if c200*c400 < c200^3 + c300^2
    q = c300/c200^(3/2) ;
    eta = c400/c200^2 ;
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
    c300 = q*c200^(3/2) ;
    c400 = eta*c200^2 ;
end
if c020 <= 0
    c020 = 0 ;
    c030 = 0 ;
    c040 = 0 ;
end
if c020*c040 < c020^3 + c030^2
    q = c030/c020^(3/2) ;
    eta = c040/c020^2 ;
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
    c030 = q*c020^(3/2) ;
    c040 = eta*c020^2 ;
end
if c002 <= 0
    c002 = 0 ;
    c003 = 0 ;
    c004 = 0 ;
end
if c002*c004 < c002^3 + c003^2
    q = c003/c002^(3/2) ;
    eta = c004/c002^2 ;
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
    c003 = q*c002^(3/2) ;
    c004 = eta*c002^2 ;
end
% HyQMOM for first direction
M1 = [ 1 0 c200 c300 c400 ] ;
[rho, up ] = three_node_hyqmom( M1 ) ;
% set default values for second direction
rho11 = 0 ;
rho12 = 1 ;
rho13 = 0 ;
rho21 = 0 ;
%rho22 = 1 ;
rho23 = 0 ;
rho31 = 0 ;
rho32 = 1 ;
rho33 = 0 ;
vp11 = 0 ;
vp12 = 0 ;
vp13 = 0 ;
vp21 = 0 ;
vp22 = 0 ;
vp23 = 0 ;
vp31 = 0 ;
vp32 = 0 ;
vp33 = 0 ;
%
Vf = zeros(3,1) ;
% set default values for third direction
rho111 = 0 ;
rho112 = 1 ;
rho113 = 0 ;
rho121 = 0 ;
rho122 = 1 ;
rho123 = 0 ;
rho131 = 0 ;
rho132 = 1 ;
rho133 = 0 ;
rho211 = 0 ;
rho212 = 1 ;
rho213 = 0 ;
rho221 = 0 ;
rho222 = 1 ;
rho223 = 0 ;
rho231 = 0 ;
rho232 = 1 ;
rho233 = 0 ;
rho311 = 0 ;
rho312 = 1 ;
rho313 = 0 ;
rho321 = 0 ;
rho322 = 1 ;
rho323 = 0 ;
rho331 = 0 ;
rho332 = 1 ;
rho333 = 0 ;
wp111 = 0 ;
wp112 = 0 ;
wp113 = 0 ;
wp121 = 0 ;
wp122 = 0 ;
wp123 = 0 ;
wp131 = 0 ;
wp132 = 0 ;
wp133 = 0 ;
wp211 = 0 ;
wp212 = 0 ;
wp213 = 0 ;
wp221 = 0 ;
wp222 = 0 ;
wp223 = 0 ;
wp231 = 0 ;
wp232 = 0 ;
wp233 = 0 ;
wp311 = 0 ;
wp312 = 0 ;
wp313 = 0 ;
wp321 = 0 ;
wp322 = 0 ;
wp323 = 0 ;
wp331 = 0 ;
wp332 = 0 ;
wp333 = 0 ;
%
Wf = zeros(3,3) ;
%
if c200 <= csmall  % case 1
    if c020 <= csmall % case 1a
        % call 3-node for 1-D HyQMOM in w statistics
        M0 = [ 1, 0, c002, c003, c004] ;
        [ N0, W0 ] = three_node_hyqmom( M0 ) ;
        % N0 has correct weights summing to 1, adjust rho's accordingly
        rho(1) = 0 ;
        rho(2) = 1 ;
        rho(3) = 0 ;
        rho22 = 1 ;
        rho221 = N0(1) ;
        rho222 = N0(2) ;
        rho223 = N0(3) ;
        %
        up = 0*up ;
        %
        wp221 = W0(1) ;
        wp222 = W0(2) ;
        wp223 = W0(3) ;
    else % case 1b
        % call 9-node for 2-D HyCQMOM in v,w statistics
        M1 = [ 1, 0, 0, c020, c011, c002, c030, c003, c040, c004] ;
        [ N1, V1, W1 ] = nine_node_10mom_hycqmom_2D( M1 ) ;
        % N1 has correct weights summing to 1, adjust rho's accordingly
        rho(1) = 0 ;
        rho(2) = 1 ;
        rho(3) = 0 ;
        rho12 = 0 ;
        rho21 = 1 ;
        rho22 = 1 ;
        rho23 = 1 ;
        rho31 = 0 ;
        rho211 = N1(1) ;
        rho212 = N1(2) ;
        rho213 = N1(3) ;
        rho221 = N1(4) ;
        rho222 = N1(5) ;
        rho223 = N1(6) ;
        rho231 = N1(7) ;
        rho232 = N1(8) ;
        rho233 = N1(9) ;
        %
        up = 0*up ;
        %
        vp21 = V1(1) ;
        vp22 = V1(5) ;
        vp23 = V1(9) ;
        %
        wp211 = W1(1) ;
        wp212 = W1(2) ;
        wp213 = W1(3) ;
        wp221 = W1(4) ;
        wp222 = W1(5) ;
        wp223 = W1(6) ;
        wp231 = W1(7) ;
        wp232 = W1(8) ;
        wp233 = W1(9) ;
    end
elseif c020 <= csmall % case 2a & case 3b (extreme for v)
    % call 9-node for 2-D HyCQMOM in u,w statistics
    M2 = [ 1, 0, 0, c200, c101, c002, c300, c003, c400, c004] ;
    [ N2, U2, W2 ] = nine_node_10mom_hycqmom_2D( M2 ) ;
    % N1 has correct weights summing to 1, adjust rho's accordingly
    rho(1) = 1 ;
    rho(2) = 1 ;
    rho(3) = 1 ;
    rho12 = 1 ;
    rho22 = 1 ;
    rho32 = 1 ;
    rho121 = N2(1) ;
    rho122 = N2(2) ;
    rho123 = N2(3) ;
    rho221 = N2(4) ;
    rho222 = N2(5) ;
    rho223 = N2(6) ;
    rho321 = N2(7) ;
    rho322 = N2(8) ;
    rho323 = N2(9) ;
    %
    up(1) = U2(1) ;
    up(2) = U2(5) ;
    up(3) = U2(9) ;
    %
    wp121 = W2(1) ;
    wp122 = W2(2) ;
    wp123 = W2(3) ;
    wp221 = W2(4) ;
    wp222 = W2(5) ;
    wp223 = W2(6) ;
    wp321 = W2(7) ;
    wp322 = W2(8) ;
    wp323 = W2(9) ;
elseif c002 <= csmall  % case 2 & 3 (extreme for w)
    % call 9-node for 2-D HyCQMOM in u,v statistics
    M3 = [ 1, 0, 0, c200, c110, c020, c300, c030, c400, c040] ;
    [ N3, U3, V3 ] = nine_node_10mom_hycqmom_2D( M3 ) ;
    % N3 has correct weights summing to 1, adjust rho's accordingly
    rho(1)= 1 ;
    rho(2)= 1 ;
    rho(3)= 1 ;
    rho11 = N3(1) ;
    rho12 = N3(2) ;
    rho13 = N3(3) ;
    rho21 = N3(4) ;
    rho22 = N3(5) ;
    rho23 = N3(6) ;
    rho31 = N3(7) ;
    rho32 = N3(8) ;
    rho33 = N3(9) ;
    %
    up(1)= U3(1) ;
    up(2)= U3(5) ;
    up(3)= U3(9) ;
    %
    vp11 = V3(1) ;
    vp12 = V3(2) ;
    vp13 = V3(3) ;
    vp21 = V3(4) ;
    vp22 = V3(5) ;
    vp23 = V3(6) ;
    vp31 = V3(7) ;
    vp32 = V3(8) ;
    vp33 = V3(9) ;
else % all 3 variances are nonzero
    % call 9-node for 2-D HyCQMOM in u,v statistics
    M4 = [ 1, 0, 0, c200, c110, c020, c300, c030, c400, c040] ;
    [ N4, ~, V4 ] = nine_node_10mom_hycqmom_2D( M4 ) ;
    % N1 has correct weights summing to 1, adjust rho's accordingly
    rho11 = N4(1)/(N4(1)+N4(2)+N4(3)) ;
    rho12 = N4(2)/(N4(1)+N4(2)+N4(3)) ;
    rho13 = N4(3)/(N4(1)+N4(2)+N4(3)) ;
    rho21 = N4(4)/(N4(4)+N4(5)+N4(6)) ;
    rho22 = N4(5)/(N4(4)+N4(5)+N4(6)) ;
    rho23 = N4(6)/(N4(4)+N4(5)+N4(6)) ;
    rho31 = N4(7)/(N4(7)+N4(8)+N4(9)) ;
    rho32 = N4(8)/(N4(7)+N4(8)+N4(9)) ;
    rho33 = N4(9)/(N4(7)+N4(8)+N4(9)) ;
    %
    Vf(1) = rho11*V4(1)+rho12*V4(2)+rho13*V4(3) ;
    Vf(2) = rho21*V4(4)+rho22*V4(5)+rho23*V4(6) ;
    Vf(3) = rho31*V4(7)+rho32*V4(8)+rho33*V4(9) ;
    %
    vp11 = V4(1)-Vf(1) ;
    vp12 = V4(2)-Vf(1) ;
    vp13 = V4(3)-Vf(1) ;
    vp21 = V4(4)-Vf(2) ;
    vp22 = V4(5)-Vf(2) ;
    vp23 = V4(6)-Vf(2) ;
    vp31 = V4(7)-Vf(3) ;
    vp32 = V4(8)-Vf(3) ;
    vp33 = V4(9)-Vf(3) ;
    %
    % compute W for third direction using NB = 3 basis matrices
    scale1 = sqrt(c200) ;
    scale2 = sqrt(c020) ;
    Rho1 = diag(rho) ;
    Rho2 = [rho11 rho12 rho13 ; rho21 rho22 rho23 ; rho31 rho32 rho33] ;
    Vp2 = [vp11 vp12 vp13 ; vp21 vp22 vp23 ; vp31 vp32 vp33] ;
    Vp2s = Vp2/scale2 ;
    %
    RAB = Rho1*Rho2 ;
    UAB = [up up up] ;
    UABs = UAB/scale1 ;
    VAB = Vp2 + diag(Vf)*ones(3) ;
    VABs = VAB/scale2 ;
    %
    % coefficient vector for scaled central moments
    C01 = RAB.*VABs ;
    % 3 basis matrices for Wf 
    Vc0 = ones(3) ;
    Vc1 = UABs ;
    Vc2 = Vp2s ;
    % coefficient matrix for Wf
    A1 = sum(sum(C01.*Vc1)) ; 
    A2 = sum(sum(C01.*Vc2)) ; 
    % scaled central moments
    c101s = c101/scale1 ; c011s = c011/scale2 ; 
    %
    if c101s^2 >= c002*(1 - small) % check for w = a*u
        c101s = sign(c101s)*sqrt(c002) ;
    elseif c011s^2 >= c002*(1 - small) % check for w = a*v
        c110s = c110/scale1/scale2 ;
        c011s = sign(c011s)*sqrt(c002) ;
        c101s = c110s*c011s ;
    end 
    %
    b0 = 0; b1 = c101s; b2 = 0;
    % solve linear system for b2
    if A2 > wsmall
        b2 = ( c011s - A1*b1 )/A2 ;
    end
    %
    Wf = b0*Vc0 + b1*Vc1 + b2*Vc2 ; 
    %
    % HyQMOM in w direction
    %
    % first find conditional variances 
    SUM002 = sum(sum(RAB.*Wf.^2)) ;
    mu2 = c002 - SUM002 ;
    mu2 = max(0,mu2) ;
    %
    q = 0 ; eta = 1 ;
    if mu2 > csmall
        % 3rd order moments are scaled with q
        SUM1 = mu2^(3/2) ;
        SUM3 = sum(sum(RAB.*Wf.^3)) ;
        q = ( c003 - SUM3 )/SUM1 ;
        %
        % 4th order moments are scaled with eta
        SUM2 = mu2^2 ;
        SUM4 = sum(sum(RAB.*Wf.^4)) + 6*SUM002*mu2 ;
        eta = ( c004 - SUM4 )/SUM2 ;
        % realizability check, revert to QMOM if needed
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
    end
    mu3 = q*mu2^(3/2) ;
    mu4 = eta*mu2^2 ;
    %
    % HyQMOM for conditional moments is w direction
    M5 = [1 0 mu2 mu3 mu4] ;
    [ rh11, up11 ] = three_node_hyqmom( M5 ) ;
    rho111 = rh11(1) ;
    rho112 = rh11(2) ;
    rho113 = rh11(3) ;
    wp111 = up11(1) ;
    wp112 = up11(2) ;
    wp113 = up11(3) ;
    %
    rh12 = rh11 ;
    up12 = up11 ;
    rho121 = rh12(1) ;
    rho122 = rh12(2) ;
    rho123 = rh12(3) ;
    wp121 = up12(1) ;
    wp122 = up12(2) ;
    wp123 = up12(3) ;
    %
    rh13 = rh11 ;
    up13 = up11 ;
    rho131 = rh13(1) ;
    rho132 = rh13(2) ;
    rho133 = rh13(3) ;
    wp131 = up13(1) ;
    wp132 = up13(2) ;
    wp133 = up13(3) ;
    % 
    rh21 = rh11 ;
    up21 = up11 ;
    wp211 = up21(1) ;
    wp212 = up21(2) ;
    wp213 = up21(3) ;
    rho211 = rh21(1) ;
    rho212 = rh21(2) ;
    rho213 = rh21(3) ;
    %
    rh22 = rh11 ;
    up22 = up11 ;
    wp221 = up22(1) ;
    wp222 = up22(2) ;
    wp223 = up22(3) ;
    rho221 = rh22(1) ;
    rho222 = rh22(2) ;
    rho223 = rh22(3) ;
    %
    rh23 = rh11 ;
    up23 = up11 ;
    wp231 = up23(1) ;
    wp232 = up23(2) ;
    wp233 = up23(3) ;
    rho231 = rh23(1) ;
    rho232 = rh23(2) ;
    rho233 = rh23(3) ;
    %
    rh31 = rh11 ;
    up31 = up11 ;
    rho311 = rh31(1) ;
    rho312 = rh31(2) ;
    rho313 = rh31(3) ;
    wp311 = up31(1) ;
    wp312 = up31(2) ;
    wp313 = up31(3) ;
    %
    rh32 = rh11 ;
    up32 = up11 ;
    rho321 = rh32(1) ;
    rho322 = rh32(2) ;
    rho323 = rh32(3) ;
    wp321 = up32(1) ;
    wp322 = up32(2) ;
    wp323 = up32(3) ;
    %
    rh33 = rh11 ;
    up33 = up11 ;
    rho331 = rh33(1) ;
    rho332 = rh33(2) ;
    rho333 = rh33(3) ;
    wp331 = up33(1) ;
    wp332 = up33(2) ;
    wp333 = up33(3) ;
end
%
N(1) = rho(1)*rho11*rho111;
N(2) = rho(1)*rho11*rho112;
N(3) = rho(1)*rho11*rho113;
N(4) = rho(1)*rho12*rho121;
N(5) = rho(1)*rho12*rho122;
N(6) = rho(1)*rho12*rho123;
N(7) = rho(1)*rho13*rho131;
N(8) = rho(1)*rho13*rho132;
N(9) = rho(1)*rho13*rho133;
N(10)= rho(2)*rho21*rho211;
N(11)= rho(2)*rho21*rho212;
N(12)= rho(2)*rho21*rho213;
N(13)= rho(2)*rho22*rho221;
N(14)= rho(2)*rho22*rho222; 
N(15)= rho(2)*rho22*rho223;
N(16)= rho(2)*rho23*rho231;
N(17)= rho(2)*rho23*rho232;
N(18)= rho(2)*rho23*rho233;
N(19)= rho(3)*rho31*rho311;
N(20)= rho(3)*rho31*rho312;
N(21)= rho(3)*rho31*rho313;
N(22)= rho(3)*rho32*rho321;
N(23)= rho(3)*rho32*rho322;
N(24)= rho(3)*rho32*rho323;
N(25)= rho(3)*rho33*rho331;
N(26)= rho(3)*rho33*rho332;
N(27)= rho(3)*rho33*rho333;
N = m000*N ;
%
U(1) = up(1);
U(2) = up(1);
U(3) = up(1);
U(4) = up(1);
U(5) = up(1);
U(6) = up(1);
U(7) = up(1);
U(8) = up(1);
U(9) = up(1);
U(10)= up(2); 
U(11)= up(2); 
U(12)= up(2); 
U(13)= up(2);
U(14)= up(2);
U(15)= up(2); 
U(16)= up(2); 
U(17)= up(2); 
U(18)= up(2); 
U(19)= up(3);
U(20)= up(3);
U(21)= up(3);
U(22)= up(3);
U(23)= up(3);
U(24)= up(3);
U(25)= up(3);
U(26)= up(3);
U(27)= up(3);
U = bu + U ;
%
V(1) = Vf(1)+vp11;
V(2) = Vf(1)+vp11;
V(3) = Vf(1)+vp11;
V(4) = Vf(1)+vp12;
V(5) = Vf(1)+vp12;
V(6) = Vf(1)+vp12;
V(7) = Vf(1)+vp13;
V(8) = Vf(1)+vp13;
V(9) = Vf(1)+vp13;
V(10)= Vf(2)+vp21;
V(11)= Vf(2)+vp21;
V(12)= Vf(2)+vp21;
V(13)= Vf(2)+vp22; 
V(14)= Vf(2)+vp22; 
V(15)= Vf(2)+vp22;
V(16)= Vf(2)+vp23; 
V(17)= Vf(2)+vp23;
V(18)= Vf(2)+vp23;
V(19)= Vf(3)+vp31;
V(20)= Vf(3)+vp31;
V(21)= Vf(3)+vp31;
V(22)= Vf(3)+vp32;
V(23)= Vf(3)+vp32;
V(24)= Vf(3)+vp32;
V(25)= Vf(3)+vp33;
V(26)= Vf(3)+vp33;
V(27)= Vf(3)+vp33;
V = bv + V ;
%
W(1) = Wf(1,1)+wp111;
W(2) = Wf(1,1)+wp112;
W(3) = Wf(1,1)+wp113;
W(4) = Wf(1,2)+wp121;
W(5) = Wf(1,2)+wp122;
W(6) = Wf(1,2)+wp123;
W(7) = Wf(1,3)+wp131;
W(8) = Wf(1,3)+wp132;
W(9) = Wf(1,3)+wp133;
W(10)= Wf(2,1)+wp211; 
W(11)= Wf(2,1)+wp212;
W(12)= Wf(2,1)+wp213;
W(13)= Wf(2,2)+wp221; 
W(14)= Wf(2,2)+wp222; 
W(15)= Wf(2,2)+wp223;
W(16)= Wf(2,3)+wp231;
W(17)= Wf(2,3)+wp232;
W(18)= Wf(2,3)+wp233;
W(19)= Wf(3,1)+wp311;
W(20)= Wf(3,1)+wp312;
W(21)= Wf(3,1)+wp313;
W(22)= Wf(3,2)+wp321;
W(23)= Wf(3,2)+wp322;
W(24)= Wf(3,2)+wp323;
W(25)= Wf(3,3)+wp331;
W(26)= Wf(3,3)+wp332;
W(27)= Wf(3,3)+wp333;
W = bw + W ;
%
end