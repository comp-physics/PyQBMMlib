function[n,u1,u2,ndx,w,x]=cqmom_2D_16node_12(mom,rmin,eabs,nodex,nodey)
% given 48 moments in 2D find 16 weights and abscissas using CQMOM
% SHB: meaning v1 | v2 or v1 given v2
% permutation (1,2) using 36 of the 48 optimal moments:
% * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
% {1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8};
% [0 1 0 2 1 0 3 2 1 0 4 3 2 1 0 5 4 3 2 1 0 6 5 4 3 2 1 0 7 6 5 4 3 2 1 0 7 6 5 3 2 1 7 6 3 2 7 3];
% [0 0 1 0 1 2 0 1 2 3 0 1 2 3 4 0 1 2 3 4 5 0 1 2 3 4 5 6 0 1 2 3 4 5 6 7 1 2 3 5 6 7 2 3 6 7 3 7];
% * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
%

% though the input specifies 48 moments, we only use 36 of them

% SHB: node x/y maximum number of abscissas in each direction (i think 4 is
% max possible?)
ndx=12;
if nodex == 0 || nodey == 0
    error('Cannot use 0 nodes!')
end
n=zeros(nodex*nodey,1);
u1=zeros(nodex*nodey,1);
u2=zeros(nodex*nodey,1);
small=10*eps('double'); % find smallest real
if mom(1) <=0
    return
elseif mom(1) < rmin(1)
    n(1)=mom(1);
    u1(1)=mom(2)/mom(1);
    u2(1)=mom(3)/mom(1);
    ndx=120101;
    return
elseif mom(1) < small*100
    nodex=min(2,nodex);
    nodey=min(2,nodey);
end
% 1D quadrature in v_1 direction
% SHB: moment are -> ( 00 10 20 30 40 50 60 70 )
m=[mom(1) mom(2) mom(4) mom(7) mom(11) mom(16) mom(22) mom(29)];
[w,x,nout1,werror]=Wheeler_moments_adaptive(m,nodex,rmin,eabs);
disp("x-dir weights");
disp(w);
disp("x-dir abscissa");
disp(x);
% SHB: nout1 is the number of abscissas used in first direction (4? 0:3 or less for this 7 moment system) 
ndx=ndx*100+nout1;
% SHB: ndx is an output, but i don't think it is really important for
% anything
if werror > 0
    error('1D quadrature failed on first step 1!')
end
% condition on direction v_1
% SHB: Build Vandermonde matrix 'V' using v_1 direction nodes
A=zeros(nout1,nout1);
for i=1:nout1
    for j=1:nout1
        A(i,j)=x(j)^(i-1);
    end
end
% SHB: 'W' weight matrix 
Nall=diag(w);
% matrix form moment orders {i,j} ->
% 01 02 03 04 05 06 07
% 11 12 13 14 15 16 17
% 21 22 23 24 25 26 27
% 31 32 33 34 35 36 37
momcall=[mom(3)  mom(6)  mom(10) mom(15) mom(21) mom(28) mom(36); ... 
         mom(5)  mom(9)  mom(14) mom(20) mom(27) mom(35) mom(42);...
         mom(8)  mom(13) mom(19) mom(26) mom(34) mom(41) mom(46);...
         mom(12) mom(18) mom(25) mom(33) mom(40) mom(45) mom(48)];
% SHB: first index is row, second is column (following above structure)
% SHB: only use full moments for the points that i have (nout1)
momc=momcall(1:nout1,:);
% SHB: just a placeholder for the conditional moments
momc1=momc;
% SHB: essentially meaningless line, could just use x1=x
x1=x(1:nout1);
% SHB: loop through all possible moments
for i=1:7
    % e.g. q = moments [01 11 21 31] or [0i 1i 2i 3i] where i is from loop above  
    q=momc(:,i);
    %compute conditional moments as solution to A.momc1 = q where A is the
    %V matrix
    momc1(:,i) = vanderls(x1,q,nout1); 
    err=A*momc1(:,i)-q;
    momc1(:,i)=momc1(:,i)-vanderls(x1,err,nout1);
    err=A*momc1(:,i)-q;
    maxerror=max(abs(err));
    if maxerror > small
        display(maxerror)
    end
end
% Weight matrix aka W (only the required size)
diagN=Nall(1:nout1,1:nout1);
% add the weighgt matix in to get the actual conditioned moments mc
mc=diagN\momc1;
nodeuv=0;
% for each x-direction quadrature point / abscissa:
for i=1:nout1
    % mom(1) = m_00; mc(i,:) = cond moment mc_i
    % this prepends mc(i,:) with a 1 and multiplies everything by mom_00
    m=mom(1)*[1 mc(i,:)];
    % compute weights and abscissas in y-dir for each x-dir abscissa 
    [w1,x1,nout2,werror]=Wheeler_moments_adaptive(m,nodey,rmin,eabs);
    disp(i);
    for k=1:nout2
        disp([x1(k) w1(k)]);
    end
    ndx=ndx*100+nout2;
    if werror > 0
        error('1D quadrature failed on second step!')
    end
    for j=1:nout2
        % w is the x-dir weights, w1 is the y-dir weights at the x
        % location, n is the total weight, stack them all into long vector
        n (nodeuv+j)=w(i)*w1(j)/mom(1);
        % stack all the absicssas together into a long vector
        % u1 for x-dir
        % u2 for y-dir
        u1(nodeuv+j)=x(i);
        u2(nodeuv+j)=x1(j);
    end
    nodeuv=nodeuv+nout2;
end
end