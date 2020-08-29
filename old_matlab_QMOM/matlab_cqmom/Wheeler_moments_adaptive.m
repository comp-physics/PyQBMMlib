function [w,x,nout,werror,sig] = Wheeler_moments_adaptive(mom,n,rmin,eabs)

cutoff = 0; werror = 0; 
if mom(1) < 0
    display('negative number density in 1-D quadrature!') 
    werror=1;
    return
elseif mom(1)==0 
    w=0;
    x=0; 
    nout=1; 
    return
end
if n==1 || mom(1) < rmin(1)
    w=mom(1); 
    x=mom(2)/mom(1); 
    nout=1;
    return
end
% Compute modified moments equal to moments 
nu=mom;
% Construct recurrence matrix
ind=n;
a=zeros(ind,1);
b=zeros(ind,1); 
sig=zeros(2*ind+1,2*ind+1);
for i=2:(2*ind+1)
    sig(2,i)=nu(i-1); 
end
a(1)=nu(2)/nu(1); 
b(1)=0;
for k=3:(ind+1)
    for l=k:(2*ind-k+3)
        sig(k,l)=sig(k-1,l+1)-a(k-2)*sig(k-1,l)-b(k-2)*sig(k-2,l); 
    end
    a(k-1)=sig(k,k+1)/sig(k,k)-sig(k-1,k)/sig(k-1,k-1); 
    b(k-1)=sig(k,k)/sig(k-1,k-1);
end
% disp('moments');
% disp(nu(:));
% disp('my sig');
% disp(sig(3,7));
% disp('sig');
% disp(sig(:,:));
% disp('a');
% disp(a(:));
% disp('b');
% disp(b(:));
% determine maximum n using diag elements of sig 
for k=(ind+1):-1:3
    if sig(k,k) <=cutoff 
        n=k-2;
        if n==1
            w=mom(1); 
            x=mom(2)/mom(1); 
            nout=1;
            return
        end
    end
end
% compute quadrature using maximum n 
a=zeros(n,1);
b=zeros(n,1);
w=zeros(n,1);
x=zeros(n,1); 
sig=zeros(2*n+1,2*n+1);
for i=2:(2*n+1)
    sig(2,i)=nu(i-1); 
end
a(1)=nu(2)/nu(1); 
b(1)=0;
for k=3:(n+1)
    for l=k:(2*n-k+3) 
        sig(k,l)=sig(k-1,l+1)-a(k-2)*sig(k-1,l)-b(k-2)*sig(k-2,l);
    end
    a(k-1)=sig(k,k+1)/sig(k,k)-sig(k-1,k)/sig(k-1,k-1); 
    b(k-1)=sig(k,k)/sig(k-1,k-1);
end

% Check if moments are not realizable (should never happen) 
bmin=min(b);
if (bmin < 0)
    display('Moments in Wheeler_moments are not realizable!') 
    werror=1;
    return
end
% Setup Jacobi matrix for n-point quadrature, adapt n using rmax and eabs 
for n1=n:-1:1
    if n1==1
        w=mom(1); 
        x=mom(2)/mom(1); 
        nout=1;
        return
    end
    z=zeros(n1,n1);
    for i=1:(n1-1)
        z(i,i)=a(i); 
        z(i,i+1)=sqrt(b(i+1)); 
        z(i+1,i)=z(i,i+1);
    end
    z(n1,n1)=a(n1);
    % Compute weights and abscissas
    [eigenvector,eigenvalue]=eig(z); 
    w=zeros(n1,1);
    x=zeros(n1,1);
    dab=zeros(n1,1);
    mab=zeros(n1,1);
    for i=1:n1 
        w(i)=mom(1)*eigenvector(1,i)^2; 
        x(i)=eigenvalue(i,i);
    end
    for i=n1:-1:2
        dab(i)=min(abs(x(i)-x(1:i-1)));
        mab(i)=max(abs(x(i)-x(1:i-1))); 
    end
    mindab=min(dab(2:n1)); 
    maxmab=max(mab(2:n1)); 
    if n1==2
        maxmab=1; 
    end
    % check conditions that weights and abscissas must both satisfy 
    if min(w)/max(w) > rmin(n1) && mindab/maxmab > eabs
        nout=n1;
        return 
    end
end
end