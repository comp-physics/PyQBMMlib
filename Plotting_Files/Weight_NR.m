function [w, xi, xid,dF,F] = Weight_NR(moments,ids,N,np)

% w = zeros(1,np);
% xi = zeros(1,np);
% xid = zeros(1,np);

w   = rand(1,np);
xi  = rand(1,np);
xid = rand(1,np);


for nn=1:10
    F = Loss_Function(moments,ids,N,np,w,xi,xid);
    dF = Loss_Function_der(moments,ids,N,np,w,xi,xid);
    
    dX = linsolve(dF,F');
    
    if (min(F) > 10^(-8))
        w_old   = w;
        xi_old  = xi;
        xid_old = xid;
        
        for ii=1:np
            w(ii)   = w_old(ii)   -dX(ii);
            xi(ii)  = xi_old(ii)  -dX(ii+np);
            xid(ii) = xid_old(ii) -dX(ii+2*np);
        end
        
    end
    


end


end