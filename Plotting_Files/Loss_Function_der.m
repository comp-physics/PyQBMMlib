function dF = Loss_Function_der(moments,ids,N,np,w,xi,xid)

%we assume N = 3*np
dF = zeros(N,N);

for pp=1:np
    for ii=1:N
        
        flag = 0.0;
        for jj=1:np
            flag = flag +w(jj)*xi(jj)^(ids(1,ii)) *xid(jj)^(ids(2,ii));
        end
        
        dF(ii,pp) = 2.0*xi(pp)^(ids(1,ii))*xid(pp)^(ids(2,ii))*(flag-moments(ii));
        if (abs(ids(1,ii)) > 0.5)
            dF(ii,pp+np) = 2.0*w(pp)*ids(1,ii)*xi(pp)^(ids(1,ii)-1)*xid(pp)^(ids(2,ii))*(flag-moments(ii));
        end
        if (abs(ids(2,ii)) > 0.5)
            dF(ii,pp+2*np) = 2.0*w(pp)*xi(pp)^ids(1,ii)*ids(2,ii)*xid(pp)^(ids(2,ii)-1)*(flag-moments(ii));
        end
        
    end
end



end