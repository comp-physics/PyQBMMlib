function F = Loss_Function(moments,ids,N,np,w,xi,xid)

F = zeros(1,N);
for ii=1:N
    flag = 0.0;
    for jj=1:np
        flag = flag +w(jj)*xi(jj)^(ids(1,ii)) *xid(jj)^(ids(2,ii));
    end
    F(ii) = F(ii) +(flag-moments(ii))^2;
end



end