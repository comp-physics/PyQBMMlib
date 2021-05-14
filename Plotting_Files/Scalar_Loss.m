function [F,a] = Scalar_Loss(xx)
global MC_moments ids N np;

F = 0.0;
for ii=1:N
    flag = 0.0;
    for jj=1:np
        flag = flag +xx(jj)*xx(jj+np)^(ids(1,ii)) *xx(jj+2*np)^(ids(2,ii));
    end
    F = F +(flag-MC_moments(ii))^2;
end
F = sqrt(F);

end