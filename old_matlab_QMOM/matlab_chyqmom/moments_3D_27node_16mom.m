function mom = moments_3D_27node_16mom(n,u1,u2,u3)
mom = zeros(16,1) ;
k1 = [0 1 0 0 2 1 1 0 0 0 3 0 0 4 0 0] ;
k2 = [0 0 1 0 0 1 0 2 1 0 0 3 0 0 4 0] ;
k3 = [0 0 0 1 0 0 1 0 1 2 0 0 3 0 0 4] ;
for i = 1:16  % moments number
    for j = 1:27 % node number
        mom(i) = mom(i) + n(j)*(u1(j)^k1(i))*(u2(j)^k2(i))*(u3(j)^k3(i)) ;
    end
end
end