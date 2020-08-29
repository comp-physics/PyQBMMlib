function f = flux_3D_27node_16mom(Nl,Ul,Vl,Wl,Nr,Ur,Vr,Wr)

f = zeros(16,1) ;
k1 = [0 1 0 0 2 1 1 0 0 0 3 0 0 4 0 0] ;
k2 = [0 0 1 0 0 1 0 2 1 0 0 3 0 0 4 0] ;
k3 = [0 0 0 1 0 0 1 0 1 2 0 0 3 0 0 4] ;
for i = 1:16  % moments number
    for j = 1:27  % nodes number
        f(i) = f(i) + Nl(j)*(Ul(j)^k1(i))*(Vl(j)^k2(i))*(Wl(j)^k3(i)) * max(Ul(j),0)...
                    + Nr(j)*(Ur(j)^k1(i))*(Vr(j)^k2(i))*(Wr(j)^k3(i)) * min(Ur(j),0);
    end
end
end