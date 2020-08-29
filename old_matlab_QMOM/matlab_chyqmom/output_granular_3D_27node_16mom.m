
function [] = output_granular_3D_27node_16mom(M1,N1,U1,V1,W1,Ycell,t,Ny,movie,k)
% output the plots of data
%
Umax = 2 ;
rhomax = 2 ;
subplot(2,4,1) ;
xx = zeros(Ny,1) ;
yy1 = zeros(Ny,1) ;  
yy2 = zeros(Ny,1) ; 
yy3 = zeros(Ny,1) ;  
yy4 = zeros(Ny,1) ;  
yy5 = zeros(Ny,1) ; 
yy6 = zeros(Ny,1) ;  
yy7 = zeros(Ny,1) ;  
yy8 = zeros(Ny,1) ; 
yy9 = zeros(Ny,1) ;  
yy10 = zeros(Ny,1) ; 
yy11 = zeros(Ny,1) ;  
yy12 = zeros(Ny,1) ;  
yy13 = zeros(Ny,1) ;
yy14 = zeros(Ny,1) ;  
yy15 = zeros(Ny,1) ; 
yy16 = zeros(Ny,1) ;  
yy17 = zeros(Ny,1) ;  
yy18 = zeros(Ny,1) ; 
yy19 = zeros(Ny,1) ;  
yy20 = zeros(Ny,1) ;  
yy21 = zeros(Ny,1) ; 
yy22 = zeros(Ny,1) ;  
yy23 = zeros(Ny,1) ; 
yy24 = zeros(Ny,1) ;  
yy25 = zeros(Ny,1) ;  
yy26 = zeros(Ny,1) ;
yy27 = zeros(Ny,1) ;
for i=1:Ny
    xx(i) = Ycell(i) ;
    yy1(i) = N1(1,i) ;
    yy2(i) = N1(2,i) ;
    yy3(i) = N1(3,i) ;
    yy4(i) = N1(4,i) ;
    yy5(i) = N1(5,i) ;
    yy6(i) = N1(6,i) ;
    yy7(i) = N1(7,i) ;
    yy8(i) = N1(8,i) ;
    yy9(i) = N1(9,i) ;
    yy10(i) = N1(10,i) ;
    yy11(i) = N1(11,i) ;
    yy12(i) = N1(12,i) ;
    yy13(i) = N1(13,i) ;
    yy14(i) = N1(14,i) ;
    yy15(i) = N1(15,i) ;
    yy16(i) = N1(16,i) ;
    yy17(i) = N1(17,i) ;
    yy18(i) = N1(18,i) ;
    yy19(i) = N1(19,i) ;
    yy20(i) = N1(20,i) ;
    yy21(i) = N1(21,i) ;
    yy22(i) = N1(22,i) ;
    yy23(i) = N1(23,i) ;
    yy24(i) = N1(24,i) ;
    yy25(i) = N1(25,i) ;
    yy26(i) = N1(26,i) ;
    yy27(i) = N1(27,i) ;
end
plot( xx, yy1,'bo', xx, yy2,'go', xx, yy3,'bo', xx, yy4,'go', xx, yy5,'ro', ...
      xx, yy6,'go', xx, yy7,'bo', xx, yy8,'go', xx, yy9,'bo', ...
      xx, yy10,'gp', xx, yy11,'rp', xx, yy12,'gp', xx, yy13,'rp', xx, yy14,'k*', ...
      xx, yy15,'rp', xx, yy16,'gp', xx, yy17,'rp', xx, yy18,'gp', ...
      xx, yy19,'bs', xx, yy20,'gs', xx, yy21,'bs', xx, yy22,'gs', xx, yy23,'rs', ...
      xx, yy24,'gs', xx, yy25,'bs', xx, yy26,'gs', xx, yy27,'bs', ...
      'Linewidth',1.0, 'MarkerSize',4.0 )
title('Normalized Weights') ;
xlim([0 1]) ;
ylim([0 rhomax/4]) ;

subplot(2,4,2) ;
for i=1:Ny
    xx(i) = Ycell(i) ;
    yy1(i) = U1(1,i) ;
    yy2(i) = U1(2,i) ;
    yy3(i) = U1(3,i) ;
    yy4(i) = U1(4,i) ;
    yy5(i) = U1(5,i) ;
    yy6(i) = U1(6,i) ;
    yy7(i) = U1(7,i) ;
    yy8(i) = U1(8,i) ;
    yy9(i) = U1(9,i) ;
    yy10(i) = U1(10,i) ;
    yy11(i) = U1(11,i) ;
    yy12(i) = U1(12,i) ;
    yy13(i) = U1(13,i) ;
    yy14(i) = U1(14,i) ;
    yy15(i) = U1(15,i) ;
    yy16(i) = U1(16,i) ;
    yy17(i) = U1(17,i) ;
    yy18(i) = U1(18,i) ;
    yy19(i) = U1(19,i) ;
    yy20(i) = U1(20,i) ;
    yy21(i) = U1(21,i) ;
    yy22(i) = U1(22,i) ;
    yy23(i) = U1(23,i) ;
    yy24(i) = U1(24,i) ;
    yy25(i) = U1(25,i) ;
    yy26(i) = U1(26,i) ;
    yy27(i) = U1(27,i) ;
end
plot( xx, yy1,'bo', xx, yy2,'go', xx, yy3,'bo', xx, yy4,'go', xx, yy5,'ro', ...
      xx, yy6,'go', xx, yy7,'bo', xx, yy8,'go', xx, yy9,'bo', ...
      xx, yy10,'gp', xx, yy11,'rp', xx, yy12,'gp', xx, yy13,'rp', xx, yy14,'k*', ...
      xx, yy15,'rp', xx, yy16,'gp', xx, yy17,'rp', xx, yy18,'gp', ...
      xx, yy19,'bs', xx, yy20,'gs', xx, yy21,'bs', xx, yy22,'gs', xx, yy23,'rs', ...
      xx, yy24,'gs', xx, yy25,'bs', xx, yy26,'gs', xx, yy27,'bs', ...
      'Linewidth',1.0, 'MarkerSize',4.0 )
title('u abscissas') ;
xlim([0 1]) ;
ylim([-Umax Umax]) ;

subplot(2,4,3) ;
for i=1:Ny
    xx(i) = Ycell(i) ;
    yy1(i) = V1(1,i) ;
    yy2(i) = V1(2,i) ;
    yy3(i) = V1(3,i) ;
    yy4(i) = V1(4,i) ;
    yy5(i) = V1(5,i) ;
    yy6(i) = V1(6,i) ;
    yy7(i) = V1(7,i) ;
    yy8(i) = V1(8,i) ;
    yy9(i) = V1(9,i) ;
    yy10(i) = V1(10,i) ;
    yy11(i) = V1(11,i) ;
    yy12(i) = V1(12,i) ;
    yy13(i) = V1(13,i) ;
    yy14(i) = V1(14,i) ;
    yy15(i) = V1(15,i) ;
    yy16(i) = V1(16,i) ;
    yy17(i) = V1(17,i) ;
    yy18(i) = V1(18,i) ;
    yy19(i) = V1(19,i) ;
    yy20(i) = V1(20,i) ;
    yy21(i) = V1(21,i) ;
    yy22(i) = V1(22,i) ;
    yy23(i) = V1(23,i) ;
    yy24(i) = V1(24,i) ;
    yy25(i) = V1(25,i) ;
    yy26(i) = V1(26,i) ;
    yy27(i) = V1(27,i) ;
end
plot( xx, yy1,'bo', xx, yy2,'go', xx, yy3,'bo', xx, yy4,'go', xx, yy5,'ro', ...
      xx, yy6,'go', xx, yy7,'bo', xx, yy8,'go', xx, yy9,'bo', ...
      xx, yy10,'gp', xx, yy11,'rp', xx, yy12,'gp', xx, yy13,'rp', xx, yy14,'k*', ...
      xx, yy15,'rp', xx, yy16,'gp', xx, yy17,'rp', xx, yy18,'gp', ...
      xx, yy19,'bs', xx, yy20,'gs', xx, yy21,'bs', xx, yy22,'gs', xx, yy23,'rs', ...
      xx, yy24,'gs', xx, yy25,'bs', xx, yy26,'gs', xx, yy27,'bs', ...
      'Linewidth',1.0, 'MarkerSize',4.0 )
title('v abscissas') ;
xlim([0 1]) ;
ylim([-Umax Umax]) ;

subplot(2,4,4) ;
for i=1:Ny
    xx(i) = Ycell(i) ;
    yy1(i) = W1(1,i) ;
    yy2(i) = W1(2,i) ;
    yy3(i) = W1(3,i) ;
    yy4(i) = W1(4,i) ;
    yy5(i) = W1(5,i) ;
    yy6(i) = W1(6,i) ;
    yy7(i) = W1(7,i) ;
    yy8(i) = W1(8,i) ;
    yy9(i) = W1(9,i) ;
    yy10(i) = W1(10,i) ;
    yy11(i) = W1(11,i) ;
    yy12(i) = W1(12,i) ;
    yy13(i) = W1(13,i) ;
    yy14(i) = W1(14,i) ;
    yy15(i) = W1(15,i) ;
    yy16(i) = W1(16,i) ;
    yy17(i) = W1(17,i) ;
    yy18(i) = W1(18,i) ;
    yy19(i) = W1(19,i) ;
    yy20(i) = W1(20,i) ;
    yy21(i) = W1(21,i) ;
    yy22(i) = W1(22,i) ;
    yy23(i) = W1(23,i) ;
    yy24(i) = W1(24,i) ;
    yy25(i) = W1(25,i) ;
    yy26(i) = W1(26,i) ;
    yy27(i) = W1(27,i) ;
end
plot( xx, yy1,'bo', xx, yy2,'go', xx, yy3,'bo', xx, yy4,'go', xx, yy5,'ro', ...
      xx, yy6,'go', xx, yy7,'bo', xx, yy8,'go', xx, yy9,'bo', ...
      xx, yy10,'gp', xx, yy11,'rp', xx, yy12,'gp', xx, yy13,'rp', xx, yy14,'k*', ...
      xx, yy15,'rp', xx, yy16,'gp', xx, yy17,'rp', xx, yy18,'gp', ...
      xx, yy19,'bs', xx, yy20,'gs', xx, yy21,'bs', xx, yy22,'gs', xx, yy23,'rs', ...
      xx, yy24,'gs', xx, yy25,'bs', xx, yy26,'gs', xx, yy27,'bs', ...
      'Linewidth',1.0, 'MarkerSize',4.0 )
title('w abscissas') ;
xlim([0 1]) ;
ylim([-Umax Umax]) ;

subplot(2,4,5) ;
for i=1:Ny
    xx(i) = Ycell(i) ;
    yy1(i) = M1(1,i) ;
end
plot(xx, yy1, 'Linewidth',2.0, 'MarkerSize',3.0 )
title('\rho') ;
xlim([0 1]) ;
ylim([0 rhomax]) ;

subplot(2,4,6) ;
for i=1:Ny
    xx(i) = Ycell(i) ;
    yy1(i) = M1(2,i)/M1(1,i) ;
end
plot(xx, yy1, 'Linewidth',2.0, 'MarkerSize',3.0 )
title('U') ;
xlim([0 1]) ;
ylim([-Umax Umax]) ;

subplot(2,4,7) ;
for i=1:Ny
    xx(i) = Ycell(i) ;
    yy1(i) = M1(3,i)/M1(1,i) ;
end
plot(xx, yy1, '-', 'Linewidth',2.0, 'MarkerSize',3.0 )
title('V') ;
xlim([0 1]) ;
ylim([-Umax Umax]) ;

subplot(2,4,8) ;
for i=1:Ny
    xx(i) = Ycell(i) ;
    yy1(i) = M1(4,i)/M1(1,i) ;
end
plot(xx, yy1, 'Linewidth',2.0, 'MarkerSize',3.0 )
title('W') ;
xlim([0 1]) ;
ylim([-Umax Umax]) ;
% 
% subplot(3,4,9) ;
% for i=1:Ny
%     xx(i) = Ycell(i) ;
%     yy1(i) = M1(5,i)/M1(1,i) - (M1(2,i)/M1(1,i))*(M1(2,i)/M1(1,i)) ;
%     yy2(i) = M1(8,i)/M1(1,i) - (M1(3,i)/M1(1,i))*(M1(3,i)/M1(1,i)) ;
%     yy3(i) = M1(10,i)/M1(1,i) - (M1(4,i)/M1(1,i))*(M1(4,i)/M1(1,i)) ;
% end
% Temp = (yy1 + yy2 + yy3)/3 ;
% % yy1 = (yy1 - Temp)./Temp ;
% % yy2 = (yy2 - Temp)./Temp ;
% % yy3 = (yy3 - Temp)./Temp ;
% yy1 = yy1.*rho ;
% yy2 = yy2.*rho ;
% yy3 = yy3.*rho ;
% plot(xx, yy1, xx, yy2, xx, yy3, 'Linewidth',2.0, 'MarkerSize',3.0 )
% title('\sigma_{uu}, \sigma_{vv}, \sigma_{ww}') ;
% xlim([0 1]) ;
% 
% xlswrite('sigma_xx', yy1);
% xlswrite('sigma_yy', yy2);
% xlswrite('sigma_zz', yy3);
% 
% subplot(3,4,10) ;
% for i=1:Ny
%     xx(i) = Ycell(i) ;
%     yy1(i) = M1(6,i)/M1(1,i) - (M1(2,i)/M1(1,i))*(M1(3,i)/M1(1,i)) ;
%     yy2(i) = M1(7,i)/M1(1,i) - (M1(2,i)/M1(1,i))*(M1(4,i)/M1(1,i)) ;
%     yy3(i) = M1(9,i)/M1(1,i) - (M1(3,i)/M1(1,i))*(M1(4,i)/M1(1,i)) ;
% end
% yy1 = yy1.*rho ;
% yy2 = yy2.*rho ;
% yy3 = yy3.*rho ;
% plot(xx, yy1, xx, yy2, xx, yy3, 'Linewidth',2.0, 'MarkerSize',3.0 )
% title('\sigma_{uv}, \sigma_{uw}, \sigma_{vw}') ;
% xlim([0 1]) ;
% 
% xlswrite('sigma_xy', yy1);
% xlswrite('sigma_xz', yy2);
% xlswrite('sigma_yz', yy3);
% 
% subplot(3,4,11) ;
% for i=1:Ny
%     xx(i) = Ycell(i) ;
%     yy1(i) = M1(5,i)/M1(1,i) - (M1(2,i)/M1(1,i))*(M1(2,i)/M1(1,i)) ;
%     yy2(i) = M1(8,i)/M1(1,i) - (M1(3,i)/M1(1,i))*(M1(3,i)/M1(1,i)) ;
%     yy3(i) = M1(10,i)/M1(1,i) - (M1(4,i)/M1(1,i))*(M1(4,i)/M1(1,i)) ;
% end
% yy4 = Temp ;
% %
% plot(xx, yy1, xx, yy2, xx, yy3, xx, yy4, 'Linewidth',2.0, 'MarkerSize',3.0 )
% title('T_u, T_v, T_w, T') ;
% xlim([0 1]) ;
% 
% xlswrite('Tx', yy1);
% xlswrite('Ty', yy2);
% xlswrite('Tz', yy3);
% xlswrite('T', yy4);
% %pause;
% 
% subplot(3,4,12) ;
% for i=1:Ny
%   xx(i) = Ycell(i) ;
%   vm = M1(3,i)/M1(1,i) ;
%   yy1(i) = 0.5*( M1(12,i) + M1(17,i) + M1(19,i) ) - M1(8,i)*vm ...
%   + 0.5*M1(1,i)*vm^3 - vm*Temp(i)*M1(1,i) ;
% end
% plot(xx, yy1, 'Linewidth',2.0, 'MarkerSize',3.0 )
% title('q_{v}') ;
% xlim([0 1]);
% xlswrite('qv', yy1);
% %pause;
% 
% % yy1 = rho.*Temp ;
% % plot(xx, yy1, 'Linewidth',2.0, 'MarkerSize',3.0 )
% % title('p') ;
% % xlim([0 1]) ;
% % pause(1)

if (movie == 1)  %% -dtiff
   print('-dpng', strcat('./movie_3/crossing_jet_1D_',num2str(k,'%03u'),'.png'))  % Create a movie in Quicktime
end
