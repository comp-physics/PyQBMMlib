
function [] = output_granular_3D_16mom(M1,Ycell,t,Ny,movie,k)
% output the plots of moments
%
Umax = 1.35 ;
rhomax = 1.8 ;

subplot(4,4,1) ;
for i=1:Ny
    xx(i) = Ycell(i) ;
    yy1(i) = M1(1,i) ;
end
plot(xx, yy1, 'Linewidth',2.0, 'MarkerSize',3.0 )
title('M_{0,0,0}') ;
xlim([0 1]) ;
ylim([0 rhomax]) ;

subplot(4,4,2) ;
for i=1:Ny
    xx(i) = Ycell(i) ;
    yy1(i) = M1(2,i) ;
end
plot(xx, yy1, 'Linewidth',2.0, 'MarkerSize',3.0 )
title('M_{1,0,0}') ;
xlim([0 1]) ;
ylim(rhomax*[-Umax Umax]) ;

subplot(4,4,3) ;
for i=1:Ny
    xx(i) = Ycell(i) ;
    yy1(i) = M1(3,i) ;
end
plot(xx, yy1, '-', 'Linewidth',2.0, 'MarkerSize',3.0 )
title('M_{0,1,0}') ;
xlim([0 1]) ;
ylim(rhomax*[-Umax Umax]) ;

subplot(4,4,4) ;
for i=1:Ny
    xx(i) = Ycell(i) ;
    yy1(i) = M1(4,i) ;
end
plot(xx, yy1, 'Linewidth',2.0, 'MarkerSize',3.0 )
title('M_{0,0,1}') ;
xlim([0 1]) ;
ylim(rhomax*[-Umax Umax]) ;

subplot(4,4,5) ;
for i=1:Ny
    xx(i) = Ycell(i) ;
    yy1(i) = M1(5,i) ;
end
plot(xx, yy1, 'Linewidth',2.0, 'MarkerSize',3.0 )
title('M_{2,0,0}') ;
xlim([0 1]) ;
ylim(rhomax*[0 Umax^2]) ;

subplot(4,4,6) ;
for i=1:Ny
    xx(i) = Ycell(i) ;
    yy1(i) = M1(6,i) ;
end
plot(xx, yy1, 'Linewidth',2.0, 'MarkerSize',3.0 )
title('M_{1,1,0}') ;
xlim([0 1]) ;
ylim(rhomax*[-Umax Umax]) ;

subplot(4,4,7) ;
for i=1:Ny
    xx(i) = Ycell(i) ;
    yy1(i) = M1(7,i) ;
end
plot(xx, yy1, 'Linewidth',2.0, 'MarkerSize',3.0 )
title('M_{1,0,1}') ;
xlim([0 1]) ;
ylim(rhomax*[-Umax Umax]) ;

subplot(4,4,8) ;
for i=1:Ny
    xx(i) = Ycell(i) ;
    yy1(i) = M1(8,i) ;
end
plot(xx, yy1, 'Linewidth',2.0, 'MarkerSize',3.0 )
title('M_{0,2,0}') ;
xlim([0 1]) ;
ylim(rhomax*[0 Umax^2]) ;

subplot(4,4,9) ;
for i=1:Ny
    xx(i) = Ycell(i) ;
    yy1(i) = M1(9,i) ;
end
plot(xx, yy1, 'Linewidth',2.0, 'MarkerSize',3.0 )
title('M_{0,1,1}') ;
xlim([0 1]) ;
ylim(rhomax*[-Umax Umax]) ;

subplot(4,4,10) ;
for i=1:Ny
    xx(i) = Ycell(i) ;
    yy1(i) = M1(10,i) ;
end
plot(xx, yy1, 'Linewidth',2.0, 'MarkerSize',3.0 )
title('M_{0,0,2}') ;
xlim([0 1]) ;
ylim(rhomax*[0 Umax^2]) ;

subplot(4,4,11) ;
for i=1:Ny
    xx(i) = Ycell(i) ;
    yy1(i) = M1(11,i) ;
end
plot(xx, yy1, 'Linewidth',2.0, 'MarkerSize',3.0 )
title('M_{3,0,0}') ;
xlim([0 1]) ;
ylim(rhomax*[-Umax^3 Umax^3]) ;

subplot(4,4,12) ;
for i=1:Ny
    xx(i) = Ycell(i) ;
    yy1(i) = M1(12,i) ;
end
plot(xx, yy1, 'Linewidth',2.0, 'MarkerSize',3.0 )
title('M_{0,3,0}') ;
xlim([0 1]) ;
ylim(rhomax*[-Umax^3 Umax^3]) ;

subplot(4,4,13) ;
for i=1:Ny
    xx(i) = Ycell(i) ;
    yy1(i) = M1(13,i) ;
end
plot(xx, yy1, 'Linewidth',2.0, 'MarkerSize',3.0 )
title('M_{0,0,3}') ;
xlim([0 1]) ;
ylim(rhomax*[-Umax^3 Umax^3]) ;

subplot(4,4,14) ;
for i=1:Ny
    xx(i) = Ycell(i) ;
    yy1(i) = M1(14,i) ;
end
plot(xx, yy1, 'Linewidth',2.0, 'MarkerSize',3.0 )
title('M_{4,0,0}') ;
xlim([0 1]) ;
ylim(rhomax*[0 Umax^4]) ;

subplot(4,4,15) ;
for i=1:Ny
    xx(i) = Ycell(i) ;
    yy1(i) = M1(15,i) ;
end
plot(xx, yy1, 'Linewidth',2.0, 'MarkerSize',3.0 )
title('M_{0,4,0}') ;
xlim([0 1]) ;
ylim(rhomax*[0 Umax^4]) ;

subplot(4,4,16) ;
for i=1:Ny
    xx(i) = Ycell(i) ;
    yy1(i) = M1(16,i) ;
end
plot(xx, yy1, 'Linewidth',2.0, 'MarkerSize',3.0 )
title('M_{0,0,4}') ;
xlim([0 1]) ;
ylim(rhomax*[0 Umax^4]) ;
% 

if (movie == 1)  %% -dtiff
   print('-dpng', strcat('./movie_3m/crossing_jet_1D_',num2str(k,'%03u'),'.png'))  % Create a movie in Quicktime
end
