clear;
close all;
clc;

load('../data/Random_Forcing/QBMM_HM_Random_Pressure_Realization32.mat')
moments_QB = moments;
T_QB = T;

load('../data/Random_Forcing/MC_HM_Random_Pressure_Realization32.mat')


load('../data/Random_Forcing/MC_Weights_4points_Realization32.mat');
abscissas4 = abscissas_hist;
weights4   = weights_hist;

load('../data/Random_Forcing/MC_Weights_9points_Realization32.mat');
abscissas9 = abscissas_hist;
weights9   = weights_hist;

Re = 1000.0;
total_times = 5001;
dt = 50.0/(total_times-1);
RHS_W4 = zeros(5,total_times);
RHS_W9 = zeros(5,total_times);
RHS_MC = zeros(5,total_times);

for tt=1:total_times

if (tt < total_times && tt > 1)
    for ii=1:5
        RHS_MC(ii,tt) = (moments(ii+1,tt+1) -moments(ii+1,tt-1))/(2.0*dt);
    end
end
    
    
    
flag1 = zeros(2,4);
flag2 = zeros(1,4);
flag1(:,:) = abscissas4(tt,:,:);
flag2(:)   = weights4(tt,:);

M10_W4 = Moment_Weights(flag1,flag2,[1,0]);
M01_W4 = Moment_Weights(flag1,flag2,[0,1]);
M20_W4 = Moment_Weights(flag1,flag2,[2,0]);
M11_W4 = Moment_Weights(flag1,flag2,[1,1]);
M02_W4 = Moment_Weights(flag1,flag2,[0,2]);
M30_W4 = Moment_Weights(flag1,flag2,[3,0]);
M03_W4 = Moment_Weights(flag1,flag2,[0,3]);
M40_W4 = Moment_Weights(flag1,flag2,[4,0]);
M04_W4 = Moment_Weights(flag1,flag2,[0,4]);

Mm12_W4 = Moment_Weights(flag1,flag2,[-1,2]);
Mm21_W4 = Moment_Weights(flag1,flag2,[-2,1]);
Mm40_W4 = Moment_Weights(flag1,flag2,[-4,0]);
Mm10_W4 = Moment_Weights(flag1,flag2,[-1,0]);

M02_W4 = Moment_Weights(flag1,flag2,[0,2]);
Mm11_W4 = Moment_Weights(flag1,flag2,[-1,1]);
Mm30_W4 = Moment_Weights(flag1,flag2,[-3,0]);
M30_W4 = Moment_Weights(flag1,flag2,[3,0]);

Mm13_W4 = Moment_Weights(flag1,flag2,[-1,3]);
Mm22_W4 = Moment_Weights(flag1,flag2,[-2,2]);
Mm41_W4 = Moment_Weights(flag1,flag2,[-4,1]);

RHS_W4(1,tt) = M01_W4;
RHS_W4(2,tt) = -1.5*Mm12_W4 -(4.0/Re)*Mm21_W4 +Mm40_W4 -pressure(tt)*Mm10_W4;
RHS_W4(3,tt) = 2.0*M11_W4;
RHS_W4(4,tt) = -0.5*M02_W4 -(4.0/Re)*Mm11_W4 +Mm30_W4 -pressure(tt);
RHS_W4(5,tt) = 2.0*(-1.5*Mm13_W4 -(4.0/Re)*Mm22_W4 +Mm41_W4 -pressure(tt)*Mm11_W4);

RHS_W4(1,tt) = M01_W4;
RHS_W4(2,tt) = -1.5*(M02_W4/M10_W4) -(4.0/Re)*(M01_W4/M20_W4) +1/M40_W4 -pressure(tt)*(1/M10_W4);
RHS_W4(3,tt) = 2.0*M11_W4;
RHS_W4(4,tt) = -0.5*M02_W4 -(4.0/Re)*(M01_W4/M10_W4) +1/M30_W4 -pressure(tt);
RHS_W4(5,tt) = 2.0*(-1.5*M03_W4/M10_W4 -(4.0/Re)*(M02_W4/M20_W4) +M01_W4/M40_W4 -pressure(tt)*M01_W4/M10_W4);

RHS_W4(1,tt) = M01_W4;
RHS_W4(2,tt) = -1.5*(M02_W4/M10_W4) -(4.0/Re)*(M01_W4/M20_W4) +Mm40_W4 -pressure(tt)*(1/M10_W4);
RHS_W4(3,tt) = 2.0*M11_W4;
RHS_W4(4,tt) = -0.5*M02_W4 -(4.0/Re)*(M01_W4/M10_W4) +Mm30_W4 -pressure(tt);
RHS_W4(5,tt) = 2.0*(-1.5*M03_W4/M10_W4 -(4.0/Re)*(M02_W4/M20_W4) +Mm41_W4 -pressure(tt)*M01_W4/M10_W4);

% RHS_W4(1,tt) = moments(3,tt);
% RHS_W4(2,tt) = -1.5*(moments(6,tt)/moments(2,tt)) -(4.0/Re)*(moments(3,tt)/moments(4,tt)) +1/moments(11,tt) -pressure(tt)*(1/moments(2,tt));
% RHS_W4(3,tt) = 2.0*moments(5,tt);
% RHS_W4(4,tt) = -0.5*moments(6,tt) -(4.0/Re)*(moments(3,tt)/moments(2,tt)) +1/moments(10,tt) -pressure(tt);
% RHS_W4(5,tt) = 2.0*(-1.5*moments(10,tt)/moments(2,tt) -(4.0/Re)*(moments(6,tt)/moments(4,tt)) +moments(3,tt)/moments(11,tt) -pressure(tt)*moments(3,tt)/moments(2,tt));



flag1 = zeros(2,9);
flag2 = zeros(1,9);
flag1(:,:) = abscissas9(tt,:,:);
flag2(:)   = weights9(tt,:);

M10_W9 = Moment_Weights(flag1,flag2,[1,0]);
M01_W9 = Moment_Weights(flag1,flag2,[0,1]);
M20_W9 = Moment_Weights(flag1,flag2,[2,0]);
M11_W9 = Moment_Weights(flag1,flag2,[1,1]);
M02_W9 = Moment_Weights(flag1,flag2,[0,2]);
M30_W9 = Moment_Weights(flag1,flag2,[3,0]);
M03_W9 = Moment_Weights(flag1,flag2,[0,3]);
M40_W9 = Moment_Weights(flag1,flag2,[4,0]);
M04_W9 = Moment_Weights(flag1,flag2,[0,4]);

Mm12_W9 = Moment_Weights(flag1,flag2,[-1,2]);
Mm21_W9 = Moment_Weights(flag1,flag2,[-2,1]);
Mm40_W9 = Moment_Weights(flag1,flag2,[-4,0]);
Mm10_W9 = Moment_Weights(flag1,flag2,[-1,0]);

M02_W9 = Moment_Weights(flag1,flag2,[0,2]);
Mm11_W9 = Moment_Weights(flag1,flag2,[-1,1]);
Mm30_W9 = Moment_Weights(flag1,flag2,[-3,0]);

Mm13_W9 = Moment_Weights(flag1,flag2,[-1,3]);
Mm22_W9 = Moment_Weights(flag1,flag2,[-2,2]);
Mm41_W9 = Moment_Weights(flag1,flag2,[-4,1]);

RHS_W9(1,tt) = M01_W9;
RHS_W9(2,tt) = -1.5*Mm12_W9 -(4.0/Re)*Mm21_W9 +Mm40_W9 -pressure(tt)*Mm10_W9;
RHS_W9(3,tt) = 2.0*M11_W9;
RHS_W9(4,tt) = -0.5*M02_W9 -(4.0/Re)*Mm11_W9 +Mm30_W9 -pressure(tt);
RHS_W9(5,tt) = 2.0*(-1.5*Mm13_W9 -(4.0/Re)*Mm22_W9 +Mm41_W9 -pressure(tt)*Mm11_W9);

% RHS_W9(1,tt) = M01_W9;
% RHS_W9(2,tt) = -1.5*(M02_W9/M10_W9) -(4.0/Re)*(M01_W9/M20_W9) +1/M40_W9 -pressure(tt)*(1/M10_W9);
% RHS_W9(3,tt) = 2.0*M11_W9;
% RHS_W9(4,tt) = -0.5*M02_W9 -(4.0/Re)*(M01_W9/M10_W9) +1/M30_W9 -pressure(tt);
% RHS_W9(5,tt) = 2.0*(-1.5*M03_W9/M10_W9 -(4.0/Re)*(M02_W9/M20_W9) +M01_W9/M04_W9 -pressure(tt)*M01_W9/M10_W9);
% 
% RHS_W9(1,tt) = M01_W9;
% RHS_W9(2,tt) = -1.5*(M02_W9/M10_W9) -(4.0/Re)*(M01_W9/M20_W9) +Mm40_W9 -pressure(tt)*(1/M10_W9);
% RHS_W9(3,tt) = 2.0*M11_W9;
% RHS_W9(4,tt) = -0.5*M02_W9 -(4.0/Re)*(M01_W9/M10_W9) +Mm30_W9 -pressure(tt);
% RHS_W9(5,tt) = 2.0*(-1.5*M03_W9/M10_W9 -(4.0/Re)*(M02_W9/M20_W9) +Mm41_W9 -pressure(tt)*M01_W9/M10_W9);


clc;
disp(tt);
end




figure(1)
for ii=1:5
   subplot(5,1,ii)
   plot(T,RHS_MC(ii,:),'Color',[0.2,0.2,0.2])
   hold on
   plot(T,RHS_W4(ii,:),'Color',[0.8,0,0])
   hold on
   plot(T,RHS_W9(ii,:),'Color',[0,0.5,1])
   xlim([0,10])
end

moments_MC = zeros(5,total_times);
moments_W4 = zeros(5,total_times);
moments_W9 = zeros(5,total_times);


moments_MC(:,1) = moments(2:6,1);
moments_W4(:,1) = moments(2:6,1);
moments_W9(:,1) = moments(2:6,1);
for tt=1:total_times-1
    for ii=1:5
        moments_MC(ii,tt+1) = moments_MC(ii,tt) +dt*RHS_MC(ii,tt);
        moments_W4(ii,tt+1) = moments_W4(ii,tt) +dt*RHS_W4(ii,tt);
        moments_W9(ii,tt+1) = moments_W9(ii,tt) +dt*RHS_W9(ii,tt);
    end
end



figure(2)
for ii=1:5
   subplot(5,1,ii)
   plot(T,moments_MC(ii,:),'Color',[0.2,0.2,0.2])
   hold on
   plot(T,moments_W4(ii,:),'Color',[0.8,0,0])
   hold on
   plot(T,moments_W9(ii,:),'Color',[0,0.5,1])
   hold on
   plot(T_QB,moments_QB(:,ii+1),'Color',[0,0.8,0])
   xlim([40,50])
end

figure(3)
subplot(5,1,1)
plot(1./moments(2,:))
hold on
plot(moments(26,:))
subplot(5,1,2)
plot(1./moments(7,:))
hold on
plot(moments(28,:))
subplot(5,1,3)
plot(1./moments(11,:))
hold on
plot(moments(25,:))
subplot(5,1,4)
plot(moments(3,:)./moments(4,:))
hold on
plot(moments(24,:))
subplot(5,1,5)
plot(moments(6,:)./moments(2,:))
hold on
plot(moments(23,:))



