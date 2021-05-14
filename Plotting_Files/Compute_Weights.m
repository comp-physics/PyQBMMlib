clear;
close all;
clc;

global MC_moments ids N np;

load(['../data/Random_Forcing/MC_HM_Random_Pressure_Realization1','.mat']);

choose_moments = [1,2,3, 4,5,6, 23,24,25, 26,27,28, 29,30,31];
%choose_moments = [1,2,3, 4,5,6, 7,8,9, 10,11,12, 13,14,15];
%choose_moments = [1,2,3, 4,5,6, 7,8,9, 10,11,12];

np = 5;
N = 15;
ids = zeros(2,N);
MC_moments = zeros(1,N);

ids(1, 1) = 0.0; ids(2, 1) = 0.0;
ids(1, 2) = 1.0; ids(2, 2) = 0.0;
ids(1, 3) = 0.0; ids(2, 3) = 1.0;

ids(1, 4) = 2.0; ids(2, 4) = 0.0;
ids(1, 5) = 1.0; ids(2, 5) = 1.0;
ids(1, 6) = 0.0; ids(2, 6) = 2.0;

% ids(1, 7) = 3.0; ids(2, 7) = 0.0;
% ids(1, 8) = 2.0; ids(2, 8) = 1.0;
% ids(1, 9) = 1.0; ids(2, 9) = 2.0;
% 
% ids(1,10) = 0.0; ids(2,10) = 3.0;
% ids(1,11) = 4.0; ids(2,11) = 0.0;
% ids(1,12) = 3.0; ids(2,12) = 1.0;

% ids(1,13) = 2.0; ids(2,13) = 2.0;
% ids(1,14) = 1.0; ids(2,14) = 3.0;
% ids(1,15) = 0.0; ids(2,15) = 4.0;

ids(1, 7) = -1.0; ids(2, 7) = 2.0;
ids(1, 8) = -2.0; ids(2, 8) = 1.0;
ids(1, 9) = -4.0; ids(2, 9) = 0.0;

ids(1,10) = -1.0; ids(2,10) = 0.0;
ids(1,11) = -1.0; ids(2,11) = 1.0;
ids(1,12) = -3.0; ids(2,12) = 0.0;

ids(1,13) = -1.0; ids(2,13) = 3.0;
ids(1,14) = -2.0; ids(2,14) = 2.0;
ids(1,15) = -4.0; ids(2,15) = 1.0;


XX = zeros(3*np,2001);

for tt=100:100
for ii=1:N
    MC_moments(ii) = moments(choose_moments(ii),tt);
end


% [w,xi,xid,dF,F] = Weight_NR(moments,ids,N,np);

x0 = zeros(1,3*np);
x0(1:np) = 1.0/np;
x0(np+1:2*np) = rand(1,np);
x0(2*np+1:3*np) = rand(1,np);
x = x0;
F = Scalar_Loss(x);
while (F > 10^(-2))
fun = @Scalar_Loss;
options = optimoptions('fsolve');
options.MaxFunctionEvaluations = 2*10^5;
options.MaxIterations = 2*10^5;
x = fsolve(fun,x0,options);
F = Scalar_Loss(x);
end
XX(:,tt) = x;

clc;
disp(tt);
end


