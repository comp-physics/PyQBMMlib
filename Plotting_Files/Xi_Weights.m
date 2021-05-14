clear;
close all;
clc;

global MC_moments ids N np;

load(['../data/Random_Forcing/MC_HM_Random_Pressure_Realization1','.mat']);

choose_moments = [1,2,4,7,11,16];
%choose_moments = [1,2,3, 4,5,6, 7,8,9, 10,11,12, 13,14,15];
%choose_moments = [1,2,3, 4,5,6, 7,8,9, 10,11,12];

np = 2;
N  = 3;
ids = zeros(2,N);
MC_moments = zeros(1,N);

ids(1, 1) = 0.0; ids(2, 1) = 0.0;
ids(1, 2) = 1.0; ids(2, 2) = 0.0;
ids(1, 3) = 2.0; ids(2, 3) = 0.0;
ids(1, 4) = 3.0; ids(2, 4) = 0.0;
ids(1, 5) = 4.0; ids(2, 5) = 0.0;
ids(1, 6) = 5.0; ids(2, 6) = 0.0;


for tt=2000:2000
for ii=1:N
    MC_moments(ii) = moments(choose_moments(ii),tt);
end


x0 = zeros(1,2*np);
x0(1:np) = 1.0/np;
x0(np+1:2*np) = rand(1,np);
x = x0;
fun = @Xi_Loss;
options = optimoptions('fsolve');
options.MaxFunctionEvaluations = 2*10^5;
options.MaxIterations = 2*10^5;
x = fsolve(fun,x0,options);
F = Xi_Loss(x);

end

