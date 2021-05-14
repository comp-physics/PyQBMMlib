clear;
close all;
clc;



file_name = '../data/Constant_Forcing/';
mc_name   = 'MC_HM_Constant_Pressure';
qbmm_name = 'QBMM_HM_Constant_Pressure';
period_name = '';
rfile_mc = {[file_name,mc_name,'20',period_name,'.mat'];...
            [file_name,mc_name,'25',period_name,'.mat'];...
            [file_name,mc_name,'30',period_name,'.mat'];...
            [file_name,mc_name,'35',period_name,'.mat'];...
            [file_name,mc_name,'40',period_name,'.mat'];...
            [file_name,mc_name,'45',period_name,'.mat'];...
            [file_name,mc_name,'50',period_name,'.mat'];...
            [file_name,mc_name,'55',period_name,'.mat'];...
            [file_name,mc_name,'60',period_name,'.mat'];...
            [file_name,mc_name,'65',period_name,'.mat'];...
            [file_name,mc_name,'70',period_name,'.mat'];...
            [file_name,mc_name,'75',period_name,'.mat'];...
            [file_name,mc_name,'80',period_name,'.mat'];...
            [file_name,mc_name,'85',period_name,'.mat'];...
            [file_name,mc_name,'90',period_name,'.mat'];...
            [file_name,mc_name,'95',period_name,'.mat'];};


total_cases = 16;
load(rfile_mc{10});

R_val  = [0.5,0.75,1.25,1.5];
Rd_val = [-0.6,-0.15,0.1,0.5];

nR  = 3;
nRd = 3;
total_weights = nR*nRd-2;
weights = zeros(1,total_weights);

MAT = zeros(total_weights,total_weights);
RHS = zeros(1,total_weights);

indices = [1,0; 0,1;...
           2,0; 1,1; 0,2;...
           3,0; 2,1; 1,2; 0,3;...
           4,0; 3,1; 2,2; 1,3; 0,4;...
           5,0; 4,1; 3,2; 2,3; 1,4; 0,5];


moment_flag = moments(:,1000);
for ii=1:total_weights
    RHS(ii) = moments(ii,1000);
    for jj=1:nR
       for kk=1:nRd
           counter = (jj-1)*nRd+kk;
           MAT(ii,counter) = R_val(jj)^indices(ii,1) *Rd_val(kk)^indices(ii,2);
       end
    end
end

weights = linsolve(MAT,RHS');

total_moments = 20;
New_moments = zeros(1,total_moments);

for ii=1:total_moments
    for jj=1:nR
       for kk=1:nRd
           counter = (jj-1)*nRd+kk;
           New_moments(ii) = New_moments(ii) +weights(counter)*R_val(jj)^indices(ii,1) *Rd_val(kk)^indices(ii,2);
       end
    end
end





