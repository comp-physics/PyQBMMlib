clear;
close all;
clc;









file_name = '../data/Sinusoidal_Forcing/';
mc_name   = 'MC_HM_Sinusoidal_Pressure';
qbmm_name = 'QBMM_HM_Sinusoidal_Pressure';
period_name = '_Period3';
rfile_mc = {[file_name,mc_name,'5',period_name,'.mat'];...
            [file_name,mc_name,'10',period_name,'.mat'];...
            [file_name,mc_name,'15',period_name,'.mat'];...
            [file_name,mc_name,'20',period_name,'.mat'];...
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
            [file_name,mc_name,'85',period_name,'.mat'];};

rfile_qbmm = {[file_name,qbmm_name,'5',period_name,'.mat'];...
            [file_name,qbmm_name,'10',period_name,'.mat'];...
            [file_name,qbmm_name,'15',period_name,'.mat'];...
            [file_name,qbmm_name,'20',period_name,'.mat'];...
            [file_name,qbmm_name,'25',period_name,'.mat'];...
            [file_name,qbmm_name,'30',period_name,'.mat'];...
            [file_name,qbmm_name,'35',period_name,'.mat'];...
            [file_name,qbmm_name,'40',period_name,'.mat'];...
            [file_name,qbmm_name,'45',period_name,'.mat'];...
            [file_name,qbmm_name,'50',period_name,'.mat'];...
            [file_name,qbmm_name,'55',period_name,'.mat'];...
            [file_name,qbmm_name,'60',period_name,'.mat'];...
            [file_name,qbmm_name,'65',period_name,'.mat'];...
            [file_name,qbmm_name,'70',period_name,'.mat'];...
            [file_name,qbmm_name,'75',period_name,'.mat'];...
            [file_name,qbmm_name,'80',period_name,'.mat'];...
            [file_name,qbmm_name,'85',period_name,'.mat'];};      
        
        
load(['../ML_Code/HM_Sinusoidal_MLQBMM',period_name,'.mat']);   

total_cases = 17;


load(rfile_mc{1});

%T_qbmm = data_qbmm(:,1)';
T_mc = T;

%input_data = zeros(15,5,size(T,2));
mc_data = zeros(total_cases,4,size(T_mc,2));
qbmm_data = zeros(total_cases,4,size(T_mc,2));

for jj=1:total_cases
    %data_qbmm = importdata(rfile_qbmm{jj});        
    load(rfile_mc{jj});
    mc_data(jj,:,:) = moments(6:9,:);
    
    load(rfile_qbmm{jj});
    for ii=1:4
        x = moments(:,5+ii);
        qbmm_data(jj,ii,:) = interp1(T,x,T_mc);
    end
    qbmm_data(jj,1,1) = 0.0;
    qbmm_data(jj,2,1) = 0.0;
    qbmm_data(jj,3,1) = 1.0;
    qbmm_data(jj,4,1) = 1.0;
    %T_qbmm = data_qbmm(:,1)';
% for ii=1:5
%     x = data_qbmm(:,ii+2);
%     input_data(jj,ii,:) = interp1(T_qbmm,x,T_mc);
% end

end

R32_ML = zeros(total_cases,1801);
R21_ML = zeros(total_cases,1801);
R30_ML = zeros(total_cases,1801);
R3g_ML = zeros(total_cases,1801);

R32_MC = zeros(total_cases,1801);
R21_MC = zeros(total_cases,1801);
R30_MC = zeros(total_cases,1801);
R3g_MC = zeros(total_cases,1801);

R32_QBMM = zeros(total_cases,1801);
R21_QBMM = zeros(total_cases,1801);
R30_QBMM = zeros(total_cases,1801);
R3g_QBMM = zeros(total_cases,1801);

for jj=1:total_cases
    R32_MC(jj,:) = mc_data(jj,1,:);
    R21_MC(jj,:) = mc_data(jj,2,:);
    R30_MC(jj,:) = mc_data(jj,3,:);
    R3g_MC(jj,:) = mc_data(jj,4,:);
    
    R32_ML(jj,:) = predictions(jj,1,:);
    R21_ML(jj,:) = predictions(jj,2,:);
    R30_ML(jj,:) = predictions(jj,3,:);
    R3g_ML(jj,:) = predictions(jj,4,:);
    
    R32_QBMM(jj,:) = qbmm_data(jj,1,:);
    R21_QBMM(jj,:) = qbmm_data(jj,2,:);
    R30_QBMM(jj,:) = qbmm_data(jj,3,:);
    R3g_QBMM(jj,:) = qbmm_data(jj,4,:);
end







xflag = zeros(1,1801);
xflag(:) = predictions(1,4,:);
plot(xflag)
hold on;
xflag(:) = output_data(1,4,:)+input_data(1,9,:);
plot(xflag)
hold on;
plot(R3g_MC(1,:))
hold on;
plot(R3g_QBMM(1,:))