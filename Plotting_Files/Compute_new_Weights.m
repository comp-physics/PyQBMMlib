clear;
close all;
clc;


file_name = '../data/Constant_Forcing/';
%file_name = '../data/Sinusoidal_Forcing/';
if ( strcmp(file_name,'../data/Constant_Forcing/') )
mc_name   = 'MC_HM_Constant_Pressure';
mc_weight = 'MC_Weights';
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
        
elseif (strcmp(file_name,'../data/Sinusoidal_Forcing/'))
mc_name   = 'MC_HM_Sinusoidal_Pressure';
mc_weight = 'MC_Weights';
period_name = '_Period3';
rfile_mc = {[file_name,mc_name,'05',period_name,'.mat'];...
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
            [file_name,mc_name,'85',period_name,'.mat'];};

 total_cases = 16;
 
end
 
 load(rfile_mc{9});
 
 total_times = 10001;
 dt = 100.0/(total_times-1);
 Re = 1000.0;
 
 Cp = 1/0.60;
 %Cp = 0.25*sin(2*pi*T/3);
 
 RHS_ref  = zeros(5,10001); 
 RHS_predmc = zeros(5,10001);
 
 for tt=1:total_times-1
    
    for kk=1:5
        RHS_ref(kk,tt)  = (moments(kk+1,tt+1)-moments(kk+1,tt))/dt;
    end
    RHS_predmc(1,tt) = moments(3,tt); 
    RHS_predmc(2,tt) = -1.5*moments(23,tt) -(4.0/Re)*moments(24,tt) +moments(25,tt) -Cp*moments(26,tt); 
    RHS_predmc(3,tt) = 2.0*moments(5,tt); 
    RHS_predmc(4,tt) = moments(6,tt) -1.5*moments(6,tt) -(4.0/Re)*moments(27,tt) +moments(28,tt) -Cp; 
    RHS_predmc(5,tt) = 2.0*(-1.5*moments(29,tt) -(4.0/Re)*moments(30,tt) +moments(31,tt) -Cp*moments(27,tt)); 
    
    
 end
 
 
figure(1)
subplot(2,3,1)
plot(T,RHS_ref(1,:))
hold on
plot(T,RHS_predmc(1,:))
 
subplot(2,3,3)
plot(T,RHS_ref(2,:))
hold on
plot(T,RHS_predmc(2,:))
 
subplot(2,3,4)
plot(T,RHS_ref(3,:))
hold on
plot(T,RHS_predmc(3,:))
 
subplot(2,3,5)
plot(T,RHS_ref(4,:))
hold on
plot(T,RHS_predmc(4,:))
 
subplot(2,3,6)
plot(T,RHS_ref(5,:))
hold on
plot(T,RHS_predmc(5,:)) 
 
 
 