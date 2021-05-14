clear;
close all;
clc;


file_name = '../data/Constant_Forcing/';
mc_name   = 'MC_HM_Constant_Pressure';
mc_weight = 'MC_Weights';
qbmm_name = 'QBMM_HM_Sinusoidal_Pressure';
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

 rfile_mc_weight = {[file_name,mc_weight,'20',period_name,'.mat'];...
            [file_name,mc_weight,'25',period_name,'.mat'];...
            [file_name,mc_weight,'30',period_name,'.mat'];...
            [file_name,mc_weight,'35',period_name,'.mat'];...
            [file_name,mc_weight,'40',period_name,'.mat'];...
            [file_name,mc_weight,'45',period_name,'.mat'];...
            [file_name,mc_weight,'50',period_name,'.mat'];...
            [file_name,mc_weight,'55',period_name,'.mat'];...
            [file_name,mc_weight,'60',period_name,'.mat'];...
            [file_name,mc_weight,'65',period_name,'.mat'];...
            [file_name,mc_weight,'70',period_name,'.mat'];...
            [file_name,mc_weight,'75',period_name,'.mat'];...
            [file_name,mc_weight,'80',period_name,'.mat'];...
            [file_name,mc_weight,'85',period_name,'.mat'];...
            [file_name,mc_weight,'90',period_name,'.mat'];...
            [file_name,mc_weight,'95',period_name,'.mat'];};
       
 total_cases = 16;
 
 
 load(rfile_mc{13});
 load(rfile_mc_weight{13});
 
 total_times = 10001;
 dt = 50/(total_times-1);
 
 moment_rec = zeros(5,total_times);
 rhs_rec = zeros(5,total_times);
 rhs_mc  = zeros(5,total_times);
 Re = 1000.0;
 Cp = 0.80;
 
 indices = zeros(2,5);
 indices = [1, 0; 0, 1; 2, 0; 1,1; 0,2]';
 
 
 for ii=1:total_times
    for jj=1:5
        for kk=1:4
            moment_rec(jj,ii) = moment_rec(jj,ii)+weights_hist(ii,kk)*abscissas_hist(ii,1,kk)^indices(1,jj) *abscissas_hist(ii,2,kk)^indices(2,jj);
        end
    end
    for kk=1:4
        rhs_rec(2,ii)    = rhs_rec(2,ii) -1.5*weights_hist(ii,kk)*abscissas_hist(ii,1,kk)^(-1) *abscissas_hist(ii,2,kk)^(2)...
                                -1.0*(4.0/Re)*weights_hist(ii,kk)*abscissas_hist(ii,1,kk)^(-2) *abscissas_hist(ii,2,kk)^(1)...
                                +1.0*weights_hist(ii,kk)*abscissas_hist(ii,1,kk)^(-4) *abscissas_hist(ii,2,kk)^(0)...
                                -1.0*(1/Cp)*weights_hist(ii,kk)*abscissas_hist(ii,1,kk)^(-1) *abscissas_hist(ii,2,kk)^(0);
                            
        rhs_rec(4,ii)    = rhs_rec(4,ii) -1.5*weights_hist(ii,kk)*abscissas_hist(ii,1,kk)^(0) *abscissas_hist(ii,2,kk)^(2)...
                                -1.0*(4.0/Re)*weights_hist(ii,kk)*abscissas_hist(ii,1,kk)^(-1) *abscissas_hist(ii,2,kk)^(1)...
                                +1.0*weights_hist(ii,kk)*abscissas_hist(ii,1,kk)^(-3) *abscissas_hist(ii,2,kk)^(0)...
                                -1.0*(1/Cp)*weights_hist(ii,kk)*abscissas_hist(ii,1,kk)^(0) *abscissas_hist(ii,2,kk)^(0);
        
        
        
        rhs_rec(5,ii)    = rhs_rec(5,ii) -1.5*weights_hist(ii,kk)*abscissas_hist(ii,1,kk)^(-1) *abscissas_hist(ii,2,kk)^(3)...
                                -1.0*(4.0/Re)*weights_hist(ii,kk)*abscissas_hist(ii,1,kk)^(-2) *abscissas_hist(ii,2,kk)^(2)...
                                +1.0*weights_hist(ii,kk)*abscissas_hist(ii,1,kk)^(-4) *abscissas_hist(ii,2,kk)^(1)...
                                -1.0*(1/Cp)*weights_hist(ii,kk)*abscissas_hist(ii,1,kk)^(-1) *abscissas_hist(ii,2,kk)^(1);
                            
                            
    end
 end
 
 for tt=1:total_times-1
        %rhs_rec(2,tt) = (moment_rec(2,tt+1)-moment_rec(2,tt))/dt;
        rhs_mc(2,tt)  = (moments(2,tt+1)-moments(2,tt))/dt;
        rhs_mc(4,tt)  = (moments(4,tt+1)-moments(4,tt))/dt;
        rhs_mc(5,tt)  = (moments(5,tt+1)-moments(5,tt))/dt;
 end
 
 
 
 moment_ev = zeros(5,total_times);
 moment_ev(:,1) = moments(1:5,1);
 
 for tt=1:total_times-1
     moment_ev(1,tt+1) = moment_ev(1,tt) +dt*moment_rec(2,tt);
     moment_ev(2,tt+1) = moment_ev(2,tt) +dt*rhs_rec(2,tt);
     moment_ev(3,tt+1) = moment_ev(3,tt) +2.0*dt*moment_rec(4,tt);
     moment_ev(4,tt+1) = moment_ev(4,tt) +dt*( moment_rec(5,tt) +rhs_rec(4,tt)  );
     moment_ev(5,tt+1) = moment_ev(5,tt) +2.0*dt*rhs_rec(5,tt);
 end
 
 
 figure(1)
 for ii=1:5
    subplot(2,3,ii)
     plot(moment_ev(ii,:))
     hold on
     plot(moment_rec(ii,:))
     hold on
     plot(moments(ii,:))
 end

 
 figure(2)
 subplot(1,3,1)
 plot(rhs_rec(2,:))
 hold on
 plot(rhs_mc(2,:))
 
  subplot(1,3,2)
 plot(rhs_rec(4,:))
 hold on
 plot(rhs_mc(4,:))
 
  subplot(1,3,3)
 plot(2.0*rhs_rec(5,:))
 hold on
 plot(rhs_mc(5,:))
 
 
        