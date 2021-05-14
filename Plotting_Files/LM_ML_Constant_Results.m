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

rfile_qbmm = {[file_name,qbmm_name,'20',period_name,'.mat'];...
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
              [file_name,qbmm_name,'85',period_name,'.mat'];...
              [file_name,qbmm_name,'90',period_name,'.mat'];...
              [file_name,qbmm_name,'95',period_name,'.mat'];};      
        
        
load(['../ML_Code/LM_MLQBMM.mat']);   

total_cases = 16;


%data_qbmm = importdata(rfile_qbmm{1}); 
load(rfile_mc{1});

%T_qbmm = data_qbmm(:,1)';
T_mc = T;

%input_data = zeros(15,5,size(T,2));
mc_data = zeros(total_cases,9,size(T_mc,2));
qbmm_data = zeros(total_cases,9,size(T_mc,2));

for jj=1:total_cases
    %data_qbmm = importdata(rfile_qbmm{jj});        
    load(rfile_mc{jj});
    mc_data(jj,:,:) = moments;
    
    load(rfile_qbmm{jj});
    for ii=1:5
        x = moments(:,ii);
        qbmm_data(jj,ii,:) = interp1(T,x,T_mc);
    end
    qbmm_data(jj,1,1) = 1.0;
    qbmm_data(jj,2,1) = 0.0;
    qbmm_data(jj,3,1) = 1.0;
    qbmm_data(jj,4,1) = 0.0;
    qbmm_data(jj,5,1) = 0.0;
    %T_qbmm = data_qbmm(:,1)';
% for ii=1:5
%     x = data_qbmm(:,ii+2);
%     input_data(jj,ii,:) = interp1(T_qbmm,x,T_mc);
% end

end

R10_ML = zeros(total_cases,1801);
R01_ML = zeros(total_cases,1801);
R20_ML = zeros(total_cases,1801);
R11_ML = zeros(total_cases,1801);
R02_ML = zeros(total_cases,1801);

R10_MC = zeros(total_cases,1801);
R01_MC = zeros(total_cases,1801);
R20_MC = zeros(total_cases,1801);
R11_MC = zeros(total_cases,1801);
R02_MC = zeros(total_cases,1801);
 
R10_QBMM = zeros(total_cases,1801);
R01_QBMM = zeros(total_cases,1801);
R20_QBMM = zeros(total_cases,1801);
R11_QBMM = zeros(total_cases,1801);
R02_QBMM = zeros(total_cases,1801);

for jj=1:total_cases
    R10_MC(jj,:) = mc_data(jj,1,:);
    R01_MC(jj,:) = mc_data(jj,2,:);
    R20_MC(jj,:) = mc_data(jj,3,:);
    R11_MC(jj,:) = mc_data(jj,4,:);
    R02_MC(jj,:) = mc_data(jj,5,:);
    
    
    R10_ML(jj,:) = predictions(jj,1,:);
    R01_ML(jj,:) = predictions(jj,2,:);
    R20_ML(jj,:) = predictions(jj,3,:);
    R11_ML(jj,:) = predictions(jj,4,:);
    R02_ML(jj,:) = predictions(jj,5,:);
    
    
    R10_QBMM(jj,:) = qbmm_data(jj,1,:);
    R01_QBMM(jj,:) = qbmm_data(jj,2,:);
    R20_QBMM(jj,:) = qbmm_data(jj,3,:);
    R11_QBMM(jj,:) = qbmm_data(jj,4,:);
    R02_QBMM(jj,:) = qbmm_data(jj,5,:);
end


% figeta = figure(1);
% set(gcf,'color','w');
% scrsz = get(groot,'ScreenSize');
% %figeta.Position = [1 scrsz(4)/15 scrsz(3)/(1.5*2.0) scrsz(4)/(1.5*2.4)];
% set(figeta,'Units','Inches','Position',[0,0,6,8],'PaperUnits','Inches','PaperSize',[6,8])
% 
% 
% for ii=1:8
%    p(ii) = subplot(4,2,ii); 
% end
% 
% iflag = [2,4,6,8,10,12,14,16];
% Plot_titles = {'$P_{ratio} = 0.20$';...
%                '$P_{ratio} = 0.30$';...
%                '$P_{ratio} = 0.40$';...
%                '$P_{ratio} = 0.50$';...
%                '$P_{ratio} = 0.60$';...
%                '$P_{ratio} = 0.70$';...
%                '$P_{ratio} = 0.80$';...
%                '$P_{ratio} = 0.90$';};
% for ii=1:8
% axes(p(ii));
% p(ii).Position = [0.04+0.50*mod(ii-1,2),0.77-0.25*floor((ii-1)/2),0.44,0.18];
% plot(T_mc,R10_MC(iflag(ii),:),'Color',[1,0,0],'linewidth',1.0)
% hold on
% plot(T_mc,R10_QBMM(iflag(ii),:),'Color',[0,0.5,0],'linewidth',1.0)
% hold on
% plot(T_mc,R10_ML(iflag(ii),:),'Color',[0,0.5,1],'linewidth',1.0)
% flag1 = max( max(R10_MC(iflag(ii),:)), max(max(R10_QBMM(iflag(ii),:)), max(R10_ML(iflag(ii),:))) );
% flag2 = min( min(R10_MC(iflag(ii),:)), min(min(R10_QBMM(iflag(ii),:)), min(R10_ML(iflag(ii),:))) );
% set(p(ii),'XTick',[0,20,40,60])
% set(p(ii),'XTickLabel',[0,20,40,60],'fontsize',7)
% ylim([1.2*flag2,flag1*1.2])
% if (ii==1)
%     legend({'MC','QBMM','ML'},'interpreter','latex','fontsize',8,'orientation','horizontal','Position',[0.02,0.49,1.,1.],'box','off')
%     text(65,flag2*1.2+(flag1-flag2)*1.8,'$\mu_{32}$','interpreter','latex','fontsize',16)
% end
% title(Plot_titles{ii},'interpreter','latex','fontsize',12)
% end
% print(figeta,'-dpdf','Moment10_Results.pdf');




% figeta = figure(2);
% set(gcf,'color','w');
% scrsz = get(groot,'ScreenSize');
% figeta.Position = [1 scrsz(4)/15 scrsz(3)/(1.5*2.0) scrsz(4)/(1.5*2.4)];
% %set(figeta,'Units','Inches','Position',[0,0,1.5,1.5],'PaperUnits','Inches','PaperSize',[1.5,1.5])
% 
% 
% 
% for ii=1:8
% subplot(4,2,ii)
% plot(T_mc,R01_MC(iflag(ii),:),'Color',[1,0,0],'linewidth',1.5)
% hold on
% plot(T_mc,R01_QBMM(iflag(ii),:),'Color',[0,0.5,0],'linewidth',1.5)
% hold on
% plot(T_mc,R01_ML(iflag(ii),:),'Color',[0,0.5,1],'linewidth',1.5)
% if (ii==1)
%     legend({'MC','ML'},'interpreter','latex','fontsize',8,'orientation','horizontal')
%     text(65,1.8,'$\mu_{01}$','interpreter','latex','fontsize',16)
% end
% title(Plot_titles{ii},'interpreter','latex','fontsize',12)
% end
% print(figeta,'-dpdf','Moment01_Results.pdf');


        
        

% figeta = figure(3);
% set(gcf,'color','w');
% scrsz = get(groot,'ScreenSize');
% figeta.Position = [1 scrsz(4)/15 scrsz(3)/(1.5*2.0) scrsz(4)/(1.5*2.4)];
% %set(figeta,'Units','Inches','Position',[0,0,1.5,1.5],'PaperUnits','Inches','PaperSize',[1.5,1.5])
% 
% 
% 
% for ii=1:8
% subplot(4,2,ii)
% plot(T_mc,R30_MC(iflag(ii),:),'Color',[1,0,0],'linewidth',1.5)
% hold on
% plot(T_mc,R30_QBMM(iflag(ii),:),'Color',[0,0.5,0],'linewidth',1.5)
% hold on
% plot(T_mc,R30_ML(iflag(ii),:),'Color',[0,0.5,1],'linewidth',1.5)
% if (ii==1)
%     legend({'MC','QBMM','ML'},'interpreter','latex','fontsize',8,'orientation','horizontal')
%     text(65,1.5,'$\mu_{30}$','interpreter','latex','fontsize',16)
% end
% title(Plot_titles{ii},'interpreter','latex','fontsize',12)
% end
% print(figeta,'-dpdf','Moment30_Results.pdf');


% figeta = figure(4);
% set(gcf,'color','w');
% scrsz = get(groot,'ScreenSize');
% figeta.Position = [1 scrsz(4)/15 scrsz(3)/(1.5*2.0) scrsz(4)/(1.5*2.4)];
% %set(figeta,'Units','Inches','Position',[0,0,1.5,1.5],'PaperUnits','Inches','PaperSize',[1.5,1.5])
% 
% 
% 
% for ii=1:8
% subplot(4,2,ii)
% plot(T_mc,R3g_MC(iflag(ii),:),'Color',[1,0,0],'linewidth',1.5)
% hold on
% plot(T_mc,R3g_QBMM(iflag(ii),:),'Color',[0,0.5,0],'linewidth',1.5)
% hold on
% plot(T_mc,R3g_ML(iflag(ii),:),'Color',[0,0.5,1],'linewidth',1.5)
% if (ii==1)
%     legend({'MC','QBMM','ML'},'interpreter','latex','fontsize',8,'orientation','horizontal')
%     text(60,4.5,'$\mu_{3(1-\gamma),0}$','interpreter','latex','fontsize',16)
% end
% title(Plot_titles{ii},'interpreter','latex','fontsize',12)
% end
% print(figeta,'-dpdf','Moment3g_Results.pdf');



% Plot L-2 error
l2_error_ml = zeros(5,total_cases);
l2_error_qbmm = zeros(5,total_cases);
l2_measure  = zeros(5,total_cases);

for ii=1:total_cases
    for jj=1:5
        for tt=1:size(T_mc,2)-1
            l2_measure(jj,ii) = l2_measure(jj,ii) +mc_data(ii,jj,tt)^2;
            l2_error_ml(jj,ii) = l2_error_ml(jj,ii) +(mc_data(ii,jj,tt)-predictions(ii,jj,tt))^2;
            l2_error_qbmm(jj,ii) = l2_error_qbmm(jj,ii) +(mc_data(ii,jj,tt)-qbmm_data(ii,jj,tt))^2;
        end
        l2_measure(jj,ii) = sqrt(l2_measure(jj,ii));
        l2_error_ml(jj,ii) = sqrt(l2_error_ml(jj,ii))/l2_measure(jj,ii);
        l2_error_qbmm(jj,ii) = sqrt(l2_error_qbmm(jj,ii))/l2_measure(jj,ii);
    end
end


pressure_ratios = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9];

figeta = figure(5);
set(gcf,'color','w');
scrsz = get(groot,'ScreenSize');
figeta.Position = [1 scrsz(4)/15 scrsz(3)/(1.5*2.0) scrsz(4)/(1.5*2.4)];
set(figeta,'Units','Inches','Position',[0,0,6,4],'PaperUnits','Inches','PaperSize',[6,4])

p1 = subplot(2,3,1);
%p1.Position = [0.10,0.55,0.30,0.30];

semilogy(pressure_ratios,l2_error_qbmm(1,1:2:16),'Color',[1,0,0],'linewidth',1.5,'Marker','*','linestyle','none')
hold on
semilogy(pressure_ratios,l2_error_ml(1,1:2:16),'Color',[0,0.5,1],'linewidth',1.5,'Marker','o','linestyle','none')
xlim([0.15,0.95])
ylim([0.0001,1])
set(gca,'YTick',[10^(-4),10^(-3),10^(-2),10^(-1),10^0,10^1])
a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'fontsize',7)
%set(gca,'YTickLabel',[10^(-3),10^(-2),10^(-1),10^0,10^1],'fontsize',7)
ylabel('$L^2$-error','interpreter','latex','fontsize',12)
title('$\mu_{1,0}$','interpreter','latex','fontsize',12)
%legend({'QBMM','QBMM-ML'},'interpreter','latex','fontsize',10,'orientation','horizontal','Position',[0.016,0.47,1.,1.],'box','off')
legend({'QBMM','QBMM-ML'},'interpreter','latex','fontsize',10,'orientation','vertical','Position',[0.004,0.30,1.,1.],'box','off')

p2 = subplot(2,3,3);
%p2.Position = [0.55,0.55,0.30,0.30];

semilogy(pressure_ratios,l2_error_qbmm(2,1:2:16),'Color',[1,0,0],'linewidth',1.5,'Marker','*','linestyle','none')
hold on
semilogy(pressure_ratios,l2_error_ml(2,1:2:16),'Color',[0,0.5,1],'linewidth',1.5,'Marker','o','linestyle','none')
xlim([0.15,0.95])
ylim([0.001,10])
set(gca,'YTick',[10^(-3),10^(-2),10^(-1),10^0,10^1])
a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'fontsize',7)
%set(gca,'YTickLabel',[10^(-3),10^(-2),10^(-1),10^0,10^1],'fontsize',7)
ylabel('$L^2$-error','interpreter','latex','fontsize',12)
title('$\mu_{0,1}$','interpreter','latex','fontsize',12)
%legend({'QBMM','QBMM-ML'},'interpreter','latex','fontsize',10,'orientation','horizontal','Position',[0.016,0.47,1.,1.],'box','off')



subplot(2,3,4)
semilogy(pressure_ratios,l2_error_qbmm(3,1:2:16),'Color',[1,0,0],'linewidth',1.5,'Marker','*','linestyle','none')
hold on
semilogy(pressure_ratios,l2_error_ml(3,1:2:16),'Color',[0,0.5,1],'linewidth',1.5,'Marker','o','linestyle','none')
xlim([0.15,0.95])
ylim([0.0001,1])
set(gca,'YTick',[10^(-4),10^(-3),10^(-2),10^(-1),10^0,10^1])
a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'fontsize',7)
title('$\mu_{2,0}$','interpreter','latex','fontsize',12)

subplot(2,3,5)
semilogy(pressure_ratios,l2_error_qbmm(4,1:2:16),'Color',[1,0,0],'linewidth',1.5,'Marker','*','linestyle','none')
hold on
semilogy(pressure_ratios,l2_error_ml(4,1:2:16),'Color',[0,0.5,1],'linewidth',1.5,'Marker','o','linestyle','none')
xlim([0.15,0.95])
ylim([0.001,10])
set(gca,'YTick',[10^(-3),10^(-2),10^(-1),10^0,10^1])
a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'fontsize',7)
xlabel('$p/p_0$','interpreter','latex','fontsize',12)
ylabel('$L^2$-error','interpreter','latex','fontsize',12)
title('$\mu_{1,1}$','interpreter','latex','fontsize',12)

subplot(2,3,6)
semilogy(pressure_ratios,l2_error_qbmm(5,1:2:16),'Color',[1,0,0],'linewidth',1.5,'Marker','*','linestyle','none')
hold on
semilogy(pressure_ratios,l2_error_ml(5,1:2:16),'Color',[0,0.5,1],'linewidth',1.5,'Marker','o','linestyle','none')
xlim([0.15,0.95])
ylim([0.0001,1])
set(gca,'YTick',[10^(-4),10^(-3),10^(-2),10^(-1),10^0])
a = get(gca,'YTickLabel');
set(gca,'YTickLabel',a,'fontsize',7)
xlabel('$p/p_0$','interpreter','latex','fontsize',12)
title('$\mu_{0,2}$','interpreter','latex','fontsize',12)
print(figeta,'-dpdf','L2_error.pdf');
