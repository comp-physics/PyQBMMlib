clear;
close all;
clc;


load(['../ML_Code/ML_Predictions/LM_Random_MLQBMM_Approach2_Weights1_Pressure','.mat']);
total_cases = 30;
total_times = 2001;

T = linspace(0,200,total_times);


R10_ML = zeros(total_cases,total_times);
R01_ML = zeros(total_cases,total_times);
R20_ML = zeros(total_cases,total_times);
R11_ML = zeros(total_cases,total_times);
R02_ML = zeros(total_cases,total_times);

R10_MC = zeros(total_cases,total_times);
R01_MC = zeros(total_cases,total_times);
R20_MC = zeros(total_cases,total_times);
R11_MC = zeros(total_cases,total_times);
R02_MC = zeros(total_cases,total_times);

R10_QBMM = zeros(total_cases,total_times);
R01_QBMM = zeros(total_cases,total_times);
R20_QBMM = zeros(total_cases,total_times);
R11_QBMM = zeros(total_cases,total_times);
R02_QBMM = zeros(total_cases,total_times);

Pressure = zeros(total_cases,total_times);

for kk=1:total_cases
   for tt=1:total_times
        R10_MC(kk,tt) = LM_MC(kk,1,tt);
        R01_MC(kk,tt) = LM_MC(kk,2,tt);
        R20_MC(kk,tt) = LM_MC(kk,3,tt);
        R11_MC(kk,tt) = LM_MC(kk,4,tt);
        R02_MC(kk,tt) = LM_MC(kk,5,tt);
        
        R10_ML(kk,tt) = predictions(kk,1,tt);
        R01_ML(kk,tt) = predictions(kk,2,tt);
        R20_ML(kk,tt) = predictions(kk,3,tt);
        R11_ML(kk,tt) = predictions(kk,4,tt);
        R02_ML(kk,tt) = predictions(kk,5,tt);
        
        R10_QBMM(kk,tt) = LM_QBMM(kk,1,tt);
        R01_QBMM(kk,tt) = LM_QBMM(kk,2,tt);
        R20_QBMM(kk,tt) = LM_QBMM(kk,3,tt);
        R11_QBMM(kk,tt) = LM_QBMM(kk,4,tt);
        R02_QBMM(kk,tt) = LM_QBMM(kk,5,tt);
        
        Pressure(kk,tt) = LM_pressure(kk,tt);
        
   end
end


% Plot L-2 error
l2_error_ml = zeros(5,total_cases);
l2_error_qbmm = zeros(5,total_cases);
l2_measure  = zeros(5,total_cases);

for ii=1:total_cases
    for jj=1:5
        for tt=1:total_times-1
            l2_measure(jj,ii) = l2_measure(jj,ii) +(LM_MC(ii,jj,tt))^2;
            l2_error_ml(jj,ii) = l2_error_ml(jj,ii) +(LM_MC(ii,jj,tt)-predictions(ii,jj,tt))^2;
            l2_error_qbmm(jj,ii) = l2_error_qbmm(jj,ii) +(LM_MC(ii,jj,tt)-LM_QBMM(ii,jj,tt))^2;
        end
        l2_measure(jj,ii) = sqrt(l2_measure(jj,ii));
        l2_error_ml(jj,ii) = sqrt(l2_error_ml(jj,ii))/l2_measure(jj,ii);
        l2_error_qbmm(jj,ii) = sqrt(l2_error_qbmm(jj,ii))/l2_measure(jj,ii);
    end
end






figeta = figure(1);
set(gcf,'color','w');
scrsz = get(groot,'ScreenSize');
%figeta.Position = [1 scrsz(4)/15 scrsz(3)/(1.5*1.5*0.8) scrsz(4)/(0.8*1.5)];
set(figeta,'Units','Inches','Position',[0,0,6.0,(1440/2560)*2.15*6.0],'PaperUnits','Inches','PaperSize',[6.0,(1440/2560)*2.15*6.0])

for ii=1:28
   p(ii) = subplot(4,7,ii); 
end

plot_cases = [21,23,28,29];
%plot_cases = [1,2,3,4];

for ii=1:4

    
    
tstart =  total_times-100;
tend   = total_times;
% tstart =  51;
% tend   = 151;
    
iflag = ii;
axes(p(iflag))
p(iflag).Position = [0.05+0.24*(ii-1),0.82,0.20,0.15/1.075]; % [left bottom width height]
semilogy([1,2,3,4,5],l2_error_qbmm(:,plot_cases(ii)),'Color',[1,0,0],'linewidth',1.5,'Marker','s','linestyle','none','MarkerFaceColor',[1,0,0],'Markersize',5)
hold on
semilogy([1,2,3,4,5],l2_error_ml(:,plot_cases(ii)),'Color',[0,0.5,1],'linewidth',1.5,'Marker','o','linestyle','none','MarkerFaceColor',[0,0.5,1],'Markersize',5)

xlim([0.5,5.5])
ylim([0.001,1])
%set(gca, 'XDir','reverse')
set(gca,'YTick',[10^(-3),10^(-2),10^(-1),10^0,10^1])
a = get(gca,'YTickLabel');  
set(gca,'YTickLabel',a,'fontsize',7)

grid on
set(p(iflag),'GridAlpha',0.2);

set(p(iflag),'XTick',[1,2,3,4,5])
set(p(iflag),'XTickLabel',[{'\mu_{1,0}','\mu_{0,1}','\mu_{2,0}','\mu_{1,1}','\mu_{0,2}'}],'fontsize',7)
if (ii == 1)
legend({'QBMM','QBMM-ML'},'interpreter','latex','fontsize',8,'orientation','horizontal','Position',[0.016,0.48,1.,1.],'box','off')
end



iflag = 4+ii;
axes(p(iflag))
p(iflag).Position = [0.05+0.24*(ii-1),0.64,0.20,0.12/1.075]; % [left bottom width height]
plot(T(tstart:4:tend),R10_MC(plot_cases(ii),tstart:4:tend),'Color',[0.2,0.2,0.2],'linewidth',1.0,'linestyle','-','Marker','o','Markersize',3) 
hold on
plot(T,R10_QBMM(plot_cases(ii),:),'Color',[0.8,0,0],'linewidth',1.0,'linestyle','--') 
hold on
plot(T,R10_ML(plot_cases(ii),:),'Color',[0,0.5,1],'linewidth',1.0) 
xlim([T(tstart),T(tend)])

ymin = min([floor(20*min(min(0.9*R10_MC(plot_cases(ii),tstart:tend))))/20,floor(20*min(min(0.9*R10_ML(plot_cases(ii),tstart:tend))))/20,floor(20*min(min(0.9*R10_QBMM(plot_cases(ii),tstart:tend))))/20]);
ymax = max([ceil(20*max(max(1.1*R10_MC(plot_cases(ii),tstart:tend))))/20,ceil(20*max(max(1.1*R10_ML(plot_cases(ii),tstart:tend))))/20,ceil(20*max(max(1.1*R10_QBMM(plot_cases(ii),tstart:tend))))/20]);
ylim([ ymin, ymax   ])

set(p(iflag),'GridAlpha',0.2);
set(p(iflag),'linewidth',1.0);
set(p(iflag),'XTick',[30,45,60])
set(p(iflag),'YTick',[ymin,  ymax])
set(p(iflag),'XTickLabel',[30,45,60],'fontsize',7)
set(p(iflag),'YTickLabel',[ymin,  ymax],'fontsize',7)

if (ii == 1)
yname1 = ylabel("$\mu_{1,0}$",'interpreter','latex','fontsize',10);
set(p(iflag),'YLabel',yname1);
ylabh = get(gca,'ylabel');
set(ylabh,'position',get(ylabh,'position') - [-0.8 0 0]);
legend({'MC','QBMM','QBMM-ML'},'interpreter','latex','fontsize',8,'orientation','horizontal','Position',[0.016,0.27,1.,1.],'box','off')
end




iflag = 8+ii;
axes(p(iflag))
p(iflag).Position = [0.05+0.24*(ii-1),0.51,0.20,0.12/1.075]; % [left bottom width height]
plot(T(tstart:4:tend),R01_MC(plot_cases(ii),tstart:4:tend),'Color',[0.2,0.2,0.2],'linewidth',1.0,'linestyle','-','Marker','o','Markersize',3) 
hold on
plot(T,R01_QBMM(plot_cases(ii),:),'Color',[0.8,0,0],'linewidth',1.0,'linestyle','--') 
hold on
plot(T,R01_ML(plot_cases(ii),:),'Color',[0,0.5,1],'linewidth',1.0) 
xlim([T(tstart),T(tend)])

ymin = min([floor(20*min(min(0.9*R01_MC(plot_cases(ii),tstart:tend))))/20,floor(20*min(min(0.9*R01_ML(plot_cases(ii),tstart:tend))))/20,floor(20*min(min(0.9*R01_QBMM(plot_cases(ii),tstart:tend))))/20]);
ymax = max([ceil(20*max(max(1.1*R01_MC(plot_cases(ii),tstart:tend))))/20,ceil(20*max(max(1.1*R01_ML(plot_cases(ii),tstart:tend))))/20,ceil(20*max(max(1.1*R01_QBMM(plot_cases(ii),tstart:tend))))/20]);
ylim([ ymin, ymax   ])

set(p(iflag),'GridAlpha',0.2);
set(p(iflag),'linewidth',1.0);
set(p(iflag),'XTick',[30,45,60])
set(p(iflag),'YTick',[ymin,  ymax])
set(p(iflag),'XTickLabel',[30,45,60],'fontsize',7)
set(p(iflag),'YTickLabel',[ymin,  ymax],'fontsize',7)

if (ii == 1)
yname1 = ylabel("$\mu_{0,1}$",'interpreter','latex','fontsize',10);
set(p(iflag),'YLabel',yname1);
ylabh = get(gca,'ylabel');
set(ylabh,'position',get(ylabh,'position') - [-0.8 0 0]);
end



iflag = 12+ii;
axes(p(iflag))
p(iflag).Position = [0.05+0.24*(ii-1),0.38,0.20,0.12/1.075]; % [left bottom width height]
plot(T(tstart:4:tend),R20_MC(plot_cases(ii),tstart:4:tend),'Color',[0.2,0.2,0.2],'linewidth',1.0,'linestyle','-','Marker','o','Markersize',3) 
hold on
plot(T,R20_QBMM(plot_cases(ii),:),'Color',[0.8,0,0],'linewidth',1.0,'linestyle','--') 
hold on
plot(T,R20_ML(plot_cases(ii),:),'Color',[0,0.5,1],'linewidth',1.0) 
xlim([T(tstart),T(tend)])

ymin = min([floor(100*min(min(0.9*R20_MC(plot_cases(ii),tstart:tend))))/100,floor(100*min(min(0.9*R20_ML(plot_cases(ii),tstart:tend))))/100,floor(100*min(min(0.9*R20_QBMM(plot_cases(ii),tstart:tend))))/100]);
ymax = max([ceil(100*max(max(1.1*R20_MC(plot_cases(ii),tstart:tend))))/100,ceil(100*max(max(1.1*R20_ML(plot_cases(ii),tstart:tend))))/100,ceil(100*max(max(1.1*R20_QBMM(plot_cases(ii),tstart:tend))))/100]);
ylim([ ymin, ymax   ])

set(p(iflag),'GridAlpha',0.2);
set(p(iflag),'linewidth',1.0);
set(p(iflag),'XTick',[30,45,60])
set(p(iflag),'YTick',[ymin,  ymax])
set(p(iflag),'XTickLabel',[30,45,60],'fontsize',7)
set(p(iflag),'YTickLabel',[ymin,  ymax],'fontsize',7)

if (ii == 1)
yname1 = ylabel("$\mu_{2,0}$",'interpreter','latex','fontsize',10);
set(p(iflag),'YLabel',yname1);
ylabh = get(gca,'ylabel');
set(ylabh,'position',get(ylabh,'position') - [-0.8 0 0]);
end



iflag = 16+ii;
axes(p(iflag))
p(iflag).Position = [0.05+0.24*(ii-1),0.250,0.20,0.12/1.075]; % [left bottom width height]
plot(T(tstart:4:tend),R11_MC(plot_cases(ii),tstart:4:tend),'Color',[0.2,0.2,0.2],'linewidth',1.0,'linestyle','-','Marker','o','Markersize',3) 
hold on
plot(T,R11_QBMM(plot_cases(ii),:),'Color',[0.8,0,0],'linewidth',1.0,'linestyle','--') 
hold on
plot(T,R11_ML(plot_cases(ii),:),'Color',[0,0.5,1],'linewidth',1.0) 
xlim([T(tstart),T(tend)])
ymin = min([floor(20*min(min(0.9*R11_MC(plot_cases(ii),tstart:tend))))/20,floor(20*min(min(0.9*R11_ML(plot_cases(ii),tstart:tend))))/20,floor(20*min(min(0.9*R11_QBMM(plot_cases(ii),tstart:tend))))/20]);
ymax = max([ceil(20*max(max(1.1*R11_MC(plot_cases(ii),tstart:tend))))/20,ceil(20*max(max(1.1*R11_ML(plot_cases(ii),tstart:tend))))/20,ceil(20*max(max(1.1*R11_QBMM(plot_cases(ii),tstart:tend))))/20]);
ylim([ ymin, ymax   ])

set(p(iflag),'GridAlpha',0.2);
set(p(iflag),'linewidth',1.0);
set(p(iflag),'XTick',[30,45,60])
set(p(iflag),'YTick',[ymin,  ymax])
set(p(iflag),'XTickLabel',[30,45,60],'fontsize',7)
set(p(iflag),'YTickLabel',[ymin,  ymax],'fontsize',7)

if (ii == 1)
yname1 = ylabel("$\mu_{1,1}$",'interpreter','latex','fontsize',10);
set(p(iflag),'YLabel',yname1);
ylabh = get(gca,'ylabel');
set(ylabh,'position',get(ylabh,'position') - [-0.8 0 0]);

end


iflag = 20+ii;
axes(p(iflag))
p(iflag).Position = [0.05+0.24*(ii-1),0.120,0.20,0.12/1.075]; % [left bottom width height]
plot(T(tstart:4:tend),R02_MC(plot_cases(ii),tstart:4:tend),'Color',[0.2,0.2,0.2],'linewidth',1.0,'linestyle','-','Marker','o','Markersize',3) 
hold on
plot(T,R02_QBMM(plot_cases(ii),:),'Color',[0.8,0,0],'linewidth',1.0,'linestyle','--') 
hold on
plot(T,R02_ML(plot_cases(ii),:),'Color',[0,0.5,1],'linewidth',1.0) 
xlim([T(tstart),T(tend)])
ymin = min([floor(20*min(min(0.9*R02_MC(plot_cases(ii),tstart:tend))))/20,floor(20*min(min(0.9*R02_ML(plot_cases(ii),tstart:tend))))/20,floor(20*min(min(0.9*R02_QBMM(plot_cases(ii),tstart:tend))))/20]);
ymax = max([ceil(20*max(max(1.1*R02_MC(plot_cases(ii),tstart:tend))))/20,ceil(20*max(max(1.1*R02_ML(plot_cases(ii),tstart:tend))))/20,ceil(20*max(max(1.1*R02_QBMM(plot_cases(ii),tstart:tend))))/20]);
ylim([ ymin, ymax   ])

set(p(iflag),'GridAlpha',0.2);
set(p(iflag),'linewidth',1.0);
set(p(iflag),'XTick',[30,45,60])
set(p(iflag),'YTick',[ymin,  ymax])
set(p(iflag),'XTickLabel',[30,45,60],'fontsize',7)
set(p(iflag),'YTickLabel',[ymin,  ymax],'fontsize',7)

if (ii == 1)
yname1 = ylabel("$\mu_{0,2}$",'interpreter','latex','fontsize',10);
set(p(iflag),'YLabel',yname1);
ylabh = get(gca,'ylabel');
set(ylabh,'position',get(ylabh,'position') - [-0.8 0 0]);

end







iflag = 24+ii;
axes(p(iflag))
p(iflag).Position = [0.05+0.24*(ii-1),0.035,0.19,0.07]; % [left bottom width height]
plot(T,Pressure(plot_cases(ii),:),'Color',[0.2,0.2,0.2],'linewidth',1.0)
xlim([T(tstart),T(tend)])
ylim([0.8*(min(min(Pressure(:,tstart:tend)))),1.1*(max(max(Pressure(:,tstart:tend))))])
%grid on;
set(p(iflag),'GridAlpha',0.2);
set(p(iflag),'linewidth',1.0);

set(p(iflag),'XTick',[190,195,200])
set(p(iflag),'YTick',[0.60,1.0,1.40])
set(p(iflag),'XTickLabel',[190,195,200],'fontsize',7)
set(p(iflag),'YTickLabel',[0.60,1.0,1.40],'fontsize',7)


xname1 = xlabel("time",'interpreter','latex','fontsize',8);
set(p(iflag),'XLabel',xname1);
%text(38.0,1.20*(max(max(Pressure(:,tstart:tend)))),'Pressure $p(t)$','interpreter','latex','fontsize',8)

if (ii == 1)
yname1 = ylabel('Pressure','interpreter','latex','fontsize',8);
set(p(iflag),'YLabel',yname1);
ylabh = get(gca,'ylabel');
set(ylabh,'position',get(ylabh,'position') - [-0.4 0 0]);
end


    
end

print(figeta,'-dpdf','Figures/LM_Random_Forcing_Results_Approach2_Weights1_Pressure.pdf');
savefig(figeta,'Figures/LM_Random_Forcing_Results_Approach2_Weights1_Pressure.fig');
