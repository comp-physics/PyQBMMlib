clear;
close all;
clc;

load(['../data/Random_Forcing/MC_HM_Random_Pressure_Realization31','.mat']);
moments_MC = moments;
total_times = 2001;
T_MC = linspace(0,200,total_times);
load(['../data/Random_Forcing/QBMM_HM_Random_Pressure_Realization31','.mat']);
moments_QBMM = zeros(31,total_times);
T_old = T;
for ii=1:31
    moments_QBMM(ii,:) = interp1(T,moments(:,ii),T_MC);

end
moments_QBMM(:,1) = moments_MC(:,1);

load(['../data/Random_Forcing/QBMM_HM_Random_Pressure_Realization32','.mat']);
moments_HYB = zeros(31,total_times);
for ii=1:31
    moments_HYB(ii,:) = interp1(T,moments(:,ii),T_MC);

end
moments_HYB(:,1) = moments_MC(:,1);
T = T_MC;

total_cases = 1;
% Plot L-2 error
l2_error_ml = zeros(5,total_cases);
l2_error_qbmm = zeros(5,total_cases);
l2_measure  = zeros(5,total_cases);

for ii=1:total_cases
    for jj=1:5
        for tt=1:total_times-1
            l2_measure(jj,ii) = l2_measure(jj,ii) +(moments_MC(jj+1,tt))^2;
            l2_error_ml(jj,ii) = l2_error_ml(jj,ii) +(moments_MC(jj+1,tt)-moments_HYB(jj+1,tt))^2;
            l2_error_qbmm(jj,ii) = l2_error_qbmm(jj,ii) +(moments_MC(jj+1,tt)-moments_QBMM(jj+1,tt))^2;
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

plot_cases = [1,1,1,1];
%plot_cases = [1,2,3,4];

for ii=1:4

    
    
tstart =  total_times-100;
tend   = total_times;

    
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
moment_case = 2;
plot(T(tstart:1:tend),moments_MC(moment_case,tstart:1:tend),'Color',[0.2,0.2,0.2],'linewidth',1.0,'linestyle','-') 
hold on
plot(T,moments_QBMM(moment_case,:),'Color',[0.8,0,0],'linewidth',1.0,'linestyle','--') 
hold on
plot(T,moments_HYB(moment_case,:),'Color',[0,0.5,1],'linewidth',1.0) 
xlim([T(tstart),T(tend)])

ymin = min([floor(20*min(min(0.9*moments_MC(moment_case,tstart:tend))))/20,floor(20*min(min(0.9*moments_HYB(moment_case,tstart:tend))))/20,floor(20*min(min(0.9*moments_QBMM(moment_case,tstart:tend))))/20]);
ymax = max([ceil(20*max(max(1.1*moments_MC(moment_case,tstart:tend))))/20,ceil(20*max(max(1.1*moments_HYB(moment_case,tstart:tend))))/20,ceil(20*max(max(1.1*moments_QBMM(moment_case,tstart:tend))))/20]);
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
moment_case = 3;
plot(T(tstart:1:tend),moments_MC(moment_case,tstart:1:tend),'Color',[0.2,0.2,0.2],'linewidth',1.0,'linestyle','-') 
hold on
plot(T,moments_QBMM(moment_case,:),'Color',[0.8,0,0],'linewidth',1.0,'linestyle','--') 
hold on
plot(T,moments_HYB(moment_case,:),'Color',[0,0.5,1],'linewidth',1.0) 
xlim([T(tstart),T(tend)])

ymin = min([floor(20*min(min(0.9*moments_MC(moment_case,tstart:tend))))/20,floor(20*min(min(0.9*moments_HYB(moment_case,tstart:tend))))/20,floor(20*min(min(0.9*moments_QBMM(moment_case,tstart:tend))))/20]);
ymax = max([ceil(20*max(max(1.1*moments_MC(moment_case,tstart:tend))))/20,ceil(20*max(max(1.1*moments_HYB(moment_case,tstart:tend))))/20,ceil(20*max(max(1.1*moments_QBMM(moment_case,tstart:tend))))/20]);
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
moment_case = 4;
plot(T(tstart:1:tend),moments_MC(moment_case,tstart:1:tend),'Color',[0.2,0.2,0.2],'linewidth',1.0,'linestyle','-') 
hold on
plot(T,moments_QBMM(moment_case,:),'Color',[0.8,0,0],'linewidth',1.0,'linestyle','--') 
hold on
plot(T,moments_HYB(moment_case,:),'Color',[0,0.5,1],'linewidth',1.0) 
xlim([T(tstart),T(tend)])

ymin = min([floor(20*min(min(0.9*moments_MC(moment_case,tstart:tend))))/20,floor(20*min(min(0.9*moments_HYB(moment_case,tstart:tend))))/20,floor(20*min(min(0.9*moments_QBMM(moment_case,tstart:tend))))/20]);
ymax = max([ceil(20*max(max(1.1*moments_MC(moment_case,tstart:tend))))/20,ceil(20*max(max(1.1*moments_HYB(moment_case,tstart:tend))))/20,ceil(20*max(max(1.1*moments_QBMM(moment_case,tstart:tend))))/20]);
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
moment_case = 4;
plot(T(tstart:1:tend),moments_MC(moment_case,tstart:1:tend),'Color',[0.2,0.2,0.2],'linewidth',1.0,'linestyle','-') 
hold on
plot(T,moments_QBMM(moment_case,:),'Color',[0.8,0,0],'linewidth',1.0,'linestyle','--') 
hold on
plot(T,moments_HYB(moment_case,:),'Color',[0,0.5,1],'linewidth',1.0) 
xlim([T(tstart),T(tend)])

ymin = min([floor(20*min(min(0.9*moments_MC(moment_case,tstart:tend))))/20,floor(20*min(min(0.9*moments_HYB(moment_case,tstart:tend))))/20,floor(20*min(min(0.9*moments_QBMM(moment_case,tstart:tend))))/20]);
ymax = max([ceil(20*max(max(1.1*moments_MC(moment_case,tstart:tend))))/20,ceil(20*max(max(1.1*moments_HYB(moment_case,tstart:tend))))/20,ceil(20*max(max(1.1*moments_QBMM(moment_case,tstart:tend))))/20]);
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
moment_case = 5;
plot(T(tstart:1:tend),moments_MC(moment_case,tstart:1:tend),'Color',[0.2,0.2,0.2],'linewidth',1.0,'linestyle','-') 
hold on
plot(T,moments_QBMM(moment_case,:),'Color',[0.8,0,0],'linewidth',1.0,'linestyle','--') 
hold on
plot(T,moments_HYB(moment_case,:),'Color',[0,0.5,1],'linewidth',1.0) 
xlim([T(tstart),T(tend)])

ymin = min([floor(20*min(min(0.9*moments_MC(moment_case,tstart:tend))))/20,floor(20*min(min(0.9*moments_HYB(moment_case,tstart:tend))))/20,floor(20*min(min(0.9*moments_QBMM(moment_case,tstart:tend))))/20]);
ymax = max([ceil(20*max(max(1.1*moments_MC(moment_case,tstart:tend))))/20,ceil(20*max(max(1.1*moments_HYB(moment_case,tstart:tend))))/20,ceil(20*max(max(1.1*moments_QBMM(moment_case,tstart:tend))))/20]);
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
plot(T,pressure(:),'Color',[0.2,0.2,0.2],'linewidth',1.0)
xlim([T(tstart),T(tend)])
ylim([0.8*(min(min(pressure(tstart:tend)))),1.1*(max(max(pressure(tstart:tend))))])
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

print(figeta,'-dpdf','LMHYB_Random_Forcing_Results.pdf');
savefig(figeta,'LMHYB_Random_Forcing_Results.fig');




%%

load(['../data/Random_Forcing/MC_HM_Random_Pressure_Realization1','.mat']);
moments_MC = moments;



load(['../ML_Code/3M_Random_MLQBMM_variable','.mat']);

moments_ML = zeros(4,total_times);
flag = zeros(1,56624);
for ii=1:4
    flag(:) = predictions(1,ii,:);
    moments_ML(ii,:) = interp1(T_old,flag,T_MC);

end

figure(2)
subplot(2,2,1)
plot(T_MC,moments_ML(1,:))
hold on
plot(T_MC,moments_MC(7,:))

subplot(2,2,2)
plot(T_MC,moments_ML(2,:))
hold on
plot(T_MC,moments_MC(10,:))

subplot(2,2,3)
plot(T_MC,moments_ML(3,:))
hold on
plot(T_MC,moments_MC(11,:))

subplot(2,2,4)
plot(T_MC,moments_ML(4,:))
hold on
plot(T_MC,moments_MC(15,:))
ylim([min(moments_MC(15,:)),max(moments_MC(15,:))])



