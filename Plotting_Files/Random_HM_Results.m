clear;
close all;
clc;


load(['../ML_Code/HM_Random_MLQBMM','.mat']);
total_cases = 25;
total_times = 2001;

T = linspace(0,200,total_times);


R32_ML = zeros(total_cases,total_times);
R21_ML = zeros(total_cases,total_times);
R30_ML = zeros(total_cases,total_times);
R3g_ML = zeros(total_cases,total_times);

R32_MC = zeros(total_cases,total_times);
R21_MC = zeros(total_cases,total_times);
R30_MC = zeros(total_cases,total_times);
R3g_MC = zeros(total_cases,total_times);

R32_QBMM = zeros(total_cases,total_times);
R21_QBMM = zeros(total_cases,total_times);
R30_QBMM = zeros(total_cases,total_times);
R3g_QBMM = zeros(total_cases,total_times);

Pressure = zeros(total_cases,total_times);

for kk=1:total_cases
   for tt=1:total_times
        R30_MC(kk,tt) = output_data(kk,1,tt)+input_data(kk,6,tt);
        R21_MC(kk,tt) = output_data(kk,2,tt)+input_data(kk,7,tt);
        R32_MC(kk,tt) = output_data(kk,3,tt)+input_data(kk,8,tt);
        R3g_MC(kk,tt) = output_data(kk,4,tt)+input_data(kk,9,tt);
        
        R30_ML(kk,tt) = predictions(kk,1,tt);
        R21_ML(kk,tt) = predictions(kk,2,tt);
        R32_ML(kk,tt) = predictions(kk,3,tt);
        R3g_ML(kk,tt) = predictions(kk,4,tt);
        
        R30_QBMM(kk,tt) = input_data(kk,6,tt);
        R21_QBMM(kk,tt) = input_data(kk,7,tt);
        R32_QBMM(kk,tt) = input_data(kk,8,tt);
        R3g_QBMM(kk,tt) = input_data(kk,9,tt);
        
        Pressure(kk,tt) = input_data(kk,10,tt);
        
   end
end


% Plot L-2 error
l2_error_ml = zeros(4,total_cases);
l2_error_qbmm = zeros(4,total_cases);
l2_measure  = zeros(4,total_cases);

for ii=1:total_cases
    for jj=1:4
        for tt=1:total_times-1
            l2_measure(jj,ii) = l2_measure(jj,ii) +(output_data(ii,jj,tt)+input_data(ii,jj+5,tt))^2;
            l2_error_ml(jj,ii) = l2_error_ml(jj,ii) +(output_data(ii,jj,tt)+input_data(ii,jj+5,tt)-predictions(ii,jj,tt))^2;
            l2_error_qbmm(jj,ii) = l2_error_qbmm(jj,ii) +(output_data(ii,jj,tt))^2;
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

for ii=1:24
   p(ii) = subplot(4,6,ii); 
end

plot_cases = [14,23,24,25];
%plot_cases = [1,2,3,4];

for ii=1:4

    
    
tstart =  total_times-100;
tend   = total_times;

    
iflag = ii;
axes(p(iflag))
p(iflag).Position = [0.05+0.24*(ii-1),0.82,0.20,0.15/1.075]; % [left bottom width height]
semilogy([1,2,3,4],l2_error_qbmm(:,plot_cases(ii)),'Color',[1,0,0],'linewidth',1.5,'Marker','s','linestyle','none','MarkerFaceColor',[1,0,0],'Markersize',5)
hold on
semilogy([1,2,3,4],l2_error_ml(:,plot_cases(ii)),'Color',[0,0.5,1],'linewidth',1.5,'Marker','o','linestyle','none','MarkerFaceColor',[0,0.5,1],'Markersize',5)

xlim([0.5,4.5])
ylim([0.001,1])
%set(gca, 'XDir','reverse')
set(gca,'YTick',[10^(-3),10^(-2),10^(-1),10^0,10^1])
a = get(gca,'YTickLabel');  
set(gca,'YTickLabel',a,'fontsize',7)

grid on
set(p(iflag),'GridAlpha',0.2);

set(p(iflag),'XTick',[1,2,3,4])
set(p(iflag),'XTickLabel',[{'\mu_{3,0}','\mu_{2,1}','\mu_{3,2}','\mu_{3(1-\gamma),0}'}],'fontsize',7)
if (ii == 1)
legend({'QBMM','QBMM-ML'},'interpreter','latex','fontsize',8,'orientation','horizontal','Position',[0.016,0.48,1.,1.],'box','off')
end



iflag = 4+ii;
axes(p(iflag))
p(iflag).Position = [0.05+0.24*(ii-1),0.62,0.20,0.13/1.075]; % [left bottom width height]
plot(T(tstart:1:tend),R30_MC(plot_cases(ii),tstart:1:tend),'Color',[0.2,0.2,0.2],'linewidth',1.0,'linestyle','-') 
hold on
plot(T,R30_QBMM(plot_cases(ii),:),'Color',[0.8,0,0],'linewidth',1.0,'linestyle','--') 
hold on
plot(T,R30_ML(plot_cases(ii),:),'Color',[0,0.5,1],'linewidth',1.0) 
xlim([T(tstart),T(tend)])

ymin = min([floor(20*min(min(0.9*R30_MC(plot_cases(ii),tstart:tend))))/20,floor(20*min(min(0.9*R30_ML(plot_cases(ii),tstart:tend))))/20,floor(20*min(min(0.9*R30_QBMM(plot_cases(ii),tstart:tend))))/20]);
ymax = max([ceil(20*max(max(1.1*R30_MC(plot_cases(ii),tstart:tend))))/20,ceil(20*max(max(1.1*R30_ML(plot_cases(ii),tstart:tend))))/20,ceil(20*max(max(1.1*R30_QBMM(plot_cases(ii),tstart:tend))))/20]);
ylim([ ymin, ymax   ])

set(p(iflag),'GridAlpha',0.2);
set(p(iflag),'linewidth',1.0);
set(p(iflag),'XTick',[30,45,60])
set(p(iflag),'YTick',[ymin,  ymax])
set(p(iflag),'XTickLabel',[30,45,60],'fontsize',7)
set(p(iflag),'YTickLabel',[ymin,  ymax],'fontsize',7)

if (ii == 1)
yname1 = ylabel("$\mu_{3,0}$",'interpreter','latex','fontsize',10);
set(p(iflag),'YLabel',yname1);
ylabh = get(gca,'ylabel');
set(ylabh,'position',get(ylabh,'position') - [-0.8 0 0]);
legend({'MC','QBMM','QBMM-ML'},'interpreter','latex','fontsize',8,'orientation','horizontal','Position',[0.016,0.27,1.,1.],'box','off')
end




iflag = 8+ii;
axes(p(iflag))
p(iflag).Position = [0.05+0.24*(ii-1),0.47,0.20,0.13/1.075]; % [left bottom width height]
plot(T(tstart:tend),R21_MC(plot_cases(ii),tstart:tend),'Color',[0.2,0.2,0.2],'linewidth',1.0,'linestyle','-') 
hold on
plot(T,R21_QBMM(plot_cases(ii),:),'Color',[0.8,0,0],'linewidth',1.0,'linestyle','--') 
hold on
plot(T,R21_ML(plot_cases(ii),:),'Color',[0,0.5,1],'linewidth',1.0) 
xlim([T(tstart),T(tend)])

ymin = min([floor(20*min(min(0.9*R21_MC(plot_cases(ii),tstart:tend))))/20,floor(20*min(min(0.9*R21_ML(plot_cases(ii),tstart:tend))))/20,floor(20*min(min(0.9*R21_QBMM(plot_cases(ii),tstart:tend))))/20]);
ymax = max([ceil(20*max(max(1.1*R21_MC(plot_cases(ii),tstart:tend))))/20,ceil(20*max(max(1.1*R21_ML(plot_cases(ii),tstart:tend))))/20,ceil(20*max(max(1.1*R21_QBMM(plot_cases(ii),tstart:tend))))/20]);
ylim([ ymin, ymax   ])

set(p(iflag),'GridAlpha',0.2);
set(p(iflag),'linewidth',1.0);
set(p(iflag),'XTick',[30,45,60])
set(p(iflag),'YTick',[ymin,  ymax])
set(p(iflag),'XTickLabel',[30,45,60],'fontsize',7)
set(p(iflag),'YTickLabel',[ymin,  ymax],'fontsize',7)

if (ii == 1)
yname1 = ylabel("$\mu_{2,1}$",'interpreter','latex','fontsize',10);
set(p(iflag),'YLabel',yname1);
ylabh = get(gca,'ylabel');
set(ylabh,'position',get(ylabh,'position') - [-0.8 0 0]);
end



iflag = 12+ii;
axes(p(iflag))
p(iflag).Position = [0.05+0.24*(ii-1),0.32,0.20,0.13/1.075]; % [left bottom width height]
plot(T(tstart:tend),R32_MC(plot_cases(ii),tstart:tend),'Color',[0.2,0.2,0.2],'linewidth',1.0,'linestyle','-') 
hold on
plot(T,R32_QBMM(plot_cases(ii),:),'Color',[0.8,0,0],'linewidth',1.0,'linestyle','--') 
hold on
plot(T,R32_ML(plot_cases(ii),:),'Color',[0,0.5,1],'linewidth',1.0) 
xlim([T(tstart),T(tend)])

ymin = min([floor(100*min(min(0.9*R32_MC(plot_cases(ii),tstart:tend))))/100,floor(100*min(min(0.9*R32_ML(plot_cases(ii),tstart:tend))))/100,floor(100*min(min(0.9*R32_QBMM(plot_cases(ii),tstart:tend))))/100]);
ymax = max([ceil(100*max(max(1.1*R32_MC(plot_cases(ii),tstart:tend))))/100,ceil(100*max(max(1.1*R32_ML(plot_cases(ii),tstart:tend))))/100,ceil(100*max(max(1.1*R32_QBMM(plot_cases(ii),tstart:tend))))/100]);
ylim([ ymin, ymax   ])

set(p(iflag),'GridAlpha',0.2);
set(p(iflag),'linewidth',1.0);
set(p(iflag),'XTick',[30,45,60])
set(p(iflag),'YTick',[ymin,  ymax])
set(p(iflag),'XTickLabel',[30,45,60],'fontsize',7)
set(p(iflag),'YTickLabel',[ymin,  ymax],'fontsize',7)

if (ii == 1)
yname1 = ylabel("$\mu_{3,2}$",'interpreter','latex','fontsize',10);
set(p(iflag),'YLabel',yname1);
ylabh = get(gca,'ylabel');
set(ylabh,'position',get(ylabh,'position') - [-0.8 0 0]);
end



iflag = 16+ii;
axes(p(iflag))
p(iflag).Position = [0.05+0.24*(ii-1),0.170,0.20,0.13/1.075]; % [left bottom width height]
plot(T(tstart:tend),R3g_MC(plot_cases(ii),tstart:tend),'Color',[0.2,0.2,0.2],'linewidth',1.0,'linestyle','-') 
hold on
plot(T,R3g_QBMM(plot_cases(ii),:),'Color',[0.8,0,0],'linewidth',1.0,'linestyle','--') 
hold on
plot(T,R3g_ML(plot_cases(ii),:),'Color',[0,0.5,1],'linewidth',1.0) 
xlim([T(tstart),T(tend)])
ymin = min([floor(20*min(min(0.9*R3g_MC(plot_cases(ii),tstart:tend))))/20,floor(20*min(min(0.9*R3g_ML(plot_cases(ii),tstart:tend))))/20,floor(20*min(min(0.9*R3g_QBMM(plot_cases(ii),tstart:tend))))/20]);
ymax = max([ceil(20*max(max(1.1*R3g_MC(plot_cases(ii),tstart:tend))))/20,ceil(20*max(max(1.1*R3g_ML(plot_cases(ii),tstart:tend))))/20,ceil(20*max(max(1.1*R3g_QBMM(plot_cases(ii),tstart:tend))))/20]);
ylim([ ymin, ymax   ])

set(p(iflag),'GridAlpha',0.2);
set(p(iflag),'linewidth',1.0);
set(p(iflag),'XTick',[30,45,60])
set(p(iflag),'YTick',[ymin,  ymax])
set(p(iflag),'XTickLabel',[30,45,60],'fontsize',7)
set(p(iflag),'YTickLabel',[ymin,  ymax],'fontsize',7)

if (ii == 1)
yname1 = ylabel("$\mu_{3(1-\gamma),0}$",'interpreter','latex','fontsize',10);
set(p(iflag),'YLabel',yname1);
ylabh = get(gca,'ylabel');
set(ylabh,'position',get(ylabh,'position') - [-0.8 0 0]);

end




iflag = 20+ii;
axes(p(iflag))
p(iflag).Position = [0.05+0.24*(ii-1),0.035,0.19,0.10]; % [left bottom width height]
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

print(figeta,'-dpdf','HM_Random_Forcing_Results.pdf');
savefig(figeta,'HM_Random_Forcing_Results.fig');
