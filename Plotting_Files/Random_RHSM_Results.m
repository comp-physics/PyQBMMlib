clear;
close all;
clc;


load(['../ML_Code/RHSM_Random_MLQBMM','.mat']);
total_cases = 30;
total_times = 2001;

T = linspace(0,200,total_times);


Rm12_ML = zeros(total_cases,total_times);
Rm21_ML = zeros(total_cases,total_times);
Rm40_ML = zeros(total_cases,total_times);
Rm10_ML = zeros(total_cases,total_times);
Rm11_ML = zeros(total_cases,total_times);
Rm30_ML = zeros(total_cases,total_times);
Rm13_ML = zeros(total_cases,total_times);
Rm22_ML = zeros(total_cases,total_times);
Rm41_ML = zeros(total_cases,total_times);

Rm12_MC = zeros(total_cases,total_times);
Rm21_MC = zeros(total_cases,total_times);
Rm40_MC = zeros(total_cases,total_times);
Rm10_MC = zeros(total_cases,total_times);
Rm11_MC = zeros(total_cases,total_times);
Rm30_MC = zeros(total_cases,total_times);
Rm13_MC = zeros(total_cases,total_times);
Rm22_MC = zeros(total_cases,total_times);
Rm41_MC = zeros(total_cases,total_times);

Pressure = zeros(total_cases,total_times);

for kk=1:total_cases
   for tt=1:total_times

        Rm12_MC(kk,tt) = output_data(kk,1,tt);
        Rm21_MC(kk,tt) = output_data(kk,2,tt);
        Rm40_MC(kk,tt) = output_data(kk,3,tt);
        Rm10_MC(kk,tt) = output_data(kk,4,tt);
        Rm11_MC(kk,tt) = output_data(kk,5,tt);
        Rm30_MC(kk,tt) = output_data(kk,6,tt);
        Rm13_MC(kk,tt) = output_data(kk,7,tt);
        Rm22_MC(kk,tt) = output_data(kk,8,tt);
        Rm41_MC(kk,tt) = output_data(kk,9,tt);
        
        
        Rm12_ML(kk,tt) = predictions(kk,1,tt);
        Rm21_ML(kk,tt) = predictions(kk,2,tt);
        Rm40_ML(kk,tt) = predictions(kk,3,tt);
        Rm10_ML(kk,tt) = predictions(kk,4,tt);
        Rm11_ML(kk,tt) = predictions(kk,5,tt);
        Rm30_ML(kk,tt) = predictions(kk,6,tt);
        Rm13_ML(kk,tt) = predictions(kk,7,tt);
        Rm22_ML(kk,tt) = predictions(kk,8,tt);
        Rm41_ML(kk,tt) = predictions(kk,9,tt);
        
        Pressure(kk,tt) = input_data(kk,6,tt);
        
   end
end

l2_error_ml = zeros(9,total_cases);
l2_measure  = zeros(9,total_cases);

for ii=1:total_cases
    for jj=1:9
        for tt=1:total_times-1
            l2_measure(jj,ii) = l2_measure(jj,ii) +(output_data(ii,jj,tt))^2;
            l2_error_ml(jj,ii) = l2_error_ml(jj,ii) +(output_data(ii,jj,tt)-predictions(ii,jj,tt))^2;
        end
        l2_measure(jj,ii) = sqrt(l2_measure(jj,ii));
        l2_error_ml(jj,ii) = sqrt(l2_error_ml(jj,ii))/l2_measure(jj,ii);
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

plot_cases = [14,23,24,25];
plot_cases = [1,2,3,4];

for ii=1:4

    
    
tstart =  total_times-100;
tend   = total_times;

    
iflag = ii;
axes(p(iflag))
p(iflag).Position = [0.05+0.24*(ii-1),0.82,0.20,0.15/1.075]; % [left bottom width height]
semilogy([1,2,3,4,5,6,7,8,9],l2_error_ml(:,plot_cases(ii)),'Color',[0,0.5,1],'linewidth',1.5,'Marker','o','linestyle','none','MarkerFaceColor',[0,0.5,1],'Markersize',5)

xlim([0.5,9.5])
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
legend({'QBMM-ML'},'interpreter','latex','fontsize',8,'orientation','horizontal','Position',[0.016,0.48,1.,1.],'box','off')
end



iflag = 4+ii;
axes(p(iflag))
p(iflag).Position = [0.05+0.24*(ii-1),0.64,0.20,0.12/1.075]; % [left bottom width height]
plot(T(tstart:1:tend),Rm12_MC(plot_cases(ii),tstart:1:tend),'Color',[0.2,0.2,0.2],'linewidth',1.0,'linestyle','-') 
hold on
plot(T,Rm12_ML(plot_cases(ii),:),'Color',[0.8,0,0],'linewidth',1.0,'linestyle','--') 
hold on
plot(T,Rm12_ML(plot_cases(ii),:),'Color',[0,0.5,1],'linewidth',1.0) 
xlim([T(tstart),T(tend)])

ymin = min([floor(20*min(min(0.9*Rm12_MC(plot_cases(ii),tstart:tend))))/20,floor(20*min(min(0.9*Rm12_ML(plot_cases(ii),tstart:tend))))/20,floor(20*min(min(0.9*Rm12_ML(plot_cases(ii),tstart:tend))))/20]);
ymax = max([ceil(20*max(max(1.1*Rm12_MC(plot_cases(ii),tstart:tend))))/20,ceil(20*max(max(1.1*Rm12_ML(plot_cases(ii),tstart:tend))))/20,ceil(20*max(max(1.1*Rm12_ML(plot_cases(ii),tstart:tend))))/20]);
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
plot(T(tstart:tend),Rm21_MC(plot_cases(ii),tstart:tend),'Color',[0.2,0.2,0.2],'linewidth',1.0,'linestyle','-') 
hold on
plot(T,Rm21_ML(plot_cases(ii),:),'Color',[0.8,0,0],'linewidth',1.0,'linestyle','--') 
hold on
plot(T,Rm21_ML(plot_cases(ii),:),'Color',[0,0.5,1],'linewidth',1.0) 
xlim([T(tstart),T(tend)])

ymin = min([floor(20*min(min(0.9*Rm21_MC(plot_cases(ii),tstart:tend))))/20,floor(20*min(min(0.9*Rm21_ML(plot_cases(ii),tstart:tend))))/20,floor(20*min(min(0.9*Rm21_ML(plot_cases(ii),tstart:tend))))/20]);
ymax = max([ceil(20*max(max(1.1*Rm21_MC(plot_cases(ii),tstart:tend))))/20,ceil(20*max(max(1.1*Rm21_ML(plot_cases(ii),tstart:tend))))/20,ceil(20*max(max(1.1*Rm21_ML(plot_cases(ii),tstart:tend))))/20]);
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
plot(T(tstart:tend),Rm40_MC(plot_cases(ii),tstart:tend),'Color',[0.2,0.2,0.2],'linewidth',1.0,'linestyle','-') 
hold on
plot(T,Rm40_ML(plot_cases(ii),:),'Color',[0.8,0,0],'linewidth',1.0,'linestyle','--') 
hold on
plot(T,Rm40_ML(plot_cases(ii),:),'Color',[0,0.5,1],'linewidth',1.0) 
xlim([T(tstart),T(tend)])

ymin = min([floor(100*min(min(0.9*Rm40_MC(plot_cases(ii),tstart:tend))))/100,floor(100*min(min(0.9*Rm40_ML(plot_cases(ii),tstart:tend))))/100,floor(100*min(min(0.9*Rm40_ML(plot_cases(ii),tstart:tend))))/100]);
ymax = max([ceil(100*max(max(1.1*Rm40_MC(plot_cases(ii),tstart:tend))))/100,ceil(100*max(max(1.1*Rm40_ML(plot_cases(ii),tstart:tend))))/100,ceil(100*max(max(1.1*Rm40_ML(plot_cases(ii),tstart:tend))))/100]);
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
plot(T(tstart:tend),Rm10_MC(plot_cases(ii),tstart:tend),'Color',[0.2,0.2,0.2],'linewidth',1.0,'linestyle','-') 
hold on
plot(T,Rm10_ML(plot_cases(ii),:),'Color',[0.8,0,0],'linewidth',1.0,'linestyle','--') 
hold on
plot(T,Rm10_ML(plot_cases(ii),:),'Color',[0,0.5,1],'linewidth',1.0) 
xlim([T(tstart),T(tend)])
ymin = min([floor(20*min(min(0.9*Rm10_MC(plot_cases(ii),tstart:tend))))/20,floor(20*min(min(0.9*Rm10_ML(plot_cases(ii),tstart:tend))))/20,floor(20*min(min(0.9*Rm10_ML(plot_cases(ii),tstart:tend))))/20]);
ymax = max([ceil(20*max(max(1.1*Rm10_MC(plot_cases(ii),tstart:tend))))/20,ceil(20*max(max(1.1*Rm10_ML(plot_cases(ii),tstart:tend))))/20,ceil(20*max(max(1.1*Rm10_ML(plot_cases(ii),tstart:tend))))/20]);
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
plot(T(tstart:tend),Rm11_MC(plot_cases(ii),tstart:tend),'Color',[0.2,0.2,0.2],'linewidth',1.0,'linestyle','-') 
hold on
plot(T,Rm11_ML(plot_cases(ii),:),'Color',[0.8,0,0],'linewidth',1.0,'linestyle','--') 
hold on
plot(T,Rm11_ML(plot_cases(ii),:),'Color',[0,0.5,1],'linewidth',1.0) 
xlim([T(tstart),T(tend)])
ymin = min([floor(20*min(min(0.9*Rm11_MC(plot_cases(ii),tstart:tend))))/20,floor(20*min(min(0.9*Rm11_ML(plot_cases(ii),tstart:tend))))/20,floor(20*min(min(0.9*Rm11_ML(plot_cases(ii),tstart:tend))))/20]);
ymax = max([ceil(20*max(max(1.1*Rm11_MC(plot_cases(ii),tstart:tend))))/20,ceil(20*max(max(1.1*Rm11_ML(plot_cases(ii),tstart:tend))))/20,ceil(20*max(max(1.1*Rm11_ML(plot_cases(ii),tstart:tend))))/20]);
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

print(figeta,'-dpdf','RHSM_Random_Forcing_Results.pdf');
savefig(figeta,'RHSM_Random_Forcing_Results.fig');


