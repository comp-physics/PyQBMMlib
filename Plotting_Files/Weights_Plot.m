clear;
close all;
clc;

load(['../ML_Code/ML_Predictions/LM_Random_MLQBMM_Approach4_Weights4','.mat']);

QBMM_Points = zeros(4,12,4001);
total_cases  = 20;
ml_dim       = 12;
total_points = 4001;
LM_QBMM = LM_MC;
for ii=1:total_cases
for tt=1:total_points
    a_flag = (LM_QBMM(ii,4,tt) -LM_QBMM(ii,1,tt)*LM_QBMM(ii,2,tt))/sqrt( LM_QBMM(ii,3,tt) -LM_QBMM(ii,1,tt)*LM_QBMM(ii,1,tt) );
   QBMM_Points(ii,1:3:ml_dim,tt) = 0.25;
   
   QBMM_Points(ii, 2,tt) = LM_QBMM(ii,1,tt) +sqrt(LM_QBMM(ii,3,tt)-LM_QBMM(ii,1,tt)*LM_QBMM(ii,1,tt));
   QBMM_Points(ii, 5,tt) = LM_QBMM(ii,1,tt) +sqrt(LM_QBMM(ii,3,tt)-LM_QBMM(ii,1,tt)*LM_QBMM(ii,1,tt));
   QBMM_Points(ii, 8,tt) = LM_QBMM(ii,1,tt) -sqrt(LM_QBMM(ii,3,tt)-LM_QBMM(ii,1,tt)*LM_QBMM(ii,1,tt));
   QBMM_Points(ii,11,tt) = LM_QBMM(ii,1,tt) -sqrt(LM_QBMM(ii,3,tt)-LM_QBMM(ii,1,tt)*LM_QBMM(ii,1,tt));
   
   QBMM_Points(ii, 3,tt) = LM_QBMM(ii,2,tt)+a_flag +sqrt(LM_QBMM(ii,5,tt)-a_flag*a_flag -LM_QBMM(ii,2,tt)*LM_QBMM(ii,2,tt));
   QBMM_Points(ii, 6,tt) = LM_QBMM(ii,2,tt)+a_flag -sqrt(LM_QBMM(ii,5,tt)-a_flag*a_flag -LM_QBMM(ii,2,tt)*LM_QBMM(ii,2,tt));
   QBMM_Points(ii, 9,tt) = LM_QBMM(ii,2,tt)-a_flag +sqrt(LM_QBMM(ii,5,tt)-a_flag*a_flag -LM_QBMM(ii,2,tt)*LM_QBMM(ii,2,tt));
   QBMM_Points(ii,12,tt) = LM_QBMM(ii,2,tt)-a_flag -sqrt(LM_QBMM(ii,5,tt)-a_flag*a_flag -LM_QBMM(ii,2,tt)*LM_QBMM(ii,2,tt));
end
end

load('../data/Random_Forcing/MC_HM_Random_Pressure_Realization31.mat');
R_samples_tot = R_samples;
Rd_samples_tot = Rd_samples;
load('../data/Random_Forcing/MC_HM_Random_Pressure_Realization32.mat');
R_samples_tot = [R_samples_tot;R_samples];
Rd_samples_tot = [Rd_samples_tot;Rd_samples];
load('../data/Random_Forcing/MC_HM_Random_Pressure_Realization33.mat');
R_samples_tot = [R_samples_tot;R_samples];
Rd_samples_tot = [Rd_samples_tot;Rd_samples];



R_samples = R_samples_tot;
Rd_samples = Rd_samples_tot;
clear R_samples_tot Rd_samples_tot;
nsamples = size(R_samples,1);


nR  = 129;
nRd = 129;

Rmin = min(min(R_samples)); Rmax = max(max(R_samples));
Rdmin = min(min(Rd_samples)); Rdmax = max(max(Rd_samples));
dR  = (Rmax-Rmin)/(nR-1);
dRd = (Rdmax-Rdmin)/(nRd-1);
R_grid  = linspace(Rmin,Rmax,nR);
Rd_grid = linspace(Rdmin,Rdmax,nRd); 
pdf = zeros(nR,nRd);
xx = zeros(nR,nRd);
yy = zeros(nR,nRd);

for ii=1:nR
    for jj=1:nRd
       xx(ii,jj) = R_grid(ii);
       yy(ii,jj) = Rd_grid(jj);
    end
end



figeta = figure(1);
set(gcf,'color','w');
scrsz = get(groot,'ScreenSize');
set(figeta,'Units','Inches','Position',[0,0,6.0,5.0],'PaperUnits','Inches','PaperSize',[6.0,5.0])
p(1) = subplot(1,1,1);

% total_times = 4001;
% xp = zeros(4,total_times);
% yp = zeros(4,total_times);
% wp = zeros(4,total_times);
% for tt=1:total_times
%     a_flag = (predictions(1,4,tt)-predictions(1,1,tt)*predictions(1,2,tt))/sqrt(predictions(1,3,tt) -predictions(1,1,tt)^2);
%     xp(jj,tt) = predictions(1,1,tt) +sqrt(predictions(1,3,tt) -predictions(1,1,tt)^2) +Weights_predictions(1,3*(jj-1)+2,tt);
%     yp(jj,tt) = predictions(1,2,tt)
% end

case_flag = 2;
tt = 3500;
pdf = zeros(nR,nRd);
for pp=1:nsamples
    rflag = 1+floor((R_samples(pp,tt)-Rmin)/dR);
    rdflag = 1+floor((Rd_samples(pp,tt)-Rdmin)/dRd);
    pdf(rflag,rdflag) = pdf(rflag,rdflag)+1;
end

pdf = pdf./nsamples;

axes(p(1))
[hf1,hc1] = contourf(xx,yy,pdf(:,:),[0.002,0.004,0.006,0.008,0.01],'edgecolor','none');
hold on
hp2 = plot(QBMM_Points(case_flag,2:3:12,tt),QBMM_Points(case_flag,3:3:12,tt),'s','Color',[0.8,0,0],'MarkerFaceColor',[0.8,0,0],'linestyle','none','Markersize',6);
hold on
hp1 = plot(Weights_predictions(case_flag,2:3:12,tt),Weights_predictions(case_flag,3:3:12,tt),'o','Color',[0,0.5,1],'MarkerFaceColor',[0,0.5,1],'linestyle','none','Markersize',6);
text(1.61*(Rmax-Rmin)+Rmin,1.15*(Rdmax-Rdmin)+Rdmin,'QBMM Scheme','interpreter','latex','fontsize',16)
text(1.38*(Rmax-Rmin)+Rmin,1.05*(Rdmax-Rdmin)+Rdmin,'Weight \hspace{1.5cm} R-value \hspace{1.5cm} $\dot{R}$-value','interpreter','latex','fontsize',14)

axes(p(1))
%set(hf1,'CData',pdf(:,:));
hc1.ZData = pdf(:,:);
% set(hp2,'XData',QBMM_Points(case_flag,2:3:12,tt))
% set(hp2,'YData',QBMM_Points(case_flag,3:3:12,tt))
% set(hp1,'XData',Weights_predictions(case_flag,2:3:12,tt))
% set(hp1,'YData',Weights_predictions(case_flag,3:3:12,tt))
c1 = colorbar;
colormap(gca,flip(bone));
caxis([0.0,0.02])
xname1 = xlabel('$R$','interpreter','latex','fontsize',12,'rot',0);
yname1 = ylabel('$\dot{R}$','interpreter','latex','fontsize',12,'rot',90);
%title1 = title('Bottom Layer - PV','interpreter','latex','fontsize',22,'rot',0);
set(gca,'XLabel',xname1);
set(gca,'YLabel',yname1);
ylabh = get(c1,'YLabel');
set(ylabh,'Position',[3.5 0.010 0])
ylabel(c1,'$f(R,\dot{R})$','interpreter','latex','fontsize',12,'rot',90)
%legend([hp1,hp2],{'QBMM','QBMM-ML'},'interpreter','latex','fontsize',14,'orientation','horizontal','Position',[-0.04,0.40,1.,1.],'box','off')


% tt = 3570;
% pdf = zeros(nR,nRd);
% for pp=1:nsamples
%     rflag = 1+floor((R_samples(pp,tt)-Rmin)/dR);
%     rdflag = 1+floor((Rd_samples(pp,tt)-Rdmin)/dRd);
%     pdf(rflag,rdflag) = pdf(rflag,rdflag)+1;
% end
% pdf = pdf./nsamples;
% 
% [hf2,hc2] = contourf(xx,yy,pdf(:,:),[0.002,0.004,0.006,0.008,0.01],'edgecolor','none');
% hold on
% hp3 = plot(QBMM_Points(case_flag,2:3:12,tt),QBMM_Points(case_flag,3:3:12,tt),'s','Color',[0.8,0,0],'MarkerFaceColor',[0.8,0,0],'linestyle','none','Markersize',6);
% hold on
% hp4 = plot(Weights_predictions(case_flag,2:3:12,tt),Weights_predictions(case_flag,3:3:12,tt),'o','Color',[0,0.5,1],'MarkerFaceColor',[0,0.5,1],'linestyle','none','Markersize',6);
% text(1.61*(Rmax-Rmin)+Rmin,1.15*(Rdmax-Rdmin)+Rdmin,'QBMM Scheme','interpreter','latex','fontsize',16)
% text(1.38*(Rmax-Rmin)+Rmin,1.05*(Rdmax-Rdmin)+Rdmin,'Weight \hspace{1.5cm} R-value \hspace{1.5cm} $\dot{R}$-value','interpreter','latex','fontsize',14)


tt = 3600;
pdf = zeros(nR,nRd);
for pp=1:nsamples
    rflag = 1+floor((R_samples(pp,tt)-Rmin)/dR);
    rdflag = 1+floor((Rd_samples(pp,tt)-Rdmin)/dRd);
    pdf(rflag,rdflag) = pdf(rflag,rdflag)+1;
end
pdf = pdf./nsamples;

[hf3,hc3] = contourf(xx,yy,pdf(:,:),[0.002,0.004,0.006,0.008,0.01],'edgecolor','none');
hold on
hp5 = plot(QBMM_Points(case_flag,2:3:12,tt),QBMM_Points(case_flag,3:3:12,tt),'s','Color',[0.8,0,0],'MarkerFaceColor',[0.8,0,0],'linestyle','none','Markersize',6);
hold on
hp6 = plot(Weights_predictions(case_flag,2:3:12,tt),Weights_predictions(case_flag,3:3:12,tt),'o','Color',[0,0.5,1],'MarkerFaceColor',[0,0.5,1],'linestyle','none','Markersize',6);
text(1.61*(Rmax-Rmin)+Rmin,1.15*(Rdmax-Rdmin)+Rdmin,'QBMM Scheme','interpreter','latex','fontsize',16)
text(1.38*(Rmax-Rmin)+Rmin,1.05*(Rdmax-Rdmin)+Rdmin,'Weight \hspace{1.5cm} R-value \hspace{1.5cm} $\dot{R}$-value','interpreter','latex','fontsize',14)



tt = 3640;
pdf = zeros(nR,nRd);
for pp=1:nsamples
    rflag = 1+floor((R_samples(pp,tt)-Rmin)/dR);
    rdflag = 1+floor((Rd_samples(pp,tt)-Rdmin)/dRd);
    pdf(rflag,rdflag) = pdf(rflag,rdflag)+1;
end
pdf = pdf./nsamples;

[hf4,hc4] = contourf(xx,yy,pdf(:,:),[0.002,0.004,0.006,0.008,0.01],'edgecolor','none');
hold on
hp7 = plot(QBMM_Points(case_flag,2:3:12,tt),QBMM_Points(case_flag,3:3:12,tt),'s','Color',[0.8,0,0],'MarkerFaceColor',[0.8,0,0],'linestyle','none','Markersize',6);
hold on
hp8 = plot(Weights_predictions(case_flag,2:3:12,tt),Weights_predictions(case_flag,3:3:12,tt),'o','Color',[0,0.5,1],'MarkerFaceColor',[0,0.5,1],'linestyle','none','Markersize',6);
text(1.61*(Rmax-Rmin)+Rmin,1.15*(Rdmax-Rdmin)+Rdmin,'QBMM Scheme','interpreter','latex','fontsize',16)
text(1.38*(Rmax-Rmin)+Rmin,1.05*(Rdmax-Rdmin)+Rdmin,'Weight \hspace{1.5cm} R-value \hspace{1.5cm} $\dot{R}$-value','interpreter','latex','fontsize',14)




tt = 3660;
pdf = zeros(nR,nRd);
for pp=1:nsamples
    rflag = 1+floor((R_samples(pp,tt)-Rmin)/dR);
    rdflag = 1+floor((Rd_samples(pp,tt)-Rdmin)/dRd);
    pdf(rflag,rdflag) = pdf(rflag,rdflag)+1;
end
pdf = pdf./nsamples;

[hf5,hc5] = contourf(xx,yy,pdf(:,:),[0.002,0.004,0.006,0.008,0.01],'edgecolor','none');
hold on
hp9 = plot(QBMM_Points(case_flag,2:3:12,tt),QBMM_Points(case_flag,3:3:12,tt),'s','Color',[0.8,0,0],'MarkerFaceColor',[0.8,0,0],'linestyle','none','Markersize',6);
hold on
hp10 = plot(Weights_predictions(case_flag,2:3:12,tt),Weights_predictions(case_flag,3:3:12,tt),'o','Color',[0,0.5,1],'MarkerFaceColor',[0,0.5,1],'linestyle','none','Markersize',6);
text(1.61*(Rmax-Rmin)+Rmin,1.15*(Rdmax-Rdmin)+Rdmin,'QBMM Scheme','interpreter','latex','fontsize',16)
text(1.38*(Rmax-Rmin)+Rmin,1.05*(Rdmax-Rdmin)+Rdmin,'Weight \hspace{1.5cm} R-value \hspace{1.5cm} $\dot{R}$-value','interpreter','latex','fontsize',14)






% 
% tt = 3680;
% pdf = zeros(nR,nRd);
% for pp=1:nsamples
%     rflag = 1+floor((R_samples(pp,tt)-Rmin)/dR);
%     rdflag = 1+floor((Rd_samples(pp,tt)-Rdmin)/dRd);
%     pdf(rflag,rdflag) = pdf(rflag,rdflag)+1;
% end
% pdf = pdf./nsamples;
% 
% [hf6,hc6] = contourf(xx,yy,pdf(:,:),[0.002,0.004,0.006,0.008,0.01],'edgecolor','none');
% hold on
% hp11 = plot(QBMM_Points(case_flag,2:3:12,tt),QBMM_Points(case_flag,3:3:12,tt),'s','Color',[0.8,0,0],'MarkerFaceColor',[0.8,0,0],'linestyle','none','Markersize',6);
% hold on
% hp12 = plot(Weights_predictions(case_flag,2:3:12,tt),Weights_predictions(case_flag,3:3:12,tt),'o','Color',[0,0.5,1],'MarkerFaceColor',[0,0.5,1],'linestyle','none','Markersize',6);
% text(1.61*(Rmax-Rmin)+Rmin,1.15*(Rdmax-Rdmin)+Rdmin,'QBMM Scheme','interpreter','latex','fontsize',16)
% text(1.38*(Rmax-Rmin)+Rmin,1.05*(Rdmax-Rdmin)+Rdmin,'Weight \hspace{1.5cm} R-value \hspace{1.5cm} $\dot{R}$-value','interpreter','latex','fontsize',14)
% 

tt = 3690;
pdf = zeros(nR,nRd);
for pp=1:nsamples
    rflag = 1+floor((R_samples(pp,tt)-Rmin)/dR);
    rdflag = 1+floor((Rd_samples(pp,tt)-Rdmin)/dRd);
    pdf(rflag,rdflag) = pdf(rflag,rdflag)+1;
end
pdf = pdf./nsamples;

[hf7,hc7] = contourf(xx,yy,pdf(:,:),[0.002,0.004,0.006,0.008,0.01],'edgecolor','none');
hold on
hp13 = plot(QBMM_Points(case_flag,2:3:12,tt),QBMM_Points(case_flag,3:3:12,tt),'s','Color',[0.8,0,0],'MarkerFaceColor',[0.8,0,0],'linestyle','none','Markersize',6);
hold on
hp14 = plot(Weights_predictions(case_flag,2:3:12,tt),Weights_predictions(case_flag,3:3:12,tt),'o','Color',[0,0.5,1],'MarkerFaceColor',[0,0.5,1],'linestyle','none','Markersize',6);
text(1.61*(Rmax-Rmin)+Rmin,1.15*(Rdmax-Rdmin)+Rdmin,'QBMM Scheme','interpreter','latex','fontsize',16)
text(1.38*(Rmax-Rmin)+Rmin,1.05*(Rdmax-Rdmin)+Rdmin,'Weight \hspace{1.5cm} R-value \hspace{1.5cm} $\dot{R}$-value','interpreter','latex','fontsize',14)




legend([hp2,hp1],{'QBMM','QBMM-ML'},'interpreter','latex','fontsize',12,'orientation','horizontal','Position',[-0.04,0.45,1.,1.],'box','off')





print(figeta,'-dpdf','Figures/Weights_Plot.pdf');
savefig(figeta,'Figures/Weights_Plot.fig');
