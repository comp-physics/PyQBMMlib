clear;
close all;
clc;

load(['../ML_Code/ML_Predictions/LM_Random_MLQBMM_Approach4_Weights5','.mat']);

QBMM_Points = zeros(4,12,4001);
total_cases  = 4;
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


eta_video = VideoWriter('PDF_Video5.mp4','MPEG-4');
%eta_video = VideoWriter('PV_history.mp4','Motion JPEG AVI');
eta_video.FrameRate = 24;
eta_video.Quality = 100;
open(eta_video);
figeta = figure(1);
set(gcf,'color','w');
scrsz = get(groot,'ScreenSize');
set(figeta,'Units','Inches','Position',[0,0,12.0,6.0],'PaperUnits','Inches','PaperSize',[12.0,6.0])
p(1) = subplot(1,2,1);
p(2) = subplot(1,2,2);

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
timestart = 2000;
timend    = 4000;
for tt=timestart:timend
%load('../data/Random_Forcing/MC_HM_Random_Pressure_Realization31.mat');
pdf = zeros(nR,nRd);
for pp=1:nsamples
    rflag = 1+floor((R_samples(pp,tt)-Rmin)/dR);
    rdflag = 1+floor((Rd_samples(pp,tt)-Rdmin)/dRd);
    pdf(rflag,rdflag) = pdf(rflag,rdflag)+1;
end
% load('../data/Random_Forcing/MC_HM_Random_Pressure_Realization31.mat');
% for pp=1:nsamples
%     rflag = 1+floor((R_samples(pp,tt)-Rmin)/dR);
%     rdflag = 1+floor((Rd_samples(pp,tt)-Rdmin)/dRd);
%     pdf(rflag,rdflag) = pdf(rflag,rdflag)+1;
% end

pdf = pdf./nsamples;

if (tt == timestart)
axes(p(1))
p(1).Position = [0.07,0.10,0.45,0.75]; % [left bottom width height]
%hf1 = pcolor(xx,yy,pdf(:,:));
%set(hf1,'EdgeColor','none')
%hp1 = plot(Weights_predictions(1,2:3:12,tt),Weights_predictions(1,3:3:12,tt),'o','Color',[0,0.5,1],'MarkerFaceColor',[0,0.5,1],'linestyle','none','Markersize',6);
%hold on
%hp2 = plot(QBMM_Points(1,2:3:12,tt),QBMM_Points(1,3:3:12,tt),'s','Color',[0.8,0,0],'MarkerFaceColor',[0.8,0,0],'linestyle','none','Markersize',6);
[hf1,hc1] = contourf(xx,yy,pdf(:,:),[0.002,0.004,0.006,0.008,0.01],'edgecolor','none');
hold on
hp2 = plot(QBMM_Points(1,2:3:12,tt),QBMM_Points(1,3:3:12,tt),'s','Color',[0.8,0,0],'MarkerFaceColor',[0.8,0,0],'linestyle','none','Markersize',6);
hold on
hp1 = plot(Weights_predictions(1,2:3:15,tt),Weights_predictions(1,3:3:15,tt),'o','Color',[0,0.5,1],'MarkerFaceColor',[0,0.5,1],'linestyle','none','Markersize',6);
text(1.61*(Rmax-Rmin)+Rmin,1.15*(Rdmax-Rdmin)+Rdmin,'QBMM Scheme','interpreter','latex','fontsize',16)
text(1.38*(Rmax-Rmin)+Rmin,1.05*(Rdmax-Rdmin)+Rdmin,'Weight \hspace{1.5cm} R-value \hspace{1.5cm} $\dot{R}$-value','interpreter','latex','fontsize',14)


hw(1) = text(1.37*(Rmax-Rmin)+Rmin,0.95*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',QBMM_Points(case_flag,1,tt)),'interpreter','latex','fontsize',14);
hw(2) = text(1.70*(Rmax-Rmin)+Rmin,0.95*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',QBMM_Points(case_flag,2,tt)),'interpreter','latex','fontsize',14);
hw(3) = text(2.04*(Rmax-Rmin)+Rmin,0.95*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',QBMM_Points(case_flag,3,tt)),'interpreter','latex','fontsize',14);


hw(4) = text(1.37*(Rmax-Rmin)+Rmin,0.85*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',QBMM_Points(case_flag,4,tt)),'interpreter','latex','fontsize',14);
hw(5) = text(1.70*(Rmax-Rmin)+Rmin,0.85*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',QBMM_Points(case_flag,5,tt)),'interpreter','latex','fontsize',14);
hw(6) = text(2.04*(Rmax-Rmin)+Rmin,0.85*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',QBMM_Points(case_flag,6,tt)),'interpreter','latex','fontsize',14);


hw(7) = text(1.37*(Rmax-Rmin)+Rmin,0.75*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',QBMM_Points(case_flag,7,tt)),'interpreter','latex','fontsize',14);
hw(8) = text(1.70*(Rmax-Rmin)+Rmin,0.75*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',QBMM_Points(case_flag,8,tt)),'interpreter','latex','fontsize',14);
hw(9) = text(2.04*(Rmax-Rmin)+Rmin,0.75*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',QBMM_Points(case_flag,9,tt)),'interpreter','latex','fontsize',14);


hw(10) = text(1.37*(Rmax-Rmin)+Rmin,0.65*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',QBMM_Points(case_flag,10,tt)),'interpreter','latex','fontsize',14);
hw(11) = text(1.70*(Rmax-Rmin)+Rmin,0.65*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',QBMM_Points(case_flag,11,tt)),'interpreter','latex','fontsize',14);
hw(12) = text(2.04*(Rmax-Rmin)+Rmin,0.65*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',QBMM_Points(case_flag,12,tt)),'interpreter','latex','fontsize',14);





%text(1.35*Rmax,1.2*Rdmax,'Weight \hspace{0.5cm} R-point \hspace{0.5cm} $\dot{R}$-points','interpreter','latex','fontsize',12)


text(1.60*(Rmax-Rmin)+Rmin,0.55*(Rdmax-Rdmin)+Rdmin,'QBMM-ML Scheme','interpreter','latex','fontsize',16)
text(1.38*(Rmax-Rmin)+Rmin,0.45*(Rdmax-Rdmin)+Rdmin,'Weight \hspace{1.5cm} R-value \hspace{1.5cm} $\dot{R}$-value','interpreter','latex','fontsize',14)

hm(1) = text(1.37*(Rmax-Rmin)+Rmin,0.35*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',Weights_predictions(case_flag,1,tt)),'interpreter','latex','fontsize',14);
hm(2) = text(1.70*(Rmax-Rmin)+Rmin,0.35*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',Weights_predictions(case_flag,2,tt)),'interpreter','latex','fontsize',14);
hm(3) = text(2.04*(Rmax-Rmin)+Rmin,0.35*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',Weights_predictions(case_flag,3,tt)),'interpreter','latex','fontsize',14);


hm(4) = text(1.37*(Rmax-Rmin)+Rmin,0.25*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',Weights_predictions(case_flag,4,tt)),'interpreter','latex','fontsize',14);
hm(5) = text(1.70*(Rmax-Rmin)+Rmin,0.25*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',Weights_predictions(case_flag,5,tt)),'interpreter','latex','fontsize',14);
hm(6) = text(2.04*(Rmax-Rmin)+Rmin,0.25*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',Weights_predictions(case_flag,6,tt)),'interpreter','latex','fontsize',14);


hm(7) = text(1.37*(Rmax-Rmin)+Rmin,0.15*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',Weights_predictions(case_flag,7,tt)),'interpreter','latex','fontsize',14);
hm(8) = text(1.70*(Rmax-Rmin)+Rmin,0.15*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',Weights_predictions(case_flag,8,tt)),'interpreter','latex','fontsize',14);
hm(9) = text(2.04*(Rmax-Rmin)+Rmin,0.15*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',Weights_predictions(case_flag,9,tt)),'interpreter','latex','fontsize',14);


hm(10) = text(1.37*(Rmax-Rmin)+Rmin,0.05*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',Weights_predictions(case_flag,10,tt)),'interpreter','latex','fontsize',14);
hm(11) = text(1.70*(Rmax-Rmin)+Rmin,0.05*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',Weights_predictions(case_flag,11,tt)),'interpreter','latex','fontsize',14);
hm(12) = text(2.04*(Rmax-Rmin)+Rmin,0.05*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',Weights_predictions(case_flag,12,tt)),'interpreter','latex','fontsize',14);

hm(13) = text(1.37*(Rmax-Rmin)+Rmin,-0.05*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',Weights_predictions(case_flag,13,tt)),'interpreter','latex','fontsize',14);
hm(14) = text(1.70*(Rmax-Rmin)+Rmin,-0.05*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',Weights_predictions(case_flag,14,tt)),'interpreter','latex','fontsize',14);
hm(15) = text(2.04*(Rmax-Rmin)+Rmin,-0.05*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',Weights_predictions(case_flag,15,tt)),'interpreter','latex','fontsize',14);




legend([hp2,hp1],{'QBMM','QBMM-ML'},'interpreter','latex','fontsize',14,'orientation','horizontal','Position',[-0.235,0.40,1.,1.],'box','off')


p(2).Position = [1.07,0.10,0.45,0.75]; % [left bottom width height]

end

axes(p(1))
%set(hf1,'CData',pdf(:,:));
hc1.ZData = pdf(:,:);
set(hp2,'XData',QBMM_Points(case_flag,2:3:12,tt))
set(hp2,'YData',QBMM_Points(case_flag,3:3:12,tt))
set(hp1,'XData',Weights_predictions(case_flag,2:3:15,tt))
set(hp1,'YData',Weights_predictions(case_flag,3:3:15,tt))
c1 = colorbar;
colormap(gca,flip(bone));
caxis([0.0,0.02])
xname1 = xlabel('$R$','interpreter','latex','fontsize',16,'rot',0);
yname1 = ylabel('$\dot{R}$','interpreter','latex','fontsize',16,'rot',90);
%title1 = title('Bottom Layer - PV','interpreter','latex','fontsize',22,'rot',0);
set(gca,'XLabel',xname1);
set(gca,'YLabel',yname1);
ylabh = get(c1,'YLabel');
set(ylabh,'Position',[3.5 0.010 0])
ylabel(c1,'$f(R,\dot{R})$','interpreter','latex','fontsize',22,'rot',90)
%legend([hp1,hp2],{'QBMM','QBMM-ML'},'interpreter','latex','fontsize',14,'orientation','horizontal','Position',[-0.04,0.40,1.,1.],'box','off')


for ii=1:12
set(hw(ii),'String',sprintf('%+1.2f',QBMM_Points(1,ii,tt)));
end

for ii=1:15
set(hm(ii),'String',sprintf('%+1.2f',Weights_predictions(1,ii,tt)));
end

frame = getframe(figeta);
writeVideo(eta_video,frame);
disp(tt);

end

close(eta_video);