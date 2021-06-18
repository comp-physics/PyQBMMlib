clear;
close all;
clc;

load(['../ML_Code/ML_Predictions/LM_Random_MLQBMM_Approach4_Weights4','.mat']);

QBMM_Points = zeros(4,12,16001);
total_cases  = 4;
ml_dim       = size(Weights_predictions,2);
abscissas    = ml_dim/3;
total_points = 16001;
total_times  = 16001;
T = linspace(0,200,total_times);
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


load('PDF_Data.mat');

eta_video = VideoWriter(['PDF_Video',num2str(abscissas),'_Plots.mp4'],'MPEG-4');
%eta_video = VideoWriter('PV_history.mp4','Motion JPEG AVI');
eta_video.FrameRate = 24;
eta_video.Quality = 100;
open(eta_video);
figeta = figure(1);
set(gcf,'color','w');
scrsz = get(groot,'ScreenSize');
set(figeta,'Units','Inches','Position',[0,0,12.0,6.0],'PaperUnits','Inches','PaperSize',[12.0,6.0])
for ii=1:6
   p(ii) = subplot(1,6,ii); 
end

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
timestart = 00001;
timend    = 01001;
flag_MC = zeros(42,total_points);
flag_QBMM = zeros(42,total_points);
flag_ML   = zeros(42,total_points);
flag_MC(:,:)   = LM_MC(case_flag,:,:);
flag_QBMM(:,:) = LM_QBMM(case_flag,:,:);
flag_ML(:,:)   = predictions(case_flag,:,:);

for tt=timestart:1:timend
%load('../data/Random_Forcing/MC_HM_Random_Pressure_Realization31.mat');
% pdf = zeros(nR,nRd);
% for pp=1:nsamples
%     rflag = 1+floor((R_samples(pp,tt)-Rmin)/dR);
%     rdflag = 1+floor((Rd_samples(pp,tt)-Rdmin)/dRd);
%     pdf(rflag,rdflag) = pdf(rflag,rdflag)+1;
% end
% load('../data/Random_Forcing/MC_HM_Random_Pressure_Realization31.mat');
% for pp=1:nsamples
%     rflag = 1+floor((R_samples(pp,tt)-Rmin)/dR);
%     rdflag = 1+floor((Rd_samples(pp,tt)-Rdmin)/dRd);
%     pdf(rflag,rdflag) = pdf(rflag,rdflag)+1;
% end

% pdf = pdf./nsamples;
% pdf =log10(pdf);
pdf = zeros(nR,nRd);
pdf(:,:) = pdf_hist(:,:,tt);

if (tt == timestart)
axes(p(1))
p(1).Position = [0.07,0.10,0.45,0.75]; % [left bottom width height]
%hf1 = pcolor(xx,yy,pdf(:,:));
%set(hf1,'EdgeColor','none')
%hp1 = plot(Weights_predictions(1,2:3:12,tt),Weights_predictions(1,3:3:12,tt),'o','Color',[0,0.5,1],'MarkerFaceColor',[0,0.5,1],'linestyle','none','Markersize',6);
%hold on
%hp2 = plot(QBMM_Points(1,2:3:12,tt),QBMM_Points(1,3:3:12,tt),'s','Color',[0.8,0,0],'MarkerFaceColor',[0.8,0,0],'linestyle','none','Markersize',6);
%[hf1,hc1] = contourf(xx,yy,pdf(:,:),[0.0005:0.001:0.10],'edgecolor','none');
[hf1,hc1] = contourf(xx,yy,pdf(:,:),[-4:0.5:-1],'edgecolor','none');

hold on
hp2 = plot(QBMM_Points(1,2:3:12,tt),QBMM_Points(1,3:3:12,tt),'s','Color',[0.8,0,0],'MarkerFaceColor',[0.8,0,0],'linestyle','none','Markersize',6);
hold on
hp1 = plot(Weights_predictions(1,2:3:ml_dim,tt),Weights_predictions(1,3:3:ml_dim,tt),'o','Color',[0,0.5,1],'MarkerFaceColor',[0,0.5,1],'linestyle','none','Markersize',6);


% text(1.61*(Rmax-Rmin)+Rmin,1.15*(Rdmax-Rdmin)+Rdmin,'QBMM Scheme','interpreter','latex','fontsize',16)
% text(1.38*(Rmax-Rmin)+Rmin,1.05*(Rdmax-Rdmin)+Rdmin,'Weight \hspace{1.5cm} R-value \hspace{1.5cm} $\dot{R}$-value','interpreter','latex','fontsize',14)
% 
% hw(1) = text(1.37*(Rmax-Rmin)+Rmin,0.95*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',QBMM_Points(case_flag,1,tt)),'interpreter','latex','fontsize',14);
% hw(2) = text(1.70*(Rmax-Rmin)+Rmin,0.95*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',QBMM_Points(case_flag,2,tt)),'interpreter','latex','fontsize',14);
% hw(3) = text(2.04*(Rmax-Rmin)+Rmin,0.95*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',QBMM_Points(case_flag,3,tt)),'interpreter','latex','fontsize',14);
% 
% hw(4) = text(1.37*(Rmax-Rmin)+Rmin,0.85*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',QBMM_Points(case_flag,4,tt)),'interpreter','latex','fontsize',14);
% hw(5) = text(1.70*(Rmax-Rmin)+Rmin,0.85*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',QBMM_Points(case_flag,5,tt)),'interpreter','latex','fontsize',14);
% hw(6) = text(2.04*(Rmax-Rmin)+Rmin,0.85*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',QBMM_Points(case_flag,6,tt)),'interpreter','latex','fontsize',14);
% 
% hw(7) = text(1.37*(Rmax-Rmin)+Rmin,0.75*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',QBMM_Points(case_flag,7,tt)),'interpreter','latex','fontsize',14);
% hw(8) = text(1.70*(Rmax-Rmin)+Rmin,0.75*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',QBMM_Points(case_flag,8,tt)),'interpreter','latex','fontsize',14);
% hw(9) = text(2.04*(Rmax-Rmin)+Rmin,0.75*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',QBMM_Points(case_flag,9,tt)),'interpreter','latex','fontsize',14);
% 
% hw(10) = text(1.37*(Rmax-Rmin)+Rmin,0.65*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',QBMM_Points(case_flag,10,tt)),'interpreter','latex','fontsize',14);
% hw(11) = text(1.70*(Rmax-Rmin)+Rmin,0.65*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',QBMM_Points(case_flag,11,tt)),'interpreter','latex','fontsize',14);
% hw(12) = text(2.04*(Rmax-Rmin)+Rmin,0.65*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',QBMM_Points(case_flag,12,tt)),'interpreter','latex','fontsize',14);


% text(1.60*(Rmax-Rmin)+Rmin,0.55*(Rdmax-Rdmin)+Rdmin,'QBMM-ML Scheme','interpreter','latex','fontsize',16)
% text(1.38*(Rmax-Rmin)+Rmin,0.45*(Rdmax-Rdmin)+Rdmin,'Weight \hspace{1.5cm} R-value \hspace{1.5cm} $\dot{R}$-value','interpreter','latex','fontsize',14)
% 
% for ii=1:abscissas
%     hm(3*(ii-1)+1) = text(1.37*(Rmax-Rmin)+Rmin,(0.35-0.1*(ii-1))*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',Weights_predictions(case_flag,3*(ii-1)+1,tt)),'interpreter','latex','fontsize',14);
%     hm(3*(ii-1)+2) = text(1.70*(Rmax-Rmin)+Rmin,(0.35-0.1*(ii-1))*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',Weights_predictions(case_flag,3*(ii-1)+2,tt)),'interpreter','latex','fontsize',14);
%     hm(3*(ii-1)+3) = text(2.04*(Rmax-Rmin)+Rmin,(0.35-0.1*(ii-1))*(Rdmax-Rdmin)+Rdmin,sprintf('%+1.2f',Weights_predictions(case_flag,3*(ii-1)+3,tt)),'interpreter','latex','fontsize',14);
% end
    

legend([hp2,hp1],{'QBMM','QBMM-ML'},'interpreter','latex','fontsize',14,'orientation','horizontal','Position',[-0.235,0.40,1.,1.],'box','off')

tleft  = 1;
tright = 401;

for ii=2:6
axes(p(ii))
p(ii).Position = [0.65,0.78-0.17*(ii-2),0.30,0.10]; % [left bottom width height]
MC_p(ii) = plot(T(1:tt),flag_MC(ii-1,1:tt),'Color',[0.2,0.2,0.2],'linewidth',1.0,'linestyle','-');
hold on
QBMM_p(ii) = plot(T(1:tt),flag_QBMM(ii-1,1:tt),'Color',[0.8,0,0],'linewidth',1.0,'linestyle','--');
hold on
ML_p(ii) = plot(T(1:tt),flag_ML(ii-1,1:tt),'Color',[0,0.5,1],'linewidth',1.0);
box on;
xlim([T(tleft),T(tright)])
ylim([min(flag_QBMM(ii-1,:)),max(flag_QBMM(ii-1,:))])
if (ii == 2)
%yname1 = ylabel("$\mu_{1,0}$",'interpreter','latex','fontsize',12);
%set(p(ii),'YLabel',yname1);
ylabh = get(gca,'ylabel');
set(ylabh,'position',get(ylabh,'position') - [-0.8 0 0]);
legend({'MC','QBMM','QBMM-ML'},'interpreter','latex','fontsize',12,'orientation','horizontal','Position',[0.3,0.44,1.,1.],'box','off')
end
if (ii == 2)
yname1 = ylabel("$\mu_{1,0}$",'interpreter','latex','fontsize',12);
set(p(ii),'YLabel',yname1);       
end
if (ii == 3)
yname1 = ylabel("$\mu_{0,1}$",'interpreter','latex','fontsize',12);
set(p(ii),'YLabel',yname1);       
end
if (ii == 4)
yname1 = ylabel("$\mu_{2,0}$",'interpreter','latex','fontsize',12);
set(p(ii),'YLabel',yname1);       
end
if (ii == 5)
yname1 = ylabel("$\mu_{1,1}$",'interpreter','latex','fontsize',12);
set(p(ii),'YLabel',yname1);       
end
if (ii == 6)
yname1 = ylabel("$\mu_{0,2}$",'interpreter','latex','fontsize',12);
set(p(ii),'YLabel',yname1);       
end


end



end

axes(p(1))
%set(hf1,'CData',pdf(:,:));
hc1.ZData = pdf(:,:);
set(hp2,'XData',QBMM_Points(case_flag,2:3:12,tt))
set(hp2,'YData',QBMM_Points(case_flag,3:3:12,tt))
set(hp1,'XData',Weights_predictions(case_flag,2:3:ml_dim,tt))
set(hp1,'YData',Weights_predictions(case_flag,3:3:ml_dim,tt))
c1 = colorbar;
colormap(gca,flip(bone));
caxis([-4,0])
xname1 = xlabel('$R$','interpreter','latex','fontsize',16,'rot',0);
yname1 = ylabel('$\dot{R}$','interpreter','latex','fontsize',16,'rot',90);
%title1 = title('Bottom Layer - PV','interpreter','latex','fontsize',22,'rot',0);
set(gca,'XLabel',xname1);
set(gca,'YLabel',yname1);
%legend([hp1,hp2],{'QBMM','QBMM-ML'},'interpreter','latex','fontsize',14,'orientation','horizontal','Position',[-0.04,0.40,1.,1.],'box','off')

set(c1,'YTick',[-4,-3,-2,-1,0])
%a = get(c1,'YTickLabel');  
%set(c1,'YTick',[-4,-3,-2,-1,0])
set(c1,'YTickLabel',[{'10^{-4}', '10^{-3}','10^{-2}','10^{-1}','10^{0}'}],'fontsize',10)

ylabh = get(c1,'YLabel');
set(ylabh,'Position',[2.5 -2.0 0])
ylabel(c1,'$f(R,\dot{R})$','interpreter','latex','fontsize',22,'rot',90)


if (tt > 401)
   tleft = tt-400;
   tright = tt;
end
for ii=2:6
    axes(p(ii));
    set(MC_p(ii),'XData',T(1:tt))
    set(MC_p(ii),'YData',flag_MC(ii-1,1:tt))
    %plot(T(1:tt),flag_MC(ii-1,1:tt),'Color',[0.2,0.2,0.2],'linewidth',1.0,'linestyle','-') 
    %hold on
    set(QBMM_p(ii),'XData',T(1:tt))
    set(QBMM_p(ii),'YData',flag_QBMM(ii-1,1:tt))
    %plot(T(1:tt),flag_QBMM(ii-1,1:tt),'Color',[0.8,0,0],'linewidth',1.0,'linestyle','--') 
    %hold on
    set(ML_p(ii),'XData',T(1:tt))
    set(ML_p(ii),'YData',flag_ML(ii-1,1:tt))
    %plot(T(1:tt),flag_ML(ii-1,1:tt),'Color',[0,0.5,1],'linewidth',1.0) 
    %box on;
    set(gca,'XLim',[T(tleft),T(tright)])
    set(gca,'YLim',[min(flag_QBMM(ii-1,:)),max(flag_QBMM(ii-1,:))])
    if (ii==2)
        if (tt == timestart)
        legend({'MC','QBMM','QBMM-ML'},'interpreter','latex','fontsize',12,'orientation','horizontal','Position',[0.54,0.44,1.,1.],'box','off')
        end
    end
end

% for ii=1:12
% set(hw(ii),'String',sprintf('%+1.2f',QBMM_Points(1,ii,tt)));
% end
% 
% for ii=1:ml_dim
% set(hm(ii),'String',sprintf('%+1.2f',Weights_predictions(1,ii,tt)));
% end



frame = getframe(figeta);
writeVideo(eta_video,frame);
disp(tt);

end

close(eta_video);