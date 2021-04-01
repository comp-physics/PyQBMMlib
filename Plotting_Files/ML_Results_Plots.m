clear;
close all;
clc;


          
rfile_mc = {['../data/Constant_Forcing/MC_HM_Constant_Pressure25.mat'];...
            ['../data/Constant_Forcing/MC_HM_Constant_Pressure30.mat'];...
            ['../data/Constant_Forcing/MC_HM_Constant_Pressure35.mat'];...
            ['../data/Constant_Forcing/MC_HM_Constant_Pressure40.mat'];...
            ['../data/Constant_Forcing/MC_HM_Constant_Pressure45.mat'];...
            ['../data/Constant_Forcing/MC_HM_Constant_Pressure50.mat'];...
            ['../data/Constant_Forcing/MC_HM_Constant_Pressure55.mat'];...
            ['../data/Constant_Forcing/MC_HM_Constant_Pressure60.mat'];...
            ['../data/Constant_Forcing/MC_HM_Constant_Pressure65.mat'];...
            ['../data/Constant_Forcing/MC_HM_Constant_Pressure70.mat'];...
            ['../data/Constant_Forcing/MC_HM_Constant_Pressure75.mat'];...
            ['../data/Constant_Forcing/MC_HM_Constant_Pressure80.mat'];...
            ['../data/Constant_Forcing/MC_HM_Constant_Pressure85.mat'];...
            ['../data/Constant_Forcing/MC_HM_Constant_Pressure90.mat'];...
            ['../data/Constant_Forcing/MC_HM_Constant_Pressure95.mat'];};

rfile_qbmm = {['../data/Constant_Forcing/QBMM_HM_Constant_Pressure25.mat'];...
            ['../data/Constant_Forcing/QBMM_HM_Constant_Pressure30.mat'];...
            ['../data/Constant_Forcing/QBMM_HM_Constant_Pressure35.mat'];...
            ['../data/Constant_Forcing/QBMM_HM_Constant_Pressure40.mat'];...
            ['../data/Constant_Forcing/QBMM_HM_Constant_Pressure45.mat'];...
            ['../data/Constant_Forcing/QBMM_HM_Constant_Pressure50.mat'];...
            ['../data/Constant_Forcing/QBMM_HM_Constant_Pressure55.mat'];...
            ['../data/Constant_Forcing/QBMM_HM_Constant_Pressure60.mat'];...
            ['../data/Constant_Forcing/QBMM_HM_Constant_Pressure65.mat'];...
            ['../data/Constant_Forcing/QBMM_HM_Constant_Pressure70.mat'];...
            ['../data/Constant_Forcing/QBMM_HM_Constant_Pressure75.mat'];...
            ['../data/Constant_Forcing/QBMM_HM_Constant_Pressure80.mat'];...
            ['../data/Constant_Forcing/QBMM_HM_Constant_Pressure85.mat'];...
            ['../data/Constant_Forcing/QBMM_HM_Constant_Pressure90.mat'];...
            ['../data/Constant_Forcing/QBMM_HM_Constant_Pressure95.mat'];};        
        
        
load('../ML_Code/HM_MLQBMM.mat');   


%data_qbmm = importdata(rfile_qbmm{1}); 
load(rfile_mc{1});

%T_qbmm = data_qbmm(:,1)';
T_mc = T;

%input_data = zeros(15,5,size(T,2));
mc_data = zeros(15,4,size(T,2));
qbmm_data = zeros(15,4,size(T,2));

for jj=1:15
    %data_qbmm = importdata(rfile_qbmm{jj});        
    load(rfile_mc{jj});
    mc_data(jj,:,:) = moments;
    
    load(rfile_qbmm{jj});
    for ii=1:4
        x = moments(:,ii);
        qbmm_data(jj,ii,:) = interp1(T,x,T_mc);
    end
    qbmm_data(jj,1,1) = 0.0;
    qbmm_data(jj,2,1) = 0.0;
    qbmm_data(jj,3,1) = 1.0;
    qbmm_data(jj,4,1) = 1.0;
    %T_qbmm = data_qbmm(:,1)';
% for ii=1:5
%     x = data_qbmm(:,ii+2);
%     input_data(jj,ii,:) = interp1(T_qbmm,x,T_mc);
% end

end

R32_ML = zeros(15,1801);
R21_ML = zeros(15,1801);
R30_ML = zeros(15,1801);
R3g_ML = zeros(15,1801);

R32_MC = zeros(15,1801);
R21_MC = zeros(15,1801);
R30_MC = zeros(15,1801);
R3g_MC = zeros(15,1801);

R32_QBMM = zeros(15,1801);
R21_QBMM = zeros(15,1801);
R30_QBMM = zeros(15,1801);
R3g_QBMM = zeros(15,1801);

for jj=1:15
    R32_MC(jj,:) = mc_data(jj,1,:);
    R21_MC(jj,:) = mc_data(jj,2,:);
    R30_MC(jj,:) = mc_data(jj,3,:);
    R3g_MC(jj,:) = mc_data(jj,4,:);
    
    R32_ML(jj,:) = predictions(jj,1,:);
    R21_ML(jj,:) = predictions(jj,2,:);
    R30_ML(jj,:) = predictions(jj,3,:);
    R3g_ML(jj,:) = predictions(jj,4,:);
    
    R32_QBMM(jj,:) = qbmm_data(jj,1,:);
    R21_QBMM(jj,:) = qbmm_data(jj,2,:);
    R30_QBMM(jj,:) = qbmm_data(jj,3,:);
    R3g_QBMM(jj,:) = qbmm_data(jj,4,:);
end


figeta = figure(1);
set(gcf,'color','w');
scrsz = get(groot,'ScreenSize');
figeta.Position = [1 scrsz(4)/15 scrsz(3)/(1.5*2.0) scrsz(4)/(1.5*2.4)];
%set(figeta,'Units','Inches','Position',[0,0,1.5,1.5],'PaperUnits','Inches','PaperSize',[1.5,1.5])


iflag = [2,4,6,8,10,12];
Plot_titles = {'$P_{ratio} = 0.30$';...
               '$P_{ratio} = 0.40$';...
               '$P_{ratio} = 0.50$';...
               '$P_{ratio} = 0.60$';...
               '$P_{ratio} = 0.70$';...
               '$P_{ratio} = 0.80$';};
for ii=1:6
subplot(3,2,ii)
plot(T_mc,R32_MC(iflag(ii),:),'Color',[1,0,0],'linewidth',1.5)
hold on
plot(T_mc,R32_QBMM(iflag(ii),:),'Color',[0,0.5,0],'linewidth',1.5)
hold on
plot(T_mc,R32_ML(iflag(ii),:),'Color',[0,0.5,1],'linewidth',1.5)
if (ii==1)
    legend({'MC','QBMM','ML'},'interpreter','latex','fontsize',8,'orientation','horizontal')
    text(65,1.0,'$\mu_{32}$','interpreter','latex','fontsize',16)
end
title(Plot_titles{ii},'interpreter','latex','fontsize',12)
end
print(figeta,'-dpdf','Moment32_Results.pdf');




figeta = figure(2);
set(gcf,'color','w');
scrsz = get(groot,'ScreenSize');
figeta.Position = [1 scrsz(4)/15 scrsz(3)/(1.5*2.0) scrsz(4)/(1.5*2.4)];
%set(figeta,'Units','Inches','Position',[0,0,1.5,1.5],'PaperUnits','Inches','PaperSize',[1.5,1.5])


iflag = [2,4,6,8,10,12];
Plot_titles = {'$P_{ratio} = 0.30$';...
               '$P_{ratio} = 0.40$';...
               '$P_{ratio} = 0.50$';...
               '$P_{ratio} = 0.60$';...
               '$P_{ratio} = 0.70$';...
               '$P_{ratio} = 0.80$';};
for ii=1:6
subplot(3,2,ii)
plot(T_mc,R21_MC(iflag(ii),:),'Color',[1,0,0],'linewidth',1.5)
hold on
plot(T_mc,R21_QBMM(iflag(ii),:),'Color',[0,0.5,0],'linewidth',1.5)
hold on
plot(T_mc,R21_ML(iflag(ii),:),'Color',[0,0.5,1],'linewidth',1.5)
if (ii==1)
    legend({'MC','ML'},'interpreter','latex','fontsize',8,'orientation','horizontal')
    text(65,1.8,'$\mu_{21}$','interpreter','latex','fontsize',16)
end
title(Plot_titles{ii},'interpreter','latex','fontsize',12)
end
print(figeta,'-dpdf','Moment21_Results.pdf');


        
        

figeta = figure(3);
set(gcf,'color','w');
scrsz = get(groot,'ScreenSize');
figeta.Position = [1 scrsz(4)/15 scrsz(3)/(1.5*2.0) scrsz(4)/(1.5*2.4)];
%set(figeta,'Units','Inches','Position',[0,0,1.5,1.5],'PaperUnits','Inches','PaperSize',[1.5,1.5])


iflag = [2,4,6,8,10,12];
Plot_titles = {'$P_{ratio} = 0.30$';...
               '$P_{ratio} = 0.40$';...
               '$P_{ratio} = 0.50$';...
               '$P_{ratio} = 0.60$';...
               '$P_{ratio} = 0.70$';...
               '$P_{ratio} = 0.80$';};
for ii=1:6
subplot(3,2,ii)
plot(T_mc,R30_MC(iflag(ii),:),'Color',[1,0,0],'linewidth',1.5)
hold on
plot(T_mc,R30_QBMM(iflag(ii),:),'Color',[0,0.5,0],'linewidth',1.5)
hold on
plot(T_mc,R30_ML(iflag(ii),:),'Color',[0,0.5,1],'linewidth',1.5)
if (ii==1)
    legend({'MC','QBMM','ML'},'interpreter','latex','fontsize',8,'orientation','horizontal')
    text(65,1.5,'$\mu_{30}$','interpreter','latex','fontsize',16)
end
title(Plot_titles{ii},'interpreter','latex','fontsize',12)
end
print(figeta,'-dpdf','Moment30_Results.pdf');


figeta = figure(4);
set(gcf,'color','w');
scrsz = get(groot,'ScreenSize');
figeta.Position = [1 scrsz(4)/15 scrsz(3)/(1.5*2.0) scrsz(4)/(1.5*2.4)];
%set(figeta,'Units','Inches','Position',[0,0,1.5,1.5],'PaperUnits','Inches','PaperSize',[1.5,1.5])


iflag = [2,4,6,8,10,12];
Plot_titles = {'$P_{ratio} = 0.30$';...
               '$P_{ratio} = 0.40$';...
               '$P_{ratio} = 0.50$';...
               '$P_{ratio} = 0.60$';...
               '$P_{ratio} = 0.70$';...
               '$P_{ratio} = 0.80$';};
for ii=1:6
subplot(3,2,ii)
plot(T_mc,R3g_MC(iflag(ii),:),'Color',[1,0,0],'linewidth',1.5)
hold on
plot(T_mc,R3g_QBMM(iflag(ii),:),'Color',[0,0.5,0],'linewidth',1.5)
hold on
plot(T_mc,R3g_ML(iflag(ii),:),'Color',[0,0.5,1],'linewidth',1.5)
if (ii==1)
    legend({'MC','QBMM','ML'},'interpreter','latex','fontsize',8,'orientation','horizontal')
    text(60,4.5,'$\mu_{3(1-\gamma),0}$','interpreter','latex','fontsize',16)
end
title(Plot_titles{ii},'interpreter','latex','fontsize',12)
end
print(figeta,'-dpdf','Moment3g_Results.pdf');



% Plot L-2 error
l2_error_ml = zeros(4,15);
l2_error_qbmm = zeros(4,15);
l2_measure  = zeros(4,15);

for ii=1:15
    for jj=1:4
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


pressure_ratios = [0.3,0.4,0.5,0.6,0.7,0.8,0.9];

figeta = figure(5);
set(gcf,'color','w');
scrsz = get(groot,'ScreenSize');
figeta.Position = [1 scrsz(4)/15 scrsz(3)/(1.5*2.0) scrsz(4)/(1.5*2.4)];
%set(figeta,'Units','Inches','Position',[0,0,1.5,1.5],'PaperUnits','Inches','PaperSize',[1.5,1.5])

subplot(2,2,1)
plot(pressure_ratios,l2_error_qbmm(1,2:2:14),'Color',[1,0,0],'linewidth',1.5,'Marker','*','linestyle','none')
hold on
plot(pressure_ratios,l2_error_ml(1,2:2:14),'Color',[0,0.5,1],'linewidth',1.5,'Marker','o','linestyle','none')
xlim([0.25,0.95])
ylabel('$L^2$-error','interpreter','latex','fontsize',14)
title('$\mu_{32}$','interpreter','latex','fontsize',16)
legend({'QBMM','QBMM-ML'},'interpreter','latex','fontsize',12)

subplot(2,2,2)
plot(pressure_ratios,l2_error_qbmm(2,2:2:14),'Color',[1,0,0],'linewidth',1.5,'Marker','*','linestyle','none')
hold on
plot(pressure_ratios,l2_error_ml(2,2:2:14),'Color',[0,0.5,1],'linewidth',1.5,'Marker','o','linestyle','none')
xlim([0.25,0.95])
title('$\mu_{21}$','interpreter','latex','fontsize',16)

subplot(2,2,3)
plot(pressure_ratios,l2_error_qbmm(3,2:2:14),'Color',[1,0,0],'linewidth',1.5,'Marker','*','linestyle','none')
hold on
plot(pressure_ratios,l2_error_ml(3,2:2:14),'Color',[0,0.5,1],'linewidth',1.5,'Marker','o','linestyle','none')
xlim([0.25,0.95])
xlabel('$p/p_0$','interpreter','latex','fontsize',14)
ylabel('$L^2$-error','interpreter','latex','fontsize',14)
title('$\mu_{30}$','interpreter','latex','fontsize',16)

subplot(2,2,4)
plot(pressure_ratios,l2_error_qbmm(4,2:2:14),'Color',[1,0,0],'linewidth',1.5,'Marker','*','linestyle','none')
hold on
plot(pressure_ratios,l2_error_ml(4,2:2:14),'Color',[0,0.5,1],'linewidth',1.5,'Marker','o','linestyle','none')
xlim([0.25,0.95])
xlabel('$p/p_0$','interpreter','latex','fontsize',14)
title('$\mu_{3(1-\gamma),0}$','interpreter','latex','fontsize',16)
print(figeta,'-dpdf','L2_error.pdf');


