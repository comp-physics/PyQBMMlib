clear;
close all;
clc;

load(['../ML_Code/ML_Predictions/LM_Random_MLQBMM_Approach4_Weights4','.mat']);

QBMM_Points = zeros(4,12,16001);
total_cases  = 4;
ml_dim       = 12;
total_points = 16001;
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
Rmin = min(min(R_samples)); Rmax = max(max(R_samples));
Rdmin = min(min(Rd_samples)); Rdmax = max(max(Rd_samples));
load('../data/Random_Forcing/MC_HM_Random_Pressure_Realization32.mat');
Rmin = min(Rmin,min(min(R_samples))); Rmax = max(Rmax,max(max(R_samples)));
Rdmin = min(Rdmin,min(min(Rd_samples))); Rdmax = max(Rdmax,max(max(Rd_samples)));
load('../data/Random_Forcing/MC_HM_Random_Pressure_Realization33.mat');
Rmin = min(Rmin,min(min(R_samples))); Rmax = max(Rmax,max(max(R_samples)));
Rdmin = min(Rdmin,min(min(Rd_samples))); Rdmax = max(Rdmax,max(max(Rd_samples)));
load('../data/Random_Forcing/MC_HM_Random_Pressure_Realization34.mat');
Rmin = min(Rmin,min(min(R_samples))); Rmax = max(Rmax,max(max(R_samples)));
Rdmin = min(Rdmin,min(min(Rd_samples))); Rdmax = max(Rdmax,max(max(Rd_samples)));
load('../data/Random_Forcing/MC_HM_Random_Pressure_Realization35.mat');
Rmin = min(Rmin,min(min(R_samples))); Rmax = max(Rmax,max(max(R_samples)));
Rdmin = min(Rdmin,min(min(Rd_samples))); Rdmax = max(Rdmax,max(max(Rd_samples)));
load('../data/Random_Forcing/MC_HM_Random_Pressure_Realization36.mat');
Rmin = min(Rmin,min(min(R_samples))); Rmax = max(Rmax,max(max(R_samples)));
Rdmin = min(Rdmin,min(min(Rd_samples))); Rdmax = max(Rdmax,max(max(Rd_samples)));
load('../data/Random_Forcing/MC_HM_Random_Pressure_Realization37.mat');
Rmin = min(Rmin,min(min(R_samples))); Rmax = max(Rmax,max(max(R_samples)));
Rdmin = min(Rdmin,min(min(Rd_samples))); Rdmax = max(Rdmax,max(max(Rd_samples)));





nR  = 129;
nRd = 129;
dR  = (Rmax-Rmin)/(nR-1);
dRd = (Rdmax-Rdmin)/(nRd-1);
R_grid  = linspace(Rmin,Rmax,nR);
Rd_grid = linspace(Rdmin,Rdmax,nRd); 
xx = zeros(nR,nRd);
yy = zeros(nR,nRd);

for ii=1:nR
    for jj=1:nRd
       xx(ii,jj) = R_grid(ii);
       yy(ii,jj) = Rd_grid(jj);
    end
end

pdf_hist = zeros(nR,nRd,total_points);


load('../data/Random_Forcing/MC_HM_Random_Pressure_Realization31.mat');
R_samples_tot = R_samples;
Rd_samples_tot = Rd_samples;
for tt=1:total_points
for pp=1:size(R_samples,1)
    rflag = 1+floor((R_samples(pp,tt)-Rmin)/dR);
    rdflag = 1+floor((Rd_samples(pp,tt)-Rdmin)/dRd);
    pdf_hist(rflag,rdflag,tt) = pdf_hist(rflag,rdflag,tt)+1;
end
end
nsamples = size(R_samples,1);
load('../data/Random_Forcing/MC_HM_Random_Pressure_Realization32.mat');
%R_samples_tot = [R_samples_tot;R_samples];
%Rd_samples_tot = [Rd_samples_tot;Rd_samples];
for tt=1:total_points
for pp=1:size(R_samples,1)
    rflag = 1+floor((R_samples(pp,tt)-Rmin)/dR);
    rdflag = 1+floor((Rd_samples(pp,tt)-Rdmin)/dRd);
    pdf_hist(rflag,rdflag,tt) = pdf_hist(rflag,rdflag,tt)+1;
end
end
nsamples = nsamples+size(R_samples,1);

load('../data/Random_Forcing/MC_HM_Random_Pressure_Realization33.mat');
%R_samples_tot = [R_samples_tot;R_samples];
%Rd_samples_tot = [Rd_samples_tot;Rd_samples];
for tt=1:total_points
for pp=1:size(R_samples,1)
    rflag = 1+floor((R_samples(pp,tt)-Rmin)/dR);
    rdflag = 1+floor((Rd_samples(pp,tt)-Rdmin)/dRd);
    pdf_hist(rflag,rdflag,tt) = pdf_hist(rflag,rdflag,tt)+1;
end
end
nsamples = nsamples+size(R_samples,1);

load('../data/Random_Forcing/MC_HM_Random_Pressure_Realization34.mat');
%R_samples_tot = [R_samples_tot;R_samples];
%Rd_samples_tot = [Rd_samples_tot;Rd_samples];
for tt=1:total_points
for pp=1:size(R_samples,1)
    rflag = 1+floor((R_samples(pp,tt)-Rmin)/dR);
    rdflag = 1+floor((Rd_samples(pp,tt)-Rdmin)/dRd);
    pdf_hist(rflag,rdflag,tt) = pdf_hist(rflag,rdflag,tt)+1;
end
end
nsamples = nsamples+size(R_samples,1);

load('../data/Random_Forcing/MC_HM_Random_Pressure_Realization35.mat');
%R_samples_tot = [R_samples_tot;R_samples];
%Rd_samples_tot = [Rd_samples_tot;Rd_samples];
for tt=1:total_points
for pp=1:size(R_samples,1)
    rflag = 1+floor((R_samples(pp,tt)-Rmin)/dR);
    rdflag = 1+floor((Rd_samples(pp,tt)-Rdmin)/dRd);
    pdf_hist(rflag,rdflag,tt) = pdf_hist(rflag,rdflag,tt)+1;
end
end
nsamples = nsamples+size(R_samples,1);

load('../data/Random_Forcing/MC_HM_Random_Pressure_Realization36.mat');
%R_samples_tot = [R_samples_tot;R_samples];
%Rd_samples_tot = [Rd_samples_tot;Rd_samples];
for tt=1:total_points
for pp=1:size(R_samples,1)
    rflag = 1+floor((R_samples(pp,tt)-Rmin)/dR);
    rdflag = 1+floor((Rd_samples(pp,tt)-Rdmin)/dRd);
    pdf_hist(rflag,rdflag,tt) = pdf_hist(rflag,rdflag,tt)+1;
end
end
nsamples = nsamples+size(R_samples,1);

load('../data/Random_Forcing/MC_HM_Random_Pressure_Realization37.mat');
%R_samples_tot = [R_samples_tot;R_samples];
%Rd_samples_tot = [Rd_samples_tot;Rd_samples];
for tt=1:total_points
for pp=1:size(R_samples,1)
    rflag = 1+floor((R_samples(pp,tt)-Rmin)/dR);
    rdflag = 1+floor((Rd_samples(pp,tt)-Rdmin)/dRd);
    pdf_hist(rflag,rdflag,tt) = pdf_hist(rflag,rdflag,tt)+1;
end
end
nsamples = nsamples+size(R_samples,1);





pdf_hist = pdf_hist./nsamples;
pdf_hist =log10(pdf_hist);


save('PDF_Data.mat','pdf_hist','Rmin','Rmax','Rdmin','Rdmax','dR','dRd','xx','yy','nR','nRd');

