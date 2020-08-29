%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   27-node quadrature method 
%                      for 1D crossing jet
%
% N.B. here x direction is inhomogeneous
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% This version transports 16 moments of the 3 velocity components:
%      1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16  
%k1 = [0 1 0 0 2 1 1 0 0 0  3  0  0  4  0  0 ] ;
%k2 = [0 0 1 0 0 1 0 2 1 0  0  3  0  0  4  0 ] ;
%k3 = [0 0 0 1 0 0 1 0 1 2  0  0  3  0  0  4 ] ;
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                           Initilization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear ; 
close all ;

alpha = 5 ; % average volume fraction
ep = 1 ; % cofficient of restitution
cfl = 0.90 ; 
Ny = 402 ; % grid + 2 (2 wall cells)

T0 = 1 ;
Ly = 1 ; % dimensionless width of domain
dp = 1/35 ; % dimensionless particle diameter
Kn = dp/(6*alpha) ; % global Knudsen number
%
vrms = sqrt(T0) ;
cs = vrms ;
% Initializing numerical parameters
%
H = Ly ;  % width of domain (dimensionless units)

Tmax = 0.6 ; % final time

Dy = H/(Ny-2) ;  % cell size
eps = 1.d-6 ; 
taff = Tmax/100 ; 
Ycell = -(Dy/2) : Dy : (H+Dy/2) ; % grid cell centers
tcol = dp/6 ; % dimensionless collision time as defined in paper
% ES-BGK parameter
% b = 0    - Regular BGK
% b = -0.5 - ES-BGK
b = 0;
verysmall = 1.d-14 ; % smallest nonzero mass
%
collision = 0 ; % (==0 no collisions)
order = 0 ; % have not implemented order == 1 (second order)
movie = 0 ;
project = 1 ; % set to 1 for projection of moments
restart = 0 ; % set to 1 to restart from saved file
%
% Initializing the grid 
M0 = zeros(16,Ny); % moments
N0 = 0*ones(27,Ny) ; % weights
U0 = zeros(27,Ny) ; % U velocity 
V0 = zeros(27,Ny) ; % V velocity
W0 = zeros(27,Ny) ; % W velocity

N1 = N0 ;
U1 = U0 ; 
V1 = V0 ;
W1 = W0 ;
NN = N0 ;
NM = M0 ;
for jj = 1:27
    NN(jj,:) = N0(jj,:)./(M0(1,:) + 1.d-16) ;
end
for jj = 1:16
    NM(jj,:) = M0(jj,:)./(M0(1,:) + 1.d-16) ;
end

% initial two waves in x direction  (change parameters to get different types of PTC)
nleft = 1 ;
nright = 1/2 ;
uleft = 0.01 ;
uright = 1 ;
for i = 1:48
    N0(1,i) = 0.125*nleft ;
    N0(3,i) = 0.125*nleft ;
    N0(7,i) = 0.125*nleft ;
    N0(9,i) = 0.125*nleft ;
    N0(19,i) = 0.125*nleft ;
    N0(21,i) = 0.125*nleft ;
    N0(25,i) = 0.125*nleft ;
    N0(27,i) = 0.125*nleft ;
    U0(1,i)  = uleft ;
    U0(3,i)  = uleft ;
    U0(7,i)  = uleft ;
    U0(9,i)  = uleft ;
    U0(19,i) = uleft ;
    U0(21,i) = uleft ;
    U0(25,i) = uleft ;
    U0(27,i) = uleft ;
    V0(1,i)  = uleft ;
    V0(3,i)  = uleft ;
    V0(7,i)  = -1 ;
    V0(9,i)  = -1 ;
    V0(19,i) = 1 ;
    V0(21,i) = 1 ;
    V0(25,i) = -1 ;
    V0(27,i) = -1 ;
    W0(1,i)  = -1 ;
    W0(3,i)  = 1 ;
    W0(7,i)  = -1 ;
    W0(9,i)  = 1 ;
    W0(19,i) = -1 ;
    W0(21,i) = 1 ;
    W0(25,i) = -1 ;
    W0(27,i) = 1 ;
end
for i = Ny-47:Ny
    N0(1,i) = 0.125*nright ;
    N0(3,i) = 0.125*nright ;
    N0(7,i) = 0.125*nright ;
    N0(9,i) = 0.125*nright ;
    N0(19,i) = 0.125*nright ;
    N0(21,i) = 0.125*nright ;
    N0(25,i) = 0.125*nright ;
    N0(27,i) = 0.125*nright ;
    U0(1,i)  = uright ;
    U0(3,i)  = uright ;
    U0(7,i)  = uright ;
    U0(9,i)  = uright ;
    U0(19,i) = uright ;
    U0(21,i) = uright ;
    U0(25,i) = uright ;
    U0(27,i) = uright ;
    V0(1,i)  = 1 ;
    V0(3,i)  = 1 ;
    V0(7,i)  = -1 ;
    V0(9,i)  = -1 ;
    V0(19,i) = 1 ;
    V0(21,i) = 1 ;
    V0(25,i) = -1 ;
    V0(27,i) = -1 ;
    W0(1,i)  = -1 ;
    W0(3,i)  = 1 ;
    W0(7,i)  = -1 ;
    W0(9,i)  = 1 ;
    W0(19,i) = -1 ;
    W0(21,i) = 1 ;
    W0(25,i) = -1 ;
    W0(27,i) = 1 ;
end

V0 = 1.0*V0 + 0 + 0.5*U0 ;
W0 = .0*W0 - 0 - 0.*U0 + 0.*V0 ;

if restart == 1
    load('initial_granular','N1','U1','V1','W1') ;
    N0 = N1 ;
    U0 = U1 ;
    V0 = V1 ;
    W0 = W1 ;
end
    
for i =1:Ny
    M0(:,i) = moments_3D_27node_16mom(N0(:,i),U0(:,i),V0(:,i),W0(:,i)) ;
    [N1(:,i),U1(:,i),V1(:,i),W1(:,i)] = twentyseven_node_16mom_hycqmom_3D(M0(:,i)) ;
end
M0(:,1) = moments_3D_27node_16mom(N0(:,Ny-1),U0(:,Ny-1),V0(:,Ny-1),W0(:,Ny-1)) ;
M0(:,Ny) = moments_3D_27node_16mom(N0(:,2),U0(:,2),V0(:,2),W0(:,2)) ;
[N1(:,1),U1(:,1),V1(:,1),W1(:,1)] = twentyseven_node_16mom_hycqmom_3D(M0(:,1)) ;
[N1(:,Ny),U1(:,Ny),V1(:,Ny),W1(:,Ny)] = twentyseven_node_16mom_hycqmom_3D(M0(:,Ny)) ;
%
%
M1 = M0 ;
for jj = 1:27
    NN(jj,:) = N1(jj,:)./(M1(1,:) + 1.d-16) ;
end
for jj = 1:16
    NM(jj,:) = M1(jj,:)./(M1(1,:) + 1.d-16) ;
end

output_granular_3D_27node_16mom(M1,N1,U1,V1,W1,Ycell,0,Ny,movie,1) ;

output_granular_3D_16mom(M1,Ycell,0,Ny,movie,1)

pause(1)
close all ;
%
t = 0 ; 
tsauv = taff ; % output first iteration
naf = 1 ;
k = 1 ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Time loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
while( t < Tmax + eps )
    display (t)
    if ( t >= 500 ) 
        movie = 1 ;
    end
    Umax = 0.01 ;
    for j=1:27   % nodes number
        for i=1:Ny
            Umax = max(Umax,abs(U0(j,i))) ; 
        end
    end
    if ( Umax > 3 )
        display(Umax)
        %break
    end
    Dt = cfl*Dy/Umax ;
    if ( collision == 1 )
        Dt = min(tcol/alpha/vrms/10,Dt) ;
    end
    t = t + Dt ; 
    tsauv = tsauv + Dt ;
    Dt2 = Dt/2 ;
    %      
    % Evaluation of the new moments using RK2 SSP
    %
    % Step 1 du RK2 SSP
    %
    % 1/2 step of spatial transport
    %
    for i=2:Ny-1          
        % Etape 1           
        Nlmoins = N0(:,i-1) ;
        Ulmoins = U0(:,i-1) ;
        Vlmoins = V0(:,i-1) ;
        Wlmoins = W0(:,i-1) ;
        Nlplus  = N0(:,i)   ;
        Ulplus  = U0(:,i)   ;
        Vlplus  = V0(:,i)   ;
        Wlplus  = W0(:,i)   ;
        Nrmoins = N0(:,i)   ;
        Urmoins = U0(:,i)   ;
        Vrmoins = V0(:,i)   ;
        Wrmoins = W0(:,i)   ;
        Nrplus  = N0(:,i+1) ;
        Urplus  = U0(:,i+1) ;
        Vrplus  = V0(:,i+1) ;
        Wrplus  = W0(:,i+1) ;
        %
        Fleft  = flux_3D_27node_16mom(Nlmoins,Ulmoins,Vlmoins,Wlmoins,Nlplus,Ulplus,Vlplus,Wlplus);
        Fright = flux_3D_27node_16mom(Nrmoins,Urmoins,Vrmoins,Wrmoins,Nrplus,Urplus,Vrplus,Wrplus); 
        % update moments by time step
        M1(:,i) = M0(:,i) - (Dt/Dy)*(Fright - Fleft) ;
    end
    % update weights and abscissas
    for i=2:Ny-1         
        [N1(:,i),U1(:,i),V1(:,i),W1(:,i)] = twentyseven_node_16mom_hycqmom_3D(M1(:,i)) ;
        if project == 1
            M1(:,i) = moments_3D_27node_16mom(N1(:,i),U1(:,i),V1(:,i),W1(:,i)) ; % projection step
        end
    end
    M1(:,1) = moments_3D_27node_16mom(N1(:,Ny-1),U1(:,Ny-1),V1(:,Ny-1),W1(:,Ny-1)) ;
    M1(:,Ny) = moments_3D_27node_16mom(N1(:,2),U1(:,2),V1(:,2),W1(:,2)) ;
    [N1(:,1),U1(:,1),V1(:,1),W1(:,1)] = twentyseven_node_16mom_hycqmom_3D(M1(:,1)) ;
    [N1(:,Ny),U1(:,Ny),V1(:,Ny),W1(:,Ny)] = twentyseven_node_16mom_hycqmom_3D(M1(:,Ny)) ;
    % Right-hand sides of moment equations
    % contribution due to BGK collisions
    if ( collision == 1 )
        for i=1:Ny
            M1(:,i) = collisions_esbgknew32_inelastic(M1(:,i),Dt2,tcol,ep,b) ;
            [N1(:,i),U1(:,i),V1(:,i),W1(:,i)] = twentyseven_node_16mom_hycqmom_3D(M1(:,i)) ;
            if project == 1
                M1(:,i) = moments_3D_27node_16mom(N1(:,i),U1(:,i),V1(:,i),W1(:,i)) ; % projection step
            end
        end
    end
    % Step 2 du RK2:  M0 are old values, M1 are 1/2-step values
    %
    % Step of spatial transport
    for i=2:Ny-1                          
        Nlmoins = N1(:,i-1) ;
        Ulmoins = U1(:,i-1) ;
        Vlmoins = V1(:,i-1) ;
        Wlmoins = W1(:,i-1) ;
        Nlplus  = N1(:,i)   ;
        Ulplus  = U1(:,i)   ;
        Vlplus  = V1(:,i)   ;
        Wlplus  = W1(:,i)   ;
        Nrmoins = N1(:,i)   ;
        Urmoins = U1(:,i)   ;
        Vrmoins = V1(:,i)   ;
        Wrmoins = W1(:,i)   ;
        Nrplus  = N1(:,i+1) ;
        Urplus  = U1(:,i+1) ;
        Vrplus  = V1(:,i+1) ;
        Wrplus  = W1(:,i+1) ;
        %
        Fleft  = flux_3D_27node_16mom(Nlmoins,Ulmoins,Vlmoins,Wlmoins,Nlplus,Ulplus,Vlplus,Wlplus);
        Fright = flux_3D_27node_16mom(Nrmoins,Urmoins,Vrmoins,Wrmoins,Nrplus,Urplus,Vrplus,Wrplus);
        % update moments by full time step
        M1(:,i) = M1(:,i) - (Dt/Dy)*(Fright - Fleft) ;
        M1(:,i) = 0.5*( M0(:,i) + M1(:,i) ) ;
    end
    % update weights and abscissas
    for i=2:Ny-1 
        [N1(:,i),U1(:,i),V1(:,i),W1(:,i)] = twentyseven_node_16mom_hycqmom_3D(M1(:,i)) ;
        if project == 1
            M1(:,i) = moments_3D_27node_16mom(N1(:,i),U1(:,i),V1(:,i),W1(:,i)) ; % projection step
        end
    end
    M1(:,1) = moments_3D_27node_16mom(N1(:,Ny-1),U1(:,Ny-1),V1(:,Ny-1),W1(:,Ny-1)) ;
    M1(:,Ny) = moments_3D_27node_16mom(N1(:,2),U1(:,2),V1(:,2),W1(:,2)) ;
    [N1(:,1),U1(:,1),V1(:,1),W1(:,1)] = twentyseven_node_16mom_hycqmom_3D(M1(:,1)) ;
    [N1(:,Ny),U1(:,Ny),V1(:,Ny),W1(:,Ny)] = twentyseven_node_16mom_hycqmom_3D(M1(:,Ny)) ;
    % Right-hand sides of moment equations
    % contribution due to BGK collisions over full time step
    if ( collision == 1 )
        for i=1:Ny
            M1(:,i) = collisions_esbgknew32_inelastic(M1(:,i),Dt,tcol,ep,b) ;
            [N1(:,i),U1(:,i),V1(:,i),W1(:,i)] = twentyseven_node_16mom_hycqmom_3D(M1(:,i)) ;
            if project == 1
                M1(:,i) = moments_3D_27node_16mom(N1(:,i),U1(:,i),V1(:,i),W1(:,i)) ; % projection step
            end
        end
    end
    for i=1:Ny
        M1(:,i) = moments_3D_27node_16mom(N1(:,i),U1(:,i),V1(:,i),W1(:,i)) ; % projection step
    end
    %
    % End of RK2 time step
    %
    if (tsauv >= taff) 
        tsauv = 0 ; 
        display(t)
        %
        fig1 = figure(1) ;
        %set(fig1,'PaperUnits','normalized','PaperPosition', [0 0 6/8 2/4],...
        %    'units','normalized', 'position',[9/16 1/8 6/16 6/8]) ;
        set(fig1,'PaperUnits','normalized','PaperPosition', [0 0 6/8 2/4],...
             'units','normalized', 'position',[1/8 1/8 4/8 4/8]) ;

        annotation(fig1,'textbox','String',{strcat('Time = ',num2str(t,'%6.2f'))},...
            'FitHeightToText','off','Position',[0.42 0.95 0.18 0.04]);

        % allow time to clear previous figure
        output_granular_3D_27node_16mom(M1,N1,U1,V1,W1,Ycell,t,Ny,movie,k) ;
        pause(1)
        close all ;
        
        fig2 = figure(2) ;
        %set(fig1,'PaperUnits','normalized','PaperPosition', [0 0 6/8 2/4],...
        %    'units','normalized', 'position',[9/16 1/8 6/16 6/8]) ;
        set(fig2,'PaperUnits','normalized','PaperPosition', [0 0 6/8 2/4],...
             'units','normalized', 'position',[1/8 1/8 4/8 4/8]) ;

        annotation(fig2,'textbox','String',{strcat('Time = ',num2str(t,'%6.2f'))},...
            'FitHeightToText','off','Position',[0.42 0.95 0.18 0.04]);

        % allow time to clear previous figure
        output_granular_3D_16mom(M1,Ycell,t,Ny,movie,k) ;
        pause(1)
        close all ;
        
        k = k + 1 ;
    end
    
    M0 = M1 ;
    N0 = N1 ;
    U0 = U1 ; 
    V0 = V1 ;
    W0 = W1 ;
    for jj = 1:27
        NN(jj,:) = N1(jj,:)./(M1(1,:) + 1.d-16) ;
    end
    for jj = 1:16
        NM(jj,:) = M1(jj,:)./(M1(1,:) + 1.d-16) ;
    end
end
%%%%%%%%%%%%%%%%%%%%%
% End of time loop
%%%%%%%%%%%%%%%%%%%%%
%
save('initial_granular','N1','U1','V1','W1') ;
%
%
%plot all moments

% fig1 = figure(1) ;
% set(fig1,'PaperUnits','normalized','PaperPosition', [0 0 6/8 2/4],...
%      'units','normalized', 'position',[1/8 1/8 6/8 6/8]) ;
% 
% annotation(fig1,'textbox','String',{strcat('Time = ',num2str(t,'%6.2f'))},...
%     'FitHeightToText','off','Position',[0.42 0.95 0.18 0.04]);
% output_moments_3D_8node(M1,Ycell,Ny) ;
% %
% fig1 = figure(2) ;
% set(fig1,'PaperUnits','normalized','PaperPosition', [0 0 6/8 2/4],...
%      'units','normalized', 'position',[1/8 1/8 6/8 6/8]) ;
% 
% annotation(fig1,'textbox','String',{strcat('Time = ',num2str(t,'%6.2f'))},...
%     'FitHeightToText','off','Position',[0.42 0.95 0.18 0.04]);
% output_central_moments_3D_8node(M1,Ycell,Ny) ;
%
fig1 = figure(3) ;
set(fig1,'PaperUnits','normalized','PaperPosition', [0 0 6/8 2/4],...
     'units','normalized', 'position',[1/8 1/8 4/8 4/8]) ;

annotation(fig1,'textbox','String',{strcat('Time = ',num2str(t,'%6.2f'))},...
    'FitHeightToText','off','Position',[0.42 0.95 0.18 0.04]);
output_granular_3D_27node_16mom(M1,N1,U1,V1,W1,Ycell,t,Ny,movie,k) ;

fig2 = figure(4) ;
        %set(fig1,'PaperUnits','normalized','PaperPosition', [0 0 6/8 2/4],...
        %    'units','normalized', 'position',[9/16 1/8 6/16 6/8]) ;
        set(fig2,'PaperUnits','normalized','PaperPosition', [0 0 6/8 2/4],...
             'units','normalized', 'position',[1/8 1/8 4/8 4/8]) ;

        annotation(fig2,'textbox','String',{strcat('Time = ',num2str(t,'%6.2f'))},...
            'FitHeightToText','off','Position',[0.42 0.95 0.18 0.04]);
        output_granular_3D_16mom(M1,Ycell,t,Ny,movie,k) ;
% matlabpool close ;
