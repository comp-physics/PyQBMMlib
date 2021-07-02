
module NN_Module
implicit none



type, public :: ML_Closure
   integer :: abscissas, ml_dim, hidden_units, input_size

   real, dimension(:,:), allocatable :: input_w, hidden_w, dense_w
   real, dimension(:), allocatable   :: bias_w, dense_b
   real, dimension(:),   allocatable :: input_state, hidden_state, cell_state, output_state
   
   contains
   procedure :: init => init_NN
   procedure :: predict => predict_NN

end type ML_Closure




contains


subroutine init_NN(this)
  class(ML_Closure), intent(inout) :: this
  integer, parameter :: mc_id = 10


open(mc_id,file='../ML_Code/Dat_Files/MC_Data0.dat',status='old')
read(mc_id,*) this%abscissas, this%ml_dim, this%hidden_units, this%input_size
close(mc_id)

allocate(this%input_w(0:this%input_size-1,0:4*this%hidden_units-1))
allocate(this%hidden_w(0:this%ml_dim-1,0:4*this%hidden_units-1))
allocate(this%bias_w(0:4*this%hidden_units-1))
allocate(this%dense_w(0:this%ml_dim-1,0:this%ml_dim-1))
allocate(this%dense_b(0:this%ml_dim-1))
allocate(this%hidden_state(0:this%hidden_units-1))
allocate(this%cell_state(0:this%hidden_units-1))
allocate(this%output_state(0:this%ml_dim-1))
allocate(this%input_state(0:this%input_size-1))

!real, dimension(0:input_size-1,0:4*hidden_units-1) :: input_w
!real, dimension(0:ml_dim-1,0:4*hidden_units-1) :: hidden_w
!real, dimension(0:4*hidden_units-1) :: bias_w
!real, dimension(0:ml_dim-1,0:ml_dim-1) :: dense_w
!real, dimension(0:ml_dim-1) :: dense_b

!real, dimension(0:hidden_units) :: hidden_state, cell_state

end subroutine



subroutine hard_sigmoid(x,y)
real, intent(in) :: x
real, intent(out) :: y

y = (x+1.0)/2.0
if (y > 1.0) then
   y = 1.0
end if
if (y < 0.0) then
   y = 0.0
end if

end subroutine



subroutine predict_NN(this,Flag_state,cp)
class(ML_Closure), intent(inout) :: this
real, dimension(0:30+this%ml_dim-1), intent(in) :: Flag_state
real, dimension(0:1), intent(in) :: cp


integer :: pp, kk, jj
real :: flag_val
real, dimension(0:this%hidden_units-1) :: input_gate, forget_gate, cell_gate, output_gate

this%input_state(0:4) = Flag_state(0:4)
this%input_state(5)   = cp(0)

! Pass input-vector, hidden state, cell state, through the input gate of the LSTM NN
do kk=0,this%hidden_units-1
   flag_val = 0.0
   do jj=0,this%input_size-1
      flag_val = flag_val +this%input_state(jj)*this%input_w(jj,kk)
   end do
   do jj=0,this%hidden_units-1
      flag_val = flag_val +this%hidden_state(jj)*this%hidden_w(jj,kk)+this%bias_w(kk)
   end do
   call hard_sigmoid(flag_val,input_gate(kk))
end do



! Pass input-vector, hidden state, cell state, through the forget gate of the LSTM NN
do kk=0,this%hidden_units-1
   flag_val = 0.0
   do jj=0,this%input_size-1
      flag_val = flag_val +this%input_state(jj)*this%input_w(jj,this%hidden_units+kk)
   end do
   do jj=0,this%hidden_units-1
      flag_val = flag_val +this%hidden_state(jj)*this%hidden_w(jj,this%hidden_units+kk)+this%bias_w(this%hidden_units+kk)
   end do
   call hard_sigmoid(flag_val,forget_gate(kk))
end do



! Pass input-vector, hidden state, cell state, through the cell gate of the LSTM NN
do kk=0,this%hidden_units-1
   flag_val = 0.0
   do jj=0,this%input_size-1
      flag_val = flag_val +this%input_state(jj)*this%input_w(jj,2*this%hidden_units+kk)
   end do
   do jj=0,this%hidden_units-1
      flag_val = flag_val +this%hidden_state(jj)*this%hidden_w(jj,2*this%hidden_units+kk)+this%bias_w(2*this%hidden_units+kk)
   end do
   cell_gate(kk) = TANH(flag_val)
end do




! Pass input-vector, hidden state, cell state, through the output gate of the LSTM NN
do kk=0,this%hidden_units-1
   flag_val = 0.0
   do jj=0,this%input_size-1
      flag_val = flag_val +this%input_state(jj)*this%input_w(jj,3*this%hidden_units+kk)
   end do
   do jj=0,this%hidden_units-1
      flag_val = flag_val +this%hidden_state(jj)*this%hidden_w(jj,3*this%hidden_units+kk)+this%bias_w(3*this%hidden_units+kk)
   end do
   call hard_sigmoid(flag_val,output_gate(kk))
end do




! Calculate new cell state
do kk=0,this%hidden_units-1
   this%cell_state(kk) = forget_gate(kk)*this%cell_state(kk) +input_gate(kk)*cell_gate(kk)
end do

! Calculate new hidden state
do kk=0,this%hidden_units-1
   this%hidden_state(kk) = output_gate(kk)*TANH(this%cell_state(kk))
end do


! Pass hidden state through final layer to extract new predictions.
do kk=0,this%ml_dim
   flag_val = 0.0
   do jj=0,this%hidden_units-1
      flag_val = flag_val +this%hidden_state(jj)*this%dense_w(jj,kk)+this%dense_b(kk)
   end do
   this%output_state(kk) = flag_val
end do



end subroutine









end module NN_Module




subroutine random_pressure(pressure,total_times,dt)
integer, intent (in) :: total_times
real, dimension(0:total_times-1), intent (out) :: pressure
real, intent (in) :: dt
integer, parameter   :: seed = 86456

real, dimension(0:2) :: cp_pha, cp_amp, cp_per
integer :: tt
real, parameter :: pi = 4.0*ATAN(1.0)

call srand(seed)

cp_pha(0) = 2.0*pi*RAND(); cp_pha(1) = 2.0*pi*RAND(); cp_pha(2) = 2.0*pi*RAND()
cp_amp(0) = 0.15+0.05*RAND(); cp_amp(1) = 0.15+0.05*RAND(); cp_amp(2) = 0.15+0.05*RAND()
cp_per(0) = 5.00; cp_per(1) = 7.00; cp_per(2) = 9.00

do tt=0,total_times-1
   pressure(tt) = 1.0          +cp_amp(0)*SIN(2.0*pi*dt*float(tt)/cp_per(0)+cp_pha(0))
   pressure(tt) = pressure(tt) +cp_amp(1)*SIN(2.0*pi*dt*float(tt)/cp_per(1)+cp_pha(1))
   pressure(tt) = pressure(tt) +cp_amp(2)*SIN(2.0*pi*dt*float(tt)/cp_per(2)+cp_pha(2))
end do

end subroutine





subroutine CHyQMOM_transform(x,ml_dim,abscissas,ids,ML_correction)
integer, intent (in) :: ml_dim, abscissas
real, dimension(0:30+ml_dim-1), intent (inout) :: x
real, dimension(0:1,0:30), intent (in) :: ids
real, dimension(0:ml_dim-1), intent (in) :: ML_correction

integer :: pp
real :: weight_val,xi_val,xid_val

x(0:29) = 0.0
do pp=0,abscissas-1
   weight_val = x(30+3*pp)   +ML_correction(3*pp)
   xi_val     = x(30+3*pp+1) +ML_correction(3*pp+1)
   xid_val    = x(30+3*pp+2) +ML_correction(3*pp+2)

   do  mm=1,30
      x(mm-1) = x(mm-1)+weight_val* xi_val**(ids(0,mm)) *xid_val**(ids(1,mm))
   end do

end do

end subroutine




subroutine CHyQMOM_inversion(x,ml_dim,abscissas,sigmaR_tol,sigmaRd_tol,QMOM)
real, intent(in) :: sigmaR_tol, sigmaRd_tol
integer, intent (in) :: ml_dim, abscissas
real, dimension(0:4), intent (inout) :: x

integer :: ii
real :: sigmaR, sigmaRd, val_flag
real, dimension(0:ml_dim-1), intent(out) :: QMOM

sigmaR = x(2) -x(0)*x(0)
if (sigmaR < sigmaR_tol) then
sigmaR = sigmaR_tol
end if
sigmaR = sqrt(sigmaR)

val_flag = (x(3)-x(0)*x(1))/sigmaR
sigmaRd = x(4) -val_flag*val_flag -x(1)*x(1)
if (sigmaRd < sigmaRd_tol) then
sigmaRd = sigmaRd_tol
end if
sigmaRd = sqrt(sigmaRd)

QMOM(0)  = 0.25
QMOM(3)  = 0.25
QMOM(6)  = 0.25
QMOM(9)  = 0.25

QMOM(1)  = x(0)+sigmaR
QMOM(4)  = x(0)+sigmaR
QMOM(7)  = x(0)-sigmaR
QMOM(10) = x(0)-sigmaR

QMOM(2)  = x(1)+val_flag+sigmaRd
QMOM(5)  = x(1)+val_flag-sigmaRd
QMOM(8)  = x(1)-val_flag+sigmaRd
QMOM(11) = x(1)-val_flag-sigmaRd

if (abscissas > 4) then
do ii =4,abscissas
QMOM(3*ii) = 0.0
QMOM(3*ii+1) = x(0)
QMOM(3*ii+2) = x(1)
end do
end if

end subroutine



subroutine RHS_eval(x,ml_dim,Re,cp,rhs)
integer, intent (in) :: ml_dim
real, intent (in) :: Re
real, intent (in) :: cp
real, dimension(0:30+ml_dim-1), intent (in) :: x
real, dimension(0:4), intent (inout) :: rhs

rhs(0) = x(1)
rhs(1) = -1.5*x(21) -(4.0/Re)*x(22) +x(23) -cp*x(24)
rhs(2) =  2.0*x(3)
rhs(3) = -0.5*x(4)  -(4.0/Re)*x(25) +x(26) -cp
rhs(4) = -3.0*x(27) -(8.0/Re)*x(28) +2.0*x(29) -2.0*cp*x(25)


end subroutine


















subroutine RK4_evolution(ml,Hybrid_state,pressure,ml_dim,total_times,abscissas,sigmaR_tol,sigmaRd_tol,ids,Re,dt,ML_correction, &
     input_w, hidden_w, bias_w, dense_w, dense_b,hidden_units, input_size, hidden_state, cell_state)
use NN_Module

type(ML_Closure), intent(inout) :: ml

integer, intent (in) :: total_times
integer, intent (in) :: ml_dim
integer, intent (in) :: abscissas
real, intent (in) :: sigmaR_tol, sigmaRd_tol
real, dimension(0:1,0:30), intent (in) :: ids
real, intent (in) :: Re, dt

integer, intent (in) :: hidden_units
integer, intent (in) :: input_size

real, dimension(0:input_size-1,0:4*hidden_units-1), intent (in) :: input_w
real, dimension(0:ml_dim-1,0:4*hidden_units-1),     intent (in) :: hidden_w
real, dimension(0:4*hidden_units-1),                intent (in) :: bias_w
real, dimension(0:ml_dim-1,0:ml_dim-1),             intent (in) :: dense_w
real, dimension(0:ml_dim-1),                        intent (in) :: dense_b

real, dimension(0:hidden_units-1), intent (inout) :: hidden_state, cell_state


real, dimension(0:30+ml_dim-1,0:total_times-1), intent (inout) :: Hybrid_state
real, dimension(0:total_times-1) :: pressure
real, dimension(0:ml_dim-1,0:total_times-1) :: ML_correction

integer :: tt

real, dimension(0:30+ml_dim-1) :: Flag_state
real, dimension(0:1) :: cp

real :: tol, dt_flag
integer :: sub_step
real, dimension(0:30+ml_dim-1) :: y1, y2


do tt=0,total_times-1

   Flag_state(:) = Hybrid_state(:,tt)
   cp(0) = pressure(tt); cp(1) = pressure(tt+1)
   call ml%predict(Flag_state,cp)

  ! call ML_Prediction(Flag_state,cp,ML_correction(:,tt),ml_dim, & 
  !      input_w, hidden_w, bias_w, dense_w, dense_b,hidden_units, input_size, hidden_state, cell_state)
   call RK4_step(Flag_state,cp,ml_dim,abscissas,sigmaR_tol,sigmaRd_tol,ids,Re,dt,ML_correction(:,tt))

   y1(:) = Flag_state(:)

   tol = 1.0
   sub_step = 2
   do while (tol > 1.0*10**(-7))
      Flag_state(:) = Hybrid_state(:,tt)
      dt_flag = dt/float(sub_step)
      do jj=0,sub_step-1
         cp(0) = pressure(tt)*float(sub_step-jj)/float(sub_step) +pressure(tt+1)*float(jj)/float(sub_step)
         cp(1) = pressure(tt)*float(sub_step-1-jj)/float(sub_step) +pressure(tt+1)*float(jj+1)/float(sub_step)
         call RK4_step(Flag_state,cp,ml_dim,abscissas,sigmaR_tol,sigmaRd_tol,ids,Re,dt,ML_correction(:,tt))
      end do
      y2(:) = Flag_state
      tol = abs(y2(4)-y1(4))/y1(4)
      y1(:) = Flag_state
   end do

   Hybrid_state(:,tt+1) = Flag_state(:)


end do

end subroutine





subroutine RK4_step(Flag_state,cp,ml_dim,abscissas,sigmaR_tol,sigmaRd_tol,ids,Re,dt,ML_correction)
integer, intent (in) :: ml_dim, abscissas
real, intent (in) :: sigmaR_tol, sigmaRd_tol
real, dimension(0:1,0:30), intent (in) :: ids
real, intent (in) :: Re, dt

real, dimension(0:ml_dim-1), intent (in) :: ML_correction

real, dimension(0:30+ml_dim-1) :: Flag_state
real, dimension(0:1) :: cp
real, dimension(0:3,0:4) :: rhs

real, dimension(0:4) :: LM_state
real, dimension(0:30+ml_dim-1) :: prev_state
integer :: mm

prev_state(:) = Flag_state(:)

call CHyQMOM_inversion(Flag_state(0:4),ml_dim,abscissas,sigmaR_tol,sigmaRd_tol,Flag_state(30:30+ml_dim-1))
call CHyQMOM_transform(Flag_state,ml_dim,abscissas,ids,ML_correction)
call RHS_eval(Flag_state,ml_dim,Re,cp(0),rhs(0,:))
do mm=0,4
   LM_state(mm) = prev_state(mm) +0.5*dt*rhs(0,mm)
end do


call CHyQMOM_inversion(LM_state(0:4),ml_dim,abscissas,sigmaR_tol,sigmaRd_tol,Flag_state(30:30+ml_dim-1))
call CHyQMOM_transform(Flag_state,ml_dim,abscissas,ids,ML_correction)
call RHS_eval(Flag_state,ml_dim,Re,0.5*(cp(0)+cp(1)),rhs(1,:))
do mm=0,4
   LM_state(mm) = prev_state(mm) +0.5*dt*rhs(1,mm)
end do

call CHyQMOM_inversion(LM_state(0:4),ml_dim,abscissas,sigmaR_tol,sigmaRd_tol,Flag_state(30:30+ml_dim-1))
call CHyQMOM_transform(Flag_state,ml_dim,abscissas,ids,ML_correction)
call RHS_eval(Flag_state,ml_dim,Re,0.5*(cp(0)+cp(1)),rhs(2,:))
do mm=0,4
   LM_state(mm) = prev_state(mm) +1.0*dt*rhs(2,mm)
end do

call CHyQMOM_inversion(LM_state(0:4),ml_dim,abscissas,sigmaR_tol,sigmaRd_tol,Flag_state(30:30+ml_dim-1))
call CHyQMOM_transform(Flag_state,ml_dim,abscissas,ids,ML_correction)
call RHS_eval(Flag_state,ml_dim,Re,cp(1),rhs(3,:))
do mm=0,4
   LM_state(mm) = prev_state(mm) +(dt/6.0)*(rhs(0,mm) +2.0*rhs(1,mm) +2.0*rhs(2,mm) +rhs(3,mm))
end do

call CHyQMOM_inversion(LM_state(0:4),ml_dim,abscissas,sigmaR_tol,sigmaRd_tol,Flag_state(30:30+ml_dim-1))
call CHyQMOM_transform(Flag_state,ml_dim,abscissas,ids,ML_correction)






end subroutine


















program Main
use NN_Module
implicit none

real, parameter :: dt = 0.01
integer, parameter :: total_times = 10001
integer, parameter :: total_cases = 1

real, dimension(0:1,0:30) :: ids
real, parameter :: sigmaR_tol  = 10.0**(-4)
real, parameter :: sigmaRd_tol = 10.0**(-4)
real, parameter :: Re = 1000.0
integer, parameter :: abscissas = 4
integer, parameter :: ml_dim = 3*abscissas

integer, parameter :: hidden_units = ml_dim
integer, parameter :: input_size   = 6

real, dimension(0:input_size-1,0:4*hidden_units-1) :: input_w
real, dimension(0:ml_dim-1,0:4*hidden_units-1) :: hidden_w
real, dimension(0:4*hidden_units-1) :: bias_w
real, dimension(0:ml_dim-1,0:ml_dim-1) :: dense_w
real, dimension(0:ml_dim-1) :: dense_b

real, dimension(0:hidden_units) :: hidden_state, cell_state

real, dimension(0:total_cases-1,0:30+ml_dim-1,0:total_times-1) :: Hybrid_MOM
real, dimension(0:total_cases-1,0:29,0:total_times-1) :: MC_data, CHYQMOM_data
real, dimension(0:total_cases-1,0:total_times-1) :: pressure
real, dimension(0:total_cases-1,0:ml_dim-1,0:total_times-1) :: ML_correction

integer :: ii, tt
real, dimension(0:30+ml_dim-1)  :: x
real, dimension(ml_dim) :: QMOM

integer, parameter :: file_id = 1
integer, parameter :: lstm_id = 2
integer, parameter :: dense_id = 3
integer, parameter :: test_id = 4
integer, parameter :: mc_id = 5
integer, parameter :: chyqmom_id = 6

type(ML_Closure) :: ml
call ml%init

ids(0,0) = 0.0; ids(1,0) = 0.0

ids(0,1) = 1.0; ids(1,1) = 0.0
ids(0,2) = 0.0; ids(1,2) = 1.0

ids(0,3) = 2.0; ids(1,3) = 0.0
ids(0,4) = 1.0; ids(1,4) = 1.0
ids(0,5) = 0.0; ids(1,5) = 2.0

ids(0,6) = 3.0; ids(1,6) = 0.0
ids(0,7) = 2.0; ids(1,7) = 1.0
ids(0,8) = 1.0; ids(1,8) = 2.0
ids(0,9) = 0.0; ids(1,9) = 3.0

ids(0,10) = 4.0; ids(1,10) = 0.0
ids(0,11) = 3.0; ids(1,11) = 1.0
ids(0,12) = 2.0; ids(1,12) = 2.0
ids(0,13) = 1.0; ids(1,13) = 3.0
ids(0,14) = 0.0; ids(1,14) = 4.0

ids(0,15) = 5.0; ids(1,15) = 0.0
ids(0,16) = 4.0; ids(1,16) = 1.0
ids(0,17) = 3.0; ids(1,17) = 2.0
ids(0,18) = 2.0; ids(1,18) = 3.0
ids(0,19) = 1.0; ids(1,19) = 4.0
ids(0,20) = 0.0; ids(1,20) = 5.0

ids(0,21) = 3.0*(1.0-1.4); ids(1,21) = 0.0

ids(0,22) = -1.0; ids(1,22) = 2.0
ids(0,23) = -2.0; ids(1,23) = 1.0
ids(0,24) = -4.0; ids(1,24) = 0.0
ids(0,25) = -1.0; ids(1,25) = 0.0
ids(0,26) = -1.0; ids(1,26) = 1.0
ids(0,27) = -3.0; ids(1,27) = 0.0
ids(0,28) = -1.0; ids(1,28) = 3.0
ids(0,29) = -2.0; ids(1,29) = 2.0
ids(0,30) = -4.0; ids(1,30) = 1.0

open(mc_id,file='../ML_Code/Dat_Files/MC_Data0.dat',status='old')
read(mc_id,*)
do tt=0,total_times-1
   read(mc_id,*) MC_data(0,:,tt), pressure(0,tt)
end do
close(mc_id)


open(chyqmom_id,file='../ML_Code/Dat_Files/CHyQMOM_Data0.dat',status='old')
read(chyqmom_id,*)
do tt=0,total_times-1
   read(chyqmom_id,*) CHYQMOM_data(0,:,tt), pressure(0,tt)
end do
close(chyqmom_id)


open(lstm_id,file='../ML_Code/Weights/LSTM_Weights.dat',status='old')
do ii=0,4*hidden_units-1
   read(lstm_id,*) ml%input_w(:,ii), ml%hidden_w(:,ii), ml%bias_w(ii)
end do
close(lstm_id)

open(dense_id,file='../ML_Code/Weights/Dense_Weights.dat',status='old')
do ii=0,ml_dim-1
   read(dense_id,*) ml%dense_w(:,ii), ml%dense_b(ii)
end do
close(dense_id)

Hybrid_MOM(:,:,:) = 0.0
do ii=0,total_cases-1
Hybrid_MOM(ii,0,0) = 1.0
Hybrid_MOM(ii,1,0) = 0.0
Hybrid_MOM(ii,2,0) = 1.01
Hybrid_MOM(ii,3,0) = 0.0
Hybrid_MOM(ii,4,0) = 0.1
!call random_pressure(pressure(ii,:),total_times,dt)
end do


do ii=0,total_cases-1
   call RK4_evolution(ml,Hybrid_MOM(ii,:,:),pressure(ii,:),ml_dim,total_times,abscissas,sigmaR_tol,sigmaRd_tol,ids,Re,dt, &
        ML_correction(ii,:,:), &
        input_w, hidden_w, bias_w, dense_w, dense_b,hidden_units,input_size, hidden_state, cell_state)
end do


print *, "Hello World"

open(file_id,file = 'Hybrid_output.dat',status = 'old')
do tt=0,total_times-1
   write(file_id,*) Hybrid_MOM(0,0:4,tt)
end do
close(file_id)

open(test_id,file = 'Hybrid_output2.dat',status = 'old')
do ii=0,4*hidden_units-1
   write(test_id,*) input_w(:,ii)
end do
close(test_id)




end program Main
