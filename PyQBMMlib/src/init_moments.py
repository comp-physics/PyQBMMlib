
import * 

# Input: moment indices (indices here, momidx in Mathematica), 
#       distribution shape (in each direction i, P[i] here)
#       and its shape parameters 
#       (x[j] here, usually two shape parameters so j = 1,2)
#
# Output: The required set of raw moments, 'moments'
#
# Notes: The required moment indices momidx depend upon 
#       inversion algorithm (e.g. wheeler/cqmom/whatever)
#       and number of quadrature points/nodes in each
#       internal coordinate direction

# Loop over each internal coordinate direction i 
# P[i] = DistributionShape[i][x[1],x[2]];
# NDF = ProductDistribution[P1, P2];

moments = Table [ Moment[NDF, indices[i]], {i,1,Nmom}]
return moments
