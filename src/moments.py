import numpy as np
from numba import njit


def get_moments(sols):
    moms = []
    for i, sol in enumerate(sols):
        mom = get_moment(sol)
        moms.append(mom)
        sol.moms = mom
        sols[i] = sol

    return sols

def get_moment(sol):
    Nt = len(sol.times)
    mom = np.zeros([Nt, sol.state.Nmom])
    if sol.state.filter:
        if sol.state.Nfilt > 0:
            # Filter based upon number of time steps
            for j in range(Nt):
                jMin = max(j - sol.state.Nfilt, 0)
                mom[j, :] = sol.state.get_quad(
                    vals=sol.save[jMin : j + 1], 
                    filt=True,
                    Nfilt=j + 1 - jMin,
                )
        elif sol.state.Tfilt > 0:
            # Filter based upon number of periods
            for j in range(Nt):
                # Get target time steps
                jTargets = []
                current_time = sol.times[j]
                t_target = \
                        current_time - \
                        sol.state.Tfilt * sol.state.periods
                # print('t_targets: ', t_target)
                for k in range(sol.state.NR0):
                    q = np.argmin(np.abs(sol.times - t_target[k]))
                    if q == 0: q = j #don't average if we cant get whole period
                    jTargets.append(q)
                    # print('jtarg: ',jTargets[i])
                    # Set to zero if they're negative (don't exist)
                    # if jTargets[i] < 0: jTargets[i] = 0
                    # Set to current step if theyre in the future (don't exist yet)
                    # if jTargets[i] > j: jTargets[i] = j

                # print('current j: ', j, 'jTargets: ', jTargets)

                # Make sure to give get_quad enough time steps
                jMin = np.min(np.array(jTargets))
                # print('min J: ', jMin)
                # Array get_quad will operate upon indexes 0:j-jMin
                # so we tell it how to far back to look for each R0
                jShift = np.array(jTargets) - j
                # print('jShift: ', jShift)

                # print('js',jMin,j)
                mom[j, :] = sol.state.get_quad(
                    vals=sol.save[jMin : j+1], 
                    filt=True,
                    Tfilt=True,
                    shifts=jShift
                )
    else:
        for j, vals in enumerate(sol.save):
            mom[j, :] = sol.state.get_quad(vals=vals)

    return mom

@njit()
def get_G(vals,
        mom,
        Nt,
        shifts,
        NR0):
    G = np.zeros(NR0)
    # print(vals)
    # print(np.shape(vals))
    for i in range(NR0):
        G[i] = 0.
        # print('shifts', shifts)
        # print('Nt ->',Nt-1,Nt-1+shifts-1)
        for q in range(Nt-1,Nt-1+shifts[i]-1,-1):
            G[i] += (
                    vals[q, i, 0] ** mom[0] * 
                    vals[q, i, 1] ** mom[1]
                )
        G[i] /= (abs(shifts[i]) + 1.)

    # print('G = ',G)
    return G
