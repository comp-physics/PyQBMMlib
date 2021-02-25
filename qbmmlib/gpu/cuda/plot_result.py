import matplotlib.pyplot as plt
import numpy as np

file_name_local = 'build/results_log.csv'
save_name = 'result.png'

def plot_result(data):
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    N = len(data[1:, 2])
    x =  data[1:, 0]

    ax[0].plot(x, data[1:, 1], label='OMP 32 core', color='b')
    ax[0].plot(x, data[1:, 2], label='CUDA', color='r')
    # linear best fit
    m_1, b_1 = np.polyfit(x, data[1:, 1], 1)
    m_2, b_2 = np.polyfit(x, data[1:, 2], 1)
    ax[0].plot(x, m_1*x+b_1, color='b', linestyle='dotted')
    ax[0].plot(x, m_2*x+b_2, color='r', linestyle='dotted')

    if_1 = np.zeros_like(data[1:, 1])
    if_2 = np.zeros_like(data[1:, 2])
    for i in range(1, N):
        if_1[i] = data[i+1, 1] / data[1, 1]
        if_2[i] = data[i+1, 2] / data[1, 2]
    
    m_1_if, b_1_if = np.polyfit(x, if_1, 1)
    m_2_if, b_2_if = np.polyfit(x, if_2, 1)

    ax[1].plot(x, if_1, color='b')
    ax[1].plot(x, if_2, color='r')
    ax[1].plot(x, m_1_if*x+b_1_if, color='b', linestyle='dotted')
    ax[1].plot(x, m_2_if*x+b_2_if, color='r', linestyle='dotted')

    ax[0].set_xlim([0, np.max(x)])
    ax[0].set_ylim([0, np.max(data[1:, 1])])
    ax[0].grid(True)
    ax[0].set_ylabel('Computation time (s)')
    ax[0].legend()
    # ax[1].set_ylim([0, np.max(x) + 100])
    ax[1].set_xlim([0, np.max(x)])
    ax[1].grid(True)
    ax[1].set_ylabel('Increase factor')

    fig.tight_layout()
    ax[0].title.set_text('Computation Time Vs Input Size')
    ax[1].set_xlabel('Input size ')
    
    plt.show()

def plot_compared_result(data1, data2):
    N = len(data1[1:, 2])
    fit_lb = int(np.ceil(5*N/7))
    x =  data1[1:, 0]
    x2 = data2[1:, 0]

    plt.plot(x, data1[1:, 1], label='c++ 10 core', color='b')
    plt.plot(x, data1[1:, 2], label='cuda total time', color='r')
    # plt.plot(x, data1[1:, 3], label='cufft overlap local', color='k')
    # linear best fit

    m_1, b_1 = np.polyfit(np.log(x[fit_lb:]), np.log(data1[fit_lb+1:, 1]), 1)
    m_2, b_2 = np.polyfit(np.log(x[fit_lb:]), np.log(data1[fit_lb+1:, 2]), 1)
    # m_3, b_3 = np.polyfit(x, data1[1:, 3], 1)
    plt.plot(x, np.exp(m_1*np.log(x) + b_1), color='b', linestyle='dotted')
    plt.plot(x, np.exp(m_2*np.log(x) + b_2), color='r', linestyle='dotted')
    # plt.plot(x, m_3*x+b_3, color='k', linestyle='dotted')

    # plt.plot(x, data2[1:, 1], label='fftw_cluster', color='c')

    plt.plot(x2, data2[1:, 2], label='cuda kernel time', color='y')
    # plt.plot(x, data2[1:, 3], label='cufft overlap cluster', color='m')
    # linear best fit
    m_1_2, b_1_2 = np.polyfit(np.log(x2[fit_lb:]), np.log(data2[fit_lb+1:, 2]), 1)
    # m_2_2, b_2_2 = np.polyfit(x, data2[1:, 2], 1)
    # m_3_2, b_3_2 = np.polyfit(x, data2[1:, 3], 1)
    plt.plot(x2, np.exp(m_1_2*np.log(x2) + b_1_2), color='y', linestyle='dotted')
    # plt.plot(x, m_2_2*x+b_2_2, color='y', linestyle='dotted')
    # plt.plot(x, m_3_2*x+b_3_2, color='m', linestyle='dotted')

    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim([1e-3, np.max(data1[1:, 1])])
    plt.title('Computation Time Vs Input Size')
    plt.xlabel('Input size')
    plt.ylabel('Time (s)')
    plt.legend()

    plt.show()


if __name__ == "__main__":

    data1 = np.genfromtxt('build/chyqmom9_total_local.csv', delimiter=',')
    data2 = np.genfromtxt('build/chyqmom9_omp_local.csv', delimiter=',')

    # start plotting
    # Top: time result. Bot: omp_time / cuda_time
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    N = len(data1[1:, 2])
    fit_lb = int(np.ceil(5*N/7))

    x1 =  data1[:, 0]
    y1 = np.minimum(data1[:, 1], data1[:, 2], data1[:, 3])
    x2 = data2[:, 0]
    y2 = np.minimum(data2[:, 1], data2[:, 2], np.minimum(data2[:, 3], data2[:, 4], data2[:, 5]))

    # time data
    ax[0].plot(x1, y1, label='c++ 10 core', color='b')
    ax[0].plot(x2, y2, label='cuda total time', color='r')
    # fitted lined
    m_1, b_1 = np.polyfit(np.log(x1[fit_lb:]), np.log(y1[fit_lb:]), 1)
    m_2, b_2 = np.polyfit(np.log(x2[fit_lb:]), np.log(y2[fit_lb:]), 1)
    ax[0].plot(x1, np.exp(m_1*np.log(x1) + b_1), color='b', linestyle='dotted')
    ax[0].plot(x2, np.exp(m_2*np.log(x2) + b_2), color='r', linestyle='dotted')

    ax[1].plot(x1, y2/y1)

    ax[0].grid(True)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].legend()
    
    ax[0].set_ylabel('Compute time (ms)')
    ax[1].grid(True)
    ax[1].set_xscale('log')
    ax[1].set_ylabel('omp/cuda')
    ax[1].set_xlabel('Input size')

    plt.show()