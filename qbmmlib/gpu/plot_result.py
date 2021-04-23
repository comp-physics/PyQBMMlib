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

    data1 = np.genfromtxt('bridges_1gpu.csv', delimiter=',')
    data2 = np.genfromtxt('bridges_2gpu.csv', delimiter=',')
    data3 = np.genfromtxt('bridges_3gpu.csv', delimiter=',')
    data4 = np.genfromtxt('bridges_4gpu.csv', delimiter=',')
    data5 = np.genfromtxt('bridges_5gpu.csv', delimiter=',')
    data6 = np.genfromtxt('bridges_6gpu.csv', delimiter=',')
    data7 = np.genfromtxt('bridges_7gpu.csv', delimiter=',')
    data8 = np.genfromtxt('bridges_8gpu.csv', delimiter=',')
    # start plotting
    # Top: time result. Bot: omp_time / cuda_time
    fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    N = len(data1[1:, 2])
    fit_lb = int(np.ceil(5*N/7))

    x1 =  data1[:, 0]
    y1 = np.minimum(data1[:, 1], data1[:, 2], data1[:, 3])
    x2 = data2[:, 0]
    y2 = np.minimum(data2[:, 1], data2[:, 2], np.minimum(data2[:, 3], data2[:, 4], data2[:, 5]))
    x3 = data3[:, 0]
    y3 = np.minimum(data3[:, 1], data3[:, 2], np.minimum(data3[:, 3], data3[:, 4], data3[:, 5]))
    x4 = data4[:, 0]
    y4 = np.minimum(data4[:, 1], data4[:, 2], np.minimum(data4[:, 3], data4[:, 4], data4[:, 5]))
    x5 = data5[:, 0]
    y5 = np.minimum(data5[:, 1], data5[:, 2], np.minimum(data5[:, 3], data5[:, 4], data5[:, 5]))
    x6 = data6[:, 0]
    y6 = np.minimum(data6[:, 1], data6[:, 2], np.minimum(data6[:, 3], data6[:, 4], data6[:, 5]))
    x7 = data7[:, 0]
    y7 = np.minimum(data7[:, 1], data7[:, 2], np.minimum(data7[:, 3], data7[:, 4], data7[:, 5]))
    x8 = data8[:, 0]
    y8 = np.minimum(data8[:, 1], data8[:, 2], np.minimum(data8[:, 3], data8[:, 4], data8[:, 5]))
    # time data
    ax[0].plot(x1, y1, label='1 GPU', color='b')
    ax[0].plot(x2, y2, label='2 GPU', color='g')
    ax[0].plot(x3, y3, label='3 GPU', color='r')
    ax[0].plot(x4, y4, label='4 GPU', color='c')
    ax[0].plot(x5, y5, label='5 GPU', color='m')
    ax[0].plot(x6, y6, label='6 GPU', color='y')
    ax[0].plot(x7, y7, label='7 GPU', color='k')
    ax[0].plot(x8, y8, label='8 GPU', color='k')
    # fitted lined
    # m_1, b_1 = np.polyfit(np.log(x1[fit_lb:]), np.log(y1[fit_lb:]), 1)
    # m_2, b_2 = np.polyfit(np.log(x2[fit_lb:]), np.log(y2[fit_lb:]), 1)
    # ax[0].plot(x1, np.exp(m_1*np.log(x1) + b_1), color='b', linestyle='dotted')
    # ax[0].plot(x2, np.exp(m_2*np.log(x2) + b_2), color='r', linestyle='dotted')

    ax[1].plot(x1, y1/y2, label='ratio 1GPU/2GPU', color='g')
    ax[1].plot(x1, y1/y3, label='ratio 1GPU/3GPU', color='r')
    ax[1].plot(x1, y1/y4, label='ratio 1GPU/4GPU', color='c')
    ax[1].plot(x1, y1/y5, label='ratio 1GPU/5GPU', color='m')
    ax[1].plot(x1, y1/y6, label='ratio 1GPU/6GPU', color='y')
    ax[1].plot(x1, y1/y7, label='ratio 1GPU/7GPU', color='k')
    ax[1].plot(x1, y1/y8, label='ratio 1GPU/8GPU', color='k')

    print((y1/y2)[-1])
    print((y1/y3)[-1])
    print((y1/y4)[-1])
    print((y1/y5)[-1])
    print((y1/y6)[-1])
    print((y1/y7)[-1])
    print((y1/y8)[-1])

    ax[0].grid(True)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].legend()
    
    ax[0].set_ylabel('Compute time (ms)')
    ax[1].grid(True)
    ax[1].set_xscale('log')
    ax[1].set_ylabel('ratio')
    ax[1].set_xlabel('Input size')

    plt.savefig('chyqmom4_multi.png', dpi=600)