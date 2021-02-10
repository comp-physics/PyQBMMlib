import matplotlib.pyplot as plt
import numpy as np

file_name_local = 'build/results_local.csv'
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
    x =  data1[1:, 0]

    plt.plot(x, data1[1:, 1], label='c++', color='b')
    plt.plot(x, data1[1:, 2], label='cuda', color='r')
    # plt.plot(x, data1[1:, 3], label='cufft overlap local', color='k')
    # linear best fit
    m_1, b_1 = np.polyfit(x, data1[1:, 1], 1)
    m_2, b_2 = np.polyfit(x, data1[1:, 2], 1)
    # m_3, b_3 = np.polyfit(x, data1[1:, 3], 1)
    plt.plot(x, m_1*x+b_1, color='b', linestyle='dotted')
    plt.plot(x, m_2*x+b_2, color='r', linestyle='dotted')
    # plt.plot(x, m_3*x+b_3, color='k', linestyle='dotted')

    # plt.plot(x, data2[1:, 1], label='fftw_cluster', color='c')
    # plt.plot(x, data2[1:, 2], label='cufft sequential_cluster', color='y')
    # plt.plot(x, data2[1:, 3], label='cufft overlap cluster', color='m')
    # linear best fit
    # m_1_2, b_1_2 = np.polyfit(x, data2[1:, 1], 1)
    # m_2_2, b_2_2 = np.polyfit(x, data2[1:, 2], 1)
    # m_3_2, b_3_2 = np.polyfit(x, data2[1:, 3], 1)
    # plt.plot(x, m_1_2*x+b_1_2, color='c', linestyle='dotted')
    # plt.plot(x, m_2_2*x+b_2_2, color='y', linestyle='dotted')
    # plt.plot(x, m_3_2*x+b_3_2, color='m', linestyle='dotted')

    plt.grid(True)
    plt.xscale('log')
    # plt.xlim([0, np.max(x)])
    plt.ylim([0, np.max(data1[1:, 1])])
    plt.title('Computation Time Vs Input Size')
    plt.xlabel('Input size')
    plt.ylabel('Time (s)')
    plt.legend()

    plt.show()

# def plot_single_result(data):


if __name__ == "__main__":

    data1 = np.genfromtxt(file_name_local, delimiter=',')
    # data2 = np.genfromtxt(file_name_remote, delimiter=',')
    # plot_result(data1)
    plot_compared_result(data1, data1)


    # data_1 = np.genfromtxt('build/results_trial2.csv', delimiter=',')
    # data_2 = np.genfromtxt('build/results_trial3.csv', delimiter=',')
    # data_3 = np.genfromtxt('build/results_trial4.csv', delimiter=',')

    # data_avg = (data_1 + data_2 + data_3)/3
    # plot_compared_result(data_avg, data_avg)