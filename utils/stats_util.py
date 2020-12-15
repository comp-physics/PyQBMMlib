import scipy.stats as stats
import scipy.special as sc
import math
import numpy as np
from stats_util import *


def raw_gaussian_moments_univar(num_moments, mu, sigma):
    """
    This function returns raw 1D-Gaussian moments as a function of
    mean (mu) and standard deviation (sigma)
    """
    moments = np.zeros(num_moments)
    for i_moment in range(num_moments):
        moments[i_moment] = stats.norm.moment(i_moment, mu, sigma)
    return moments


def raw_gaussian_moments_bivar(indices, mu1, mu2, sig1, sig2):
    """
    This function returns raw 2D-Gaussian moments as a function of
    means (mu_1,mu_2) and standard deviations (sigma_1,sigma_2)
    """

    num_moments = len(indices)
    moments = np.zeros(num_moments)
    for i_moment in range(num_moments):
        i = indices[i_moment][0]
        j = indices[i_moment][1]
        moments[i_moment] = (
            (1.0 / math.pi)
            * 2 ** ((-4.0 + i + j) / 2.0)
            * math.exp(
                -(mu1 ** 2.0 / (2.0 * sig1 ** 2)) - (mu2 ** 2.0 / (2 * sig2 ** 2.0))
            )
            * sig1 ** (-1.0 + i)
            * sig2 ** (-1 + j)
            * (
                -math.sqrt(2.0)
                * (-1.0 + (-1.0) ** i)
                * mu1
                * sc.gamma(1.0 + i / 2.0)
                * sc.hyp1f1(1 + i / 2.0, 3.0 / 2.0, mu1 ** 2.0 / (2.0 * sig1 ** 2.0))
                + (1.0 + (-1.0) ** i)
                * sig1
                * sc.gamma((1.0 + i) / 2.0)
                * sc.hyp1f1(
                    (1.0 + i) / 2.0, 1.0 / 2.0, mu1 ** 2.0 / (2.0 * sig1 ** 2.0)
                )
            )
            * (
                -math.sqrt(2.0)
                * (-1 + (-1) ** j)
                * mu2
                * sc.gamma(1.0 + j / 2.0)
                * sc.hyp1f1(1.0 + j / 2.0, 3.0 / 2.0, mu2 ** 2.0 / (2.0 * sig2 ** 2.0))
                + (1.0 + (-1) ** j)
                * sig2
                * sc.gamma((1.0 + j) / 2.0)
                * sc.hyp1f1(
                    (1.0 + j) / 2.0, 1.0 / 2.0, mu2 ** 2.0 / (2.0 * sig2 ** 2.0)
                )
            )
        )
    return moments


def raw_gaussian_moments_trivar(indices, mu1, mu2, mu3, sig1, sig2, sig3):
    num_moments = len(indices)
    moments = np.zeros(num_moments)

    i = 0
    for idx2 in indices:
        idx = idx2.tolist()
        if idx == [0, 0, 0]:
            moments[i] = 1
        if idx == [0, 0, 1]:
            moments[i] = mu3
        if idx == [0, 0, 2]:
            moments[i] = mu3 ** 2 + sig3 ** 2
        if idx == [0, 0, 3]:
            moments[i] = mu3 ** 3 + 3 * mu3 * sig3 ** 2
        if idx == [0, 0, 4]:
            moments[i] = mu3 ** 4 + 6 * mu3 ** 2 * sig3 ** 2 + 3 * sig3 ** 4
        if idx == [0, 1, 0]:
            moments[i] = mu2
        if idx == [0, 1, 1]:
            moments[i] = mu2 * mu3
        if idx == [0, 1, 2]:
            moments[i] = mu2 * mu3 ** 2 + mu2 * sig3 ** 2
        if idx == [0, 1, 3]:
            moments[i] = mu2 * mu3 ** 3 + 3 * mu2 * mu3 * sig3 ** 2
        if idx == [0, 2, 0]:
            moments[i] = mu2 ** 2 + sig2 ** 2
        if idx == [0, 2, 1]:
            moments[i] = mu2 ** 2 * mu3 + mu3 * sig2 ** 2
        if idx == [0, 2, 2]:
            moments[i] = (
                mu2 ** 2 * mu3 ** 2
                + mu3 ** 2 * sig2 ** 2
                + mu2 ** 2 * sig3 ** 2
                + sig2 ** 2 * sig3 ** 2
            )
        if idx == [0, 3, 0]:
            moments[i] = mu2 ** 3 + 3 * mu2 * sig2 ** 2
        if idx == [0, 3, 1]:
            moments[i] = mu2 ** 3 * mu3 + 3 * mu2 * mu3 * sig2 ** 2
        if idx == [0, 4, 0]:
            moments[i] = mu2 ** 4 + 6 * mu2 ** 2 * sig2 ** 2 + 3 * sig2 ** 4
        if idx == [1, 0, 0]:
            moments[i] = mu1
        if idx == [1, 0, 1]:
            moments[i] = mu1 * mu3
        if idx == [1, 0, 2]:
            moments[i] = mu1 * mu3 ** 2 + mu1 * sig3 ** 2
        if idx == [1, 0, 3]:
            moments[i] = mu1 * mu3 ** 3 + 3 * mu1 * mu3 * sig3 ** 2
        if idx == [1, 1, 0]:
            moments[i] = mu1 * mu2
        if idx == [1, 1, 1]:
            moments[i] = mu1 * mu2 * mu3
        if idx == [1, 1, 2]:
            moments[i] = mu1 * mu2 * mu3 ** 2 + mu1 * mu2 * sig3 ** 2
        if idx == [1, 2, 0]:
            moments[i] = mu1 * mu2 ** 2 + mu1 * sig2 ** 2
        if idx == [1, 2, 1]:
            moments[i] = mu1 * mu2 ** 2 * mu3 + mu1 * mu3 * sig2 ** 2
        if idx == [1, 3, 0]:
            moments[i] = mu1 * mu2 ** 3 + 3 * mu1 * mu2 * sig2 ** 2
        if idx == [2, 0, 0]:
            moments[i] = mu1 ** 2 + sig1 ** 2
        if idx == [2, 0, 1]:
            moments[i] = mu1 ** 2 * mu3 + mu3 * sig1 ** 2
        if idx == [2, 0, 2]:
            moments[i] = (
                mu1 ** 2 * mu3 ** 2
                + mu3 ** 2 * sig1 ** 2
                + mu1 ** 2 * sig3 ** 2
                + sig1 ** 2 * sig3 ** 2
            )
        if idx == [2, 1, 0]:
            moments[i] = mu1 ** 2 * mu2 + mu2 * sig1 ** 2
        if idx == [2, 1, 1]:
            moments[i] = mu1 ** 2 * mu2 * mu3 + mu2 * mu3 * sig1 ** 2
        if idx == [2, 2, 0]:
            moments[i] = (
                mu1 ** 2 * mu2 ** 2
                + mu2 ** 2 * sig1 ** 2
                + mu1 ** 2 * sig2 ** 2
                + sig1 ** 2 * sig2 ** 2
            )
        if idx == [3, 0, 0]:
            moments[i] = mu1 ** 3 + 3 * mu1 * sig1 ** 2
        if idx == [3, 0, 1]:
            moments[i] = mu1 ** 3 * mu3 + 3 * mu1 * mu3 * sig1 ** 2
        if idx == [3, 1, 0]:
            moments[i] = mu1 ** 3 * mu2 + 3 * mu1 * mu2 * sig1 ** 2
        if idx == [4, 0, 0]:
            moments[i] = mu1 ** 4 + 6 * mu1 ** 2 * sig1 ** 2 + 3 * sig1 ** 4
        i = i + 1

    return moments
