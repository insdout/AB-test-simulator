import numpy as np
import scipy as sp


def get_sample_size(min_detectable_change, variance_0, variance_1, alpha, beta):
    z_a = sp.stats.norm.ppf(alpha/2)
    z_b = sp.stats.norm.ppf(beta)
    n = int(np.ceil(((z_a*np.sqrt(2*variance_0) + z_b*np.sqrt(variance_0 + variance_1))**2)/min_detectable_change**2))
    return n


def design_binomial_experiment(min_detectable_change, p_0, alpha, beta):
    p_1 = p_0 + min_detectable_change
    variance_0 = p_0*(1 - p_0)
    variance_1 = p_1*(1 - p_1)
    n = get_sample_size(min_detectable_change, variance_0, variance_1, alpha, beta)
    return n


if __name__ == '__main__':
    p1 = 0.2
    min_detectable_change = 0.05
    alpha = 0.05
    beta = 0.2
    print(design_binomial_experiment(min_detectable_change, p1, alpha, beta))
