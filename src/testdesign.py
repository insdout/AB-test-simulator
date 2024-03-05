import numpy as np
import scipy.stats as sp


def get_sample_size(mde: float,
                    variance_0: float, variance_1: float,
                    alpha: float, beta: float) -> int:
    """
    Calculate the sample size needed for a binomial experiment.

    Args:
        mde (float): Minimum detectable change.
        variance_0 (float): Variance of the control group.
        variance_1 (float): Variance of the treatment group.
        alpha (float): Significance level.
        beta (float): Power of the test.

    Returns:
        int: Sample size needed for the experiment.
    """
    z_a = sp.norm.ppf(alpha/2)
    z_b = sp.norm.ppf(beta)
    n = int(np.ceil(((z_a*np.sqrt(2*variance_0) +
                      z_b*np.sqrt(variance_0 + variance_1))**2) / mde**2))
    return n


def design_binomial_experiment(mde: float, p_0: float, alpha: float,
                               beta: float) -> int:
    """
    Design a binomial experiment and calculate the required sample size.

    Args:
        mde (float): Minimum detectable change.
        p_0 (float): Probability of success in the control group.
        alpha (float): Significance level.
        beta (float): Power of the test.

    Returns:
        int: Sample size needed for the experiment.
    """
    p_1 = p_0 + mde
    variance_0 = p_0 * (1 - p_0)
    variance_1 = p_1 * (1 - p_1)
    n = get_sample_size(mde, variance_0, variance_1, alpha, beta)
    return n
