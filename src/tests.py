import numpy as np
import scipy.stats as stats
from src.utils import get_ctrs_hat


def t_test_clicks(results: dict[str, np.ndarray]) -> np.ndarray:
    """
    Perform two-sample T-test for clicks data in A/B test results.

    Args:
        results (dict[str, np.ndarray]): A dictionary containing
            A/B test results.

    Returns:
        np.ndarray: An array containing the p-values of T-test
            for each experiment.
    """
    a = results['clicks_0']
    b = results['clicks_1']
    n_runs = a.shape[0]
    result = np.zeros(n_runs)
    for i in range(n_runs):
        result[i] = stats.ttest_ind(a[i], b[i], alternative='two-sided').pvalue
    return result


def t_test_ctr(results: dict[str, np.ndarray]) -> np.ndarray:
    """
    Perform two-sample T-test for CTR data in A/B test results.

    Args:
        results (dict[str, np.ndarray]): A dictionary containing
            A/B test results.

    Returns:
        np.ndarray: An array containing the p-values of T-test
            for each experiment.
    """
    ctrs_hat = get_ctrs_hat(results)
    a = ctrs_hat['ctrs_0_hat']
    b = ctrs_hat['ctrs_1_hat']
    n_runs = a.shape[0]
    result = np.zeros(n_runs)
    for i in range(n_runs):
        result[i] = stats.ttest_ind(a[i], b[i], alternative='two-sided').pvalue
    return result


def mw_test(results: dict[str, np.ndarray]) -> np.ndarray:
    """
    Perform Mann-Whitney U test for clicks data in A/B test results.

    Args:
        results (dict[str, np.ndarray]): A dictionary containing
            A/B test results.

    Returns:
        np.ndarray: An array containing the p-values of Mann-Whitney U test
            for each experiment.
    """
    a = results['clicks_0']
    b = results['clicks_1']
    n_runs = a.shape[0]
    result = np.zeros(n_runs)
    for i in range(n_runs):
        result[i] = stats.mannwhitneyu(
            a[i],
            b[i],
            alternative='two-sided'
        ).pvalue
    return result


def binom_test(results: dict[str, np.ndarray]) -> np.ndarray:
    """
    Perform binomial test for A/B test results.

    Args:
        results (dict[str, np.ndarray]): A dictionary containing
            A/B test results.

    Returns:
        np.ndarray: An array containing the p-values of binomial test
            for each experiment.
    """
    clicks_0 = results['clicks_0'].sum(axis=1)
    clicks_1 = results['clicks_1'].sum(axis=1)
    n_0 = results['views_0'].sum(axis=1)
    n_1 = results['views_1'].sum(axis=1)
    global_ctr_0 = clicks_0 / n_0
    global_ctr_1 = clicks_1 / n_1

    ctr_H0 = (clicks_0 + clicks_1) / (n_0 + n_1)
    se = np.sqrt(ctr_H0 * (1 - ctr_H0) * (1/n_0 + 1/n_1))
    z_stat = (global_ctr_0 - global_ctr_1) / se
    result = 2 * np.minimum(
        stats.norm(0, 1).cdf(z_stat),
        1 - stats.norm(0, 1).cdf(z_stat)
    )
    return result


def bootstrap_test(results: dict[str, np.ndarray],
                   n_bootstrap: int = 1000) -> np.ndarray:
    """
    Perform bootstrap test for A/B test results.

    Args:
        results (dict[str, np.ndarray]): A dictionary containing
            A/B test results.
        n_bootstrap (int): Number of bootstrap samples. Defaults to 2000.

    Returns:
        np.ndarray: An array containing the p-values of bootstrap test
            for each experiment.
    """
    clicks_0 = results['clicks_0']
    clicks_1 = results['clicks_1']
    views_0 = results['views_0']
    views_1 = results['views_1']
    ctrs_0_hat = clicks_0 / views_0
    ctrs_1_hat = clicks_1 / views_1

    poisson_bootstraps = stats.poisson(1).rvs(
        (n_bootstrap, ctrs_0_hat.shape[1])
        ).astype(int)

    values_0 = np.matmul(ctrs_0_hat * views_0, poisson_bootstraps.T)
    weights_0 = np.matmul(views_0, poisson_bootstraps.T)

    values_1 = np.matmul(ctrs_1_hat * views_1, poisson_bootstraps.T)
    weights_1 = np.matmul(views_1, poisson_bootstraps.T)

    deltas = values_1 / weights_1 - values_0 / weights_0

    positions = np.sum(deltas < 0, axis=1)

    return 2 * np.minimum(positions, n_bootstrap - positions) / n_bootstrap
