from collections import defaultdict
import numpy as np


def get_ctrs_hat(results: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Calculate estimated CTRs for control and treatment groups.

    Args:
        results (Dict[str, np.ndarray]): A dictionary containing
            A/B test results.

    Returns:
        dict[str, np.ndarray]: A dictionary containing estimated CTRs for
            control and treatment groups.
    """
    n_runs = results['clicks_0'].shape[0]
    ctrs_0 = []
    ctrs_1 = []
    for i in range(n_runs):
        ctrs_0.append(results['clicks_0'][i] / results['views_0'][i])
        ctrs_1.append(results['clicks_1'][i] / results['views_1'][i])
    return {
        'ctrs_0_hat': np.array(ctrs_0),
        'ctrs_1_hat': np.array(ctrs_1)
    }


def apply_tests(
        results: dict[str, np.ndarray],
        test_config: dict[str, callable]
        ) -> dict[str, dict[str, np.ndarray]]:
    """
    Apply statistical tests to A/B test results.

    Args:
        results (Dict[str, np.ndarray]): A dictionary containing
            A/B test results.
        test_config (Dict[str, callable]): A dictionary containing test names
            as keys and corresponding test functions as values.

    Returns:
        dict[str, dict[str, np.ndarray]]: A dictionary containing test results
            for each test.
    """
    test_results = defaultdict(dict)
    for test_name, test_function in test_config.items():
        if test_function:
            test_results[test_name]['p_vals'] = test_function(results)
    return test_results


def empirical_cdf(p_vals: list[float]) -> tuple[list[float], list[float]]:
    """
    Calculate empirical cumulative distribution function (CDF) of p-values.

    Args:
        p_vals (list[float]): A list of p-values.

    Returns:
        tuple[list[float], list[float]]: A tuple containing sorted p-values
            and corresponding probabilities.
    """
    p_vals_sorted = sorted(p_vals)
    n = len(p_vals)
    probs = [(i + 1) / n for i in range(n)]
    return p_vals_sorted + [1], probs + [1]
