from collections import defaultdict
import numpy as np
from tests import t_test, mw_test


def get_ctrs_hat(results):
    n_runs = results['clicks_0'].shape[0]
    ctrs_0 = []
    ctrs_1 = []
    for i in range(n_runs):
        ctrs_0.append(results['clicks_0'][i]/results['views_0'][i])
        ctrs_1.append(results['clicks_1'][i]/results['views_1'][i])
    return {
        'ctrs_0_hat': np.array(ctrs_0),
        'ctrs_1_hat': np.array(ctrs_1)
        }


def apply_tests(results, test_config={'t_test': t_test, 'mw_test': mw_test}):
    ctrs_hat = get_ctrs_hat(results)
    ctrs_a = ctrs_hat['ctrs_0_hat']
    ctrs_b = ctrs_hat['ctrs_1_hat']
    n_runs = ctrs_a.shape[0]
    results = defaultdict(dict)
    for test_name, test_function in test_config.items():
        if test_function:
            results[test_name]['p_vals'] = np.zeros(n_runs)
            for i in range(n_runs):
                results[test_name]['p_vals'][i] = test_function(
                    ctrs_a[i],
                    ctrs_b[i]
                )
    return results


def empirical_cdf(p_vals):
    p_vals_sorted = sorted(p_vals)
    n = len(p_vals)
    probs = [(i+1)/n for i in range(n)]
    print(probs[-1])
    return p_vals_sorted, probs
