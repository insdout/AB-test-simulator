from collections import defaultdict
import numpy as np


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


def apply_tests(results, test_config):
    test_results = defaultdict(dict)
    for test_name, test_function in test_config.items():
        if test_function:
            test_results[test_name]['p_vals'] = test_function(results)
    return test_results


def empirical_cdf(p_vals):
    p_vals_sorted = sorted(p_vals)
    n = len(p_vals)
    probs = [(i+1)/n for i in range(n)]
    return p_vals_sorted + [1], probs +[1]
