import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import empirical_cdf
import streamlit as st


def plot_ctr(results: dict[str, np.ndarray],
             i: int, figsize: tuple[int, int] = (4, 3)) -> None:
    """
    Plot the ground truth user click-through rate (CTR) distribution.

    Args:
        results (dict[str, np.ndarray]): dictionary containing arrays of CTRs
            for both control and treatment groups.
        i (int): Index of the experiment to plot.
        figsize (tuple[int, int], optional): Figure size. Defaults to (4, 3).
    """
    ctrs_0 = results['ctrs_0'][i]
    ctrs_1 = results['ctrs_1'][i]

    sns.set_theme(style="darkgrid")
    sns.set_palette('rocket')
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot([ctrs_0, ctrs_1], kde=False, stat="probability",
                 common_norm=False, multiple="layer", bins=50, ax=ax)
    ax.set_title('Ground truth user CTR distribution')
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_views(results: dict[str, np.ndarray], i: int,
               figsize: tuple[int, int] = (4, 3)) -> None:
    """
    Plot the ground truth user views distribution.

    Args:
        results (dict[str, np.ndarray]): dictionary containing arrays of views
            for both control and treatment groups.
        i (int): Index of the experiment to plot.
        figsize (tuple[int, int], optional): Figure size. Defaults to (4, 3).
    """
    views_0 = results['views_0'][i]
    views_1 = results['views_1'][i]

    sns.set_theme(style="darkgrid")
    sns.set_palette('rocket')
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot([views_0, views_1], kde=False, stat="probability",
                 common_norm=False, multiple="layer", bins=range(0, 30), ax=ax)
    ax.set_title('Ground truth user views distribution')
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_p_hist(p_vals: np.ndarray, figsize: tuple[int, int] = (5, 4),
                fontsize: int = 10) -> None:
    """
    Plot the distribution of p-values.

    Args:
        p_vals (np.ndarray): Array of p-values.
        figsize (tuple[int, int], optional): Figure size. Defaults to (5, 4).
        fontsize (int, optional): Font size. Defaults to 10.
    """
    sns.set_theme(style="darkgrid")
    sns.set_palette('rocket')
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(p_vals, bins=20, density=True)
    ax.set_title('p-values distribution', fontsize=fontsize)
    ax.set_ylabel('Probability', fontsize=fontsize)
    ax.set_xlabel('p-value', fontsize=fontsize)
    ax.set_xlim(right=1)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_p_hist_all(results_pvals: dict[str, dict[str, np.ndarray]],
                    figsize: tuple[int, int] = (5, 4),
                    fontsize: int = 10, hist_alpha: float = 0.5) -> None:
    """
    Plot the distribution of p-values for multiple tests.

    Args:
        results_pvals (dict[str, dict[str, np.ndarray]]):
            dictionary containing arrays of p-values for multiple tests.
        figsize (tuple[int, int], optional): Figure size. Defaults to (5, 4).
        fontsize (int, optional): Font size. Defaults to 10.
        hist_alpha (float, optional): Transparency of histogram bars.
            Defaults to 0.5.
    """
    sns.set_theme(style="darkgrid")
    sns.set_palette('rocket')
    fig, ax = plt.subplots(figsize=figsize)
    for test_name in results_pvals.keys():
        ax.hist(results_pvals[test_name]['p_vals'], bins=20,
                density=True, label=test_name, alpha=hist_alpha)
    ax.set_title('p-values distribution', fontsize=fontsize)
    ax.set_ylabel('Probability', fontsize=fontsize)
    ax.set_xlabel('p-value', fontsize=fontsize)
    ax.set_xlim(right=1)
    ax.legend(loc='lower right')
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def plot_p_cdf(p_vals: np.ndarray, alpha: float = 0.05,
               figsize: tuple[int, int] = (5, 4),
               fontsize: int = 10, label_fontsize: int = 10) -> None:
    """
    Plot the empirical cumulative distribution function (CDF) of p-values.

    Args:
        p_vals (np.ndarray): Array of p-values.
        alpha (float, optional): Threshold for statistical significance.
            Defaults to 0.05.
        figsize (tuple[int, int], optional): Figure size. Defaults to (5, 4).
        fontsize (int, optional): Font size. Defaults to 10.
        label_fontsize (int, optional): Font size for labels. Defaults to 10.
    """
    sns.set_theme(style="darkgrid")
    sns.set_palette('rocket')

    plt.figure(figsize=figsize)
    p_vals_sorted, probs = empirical_cdf(p_vals)

    plt.plot(p_vals_sorted, probs, lw=3)
    plt.plot([0, 1], [0, 1], color='gray', lw=1)
    plt.plot([alpha, alpha], [0, 1], color='gray', lw=1)
    plt.xlim(right=1)
    plt.ylim(bottom=0)

    plt.ylabel('Probability', fontsize=label_fontsize)
    plt.xlabel('p-value', fontsize=label_fontsize)
    plt.title('Empirical CDF', fontsize=label_fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    st.pyplot(plt.gcf(), use_container_width=True)


def plot_p_cdf_all(p_vals_dict, alpha=0.05, figsize=(5, 4),
                   fontsize=10, label_fontsize=10):
    sns.set_theme(style="darkgrid")
    sns.set_palette('rocket')

    plt.figure(figsize=figsize)

    for test_name in p_vals_dict.keys():
        p_vals_sorted, probs = empirical_cdf(p_vals_dict[test_name]['p_vals'])
        plt.plot(p_vals_sorted, probs, label=test_name, lw=3)
    plt.plot([0, 1], [0, 1], color='gray', lw=1)
    plt.plot([alpha, alpha], [0, 1], color='gray', lw=1)
    plt.xlim(right=1)
    plt.ylim(bottom=0)

    plt.ylabel('Probability', fontsize=label_fontsize)
    plt.xlabel('p-value', fontsize=label_fontsize)
    plt.title('Empirical CDF', fontsize=label_fontsize)
    plt.legend()

    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    st.pyplot(plt.gcf(), use_container_width=True)


def plot_power(tests_results, alpha=0.05, figsize=(6, 2), fontsize=10,
               label_fontsize=10):
    sns.set_theme(style="darkgrid")
    sns.set_palette('rocket')
    powers = dict()
    for test_name in tests_results.keys():
        powers[test_name] = np.mean(tests_results[test_name]['p_vals'] < alpha)

    plt.figure(figsize=figsize)
    plt.barh(list(powers.keys()), list(powers.values()),
             color=sns.color_palette(palette='rocket',
                                     n_colors=len(list(powers.keys()))+2))
    plt.xlim(left=0, right=1)

    plt.xlabel('Power', fontsize=label_fontsize)
    plt.ylabel('Test Name', fontsize=label_fontsize)
    plt.title('Statistical Power of Tests', fontsize=label_fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.tight_layout()
    st.pyplot(plt.gcf(), use_container_width=True)
