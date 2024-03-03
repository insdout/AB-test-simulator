import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import empirical_cdf
import streamlit as st


def plot_ctr(results, i, figsize=(4, 3)):
    ctrs_0 = results['ctrs_0'][i]
    ctrs_1 = results['ctrs_1'][i]

    sns.set_theme(style="darkgrid")
    sns.set_palette('rocket')
    fig, ax = plt.subplots(figsize=figsize)  # Create one figure
    sns.histplot([ctrs_0, ctrs_1], kde=False, stat="probability",
                 common_norm=False, multiple="layer", bins=50, ax=ax)
    ax.set_title('Ground truth user CTR distribution')  # Set title on the axes
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)  # Close the figure after use to avoid memory leakage


def plot_views(results, i, figsize=(4, 3)):
    views_0 = results['views_0'][i]
    views_1 = results['views_1'][i]

    sns.set_theme(style="darkgrid")
    sns.set_palette('rocket')
    fig, ax = plt.subplots(figsize=figsize)  # Create one figure
    sns.histplot([views_0, views_1], kde=False, stat="probability",
                 common_norm=False, multiple="layer", bins=range(0, 30), ax=ax)
    ax.set_title('Ground truth user views distribution')  # Set title on the axes
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)  # Close the figure after use to avoid memory leakage


def plot_p_hist(p_vals, figsize=(5, 4), fontsize=10):
    sns.set_theme(style="darkgrid")
    sns.set_palette('rocket')
    fig, ax = plt.subplots(figsize=figsize)  # Create one figure
    ax.hist(p_vals, bins=20, density=True)
    ax.set_title('p-values distribution', fontsize=fontsize)  # Set title on the axes
    ax.set_ylabel('Probability', fontsize=fontsize)  # Set ylabel on the axes
    ax.set_xlabel('p-value', fontsize=fontsize)  # Set xlabel on the axes
    ax.set_xlim(right=1)  # Set xlim on the axes
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)  # Close the figure after use to avoid memory leakage


def plot_p_hist_all(results_pvals, figsize=(5, 4), fontsize=10, hist_alpha=0.5):
    sns.set_theme(style="darkgrid")
    sns.set_palette('rocket')
    fig, ax = plt.subplots(figsize=figsize)  # Create one figure
    for test_name in results_pvals.keys():
        ax.hist(results_pvals[test_name]['p_vals'], bins=20, density=True, label=test_name, alpha=hist_alpha)  # Fixed the indexing and label typo
    ax.set_title('p-values distribution', fontsize=fontsize)  # Set title on the axes
    ax.set_ylabel('Probability', fontsize=fontsize)  # Set ylabel on the axes
    ax.set_xlabel('p-value', fontsize=fontsize)  # Set xlabel on the axes
    ax.set_xlim(right=1)  # Set xlim on the axes
    ax.legend(loc='lower right')  # Add legend
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)  # Close the figure after use to avoid memory leakage


def plot_p_cdf(p_vals, alpha=0.05, figsize=(5, 4),
               fontsize=10, label_fontsize=10):
    sns.set_theme(style="darkgrid")
    sns.set_palette('rocket')

    plt.figure(figsize=figsize)
    p_vals_sorted, probs = empirical_cdf(p_vals)
    # plt.plot(p_vals_sorted, probs)
    plt.plot(p_vals_sorted, probs, lw=3)
    plt.plot([0, 1], [0, 1], color='gray', lw=1)
    plt.plot([alpha, alpha], [0, 1], color='gray', lw=1)
    plt.xlim(right=1)
    plt.ylim(bottom=0)
    # Set smaller font size for the title
    plt.ylabel('Probability', fontsize=label_fontsize)
    plt.xlabel('p-value', fontsize=label_fontsize)
    plt.title('Empirical CDF', fontsize=label_fontsize)
    # Set smaller font size for ticks
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
    # plt.plot(p_vals_sorted, probs)
    plt.plot([0, 1], [0, 1], color='gray', lw=1)
    plt.plot([alpha, alpha], [0, 1], color='gray', lw=1)
    plt.xlim(right=1)
    plt.ylim(bottom=0)
    # Set smaller font size for the title
    plt.ylabel('Probability', fontsize=label_fontsize)
    plt.xlabel('p-value', fontsize=label_fontsize)
    plt.title('Empirical CDF', fontsize=label_fontsize)
    plt.legend()
    # Set smaller font size for ticks
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    st.pyplot(plt.gcf(), use_container_width=True)


def plot_power(tests_results, alpha=0.05, figsize=(6, 2), fontsize=10, label_fontsize=10):
    sns.set_theme(style="darkgrid")
    sns.set_palette('rocket')
    powers = dict()
    for test_name in tests_results.keys():
        powers[test_name] = np.mean(tests_results[test_name]['p_vals'] < alpha)

    plt.figure(figsize=figsize)
    plt.barh(list(powers.keys()), list(powers.values()),
             color=sns.color_palette(palette='rocket', n_colors=len(list(powers.keys()))+2))
    plt.xlim(left=0, right=1)

    plt.xlabel('Power', fontsize=label_fontsize)
    plt.ylabel('Test Name', fontsize=label_fontsize)
    plt.title('Statistical Power of Tests', fontsize=label_fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.tight_layout()  # Adjust layout for better spacing
    st.pyplot(plt.gcf(), use_container_width=True)
