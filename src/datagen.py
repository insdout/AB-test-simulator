import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st


class ABTestGenerator:
    def __init__(self, base_ctr: float, uplift: float, beta: float, skew: float, traffic_per_day: int):
        self.base_ctr = base_ctr
        self.uplift = uplift
        self.beta = beta
        self.skew = skew
        self.traffic_per_day = traffic_per_day

    def generate_n_experiment(self, num_users: int, n_runs: int) -> dict[np.ndarray]:
        alpha_0 = self.base_ctr * self.beta / (1 - self.base_ctr)
        alpha_1 = (self.base_ctr + self.uplift) * self.beta / (1 - self.base_ctr - self.uplift)

        views_0 = np.exp(stats.norm(1, self.skew).rvs((n_runs, num_users))).astype(np.int64) + 1
        views_1 = np.exp(stats.norm(1, self.skew).rvs((n_runs, num_users))).astype(np.int64) + 1

        # Generate random dates within a date range
        days_needed = int(np.ceil(num_users / self.traffic_per_day))
        start_date = pd.to_datetime('2024-01-01')
        end_date = start_date + pd.Timedelta(days=days_needed)
        days = np.random.choice(pd.date_range(start=start_date, end=end_date), size=num_users, replace=True)

        ctrs_0 = stats.beta.rvs(a=alpha_0, b=self.beta, size=(n_runs, num_users))
        ctrs_1 = stats.beta.rvs(a=alpha_1, b=self.beta, size=(n_runs, num_users))

        clicks_0 = stats.binom(n=views_0.flatten(), p=ctrs_0.flatten()).rvs().reshape(n_runs, num_users)
        clicks_1 = stats.binom(n=views_1.flatten(), p=ctrs_1.flatten()).rvs().reshape(n_runs, num_users)

        return {
            'days': days,
            'ctrs_0': ctrs_0,
            'ctrs_1': ctrs_1,
            'clicks_0': clicks_0,
            'clicks_1': clicks_1,
            'views_0': views_0,
            'views_1': views_1
        }


def plot_mean_ctr_per_day(days_list, ctrs_0, ctrs_1):

    ctr_df = pd.DataFrame({'Date': days_list, 'CTR_0': ctrs_0, 'CTR_1': ctrs_1})
    print(ctr_df.head())
    mean_ctr_per_day = ctr_df.groupby('Date').mean()
    print(mean_ctr_per_day.head())
    plt.plot(mean_ctr_per_day.index, mean_ctr_per_day['CTR_0'], label='CTR_0')
    plt.plot(mean_ctr_per_day.index, mean_ctr_per_day['CTR_1'], label='CTR_1')
    plt.xlabel('Date')
    plt.ylabel('Mean CTR')
    plt.title('Mean CTR per Day')
    # Format x-axis labels to show only day and month
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    # Rotate x-axis labels by 45 degrees
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    base_ctr = 0.02
    uplift = 0.01
    beta = 250
    skew = 0.4
    traffic_per_day = 100
    num_users = 6000
    n_runs = 1000

    ab_test_generator = ABTestGenerator(base_ctr, uplift, beta, skew, traffic_per_day)
    experiments = ab_test_generator.generate_n_experiment(num_users, n_runs)

    # Accessing the data from the experiments dictionary
    for i in range(n_runs):
        break
        days, ctrs_0, ctrs_1, clicks_0, clicks_1, views_0, views_1 = experiments['days'], experiments['ctrs_0'][i], experiments['ctrs_1'][i], experiments['clicks_0'][i], experiments['clicks_1'][i], experiments['views_0'][i], experiments['views_1'][i]
        print(np.mean(ctrs_0), np.mean(ctrs_1), len(ctrs_0), sum(views_0), sum(views_1))
        print(np.sum(clicks_0)/np.sum(views_0), np.sum(clicks_1)/np.sum(views_1))
        print(ctrs_0.shape)
        plot_mean_ctr_per_day(days, ctrs_0, ctrs_1)
