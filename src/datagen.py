import numpy as np
import scipy.stats as stats


class ABTestGenerator:
    def __init__(self, base_ctr: float, uplift: float, beta: float, skew: float):
        self.base_ctr = base_ctr
        self.uplift = uplift
        self.beta = beta
        self.skew = skew

    def generate_n_experiment(self, num_users: int, n_runs: int) -> dict[np.ndarray]:
        '''
        alpha_0 = success_rate * beta / (1 - success_rate)
        alpha_1 = success_rate * (1 + uplift) * beta / (1 - success_rate * (1 + uplift))
        '''
        alpha_0 = self.base_ctr * self.beta / (1 - self.base_ctr)
        alpha_1 = (self.base_ctr + self.uplift) * self.beta / (1 - self.base_ctr - self.uplift)

        views_0 = np.exp(stats.norm(1, self.skew).rvs((n_runs, num_users))).astype(int) + 1
        views_1 = np.exp(stats.norm(1, self.skew).rvs((n_runs, num_users))).astype(int) + 1

        ctrs_0 = stats.beta.rvs(a=alpha_0, b=self.beta, size=(n_runs, num_users)).astype(np.float16)
        ctrs_1 = stats.beta.rvs(a=alpha_1, b=self.beta, size=(n_runs, num_users)).astype(np.float16)

        clicks_0 = stats.binom(n=views_0, p=ctrs_0).rvs()
        clicks_1 = stats.binom(n=views_1, p=ctrs_1).rvs()
        if clicks_0.ndim == 1:
            clicks_0 = np.expand_dims(clicks_0, 0)
            clicks_1 = np.expand_dims(clicks_1, 0)
        print(views_0.shape, ctrs_0.shape, clicks_0.shape, clicks_0.ndim)
        return {
            'ctrs_0': ctrs_0,
            'ctrs_1': ctrs_1,
            'clicks_0': clicks_0,
            'clicks_1': clicks_1,
            'views_0': views_0,
            'views_1': views_1
        }
