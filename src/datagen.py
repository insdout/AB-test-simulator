import numpy as np
import scipy.stats as stats


class ABTestGenerator:
    def __init__(self, base_ctr: float, uplift: float,
                 beta: float, skew: float):
        """
        Initialize the ABTestGenerator object.

        Args:
            base_ctr (float): The base click-through rate (CTR)
                of the control group.
            uplift (float): The uplift or change in CTR due to the treatment.
            beta (float): The beta parameter of the beta distribution used
                for generating CTR.
            skew (float): The skew parameter of the log-normal distribution
                used for generating views.
        """
        self.base_ctr = base_ctr
        self.uplift = uplift
        self.beta = beta
        self.skew = skew

    def generate_n_experiment(self, num_users: int,
                              n_runs: int) -> dict[np.ndarray]:
        """
        Generate data for A/B testing experiments.

        Args:
            num_users (int): The number of users or samples in each experiment.
            n_runs (int): The number of experiments to run.

        Returns:
            dict: A dictionary containing arrays of CTRs, clicks, and views
                for both control and treatment groups.
        """
        nominator = self.base_ctr * self.beta
        denominator = (1 - self.base_ctr)
        alpha_0 = nominator / denominator

        nominator = (self.base_ctr + self.uplift) * self.beta
        denominator = (1 - self.base_ctr - self.uplift)
        alpha_1 = nominator / denominator

        views_0 = np.exp(
            stats.norm(1, self.skew).rvs((n_runs, num_users))
        ).astype(int) + 1

        views_1 = np.exp(
            stats.norm(1, self.skew).rvs((n_runs, num_users))
        ).astype(int) + 1

        ctrs_0 = stats.beta.rvs(
            a=alpha_0,
            b=self.beta,
            size=(n_runs, num_users)
        ).astype(np.float16)

        ctrs_1 = stats.beta.rvs(
            a=alpha_1,
            b=self.beta,
            size=(n_runs, num_users)
        ).astype(np.float16)

        clicks_0 = stats.binom(n=views_0, p=ctrs_0).rvs()
        clicks_1 = stats.binom(n=views_1, p=ctrs_1).rvs()
        if clicks_0.ndim == 1:
            clicks_0 = np.expand_dims(clicks_0, 0)
            clicks_1 = np.expand_dims(clicks_1, 0)
        return {
            'ctrs_0': ctrs_0,
            'ctrs_1': ctrs_1,
            'clicks_0': clicks_0,
            'clicks_1': clicks_1,
            'views_0': views_0,
            'views_1': views_1
        }
