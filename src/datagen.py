import numpy as np
import scipy.stats as stats


class ABTestGenerator:
    def __init__(self, base_ctr: float, uplift: float, beta: float):
        self.base_ctr = base_ctr
        self.uplift = uplift
        self.beta = beta

    def generate_single_experiment(self, num_users: int) -> tuple[np.ndarray]:
        alpha_0 = self.base_ctr * self.beta / (1 - self.base_ctr)
        alpha_1 = (self.base_ctr + self.uplift) * self.beta / (1 - self.base_ctr - self.uplift)

        ctrs_0 = stats.beta.rvs(a=alpha_0, b=self.beta, size=num_users)
        ctrs_1 = stats.beta.rvs(a=alpha_1, b=self.beta, size=num_users)

        clicks_0 = stats.binom(n=1, p=ctrs_0).rvs(size=num_users)
        clicks_1 = stats.binom(n=1, p=ctrs_1).rvs(size=num_users)

        return ctrs_0, ctrs_1, clicks_0, clicks_1

    def generate_n_experiment(
            self,
            num_users: int,
            n_runs: int
    ) -> tuple[list[np.ndarray]]:

        ctrs_0_list, ctrs_1_list, clicks_0_list, clicks_1_list = [], [], [], []

        for _ in range(n_runs):
            ctrs_0, ctrs_1, clicks_0, clicks_1 = self.generate_single_experiment(num_users)
            ctrs_0_list.append(ctrs_0)
            ctrs_1_list.append(ctrs_1)
            clicks_0_list.append(clicks_0)
            clicks_1_list.append(clicks_1)

        return ctrs_0_list, ctrs_1_list, clicks_0_list, clicks_1_list


if __name__ == '__main__':
    # Example usage:
    base_ctr = 0.1
    uplift = 0.02
    beta = 2.0
    num_users = 10000
    n_runs = 5

    ab_test_generator = ABTestGenerator(base_ctr, uplift, beta)
    ctrs_0_list, ctrs_1_list, clicks_0_list, clicks_1_list = ab_test_generator.generate_n_experiment(num_users, n_runs)
    print(np.mean(ctrs_0_list), np.mean(clicks_1_list))
