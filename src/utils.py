import numpy as np
import matplotlib.pyplot as plt


class EmpiricalCDF:
    def __init__(self, data):
        self.sorted_data, self.probabilities = self._calculate_cdf(data)

    def _calculate_cdf(self, data):
        sorted_data = np.sort(data)
        n = len(sorted_data)
        probabilities = np.arange(1, n + 1) / n
        return sorted_data, probabilities

    def calculate_probability(self, x):
        idx = np.searchsorted(self.sorted_data, x, side='right')
        if idx == 0:
            return 0.0
        elif idx == len(self.sorted_data):
            return 1.0
        else:
            x0 = self.sorted_data[idx - 1]
            x1 = self.sorted_data[idx]
            p0 = self.probabilities[idx - 1]
            p1 = self.probabilities[idx]
            return p0 + (p1 - p0) * (x - x0) / (x1 - x0)

    def plot_cdf(self):
        plt.plot(self.sorted_data, self.probabilities, marker='.', linestyle='none')
        plt.xlabel('Data')
        plt.ylabel('Cumulative Probability')
        plt.title('Empirical CDF')
        plt.grid(True)
        plt.show()

# Example usage:
data = np.random.normal(loc=0, scale=1, size=1000)  # Generate example data (normally distributed)
empirical_cdf = EmpiricalCDF(data)

# Calculate the probability of a specific value
x = 0.0
prob_x = empirical_cdf.calculate_probability(x)
print(f"The probability of x={x} according to the empirical CDF is: {prob_x:.4f}")

# Plot the empirical CDF
empirical_cdf.plot_cdf()


