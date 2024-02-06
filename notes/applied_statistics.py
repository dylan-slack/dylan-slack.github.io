"""
Applied statistics study
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

# Seed for reproducibility
np.random.seed(1)

# Create data from multiple normal distributions
data1 = np.random.normal(-5, 0.5, 100_000)  # First hump
data2 = np.random.normal(0, 2, 100_000)   # Second hump
data3 = np.random.normal(5, 1.5, 100_000) # Third hump

# Combine the data into a single dataset
combined_data = np.concatenate([data1, data2, data3])

# Plotting the result
plt.hist(combined_data, bins=30, alpha=0.7)

true_mean = np.mean(combined_data)

n_samples = 100
attempts = 1_000

n_in = 0

for _ in range(attempts):
    sample = np.random.choice(combined_data, size=n_samples, replace=True)
    ci = 1.96 * stats.sem(sample)
    in_interval = (np.mean(sample) - ci) <= true_mean <= (np.mean(sample) + ci)
    n_in += int(in_interval)

print(f"Pct In {n_in / attempts}")

# p value

from scipy.stats import ttest_ind

n_sig = 0
thresh = 0.05
attempts = 10_000
n_samples = 1_000

for _ in range(attempts):
    s = np.random.choice(combined_data, size=n_samples)
    s2 = np.random.choice(combined_data, size=n_samples)
    r = ttest_ind(s, s2).pvalue

    if r < thresh:
        n_sig += 1

print(f"Pct in {n_sig / attempts}")



