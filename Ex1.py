import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

# 1D Case
mu, sigma = 1, 0.2  # mean and standard deviation
sample = 1000
bins = 30
e_gaussian = np.random.normal(mu, sigma, sample)

fig1 = plt.figure()
# the experiment
ax1 = fig1.add_subplot(121)
count, x, ignored = ax1.hist(e_gaussian, bins, density=True)
g_gaussian = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))

# the ground truth
ax2 = fig1.add_subplot(122)
ax2.plot(x, g_gaussian, color='r')
# show
plt.show()