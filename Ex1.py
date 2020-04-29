import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

# 1D Case
mu, sigma = 1, 0.2  # mean and standard deviation
sample = 1000
bins = 30
e_gaussian = np.sort(np.random.normal(mu, sigma, sample))

fig1 = plt.figure(figsize=(40, 20))
# the experiment
ax1 = fig1.add_subplot(121)
count, x, ignored = ax1.hist(e_gaussian, bins, density=True)
g_gaussian = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(-(e_gaussian - mu)**2 / (2 * sigma**2))

# the ground truth
ax2 = fig1.add_subplot(122)
ax2.plot(e_gaussian, g_gaussian, color='r')
# show result
plt.show()


# 2D Case
def Gaussian_Distribution(mean, cov, sample):
    # get distribution data of 2D Gaussian
    data = np.random.multivariate_normal(mean, cov, sample).T
    # get PDF of 2D Gaussian
    Gaussian = multivariate_normal(mean=mean, cov=cov)
    return data, Gaussian


mean = [0.5, -0.2]
cov = [[2.0, 0.3], [0.3, 0.5]]
sample = 1000
bins = 30
data, Gaussian = Gaussian_Distribution(mean, cov, sample)
# set 2D space
X, Y = np.meshgrid(np.linspace(-6, 6, sample), np.linspace(-6, 6, sample))
# get 2D data
d = np.dstack([X, Y])
# compute joint Gaussian in 2D
Z = Gaussian.pdf(d).reshape(sample, sample)


# draw the ground truth
fig2 = plt.figure(figsize=(20, 20))
ax3 = Axes3D(fig2)
ax3.plot_surface(X, Y, Z, cmap='Reds_r')


# draw the experiment
fig3 = plt.figure(figsize=(20, 20))
ax4 = fig3.gca(projection='3d')
hist, xedges, yedges = np.histogram2d(data.T[:, 0], data.T[:, 1], bins=30, range=[[-6, 6], [-3, 3]])

# xpos, ypos, zpos: x,y,z coordinates of each bar
# dx, dy, dz: Width, Depth, Height of each bar
xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1])
xpos = xpos.flatten('F')
ypos = ypos.flatten('F')
zpos = np.zeros_like(xpos)
dx = 0.5 * np.ones_like(zpos)
dy = dx.copy()
dz = hist.flatten()

ax4.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
plt.show()
