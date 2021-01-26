import GPy
import numpy as np
import matplotlib.pyplot as plt

# sample_size = 5
# X = np.random.uniform(0, 1., (sample_size, 1))
# Y = np.sin(X) + np.random.randn(sample_size, 1)*0.05
# 
# kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)
# model = GPy.models.GPRegression(X,Y,kernel, noise_var=1e-10)
# 
# testX = np.linspace(0, 1, 100).reshape(-1, 1)
# posteriorTestY = model.posterior_samples_f(testX, full_cov=True, size=3)
# simY, simMse = model.predict(testX)

# plt.plot()
# plt.plot(testX, posteriorTestY[,,3])
# plt.plot(X, Y, 'ok', markersize=10)
# plt.plot(testX, simY - 3 * simMse ** 0.5, '--g')
# plt.plot(testX, simY + 3 * simMse ** 0.5, '--g')



def gp_sample_pred(testX, model):
  posteriorY = model.posterior_samples_f(testX, full_cov=True, size =1)
  return posteriorY
  # simY, simMse = posteriorY.predict(testX)
  # return simY, simMse
  
