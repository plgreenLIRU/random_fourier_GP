from matplotlib import pyplot as plt
import numpy as np
from RFFGaussianProcess import RFFGaussianProcess1D

X = np.linspace(0, 10, 100)
X_star = np.linspace(0, 10, 100)
y = np.sin(X) + 0.1 * np.random.randn(len(X))

m = RFFGaussianProcess1D(n_features=10, noise_var=0.01)
m.train(X=X, y=y)
y_star_mean, y_star_var = m.predict(X_star)

fig, ax = plt.subplots()
ax.plot(X, y, 'o')
ax.plot(X_star, y_star_mean, color='red')
ax.plot(X_star, y_star_mean + 3 * np.sqrt(y_star_var), color='red')
ax.plot(X_star, y_star_mean - 3 * np.sqrt(y_star_var), color='red')
plt.show()
