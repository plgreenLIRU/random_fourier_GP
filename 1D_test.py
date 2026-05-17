from matplotlib import pyplot as plt
import numpy as np
from random_fourier_gp import RFFGP

X = np.linspace(0, 10, 100)
X_star = np.linspace(-1, 11, 100)
y = np.sin(X) + 0.1 * np.random.randn(len(X))

m = RFFGP(input_dim=1, noise_var=0.01)
m.train(X=X, y=y)
y_star_mean, y_star_var = m.predict(X_star)

fig, ax = plt.subplots()
ax.plot(X, y, 'o', color='black')
ax.plot(X_star, y_star_mean, color='red')
ax.plot(X_star, y_star_mean + 3 * np.sqrt(y_star_var), color='red')
ax.plot(X_star, y_star_mean - 3 * np.sqrt(y_star_var), color='red')
plt.show()
