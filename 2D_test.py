import numpy as np
from random_fourier_gp import RFFGP
from matplotlib import pyplot as plt

np.random.seed(0)

# ============================================================
# Generate training data
# ============================================================

N = 200

x1 = np.random.uniform(-4, 4, N)
x2 = np.random.uniform(-4, 4, N)

X_train = np.column_stack([x1, x2])

# target function
y_train = np.sin(x1) * np.cos(x2)

# noisy observations
y_train += 0.1 * np.random.randn(N)

# ============================================================
# Train model
# ============================================================

model = RFFGP(
    input_dim=2,
    n_features=1000,
    lengthscale=1.0,
    noise_var=1e-2
)

model.train(X_train, y_train)

# ============================================================
# Prediction grid
# ============================================================

grid_size = 80

x1_test = np.linspace(-4, 4, grid_size)
x2_test = np.linspace(-4, 4, grid_size)

X1, X2 = np.meshgrid(x1_test, x2_test)

X_test = np.column_stack([
    X1.ravel(),
    X2.ravel()
])

mean, var = model.predict(X_test)

Z = mean.reshape(grid_size, grid_size)

# ============================================================
# Surface plot
# ============================================================

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X1, X2, Z, alpha=0.8)

ax.scatter(
    X_train[:, 0],
    X_train[:, 1],
    y_train,
    s=10
)

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("f(x1,x2)")

ax.set_title("RFF Gaussian Process Surface")

plt.show()

# ============================================================
# Predicted vs true with 3-sigma error bars
# ============================================================

N_test = 100

x1_true = np.random.uniform(-4, 4, N_test)
x2_true = np.random.uniform(-4, 4, N_test)

X_true = np.column_stack([x1_true, x2_true])

# noiseless ground truth
y_true = np.sin(x1_true) * np.cos(x2_true)

# model predictions
y_pred, y_var = model.predict(X_true)

y_std = np.sqrt(y_var)

# ============================================================
# Plot predicted vs true
# ============================================================

plt.figure(figsize=(7, 7))

plt.errorbar(
    y_true,
    y_pred,
    yerr=3 * y_std,
    fmt='o',
    alpha=0.7,
    capsize=3
)

# ideal prediction line
lims = [
    min(y_true.min(), y_pred.min()),
    max(y_true.max(), y_pred.max())
]

plt.plot(lims, lims, 'k--', linewidth=2)

plt.xlabel("True value")
plt.ylabel("Predicted mean")

plt.title("Predicted vs True with 3σ Error Bars")

plt.grid(True)

plt.show()
