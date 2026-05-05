import numpy as np

class RFFGaussianProcess1D:
    def __init__(self, n_features=100, lengthscale=1.0, noise_var=1e-2, seed=None):
        """
        Random Fourier Feature GP (Bayesian linear regression view)

        Parameters:
        - n_features: number of frequencies (Ns)
        - lengthscale: l in beta = l^{-2}
        - noise_var: sigma^2
        """
        self.n_features = n_features
        self.lengthscale = lengthscale
        self.noise_var = noise_var

        self.rng = np.random.default_rng(seed)

        # spectral density: w ~ N(0, beta), beta = l^{-2}
        self.beta = 1.0 / (lengthscale ** 2)

        self.w = None
        self.mu = None
        self.Sigma = None
        self.X_train = None
        self.Phi = None

    def _sample_features(self):
        # 1D frequencies
        self.w = self.rng.normal(loc=0.0, scale=np.sqrt(self.beta), size=self.n_features)

    def _phi(self, x):
        """
        Compute RFF feature vector for 1D input x
        shape: (2 * n_features,)
        """
        x = np.atleast_1d(x)

        cos_part = np.cos(np.outer(x, self.w))   # (N, Ns)
        sin_part = np.sin(np.outer(x, self.w))   # (N, Ns)

        # concatenate cos and sin features
        return np.hstack([cos_part, sin_part]) * np.sqrt(1.0 / self.n_features)

    def train(self, X, y):
        """
        Fit Bayesian linear regression in feature space
        """
        X = np.asarray(X).ravel()
        y = np.asarray(y).ravel()

        self.X_train = X

        # sample random features
        self._sample_features()

        # build design matrix Phi
        self.Phi = self._phi(X)   # shape (N, 2Ns)

        # posterior covariance matrix
        self.Sigma = np.linalg.inv(self.Phi.T @ self.Phi + self.noise_var * np.eye(self.Phi.shape[1])) * self.noise_var

        # posterior mean
        self.mu = (1 / self.noise_var) * self.Sigma @ self.Phi.T @ y

    def predict(self, X_test):
        """
        Predict mean and variance
        """
        X_test = np.asarray(X_test).ravel()
        Phi_star = self._phi(X_test)

        # predictive mean
        pred_mean = Phi_star @ self.mu

        # predictive variance
        pred_cov = Phi_star @  self.Sigma @ Phi_star.T + np.eye(len(X_test)) * self.noise_var

        return pred_mean, np.diag(pred_cov)
