import numpy as np

class RFFGP:
    def __init__(self, input_dim, n_features=100, lengthscale=1.0, noise_var=1e-2, seed=None):
        """
        Random Fourier Feature GP (Bayesian linear regression view)

        Parameters:
        - n_features: number of frequencies (Ns)
        - lengthscale: l in beta = l^{-2}
        - noise_var: sigma^2
        """
        self.input_dim = input_dim
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
        self.w = np.random.normal(0, np.sqrt(self.beta), size=(self.n_features, self.input_dim))

    def _phi(self, X):
        """
        Compute RFF feature matrix for multidimensional inputs.

        Parameters
        ----------
        X : array, shape (N, D)

        Returns
        -------
        Phi : array, shape (N, 2 * n_features)
        """
        X = np.atleast_2d(X)   # (N, D)

        # projections: (N, Ns)
        proj = X @ self.w.T

        cos_part = np.cos(proj)
        sin_part = np.sin(proj)

        return np.hstack([cos_part, sin_part]) * np.sqrt(1.0 / self.n_features)

    def train(self, X, y):
        """
        Fit Bayesian linear regression in feature space
        """
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X[:, None]
        self.X_train = X
        
        y = np.asarray(y).ravel()

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
        Predict posterior mean and variance

        Parameters
        ----------
        X_test : array, shape (N, D)

        Returns
        -------
        pred_mean : shape (N,)
        pred_var : shape (N,)
        """
        X_test = np.asarray(X_test)

        # ensure shape (N, D)
        if X_test.ndim == 1:
            X_test = X_test[:, None]

        Phi_star = self._phi(X_test)   # (N, 2Ns)

        # predictive mean
        pred_mean = Phi_star @ self.mu

        # predictive variance
        #
        # diag(Phi Sigma Phi^T)
        #
        pred_var = np.sum(
            (Phi_star @ self.Sigma) * Phi_star,
            axis=1
        ) + self.noise_var

        return pred_mean, pred_var
