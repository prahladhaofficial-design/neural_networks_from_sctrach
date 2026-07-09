import numpy as np

class KernelSVR:
    """
    Kernel SVR implementation from scratch
    """
    def __init__(self, kernel_fn, C=1.0, epsilon=0.1):
        self.kernel_fn = kernel_fn
        self.C = C
        self.epsilon = epsilon
        self.alpha = None
        self.alpha_star = None
        self.X_train = None
        self.y_train = None
        self.K = None

    def fit(self, X, y):
        n = X.shape[0]
        self.X_train = X
        self.y_train = y
        self.K = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                self.K[i,j] = self.kernel_fn(X[i], X[j])

        # Solve using simplified dual formulation (gradient descent)
        self.alpha = np.zeros(n)
        self.alpha_star = np.zeros(n)
        lr = 0.001
        for _ in range(500):
            y_pred = self.predict(X)
            error = y_pred - y
            self.alpha -= lr * error
            self.alpha_star -= lr * (-error)
            # clip alpha to [0, C]
            self.alpha = np.clip(self.alpha, 0, self.C)
            self.alpha_star = np.clip(self.alpha_star, 0, self.C)

    def predict(self, X):
        n_train = self.X_train.shape[0]
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            s = 0
            for j in range(n_train):
                s += (self.alpha[j] - self.alpha_star[j]) * self.kernel_fn(self.X_train[j], X[i])
            y_pred[i] = s
        return y_pred
