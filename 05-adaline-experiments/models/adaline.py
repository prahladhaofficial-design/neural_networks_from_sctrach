import numpy as np

class Adaline:

    def __init__(self, learning_rate=1.0, max_iterations=1000):
        """
        learning_rate : gradient descent step size
        max_iterations : number of epochs
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None


    def fit(self, X, y, X_val=None, y_val=None):

        n_samples, n_features = X.shape

        y = y.flatten()

        if y_val is not None:
            y_val = y_val.flatten()  

        self.weights = np.zeros(n_features)
        self.bias = 0

        train_mse_history = []
        val_mse_history = []

        for epoch in range(self.max_iterations):

            y_pred = np.dot(X, self.weights) + self.bias
            error = y - y_pred

            mse = np.mean(error**2)
            train_mse_history.append(mse)

            # gradient descent update
            self.weights += self.learning_rate * np.dot(X.T, error) / n_samples
            self.bias += self.learning_rate * np.mean(error)

            # validation error
            if X_val is not None:

                val_pred = np.dot(X_val, self.weights) + self.bias
                val_error = y_val - val_pred     
                val_mse = np.mean(val_error**2)

                val_mse_history.append(val_mse)

        if X_val is not None:
            return train_mse_history, val_mse_history

        return train_mse_history


    def predict(self, X):

        return np.dot(X, self.weights) + self.bias


    def score(self, X, y):

        y = y.flatten()

        y_pred = self.predict(X)

        return np.mean((y - y_pred)**2)
