import numpy as np

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.shape[0]


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mae_derivative(y_true, y_pred):
    return np.sign(y_pred - y_true) / y_true.shape[0]


def huber(y_true, y_pred, delta=1.0):

    error = y_true - y_pred
    abs_error = np.abs(error)

    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic

    return np.mean(0.5 * quadratic**2 + delta * linear)


def huber_derivative(y_true, y_pred, delta=1.0):

    error = y_pred - y_true

    grad = np.where(
        np.abs(error) <= delta,
        error,
        delta * np.sign(error)
    )

    return grad / y_true.shape[0]

def cross_entropy(y_true, y_pred):

    eps = 1e-12
    y_pred = np.clip(y_pred, eps, 1 - eps)

    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def cross_entropy_derivative(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[0]

LOSSES = {
    "mse": (mse, mse_derivative),
    "mae": (mae, mae_derivative),
    "huber": (huber, huber_derivative),
    "cross_entropy": (cross_entropy, cross_entropy_derivative)
}
