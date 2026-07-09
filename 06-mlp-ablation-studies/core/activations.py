import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s*(1-s)


def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2


def relu(x):
    return np.maximum(0,x)

def relu_derivative(x):
    return (x>0).astype(float)


def leaky_relu(x, alpha=0.01):
    return np.where(x>0,x,alpha*x)

def leaky_relu_derivative(x, alpha=0.01):
    dx = np.ones_like(x)
    dx[x<0] = alpha
    return dx

def identity(x):
    return x

def identity_derivative(x):
    return np.ones_like(x)

ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    "relu": (relu, relu_derivative),
    "leaky_relu": (leaky_relu, leaky_relu_derivative),
    "identity": (identity, identity_derivative)
}
