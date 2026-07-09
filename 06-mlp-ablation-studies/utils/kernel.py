import numpy as np

def linear_kernel(xi, xj):
    return np.dot(xi, xj)

def poly_kernel(xi, xj, c=1, d=2):
    return (np.dot(xi, xj) + c)**d

def rbf_kernel(xi, xj, gamma=0.1):
    return np.exp(-gamma * np.linalg.norm(xi-xj)**2)

def neural_kernel(xi, xj, feature_extractor):
    phi_i = feature_extractor(xi.reshape(1,-1))
    phi_j = feature_extractor(xj.reshape(1,-1))
    return np.dot(phi_i.flatten(), phi_j.flatten())
