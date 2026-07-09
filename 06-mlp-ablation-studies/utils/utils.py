import pandas as pd
import numpy as np


def load_dataset(path):

    df = pd.read_csv(path)

    X = df.drop(columns=["mess_duration"]).values
    y = df["mess_duration"].values

    return X, y


def normalize(X):

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)

    return (X - mean) / std


def train_test_split(X, y, test_ratio=0.2):

    n = X.shape[0]

    indices = np.random.permutation(n)

    split = int(n * (1 - test_ratio))

    train_idx = indices[:split]
    test_idx = indices[split:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def pca_2d(X):

    X_centered = X - np.mean(X, axis=0)

    cov = np.cov(X_centered.T)

    eig_vals, eig_vecs = np.linalg.eig(cov)

    idx = np.argsort(eig_vals)[::-1]

    eig_vecs = eig_vecs[:, idx]

    W = eig_vecs[:, :2]

    return X_centered @ W
