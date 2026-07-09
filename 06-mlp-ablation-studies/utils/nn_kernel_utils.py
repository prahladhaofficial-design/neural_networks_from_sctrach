# utils/nn_kernel_utils.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# --------------------------
# t-SNE Plot
# --------------------------
def plot_tsne(features, labels, title="t-SNE plot"):
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(features)
    plt.figure(figsize=(6,6))
    plt.scatter(tsne_results[:,0], tsne_results[:,1], c=labels.flatten(), cmap='viridis', s=15)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()

# --------------------------
# Compute Kernel Matrix
# --------------------------
def compute_kernel_matrix(X, kernel_fn):
    n = X.shape[0]
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i,j] = kernel_fn(X[i], X[j])
    return K

# --------------------------
# Decision Boundary Plot
# --------------------------
def plot_decision_boundary(model, X, y, save_path=None, title="Decision Boundary", step=0.1):
    """
    X: 2D data (N, 2)
    model: trained model with .predict() method
    """
    # Only works for 2D data
    if X.shape[1] != 2:
        print("Decision boundary plot only works for 2D features.")
        return

    x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
    y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = model.predict(grid)
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(6,6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
    plt.scatter(X[:,0], X[:,1], c=y.flatten(), s=15, cmap="viridis")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
