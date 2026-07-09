# nn_kernel_methods.py
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.manifold import TSNE

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from utils.path_setup import *
from models.mlp import MLP
from utils.utils import load_dataset, normalize, train_test_split

# -----------------------------
# Kernel functions
# -----------------------------
def linear_kernel(x_i, x_j):
    return np.dot(x_i, x_j)

def polynomial_kernel(x_i, x_j, c=1, d=2):
    return (np.dot(x_i, x_j) + c) ** d

def rbf_kernel(x_i, x_j, gamma=0.1):
    diff = x_i - x_j
    return np.exp(-gamma * np.dot(diff, diff))

def neural_kernel(x_i, x_j, feature_extractor):
    phi_i = feature_extractor(x_i.reshape(1,-1))
    phi_j = feature_extractor(x_j.reshape(1,-1))
    return np.dot(phi_i.flatten(), phi_j.flatten())

# -----------------------------
# Compute precomputed kernel matrix
# -----------------------------
def compute_precomputed_kernel(X1, X2, kernel_fn):
    n1, n2 = X1.shape[0], X2.shape[0]
    K = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            K[i, j] = kernel_fn(X1[i], X2[j])
    return K

# -----------------------------
# t-SNE plot
# -----------------------------
def plot_tsne(features, labels, title, save_path):
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    plt.figure(figsize=(6,6))
    plt.scatter(features_2d[:,0], features_2d[:,1], c=labels.flatten(), cmap="viridis")
    plt.title(title)
    plt.colorbar()
    plt.savefig(save_path)
    plt.close()

# -----------------------------
# Main experiment
# -----------------------------
if __name__ == "__main__":

    dataset_path = os.path.join(PROJECT_ROOT, "data", "iith_mess_dataset_5000.csv")
    plots_root = os.path.join(PROJECT_ROOT, "plots", "nn_kernel")
    results_root = os.path.join(PROJECT_ROOT, "results")

    os.makedirs(plots_root, exist_ok=True)
    os.makedirs(results_root, exist_ok=True)

    print("Loading dataset...")
    X, y = load_dataset(dataset_path)
    X = normalize(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_ratio=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_ratio=0.5)

    y_train = y_train.reshape(-1,1)
    y_val = y_val.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    # -----------------------------
    # Train MLP
    # -----------------------------
    input_dim = X_train.shape[1]
    architecture = [input_dim, 256,128,128,64, 1]

    mlp = MLP(
        layer_sizes=architecture,
        activations=["relu","relu","relu","relu","identity"],
        loss="huber",
        learning_rate=0.01,
        optimizer="adam",
        batch_size=32,
        weight_init="he"
    )

    print("Training MLP...")
    history = mlp.fit(X_train, y_train, X_val, y_val, epochs=100, use_validation=True)

    # -----------------------------
    # Extract features from penultimate layer
    # -----------------------------
    phi_train = mlp.extract_features(X_train, layer_index=-2)
    phi_val = mlp.extract_features(X_val, layer_index=-2)
    phi_test = mlp.extract_features(X_test, layer_index=-2)

    plot_tsne(phi_train, y_train, "t-SNE of NN Features (Train)", os.path.join(plots_root, "tsne_train.png"))

    # -----------------------------
    # Define kernels
    # -----------------------------
    kernels = {
        "linear": linear_kernel,
        "poly2": lambda x,y: polynomial_kernel(x,y,d=2),
        "poly3": lambda x,y: polynomial_kernel(x,y,d=3),
        "rbf0.01": lambda x,y: rbf_kernel(x,y,gamma=0.01),
        "rbf0.1": lambda x,y: rbf_kernel(x,y,gamma=0.1),
        "rbf1.0": lambda x,y: rbf_kernel(x,y,gamma=1.0),
        "neural": lambda x,y: neural_kernel(x,y, lambda xx: mlp.extract_features(xx, layer_index=-2))
    }

    # -----------------------------
    # Train SVR for each kernel and save results
    # -----------------------------
    results = []

    for name, kernel_fn in kernels.items():
        print(f"\nProcessing kernel: {name}")

        K_train = compute_precomputed_kernel(X_train, X_train, kernel_fn)
        K_test = compute_precomputed_kernel(X_test, X_train, kernel_fn)

        svr_model = SVR(kernel="precomputed")
        svr_model.fit(K_train, y_train.flatten())

        y_pred = svr_model.predict(K_test)

        mse = np.mean((y_test.flatten() - y_pred)**2)
        print(f"{name} kernel MSE on test set: {mse:.4f}")

        results.append({
            "kernel": name,
            "mse": mse
        })

        # Plot Predicted vs True
        plt.figure()
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Predicted vs True ({name} kernel)")
        plt.savefig(os.path.join(plots_root, f"{name}_pred_vs_true.png"))
        plt.close()

    # -----------------------------
    # Save results CSV
    # -----------------------------
    results_df = pd.DataFrame(results)
    results_file = os.path.join(results_root, "nn_kernel_results.csv")
    results_df.to_csv(results_file, index=False)
    print("Results saved to:", results_file)

    # -----------------------------
    # Plot Kernel Comparison
    # -----------------------------
    plt.figure(figsize=(8,5))
    kernels_list = [r["kernel"] for r in results]
    mses = [r["mse"] for r in results]
    plt.bar(kernels_list, mses)
    plt.ylabel("Test MSE")
    plt.xlabel("Kernel")
    plt.title("Kernel Method Comparison")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plot_file = os.path.join(plots_root, "kernel_comparison.png")
    plt.savefig(plot_file)
    plt.close()
    print("Kernel comparison plot saved to:", plot_file)
