import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from utils.path_setup import *
from utils.utils import load_dataset, normalize, train_test_split, pca_2d

from models.mlp import MLP
from models.adaline import Adaline


# -------------------------------------------------
# Experiment
# -------------------------------------------------

def run_success_analysis(dataset_path, plots_root, results_root):

    print("Loading dataset...")

    X, y = load_dataset(dataset_path)

    X = normalize(X)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_ratio=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_ratio=0.5)

    y_train = y_train.reshape(-1,1)
    y_val = y_val.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    input_dim = X_train.shape[1]

    architecture = [input_dim, 64, 64, 1]

    plot_dir = os.path.join(plots_root, "success_analysis")
    os.makedirs(plot_dir, exist_ok=True)

    # -------------------------------------------------
    # Train MLP
    # -------------------------------------------------

    print("Training MLP...")

    mlp = MLP(
        layer_sizes=architecture,
        activations=["relu","relu","identity"],
        loss="mse",
        learning_rate=0.001,
        optimizer="adam",
        batch_size=32,
        weight_init="he"
    )

    mlp.fit(
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=100,
        use_validation=True
    )

    # -------------------------------------------------
    # Extract Hidden Layer Features
    # -------------------------------------------------

    print("Extracting hidden features...")

    hidden_features = mlp.extract_features(X_test, layer_index=2)

    features_2d = pca_2d(hidden_features)

    # -------------------------------------------------
    # Plot Feature Visualization
    # -------------------------------------------------

    plt.figure()

    plt.scatter(features_2d[:,0], features_2d[:,1], c=y_test)

    plt.xlabel("PC1")
    plt.ylabel("PC2")

    plt.title("MLP Learned Feature Representation (PCA)")

    plt.savefig(os.path.join(plot_dir, "mlp_feature_visualization.png"))

    plt.close()

    # -------------------------------------------------
    # Train Adaline
    # -------------------------------------------------

    print("Training Adaline...")

    adaline = Adaline(learning_rate=0.01, max_iterations=200)

    adaline.fit(X_train, y_train)

    # -------------------------------------------------
    # Evaluate models
    # -------------------------------------------------

    mlp_test_loss = mlp.evaluate(X_test, y_test)

    adaline_test_loss = adaline.score(X_test, y_test)

    mlp_params = mlp.count_parameters()

    # -------------------------------------------------
    # Comparison Table
    # -------------------------------------------------

    comparison = pd.DataFrame({

        "Model": ["Adaline", "MLP"],

        "Test_MSE": [adaline_test_loss, mlp_test_loss],

        "Number_of_Parameters": [X_train.shape[1] + 1, mlp_params]

    })

    results_file = os.path.join(results_root, "model_comparison.csv")

    comparison.to_csv(results_file, index=False)

    print("\nSuccess Analysis completed")

    print("Feature plot saved to:", plot_dir)

    print("Comparison table saved to:", results_file)


# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":

    dataset_path = os.path.join(PROJECT_ROOT, "data", "iith_mess_dataset_5000.csv")

    plots_root = os.path.join(PROJECT_ROOT, "plots")

    results_root = os.path.join(PROJECT_ROOT, "results")

    run_success_analysis(dataset_path, plots_root, results_root)
