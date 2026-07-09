import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)
from utils.path_setup import *
from models.mlp import MLP
from utils.utils import load_dataset, normalize, train_test_split


# -------------------------------------------------
# Define architectures for depth experiment
# -------------------------------------------------

def get_architectures(input_dim, output_dim):

    return {
        "depth1": [input_dim, 64, output_dim],
        "depth2": [input_dim, 64, 64, output_dim],
        "depth3": [input_dim, 64, 64, 64, output_dim],
        "depth4": [input_dim, 64, 64, 64, 64, output_dim]
    }


# -------------------------------------------------
# Depth Ablation Experiment
# -------------------------------------------------

def run_depth_ablation(dataset_path, plots_root, results_root):

    print("\nLoading dataset:", dataset_path)

    # ---------------------------
    # Load and preprocess data
    # ---------------------------

    X, y = load_dataset(dataset_path)

    X = normalize(X)

    # train / val / test split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_ratio=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_ratio=0.5)

    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    input_dim = X_train.shape[1]
    output_dim = 1

    architectures = get_architectures(input_dim, output_dim)

    # ---------------------------
    # Create folders
    # ---------------------------

    plot_dir = os.path.join(plots_root, "depth_ablation")
    os.makedirs(plot_dir, exist_ok=True)

    os.makedirs(results_root, exist_ok=True)

    results = []

    # ---------------------------
    # Run experiment
    # ---------------------------

    for name, arch in architectures.items():

        depth = len(arch) - 2

        print("\n===============================")
        print("Training architecture:", arch)

        activations = ["relu"] * depth + ["identity"]

        model = MLP(
            layer_sizes=arch,
            activations=activations,
            loss="mse",
            learning_rate=0.01,
            optimizer="adam",
            batch_size=32,
            weight_init="he"
        )

        history = model.fit(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=100,
            use_validation=True
        )

        # ---------------------------
        # Evaluate test error
        # ---------------------------

        test_error = model.evaluate(X_test, y_test)

        avg_epoch_time = np.mean(history["epoch_time"])

        print("Test Error:", test_error)

        results.append({
            "depth": depth,
            "test_error": test_error,
            "avg_epoch_time": avg_epoch_time,
            "parameters": model.count_parameters()
        })

        # ---------------------------
        # Plot training curves
        # ---------------------------

        plt.figure()

        plt.plot(history["train_loss"], label="Train")
        plt.plot(history["val_loss"], label="Validation")

        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title(f"Depth {depth}")

        plt.legend()

        plt.savefig(os.path.join(plot_dir, f"loss_depth_{depth}.png"))

        plt.close()

    # ---------------------------
    # Summary Plot
    # ---------------------------

    depths = [r["depth"] for r in results]
    errors = [r["test_error"] for r in results]

    plt.figure()

    plt.plot(depths, errors, marker="o")

    plt.xlabel("Network Depth")
    plt.ylabel("Test Error (MSE)")
    plt.title("Test Error vs Depth")

    plt.savefig(os.path.join(plot_dir, "test_error_vs_depth.png"))

    plt.close()

    # ---------------------------
    # Save results
    # ---------------------------

    results_df = pd.DataFrame(results)

    results_file = os.path.join(results_root, "depth_ablation_results.csv")

    results_df.to_csv(results_file, index=False)

    print("\nDepth ablation experiment completed.")
    print("Results saved to:", results_file)


# -------------------------------------------------
# Run Script
# -------------------------------------------------

if __name__ == "__main__":

    dataset_path = os.path.join(PROJECT_ROOT, "data", "iith_mess_dataset_5000.csv")

    plots_root = os.path.join(PROJECT_ROOT, "plots")

    results_root = os.path.join(PROJECT_ROOT, "results")

    run_depth_ablation(dataset_path, plots_root, results_root)
