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
# Architectures for width experiment
# -------------------------------------------------

def get_architectures(input_dim, output_dim):

    widths = [16, 32, 64, 128, 256]

    architectures = {}

    for w in widths:
        architectures[f"width{w}"] = [input_dim, w, w, output_dim]

    return architectures


# -------------------------------------------------
# Width Ablation Experiment
# -------------------------------------------------

def run_width_ablation(dataset_path, plots_root, results_root):

    print("\nLoading dataset:", dataset_path)

    # -------------------------
    # Load dataset
    # -------------------------

    X, y = load_dataset(dataset_path)

    X = normalize(X)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_ratio=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_ratio=0.5)

    y_train = y_train.reshape(-1,1)
    y_val = y_val.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    input_dim = X_train.shape[1]
    output_dim = 1

    architectures = get_architectures(input_dim, output_dim)

    # -------------------------
    # Create folders
    # -------------------------

    plot_dir = os.path.join(plots_root, "width_ablation")
    os.makedirs(plot_dir, exist_ok=True)

    os.makedirs(results_root, exist_ok=True)

    results = []

    # -------------------------
    # Run experiment
    # -------------------------

    for name, arch in architectures.items():

        width = arch[1]

        print("\n===============================")
        print("Training architecture:", arch)

        activations = ["relu", "relu", "identity"]

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

        # -------------------------
        # Evaluate test error
        # -------------------------

        test_error = model.evaluate(X_test, y_test)

        avg_epoch_time = np.mean(history["epoch_time"])

        params = model.count_parameters()

        print("Width:", width)
        print("Parameters:", params)
        print("Test Error:", test_error)

        results.append({
            "width": width,
            "parameters": params,
            "test_error": test_error,
            "avg_epoch_time": avg_epoch_time
        })

        # -------------------------
        # Plot training curves
        # -------------------------

        plt.figure()

        plt.plot(history["train_loss"], label="Train")
        plt.plot(history["val_loss"], label="Validation")

        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title(f"Width {width}")

        plt.legend()

        plt.savefig(os.path.join(plot_dir, f"loss_width_{width}.png"))

        plt.close()

    # -------------------------
    # Plot 1 : Test error vs width
    # -------------------------

    widths = [r["width"] for r in results]
    errors = [r["test_error"] for r in results]

    plt.figure()

    plt.plot(widths, errors, marker="o")

    plt.xlabel("Hidden Layer Width")
    plt.ylabel("Test Error (MSE)")
    plt.title("Test Error vs Width")

    plt.savefig(os.path.join(plot_dir, "test_error_vs_width.png"))

    plt.close()

    # -------------------------
    # Plot 2 : Test error vs parameters (log scale)
    # -------------------------

    params = [r["parameters"] for r in results]

    plt.figure()

    plt.plot(params, errors, marker="o")

    plt.xscale("log")

    plt.xlabel("Number of Parameters (log scale)")
    plt.ylabel("Test Error (MSE)")
    plt.title("Test Error vs Model Size")

    plt.savefig(os.path.join(plot_dir, "test_error_vs_parameters.png"))

    plt.close()

    # -------------------------
    # Save results
    # -------------------------

    results_df = pd.DataFrame(results)

    results_file = os.path.join(results_root, "width_ablation_results.csv")

    results_df.to_csv(results_file, index=False)

    print("\nWidth ablation experiment completed.")
    print("Results saved to:", results_file)


# -------------------------------------------------
# Run script
# -------------------------------------------------

if __name__ == "__main__":

    dataset_path = os.path.join(PROJECT_ROOT, "data", "iith_mess_dataset_5000.csv")

    plots_root = os.path.join(PROJECT_ROOT, "plots")

    results_root = os.path.join(PROJECT_ROOT, "results")

    run_width_ablation(dataset_path, plots_root, results_root)
