import os
import sys
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
# Loss configurations
# -------------------------------------------------

def get_loss_configs():

    return [
        "mse",
        "mae",
        "huber"
    ]


# -------------------------------------------------
# Main experiment
# -------------------------------------------------

def run_loss_analysis(dataset_path, plots_root, results_root):

    print("Loading dataset...")

    X, y = load_dataset(dataset_path)

    X = normalize(X)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_ratio=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_ratio=0.5)

    y_train = y_train.reshape(-1,1)
    y_val = y_val.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    input_dim = X_train.shape[1]
    output_dim = 1

    architecture = [input_dim, 64, 64, output_dim]

    losses = get_loss_configs()

    plot_dir = os.path.join(plots_root, "loss_analysis")
    os.makedirs(plot_dir, exist_ok=True)

    results = []

    # -------------------------------------------------
    # Run experiments
    # -------------------------------------------------

    for loss_name in losses:

        print("\n=============================")
        print("Training with loss:", loss_name)

        model = MLP(
            layer_sizes=architecture,
            activations=["relu", "relu", "identity"],
            loss=loss_name,
            learning_rate=0.001,
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

        train_error = model.evaluate(X_train, y_train)
        val_error = model.evaluate(X_val, y_val)
        test_error = model.evaluate(X_test, y_test)

        results.append({
            "loss_function": loss_name,
            "train_error": train_error,
            "val_error": val_error,
            "test_error": test_error
        })

        # -------------------------------------------------
        # Plot loss curves
        # -------------------------------------------------

        plt.figure()

        plt.plot(history["train_loss"], label="Train")
        plt.plot(history["val_loss"], label="Validation")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.title(f"Loss Function: {loss_name}")

        plt.legend()

        plt.savefig(os.path.join(plot_dir, f"{loss_name}_loss_curve.png"))

        plt.close()

    # -------------------------------------------------
    # Save results
    # -------------------------------------------------

    results_df = pd.DataFrame(results)

    results_file = os.path.join(results_root, "loss_function_analysis.csv")

    results_df.to_csv(results_file, index=False)

    print("\nLoss function analysis completed.")
    print("Results saved to:", results_file)


# -------------------------------------------------

if __name__ == "__main__":

    dataset_path = os.path.join(PROJECT_ROOT, "data", "iith_mess_dataset_5000.csv")

    plots_root = os.path.join(PROJECT_ROOT, "plots")

    results_root = os.path.join(PROJECT_ROOT, "results")

    run_loss_analysis(dataset_path, plots_root, results_root)
