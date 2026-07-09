import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from utils.path_setup import *
from models.mlp import MLP
from utils.utils import load_dataset, normalize, train_test_split


# -------------------------------------------------
# Optimizers
# -------------------------------------------------

def get_optimizers():

    return [
        "sgd",
        "momentum",
        "adam",
        "adagrad",
        "nesterov",
        "rmsprop",
        "muon"
    ]


# -------------------------------------------------
# Learning rates for sensitivity test
# -------------------------------------------------

def get_learning_rates():

    return [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]


# -------------------------------------------------
# Main Experiment
# -------------------------------------------------

def run_optimizer_experiment(dataset_path, plots_root, results_root):

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

    optimizers = get_optimizers()

    plot_dir = os.path.join(plots_root, "optimizers")
    os.makedirs(plot_dir, exist_ok=True)

    results = []

    # -------------------------------------------------
    # PART A & B: Training comparison
    # -------------------------------------------------

    for opt in optimizers:

        print("\n=============================")
        print("Optimizer:", opt)

        model = MLP(
            layer_sizes=architecture,
            activations=["relu","relu","identity"],
            loss="mse",
            learning_rate=0.001,
            optimizer=opt,
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

        # -----------------------------------------
        # Convergence plot
        # -----------------------------------------

        plt.figure()

        plt.plot(history["train_loss"], label="Train")
        plt.plot(history["val_loss"], label="Validation")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plt.title(f"Optimizer: {opt}")

        plt.legend()

        plt.savefig(os.path.join(plot_dir, f"{opt}_convergence.png"))

        plt.close()

        # -----------------------------------------
        # Weight update magnitudes (Epoch vs Update)
        # -----------------------------------------

        update_magnitudes = history["update_magnitude"]

        plt.figure()

        plt.plot(update_magnitudes)

        plt.xlabel("Epoch")
        plt.ylabel("Update Magnitude")

        plt.title(f"Epoch vs Update Magnitude: {opt}")

        plt.savefig(os.path.join(plot_dir, f"{opt}_update_magnitude.png"))

        plt.close()

        # -----------------------------------------
        # Time to reach 90% best validation
        # -----------------------------------------

        best_val = min(history["val_loss"])

        threshold = best_val * 1.1

        epoch_90 = None

        for i, v in enumerate(history["val_loss"]):

            if v <= threshold:
                epoch_90 = i + 1
                break

        test_error = model.evaluate(X_test, y_test)

        results.append({
            "optimizer": opt,
            "best_val_loss": best_val,
            "epoch_to_90pct": epoch_90,
            "test_error": test_error
        })

    # -------------------------------------------------
    # PART C: Learning rate sensitivity
    # -------------------------------------------------

    learning_rates = get_learning_rates()

    lr_results = []

    for opt in optimizers:

        final_losses = []

        for lr in learning_rates:

            print(f"{opt} | LR={lr}")

            model = MLP(
                layer_sizes=architecture,
                activations=["relu","relu","identity"],
                loss="mse",
                learning_rate=lr,
                optimizer=opt,
                batch_size=32,
                weight_init="he"
            )

            history = model.fit(
                X_train,
                y_train,
                X_val,
                y_val,
                epochs=50,
                use_validation=True
            )

            final_losses.append(history["val_loss"][-1])

        # Plot LR sensitivity

        plt.figure()

        plt.plot(learning_rates, final_losses, marker="o")

        plt.xscale("log")

        plt.xlabel("Learning Rate")
        plt.ylabel("Final Validation Loss")

        plt.title(f"LR Sensitivity: {opt}")

        plt.savefig(os.path.join(plot_dir, f"{opt}_lr_sensitivity.png"))

        plt.close()

        for lr, loss_val in zip(learning_rates, final_losses):

            lr_results.append({
                "optimizer": opt,
                "learning_rate": lr,
                "val_loss": loss_val
            })

    # -------------------------------------------------
    # Save results
    # -------------------------------------------------

    results_df = pd.DataFrame(results)

    results_file = os.path.join(results_root, "optimizer_results.csv")

    results_df.to_csv(results_file, index=False)

    lr_df = pd.DataFrame(lr_results)

    lr_file = os.path.join(results_root, "optimizer_lr_sensitivity.csv")

    lr_df.to_csv(lr_file, index=False)

    print("\nExperiment completed.")
    print("Results saved to:", results_file)


# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":

    dataset_path = os.path.join(PROJECT_ROOT, "data", "iith_mess_dataset_5000.csv")

    plots_root = os.path.join(PROJECT_ROOT, "plots")

    results_root = os.path.join(PROJECT_ROOT, "results")

    run_optimizer_experiment(dataset_path, plots_root, results_root)
