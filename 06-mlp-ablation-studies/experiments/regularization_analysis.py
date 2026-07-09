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
# Experiment
# -------------------------------------------------

def run_regularization_experiment(dataset_path, plots_root, results_root):

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

    lambdas = [0.001, 0.01, 0.1]

    plot_dir = os.path.join(plots_root, "regularization")
    os.makedirs(plot_dir, exist_ok=True)

    results = []
    sparsity_results = []

    configs = []

    configs.append(("none", 0))

    for l in lambdas:
        configs.append(("l1", l))
        configs.append(("l2", l))

    # -------------------------------------------------
    # Run experiments
    # -------------------------------------------------

    for reg, lam in configs:

        print("\n===============================")
        print("Regularization:", reg, "lambda:", lam)

        model = MLP(
            layer_sizes=architecture,
            activations=["relu","relu","identity"],
            loss="mse",
            learning_rate=0.001,
            optimizer="adam",
            batch_size=32,
            weight_init="he",
            regularization=None if reg=="none" else reg,
            lambda_reg=lam
        )

        history = model.fit(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=100,
            use_validation=True
        )

        # -------------------------------------------------
        # Train vs Validation loss plot
        # -------------------------------------------------

        plt.figure()

        plt.plot(history["train_loss"], label="Train")
        plt.plot(history["val_loss"], label="Validation")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        title = f"{reg}_lambda_{lam}"
        plt.title(title)

        plt.legend()

        plt.savefig(os.path.join(plot_dir, f"{title}_train_val_curve.png"))

        plt.close()

        # -------------------------------------------------
        # Evaluation
        # -------------------------------------------------

        test_loss = model.evaluate(X_test, y_test)

        results.append({
            "regularization": reg,
            "lambda": lam,
            "test_loss": test_loss
        })

        # -------------------------------------------------
        # L1 Sparsity
        # -------------------------------------------------

        if reg == "l1":

            sparsity = model.weight_sparsity()

            sparsity_results.append({
                "lambda": lam,
                "sparsity": sparsity
            })

    # -------------------------------------------------
    # Plot L1 Sparsity vs Lambda
    # -------------------------------------------------

    lamb = [x["lambda"] for x in sparsity_results]
    spars = [x["sparsity"] for x in sparsity_results]

    plt.figure()

    plt.plot(lamb, spars, marker="o")

    plt.xlabel("Lambda")
    plt.ylabel("Weight Sparsity (<1e-3)")

    plt.title("L1 Sparsity vs Lambda")

    plt.xscale("log")

    plt.savefig(os.path.join(plot_dir, "l1_sparsity_vs_lambda.png"))

    plt.close()

    # -------------------------------------------------
    # Save Results
    # -------------------------------------------------

    results_df = pd.DataFrame(results)

    results_file = os.path.join(results_root, "regularization_results.csv")

    results_df.to_csv(results_file, index=False)

    sparsity_df = pd.DataFrame(sparsity_results)

    sparsity_file = os.path.join(results_root, "l1_sparsity_results.csv")

    sparsity_df.to_csv(sparsity_file, index=False)

    print("\nRegularization experiment completed")
    print("Results saved to:", results_file)


# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":

    dataset_path = os.path.join(PROJECT_ROOT, "data", "iith_mess_dataset_5000.csv")

    plots_root = os.path.join(PROJECT_ROOT, "plots")

    results_root = os.path.join(PROJECT_ROOT, "results")

    run_regularization_experiment(dataset_path, plots_root, results_root)
