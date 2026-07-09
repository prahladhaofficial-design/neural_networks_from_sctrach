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


# Activation configurations
def get_activation_configs():

    return {
        "sigmoid": ["sigmoid", "sigmoid", "sigmoid"],
        "tanh": ["tanh", "tanh", "tanh"],
        "relu": ["relu", "relu", "sigmoid"],
        "leaky_relu": ["leaky_relu", "leaky_relu", "sigmoid"]
    }


# Dead neuron analysis
def compute_dead_neurons(model, X):

    activations = model.forward(X)

    dead_counts = []

    for layer_act in activations[1:-1]:

        dead = np.sum(np.all(layer_act == 0, axis=0))

        dead_counts.append(dead)

    return dead_counts


# Gradient statistics
def gradient_statistics(grads_W):

    means = []
    stds = []

    for g in grads_W:
        means.append(np.mean(g))
        stds.append(np.std(g))

    return means, stds


# Main experiment
def run_activation_experiment(dataset_path, plots_root, results_root):

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

    configs = get_activation_configs()

    plot_dir = os.path.join(plots_root, "activation_comparison")
    os.makedirs(plot_dir, exist_ok=True)

    results = []

    # Run configurations
    for name, activations in configs.items():

        print("\n=============================")
        print("Activation config:", name)

        model = MLP(
            layer_sizes=architecture,
            activations=activations,
            loss="mse",
            learning_rate=0.01,
            optimizer="adam",
            batch_size=32,
            weight_init="xavier"
        )

        history = model.fit(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=100,
            use_validation=True
        )

        test_error = model.evaluate(X_test, y_test)

        # Gradient statistics
        grads_W, grads_b = model.backward(X_train[:64], y_train[:64])

        grad_mean, grad_std = gradient_statistics(grads_W)

        # Dead neurons
        dead_neurons = None

        if "relu" in name:
            dead_neurons = compute_dead_neurons(model, X_train)

        # Vanishing gradient measurement
        grad_magnitude = [np.linalg.norm(g) for g in grads_W]

        # Save results
        results.append({
            "activation": name,
            "test_error": test_error,
            "grad_mean_layer1": grad_mean[0],
            "grad_mean_layer2": grad_mean[1],
            "grad_std_layer1": grad_std[0],
            "grad_std_layer2": grad_std[1],
            "grad_norm_layer1": grad_magnitude[0],
            "grad_norm_layer2": grad_magnitude[1],
            "dead_neurons": dead_neurons
        })

        # Plot loss curves
        plt.figure()

        plt.plot(history["train_loss"], label="Train")
        plt.plot(history["val_loss"], label="Validation")

        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")

        plt.title(f"Activation: {name}")

        plt.legend()

        plt.savefig(os.path.join(plot_dir, f"loss_{name}.png"))

        plt.close()

    # Save results
    results_df = pd.DataFrame(results)

    results_file = os.path.join(results_root, "activation_comparison_results.csv")

    results_df.to_csv(results_file, index=False)

    print("\nActivation comparison completed.")
    print("Results saved to:", results_file)


# -------------------------------------------------

if __name__ == "__main__":

    dataset_path = os.path.join(PROJECT_ROOT, "data", "iith_mess_dataset_5000.csv")

    plots_root = os.path.join(PROJECT_ROOT, "plots")

    results_root = os.path.join(PROJECT_ROOT, "results")

    run_activation_experiment(dataset_path, plots_root, results_root)
