import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)
from utils.path_setup import *

from models.adaline import Adaline
from utils.utils import load_dataset, normalize, train_test_split, pca_2d


DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "iith_mess_dataset_5000.csv")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


#-------------------------------------------------------
# Load Dataset
#-------------------------------------------------------

X, y = load_dataset(DATASET_PATH)

X = normalize(X)

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Save dataset info
dataset_info = pd.DataFrame({
    "num_samples":[len(X)],
    "num_features":[X.shape[1]],
    "train_samples":[len(X_train)],
    "test_samples":[len(X_test)]
})

dataset_info.to_csv(os.path.join(RESULTS_DIR,"dataset_info.csv"),index=False)


#-------------------------------------------------------
# 1 Dataset Plot
#-------------------------------------------------------

X_2d = pca_2d(X)

plt.figure()

plt.scatter(X_2d[:,0], X_2d[:,1], c=y)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title("Dataset Visualization (PCA Projection)")

plt.savefig(os.path.join(PLOTS_DIR,"dataset_plot.png"))

plt.close()


#-------------------------------------------------------
# 2 Train ADALINE
#-------------------------------------------------------

model = Adaline(learning_rate=0.01, max_iterations=200)

history = model.fit(X_train, y_train)

plt.figure()

plt.plot(history)

plt.xlabel("Epoch")
plt.ylabel("MSE")

plt.title("MSE vs Epoch on training data")

plt.savefig(os.path.join(PLOTS_DIR,"mse_vs_epoch_on_training_data.png"))

plt.close()

# Save MSE history
mse_df = pd.DataFrame({
    "epoch":np.arange(1,len(history)+1),
    "train_mse":history
})

mse_df.to_csv(os.path.join(RESULTS_DIR,"training_mse_history.csv"),index=False)


#-------------------------------------------------------
# 3 Decision Boundary using PCA
#-------------------------------------------------------

X_train_2d = pca_2d(X_train)

model = Adaline(learning_rate=0.01, max_iterations=200)

model.fit(X_train_2d, y_train)

x_min, x_max = X_train_2d[:,0].min()-1, X_train_2d[:,0].max()+1
y_min, y_max = X_train_2d[:,1].min()-1, X_train_2d[:,1].max()+1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

grid = np.c_[xx.ravel(), yy.ravel()]

Z = model.predict(grid)

Z = Z.reshape(xx.shape)

plt.figure()

plt.contourf(xx, yy, Z, alpha=0.3)

plt.scatter(X_train_2d[:,0], X_train_2d[:,1], c=y_train)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title("ADALINE Decision Boundary (PCA)")

plt.savefig(os.path.join(PLOTS_DIR,"decision_boundary.png"))

plt.close()


#-------------------------------------------------------
# 4 Training vs Validation Error
#-------------------------------------------------------

model = Adaline(learning_rate=0.01, max_iterations=200)

train_error, val_error = model.fit(X_train, y_train, X_test, y_test)

plt.figure()

plt.plot(train_error, label="Training Error")
plt.plot(val_error, label="Validation Error")

plt.legend()

plt.xlabel("Epoch")
plt.ylabel("MSE")

plt.title("Training vs Validation Error")

plt.savefig(os.path.join(PLOTS_DIR, "train_val_error.png"))

plt.close()

# Save errors
error_df = pd.DataFrame({
    "epoch":np.arange(1,len(train_error)+1),
    "train_error":train_error,
    "validation_error":val_error
})

error_df.to_csv(os.path.join(RESULTS_DIR,"train_val_error.csv"),index=False)


#-------------------------------------------------------
# 5 Learning Rate Experiment
#-------------------------------------------------------

learning_rates = [0.01, 0.1, 1.0, 10.0]

fig = plt.figure(figsize=(10,10))

histories = []
lr_results = []

for i, lr in enumerate(learning_rates):

    model = Adaline(learning_rate=lr, max_iterations=200)

    history = model.fit(X_train, y_train)

    histories.append(history)

    lr_results.append({
        "learning_rate":lr,
        "final_mse":history[-1]
    })

    ax = plt.subplot(3,2,i+1)

    ax.plot(history)

    ax.set_title(f"Learning Rate = {lr}")

    ax.set_xlabel("Epoch")

    ax.set_ylabel("MSE")


ax = plt.subplot(3,1,3)

for lr, history in zip(learning_rates, histories):

    ax.plot(history, label=f"lr={lr}")

ax.set_title("Learning Rate Comparison")

ax.set_xlabel("Epoch")

ax.set_ylabel("MSE")

ax.legend()

plt.tight_layout()

plt.savefig(os.path.join(PLOTS_DIR, "learning_rate_comparison.png"))

plt.close()

# Save LR results
lr_df = pd.DataFrame(lr_results)

lr_df.to_csv(os.path.join(RESULTS_DIR,"learning_rate_results.csv"),index=False)


#-------------------------------------------------------
# 6 Training Data Size Experiment
#-------------------------------------------------------

sizes = np.arange(0.1,1.1,0.1)

performance = []

for s in sizes:

    n = int(len(X_train)*s)

    X_sub = X_train[:n]
    y_sub = y_train[:n]

    model = Adaline(learning_rate=0.01,max_iterations=200)

    model.fit(X_sub,y_sub)

    mse = model.score(X_test,y_test)

    performance.append(mse)

plt.figure()

plt.plot(sizes*100,performance)

plt.xlabel("Training Data (%)")
plt.ylabel("Performance")

plt.title("Training Size vs Final MSE")

plt.savefig(os.path.join(PLOTS_DIR,"training_size_vs_final_mse_on_test_data.png"))

plt.close()

# Save training size results
size_df = pd.DataFrame({
    "training_percentage":sizes*100,
    "test_mse":performance
})

size_df.to_csv(os.path.join(RESULTS_DIR,"training_size_results.csv"),index=False)


print("All experiments completed.")
print("Plots saved in:",PLOTS_DIR)
print("Results saved in:",RESULTS_DIR)
