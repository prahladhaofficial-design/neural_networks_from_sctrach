# ADALINE Experiment Report

## 1. Folder Structure

The **Module-5** folder is organized as follows:

```
Module-5/
 data/
    iith_mess_dataset_5000.csv
 experiments/
    adaline_experiments.py
 models/
    __init__.py
    adaline.py
 plots/
    dataset_plot.png
    decision_boundary.png
    learning_rate_comparison.png
    mse_vs_epoch_on_training_data.png
    train_val_error.png
    training_size_vs_final_mse_on_test_data.png
 results/
    dataset_info.csv
    learning_rate_results.csv
    train_val_error.csv
    training_mse_history.csv
    training_size_results.csv
 Theory/
    AdalineAnalysis.pdf
 utils/
     __init__.py
     path_setup.py
     utils.py
```

**Description of each folder:**

* **data/**: Contains the dataset `iith_mess_dataset_5000.csv` used for training and testing.
* **experiments/**: Contains experiment scripts like `adaline_experiments.py` which run training/testing for the model.
* **models/**: Implements the machine learning model; `adaline.py` defines the ADALINE model.
* **plots/**: Stores all generated plots from experiments (training error, decision boundaries, learning rate comparison, etc.).
* **results/**: Contains CSV files storing the numerical outputs from experiments (training MSE, learning rate results, train-validation errors, etc.).
* **Theory/**: Contains theoretical analysis in PDF format (`AdalineAnalysis.pdf`) explaining ADALINE behavior, error convergence, etc.
* **utils/**: Contains helper functions and path setups to facilitate experiments.

---

## 2. Experiments Conducted

The experiments in this Module-5 folder primarily focus on analyzing the performance of the **ADALINE model** on the given dataset. The key experiments include:

1. **Learning Rate Analysis**:

   * Testing different learning rates to observe convergence speed and stability.
   * Recorded results are in `learning_rate_results.csv` and visualized in `learning_rate_comparison.png`.

2. **Training vs Validation Error**:

   * Monitored the MSE over epochs on training and validation data.
   * Helps identify overfitting or underfitting.
   * Plots: `train_val_error.png`
   * Data: `train_val_error.csv`

3. **MSE vs Epoch**:

   * Tracked the mean squared error (MSE) progression during training.
   * Plot: `mse_vs_epoch_on_training_data.png`
   * Data: `training_mse_history.csv`

4. **Dataset Analysis**:

   * Understanding the distribution and features of the dataset.
   * Plot: `dataset_plot.png`
   * Data summary: `dataset_info.csv`

5. **Effect of Training Size**:

   * Tested how increasing training dataset size affects the final MSE on test data.
   * Plot: `training_size_vs_final_mse_on_test_data.png`
   * Data: `training_size_results.csv`

6. **Decision Boundary Visualization**:

   * Visualized the learned decision boundary of the ADALINE model on the dataset.
   * Plot: `decision_boundary.png`

All experiments are executed through `adaline_experiments.py` in the **experiments** folder.

---

## 3. Plots, Results, and Analysis

* **Plots Folder**: Contains visualizations for dataset distribution, learning rate comparison, training/validation error, decision boundary, and training size effects.
* **Results Folder**: Contains CSV files corresponding to each experiment, useful for further numerical analysis and reproducibility.

These provide both a qualitative (plots) and quantitative (CSV) understanding of model behavior.

---

## 4. Main Results and Observations

From the experiments conducted in the ADALINE experiment suite, the following observations were made:

1. **Learning Rate**:

   * Optimal learning rate balances convergence speed and stability.
   * Too high a learning rate caused divergence/oscillations in training MSE.
   * Too low a learning rate slowed convergence.

2. **Training vs Validation Error**:

   * Training MSE consistently decreased with epochs.
   * Validation MSE helps detect overfitting; minimal gap indicates good generalization.

3. **MSE Progression**:

   * MSE decreases exponentially during initial epochs and stabilizes later.
   * Indicates that the ADALINE model is learning effectively from the dataset.

4. **Dataset Insights**:

   * Dataset has 5000 samples.
   * Feature distribution and target behavior visualized in `dataset_plot.png`.

5. **Effect of Training Size**:

   * Larger training size reduces test MSE.
   * Suggests that ADALINE benefits from more data, as expected for linear models.

6. **Decision Boundary**:

   * ADALINE produces a linear boundary separating classes.
   * Boundary aligns well with data distribution, showing effective learning.

---

## 5. Conclusion

The Module-5 folder provides a comprehensive analysis of the **ADALINE model** on the IITH Mess dataset. Through systematic experiments:

* We evaluated learning rate effects, dataset size impacts, and error trends.
* Visualizations in **plots/** complemented numerical results in **results/**.

And also in this one all scripts, utilities, and theoretical explanations are well-organized to reproduce and extend the experiments efficiently.
