# Neural Network Foundations From Scratch

A from-scratch neural-network laboratory covering XOR backpropagation, synthetic data generation, ADALINE regression, and multilayer perceptron ablation studies.

## Overview

Neural networks are easiest to trust when their training mechanics are visible. This project breaks the learning process into small, inspectable experiments before moving into larger tabular regression studies.
The repository connects theory with implementation by showing how gradients, activations, losses, optimizer choices, and model capacity change learning behavior.

## Project Design

- Start with a controlled XOR classifier to expose forward propagation, backward propagation, and gradient-based updates.
- Generate a synthetic tabular dataset with weather, sleep, time, and contextual variables so regression experiments have a richer feature space.
- Use ADALINE as a linear baseline and compare it with multilayer perceptrons.
- Run focused ablations where only one design choice changes at a time.

## Key Capabilities

- Noisy XOR classification with a two-layer neural network.
- Synthetic dataset generation for a realistic regression-style problem.
- ADALINE implementation with learning-rate and training-size studies.
- MLP implementation with configurable depth, width, activation functions, losses, optimizers, initialization, and regularization.
- Experiment outputs saved as plots and CSV files for review without rerunning every training job.

## Repository Structure

```text
01-backpropagation-xor/   Backpropagation notebook and supporting notes.
04-data-generation/       Synthetic dataset generator.
05-adaline-experiments/   Linear baseline, training scripts, results, and plots.
06-mlp-ablation-studies/  MLP implementation and ablation experiments.
docs/                     Theory notes and converted technical briefs.
requirements.txt          Python environment dependencies.
```

## Running Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook 01-backpropagation-xor/backpropagation_xor.ipynb
cd 05-adaline-experiments && python experiments/adaline_experiments.py
cd ../06-mlp-ablation-studies && python experiments/optimizer_comparison.py
```

## Results and Artifacts

- `05-adaline-experiments/results/` contains dataset summaries, learning-rate sweeps, and train-validation logs.
- `05-adaline-experiments/plots/` contains dataset views, decision boundaries, and training curves.
- `06-mlp-ablation-studies/results/` contains model-comparison and ablation tables.
- `06-mlp-ablation-studies/plots/` contains optimizer, regularization, depth, width, and loss-function visualizations.

## Reproducibility Notes

- Run module experiments in order when comparing perceptron, MLP, activation, and optimizer behavior.
- Keep seeds, learning rates, and epoch counts recorded for every training-curve comparison.
- Regenerate plots after changing network depth, activation functions, or optimizer settings.
- Review the dataset note before rerunning experiments that depend on local data files.

## Detailed Workflow

- Start with the XOR notebook to inspect how the network transforms inputs into logits and how error terms move backward through the layers.
- Use the data-generation script to create the synthetic regression dataset used by the ADALINE and MLP studies.
- Run the ADALINE experiments as the linear baseline and inspect how learning rate changes convergence.
- Run the MLP ablations to compare nonlinear capacity, optimizer behavior, and regularization.
- Use the CSV outputs as the source of truth for comparing experiments instead of relying only on notebook screenshots.

## Implementation Details

- The ADALINE implementation keeps the linear prediction rule explicit, which makes it easy to compare against the later nonlinear model.
- The MLP study separates core concepts into small modules: activations, losses, optimizers, initialization, model code, and experiment runners.
- The ablation scripts are intentionally independent so each experimental factor can be rerun without triggering the full suite.
- The result folders preserve numeric outputs while the plot folders preserve training curves and comparison graphics.
- The docs folder contains supporting theory for approximation, gradient flow, kernels, and network behavior.

## Data and Inputs

- The main tabular dataset is generated locally and stored as CSV files.
- Feature columns are designed to simulate measurable context variables rather than arbitrary random noise.
- The 5,000-row dataset is used for the larger regression comparisons.
- Because the data is generated, experiments can be rerun and extended without depending on private external data.

## How To Interpret the Results

- Use ADALINE results to understand the limits of a linear predictor on the generated regression task.
- Use the MLP model-comparison results to inspect how nonlinear representation improves the fit.
- Use optimizer plots to compare convergence speed and stability.
- Use width and depth ablations to reason about underfitting, overfitting, and model capacity.
- Use regularization plots to inspect the trade-off between flexibility and stability.

## Limitations and Assumptions

- The generated dataset is useful for controlled experiments, but it is not a production data source.
- The implementations emphasize clarity and learning value rather than maximum runtime performance.
- Some experiments are computationally heavier than others, especially the wider MLP configurations.
- The project is best reviewed by reading the results folders together with the source code.


## Review Guide

Use this checklist when reviewing the repository on GitHub:

- Start with the overview to understand the project objective before opening notebooks or source files.
- Read the repository structure section to identify the main entry point.
- Inspect generated outputs before rerunning experiments; they provide a quick view of the current project state.
- Review scripts and notebooks together when both are present, since notebooks often explain decisions while scripts preserve reusable execution paths.
- Check the limitations section before interpreting results too strongly.
- Use the possible extensions section to understand how the project can grow without changing its core purpose.

## Troubleshooting

Common issues and fixes:

- If imports fail, confirm that the virtual environment is activated and dependencies were installed from `requirements.txt`.
- If a notebook cell fails because a variable is missing, restart the kernel and run the notebook from the first cell.
- If a script cannot find a data file, run it from the repository root unless the README gives a different working directory.
- If plots are not regenerated, check that the output folder exists and that the script has write permission.
- If a package version causes errors, create a fresh environment and reinstall dependencies before debugging the source code.
- If GPU-related packages are unavailable, run CPU-compatible sections first and treat heavier deep-learning experiments separately.

## Development Notes

The repository is organized to be understandable before it is optimized. Most code favors clear structure, visible intermediate steps, and saved artifacts over highly compressed implementations. When extending the project, prefer small changes that keep the original experiment reproducible.

Recommended development practices:

- Keep new data files under `data/`, `datasets/`, or the existing local data folder for that project.
- Keep generated figures under `figures/`, `plots/`, `outputs/`, or `results/` depending on the existing convention.
- Add new experiments as separate scripts or clearly labeled notebook sections.
- Avoid overwriting existing outputs unless the experiment is intentionally being refreshed.
- Document any new dependency in `requirements.txt`.
- Prefer relative paths so the repository works after cloning on another machine.

## Project Status

This repository is prepared as a standalone GitHub project. The current version includes the core implementation, documentation, and available artifacts needed to understand the work. Future improvements can focus on stronger automation, more tests, richer visual summaries, and cleaner packaging where appropriate.

## Possible Extensions

- Add automated experiment configuration files for every ablation.
- Wrap the ADALINE and MLP models in a unified estimator interface.
- Add tests for gradient calculations and optimizer updates.
- Export a compact experiment report from the saved CSV files.

## Tech Stack

Python, NumPy, Pandas, Matplotlib, scikit-learn-style evaluation, Jupyter Notebook.

## Author

Prahladha, IIT Hyderabad
