# Deep Learning Project 1

## Artificial Neural Network with Backpropagation (From Scratch)

This project implements a **2-layer Artificial Neural Network (ANN)** from scratch using **NumPy** without relying on deep learning frameworks such as TensorFlow or PyTorch.

The goal of this project is to understand the **core mechanics of neural networks**, including:

* Forward Propagation
* Backpropagation
* Gradient Descent Optimization
* Weight Initialization
* Loss Functions
* Regularization
* Training behavior of neural networks

Two learning tasks are explored:

1. **Classification**  XOR dataset with Gaussian noise
2. **Regression**  Cosine function approximation

---

# Project Overview

The notebook implements a configurable neural network class that supports:

* Multiple activation functions
* Different gradient descent optimizers
* Weight initialization methods
* L2 regularization
* Learning rate decay
* Training and testing metrics tracking
* Visualization of loss and accuracy

The goal of the experiments is to observe how neural networks behave under different training conditions.

---

# Neural Network Architecture

The implemented neural network is a **2-layer fully connected network**:

Input Layer  Hidden Layer  Output Layer

### Configuration

| Component             | Options                        |
| --------------------- | ------------------------------ |
| Task                  | Classification / Regression    |
| Activation            | Sigmoid / Tanh / ReLU          |
| Optimizer             | Batch GD / SGD / Mini-Batch GD |
| Weight Initialization | Random / Xavier / He           |
| Regularization        | L2 Regularization              |
| Learning Rate Decay   | Supported                      |

---

# Datasets Used

Two datasets are generated programmatically in the notebook.

---

# 1. XOR Dataset (Classification)

The XOR problem is a **non-linearly separable classification task**.


Base XOR patterns:

```
(0,0)  0
(0,1)  1
(1,0)  1
(1,1)  0
```

Gaussian noise is added to the inputs.

---

# Network Configuration (XOR)

| Layer  | Nodes |
| ------ | ----- |
| Input  | 2     |
| Hidden | 4     |
| Output | 2     |

Activation Function: **tanh**
Loss Function: **Cross Entropy**
Optimizer: **Gradient Descent**

---

# Training Behavior

During training:

* Training loss decreases steadily across epochs.
* Testing loss follows a similar pattern.
* Accuracy increases gradually as the model learns the XOR decision boundary.

### Key Observations

* The neural network successfully learns the **non-linear XOR boundary**.
* Training and testing accuracy become high after sufficient epochs.
* Loss curves show **stable convergence**.

This demonstrates that a neural network with a hidden layer can solve problems that are not linearly separable.

---

# Experiment 1: Effect of Training Sample Size

The model was trained using different dataset sizes.

Sample sizes tested:

n = 100
n = 500
n = 2000

---

## Observations

### Small Dataset (n = 100)

* Training is less stable.
* Higher variance in results.
* Testing accuracy fluctuates.

### Medium Dataset (n = 500)

* Training becomes more stable.
* Improved generalization performance.

### Large Dataset (n = 2000)

* Training becomes smoother.
* Higher testing accuracy.
* Better generalization.

### Conclusion

Increasing the dataset size:

* Improves model stability
* Reduces variance
* Enhances generalization performance.

---

# Experiment 2: Effect of Batch Size

In this experiment, the dataset size is fixed and the **batch size (m)** is varied.

Batch sizes tested:

m = 1
m = 32
m = 128

| Batch Size | Optimizer                         |
| ---------- | --------------------------------- |
| 1          | Stochastic Gradient Descent (SGD) |
| 32         | Mini-Batch Gradient Descent       |
| 128        | Large Mini-Batch                  |

---

## Observations

### Batch Size = 1 (SGD)

* Frequent parameter updates.
* High variance in gradient updates.
* Loss curve tends to be noisy.
* Faster but unstable training.

### Batch Size = 32

* More stable training.
* Good balance between stability and speed.
* Often gives the best practical performance.

### Batch Size = 128

* Very smooth loss curves.
* More stable gradient updates.
* Slower convergence due to fewer updates.

### Conclusion

* Small batch sizes introduce noise in gradients.
* This noise can sometimes help escape poor local minima.
* Medium batch sizes often provide the best balance between stability and speed.

---

# 2. Cosine Dataset (Regression)

The regression task aims to approximate the function:

y = cos(x)

Noise is added to the outputs

---

# Network Configuration (Cosine Regression)

| Layer  | Nodes |
| ------ | ----- |
| Input  | 1     |
| Hidden | 10    |
| Output | 1     |

Activation Function: **tanh**
Loss Function: **Mean Squared Error (MSE)**

---

# Training Behavior

During training:

* Training loss decreases steadily across epochs.
* Testing loss follows a similar pattern.
* The neural network learns to approximate the cosine function.

### Key Observations

* The network successfully approximates the cosine curve.
* The model handles noise reasonably well.
* Loss curves indicate smooth convergence.

---

# Experiment 1: Effect of Training Sample Size

Sample sizes tested:

n = 100
n = 500
n = 2000

---

## Observations

### Small Dataset (n = 100)

* Higher training and testing loss.
* The network struggles to accurately approximate the cosine curve.

### Medium Dataset (n = 500)

* Improved approximation of the cosine function.
* Lower testing loss.

### Large Dataset (n = 2000)

* Best regression performance.
* Smooth cosine approximation.
* Lowest test loss.

### Conclusion

Increasing training data:

* Improves regression accuracy
* Reduces approximation error
* Improves generalization.

---

# Experiment 2: Effect of Batch Size

Batch sizes tested:

m = 1
m = 32
m = 128

---

## Observations

### Batch Size = 1 (SGD)

* Very noisy loss curve.
* Many frequent updates.

### Batch Size = 32

* Stable convergence.
* Balanced training behavior.

### Batch Size = 128

* Smooth loss curves.
* Slower convergence.

### Conclusion

* Larger batch sizes produce smoother training curves.
* Smaller batch sizes allow more frequent parameter updates.

---

# Visualizations

The notebook generates several plots to analyze training behavior:

* Training vs Testing Loss
* Training vs Testing Accuracy
* Effect of Batch Size on Loss
* Effect of Batch Size on Accuracy

These plots help visualize the convergence behavior of the neural network.

---

# Overall Conclusions

From all experiments, the following insights were observed:

### 1. Effect of Dataset Size

Increasing the dataset size:

* Improves model generalization
* Reduces training variance
* Produces more stable learning behavior.

---

### 2. Effect of Batch Size

Batch size affects the training dynamics:

Small batches:

* Noisy gradients
* Faster updates

Medium batches:

* Best balance between speed and stability

Large batches:

* Smooth training
* Slower convergence.

---

### 3. Neural Network Capability

The implemented neural network successfully demonstrates:

* Learning of **non-linear decision boundaries (XOR classification)**.
* Approximation of **continuous functions (cosine regression)**.
* The effect of different gradient descent strategies.

---

# Plots

* you can find all the plots in the colab notebook

---

# How to Run

1. Open the notebook in **Google Colab**.
2. Run all cells sequentially.
3. The experiments will generate training results and plots automatically.

---

# Summary

This project demonstrates the internal working of neural networks by implementing **forward propagation, backpropagation, and gradient descent from scratch**.

The experiments highlight important deep learning concepts such as:

* Effect of dataset size
* Impact of batch size
* Behavior of gradient descent methods
* Neural network learning dynamics

These experiments provide a deeper understanding of how neural networks learn and converge during training.
