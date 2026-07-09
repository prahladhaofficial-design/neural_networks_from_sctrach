# Deep Learning Project 1


Project 1
Deadline: 9th March, 2026
Project Version: (A1 <roll-number>.zip) Total: 200 Evaluation Points
Please read the following instructions before starting the project:
 It is strongly recommended that you work on this homework on an individual basis. If you have
any modules or concerns, feel free to contact the instructor or the reviewers at least 2 days before deadline.
 Use numpy for basic functions like log, sqrt, power. Do not use other built-in functions and implimen-
tation should be from scratch.
 You can also use markdown cells in the notebook to explain theoretical problems and observations.
 Some modules have specific project version requirements as mentioned in the respective problem state-
ments. Please carefully follow those guidelines (e.g., project version of .py files, .pdf reports, plots, etc.).
 Ensure that all submitted code runs without errors. Non-executable code will receive zero evaluation points for
the corresponding module.
 Include a README.md file clearly describing the folder structure of the submitted zip file, indicating
where the solution for each module is located, along with instructions to run the code.
1 Artificial Neural Network with Backpropagation (20 Evaluation Points)
a) Implement a simple Artificial Neural Network (ANN) from scratch (i.e., without using built-in func-
tions). Implement the back-propagation algorithm to learn the weights of an ANN with 2 input nodes,
2 hidden nodes and 1 output node. Train your network to learn the following functions:
i) XOR (10)
ii) Cosine (10)
Choose the activation functions and loss functions appropriately. The number of nodes in the network
should be a configurable parameter in your code. Generate the dataset by perturbing the input binary
digits with a small amount of Gaussian noise. Use 80% of the dataset for training and the remaining
for testing. Report the following:
i) Training and testing loss over training iterations. Plot these values.
ii) Training and testing accuracy over training iterations. Plot these values.
iii) Print the training and test loss and accuracy at the end of training.
iv) Deterministic Gradient Descent: Experiment with the number of training samples n and see how
it affects performance. In other words, repeat the training and testing for three different choices
of n.
v) Stochastic Gradient Descent: Experiment with the batch size m (for a fixed n) and see how it
affects performance. In other words, repeat the training and testing for three different choices of
m.
1


2 Universal Approximation Theorem (15 evaluation points)
(a) State a formal version of the Universal Approximation Theorem for a feedforward neural network with
a single hidden layer and a sigmoid (or any non-polynomial continuous) activation function. Clearly
specify: (4)
 the class of target functions being approximated,
 assumptions on the domain,
 the notion of approximation and the norm used.
(b) Outline the main idea behind a proof of the Universal Approximation Theorem. A full formal proof
is not required. Your answer should briefly explain the intuition or the key theoretical argument
(e.g., constructive approximation using basis-like functions, or a connection to classical results from
approximation theory such as the StoneWeierstrass theorem). (4)
(c) Discuss two important limitations of the Universal Approximation Theorem from a practical machine
learning perspective. For each limitation, explain why it is relevant when designing or training neural
networks. (4)
(d) Briefly comment on how the Universal Approximation property changes when using ReLU activations
and allowing network depth to increase while keeping the width relatively small. (3)
3 One Step of Gradient Descent in a Deep Neural Network (25 evaluation points)
Consider the following feedforward neural network with three affine layers (four layers including the input),
ReLU activations in the hidden layers, and two output classes.
Network architecture
(cid:20) (cid:21)
1
 Input: x =  R2
1
 Layer 1 (Affine + ReLU):
z(1) = W (1)x + b(1), a(1) = ReLU(z(1))
(cid:20) (cid:21) (cid:20) (cid:21)
1 0 0
W (1) = , b(1) =
0 1 0
 Layer 2 (Affine + ReLU):
z(2) = W (2)a(1) + b(2), a(2) = ReLU(z(2))
(cid:20) (cid:21) (cid:20) (cid:21)
1 1 0
W (2) = , b(2) =
1 1 0
 Layer 3 (Output layer, affine only):
s = W (3)a(2) + b(3)
(cid:20) (cid:21) (cid:20) (cid:21)
1 0 0
W (3) = , b(3) =
0 1 0
There are two classes, and the true class label is y = 1 (using 1-based indexing). The learning rate is  = 0.1.
Assume the derivative of the ReLU function is
(cid:40)
1, t > 0
ReLU(t) =
0, t  0
2


Part A: Softmax with Cross-Entropy Loss (12 Evaluation Points)
The softmax function is defined as
esi
q =
i (cid:80) esj
j
and the cross-entropy loss for a single example is
L =  log q
CE y
(a) Perform the forward pass and compute z(1), a(1), z(2), a(2), s, and the softmax probabilities q. (3)
(b) Compute the cross-entropy loss L . (1)
CE
(c) Compute the gradient of the loss with respect to the logits s. (1)
(d) Using backpropagation, compute the gradients with respect to all weights and biases:
 ,  ,  ,  ,  ,
W (3) b(3) W (2) b(2) W (1) b(1)
(4)
(e) Perform one gradient descent update step for all parameters and report the updated values. (3)
Part B: Multiclass Hinge Loss (13 evaluation points)
The multiclass hinge loss is defined as
(cid:88)
L = max (0, s  s + 1)
hinge j y
j=y
(a) Compute the hinge loss for the given example. (2)
(b) Compute the gradient of the hinge loss with respect to the logits s. (2)
(c) Using backpropagation, compute the gradients with respect to all weights and biases. (6)
(d) Perform one gradient descent update step for all parameters and report the updated values. (3)
4 A Voyage into Neural Networks
This project takes you on a journey through the historical development of neural networks, from the
simple perceptron (1958) to ideas in modern deep learning. You will implement each model from scratch,
understand why each advancement was necessary, and empirically demonstrate the limitations and capabil-
ities of each approach. Side by side, this will also require you to create your own dataset. Instructions for
both parts will be specified incrementally.
Dataset: IIT-H Campus Life Prediction
You will create and progressively augment a tabular dataset based on student life at IIT-H. The dataset will
be used to predict mess duration (how long a student would spend in the mess for a particular meal).
Dataset Construction
Collect/simulate data with 500 samples collected over 15 days containing:
 student ID (integer)
 day of week (0-6, Monday=0)
 meal time (0: breakfast, 1: lunch, 2: dinner)
3


 meal type (0: vegetarian, 1: non-vegetarian)
 is weekend (binary)
 class day (0: if class is there that day, 1: otherwise)
 project deadline (0: if there is an project deadline that day, 1: otherwise)
 weather features (temperature (C), humidity, wind speed, rain, air quality)
 rising time (minutes from 00:00:00)
 sleeping time (minutes from 00:00:00)
 mess duration (target; minutes)
Then extend this dataset to 5000 samples by adding Gaussian noise to the weather features, rising time,
and sleeping time features.
Project Version Requirement: Include a data generation.py script and dataset description.pdf ex-
plaining your data collection/generation methodology.
5 ADALINE (Adaptive Linear Neuron) (1960)
5.1 Theory (20 points)
a) Discuss the ADALINE work introduced by Widrow & Hoff (1960) and provide the ADALINE learning
algorithm.
b) Show that ADALINE has a strong convergence guarantee to the global least-squares solution, and is
independent of data separability.
5.2 Implementation (10 points)
Implement the ADALINE with the following specifications:
1 class Adaline :
2 def __init__ ( self , learning_rate =1.0 , max_iterations =1000) :
3 """
4 Initialize the perceptron
5 learning_rate : step size for weight updates
6 max_iterations : maximum training iterations
7 """
8 pass
9
10 def fit ( self , X , y):
11 """
12 Train the perceptron
13 X: ( n_samples , n_features ) array
14 y: ( n_samples ,)
15 Returns : training history ( mistakes per epoch )
16 """
17 pass
18
19 def predict ( self , X):
20 """ Return predicted outputs """
21 pass
22
23 def score ( self , X , y):
24 """ Return mean square error """
25 pass
4


5.3 Experiments (10 points)
Using the dataset you have collected:
a) Plot the dataset;
b) Train Adaline and plot:
 Mean-Squared error vs. epoch
 Decision boundary in 2D (project to 2 principal components)
 Training and validation error vs. epoch
c) Experiment with learning rates   {0.01, 0.1, 1.0, 10.0}. Plot convergence curves and discuss the effect.
d) Vary the amount of training data from 10% to 100% in increments of 10%. Plot final accuracy vs.
training set size.
Deliverables: Code (adaline.py), plots, and analysis in report.
6 Multi-Layer Perceptron with Backpropagation
6.1 Theory (10 points)
a) Explain the vanishing gradient problem. Show mathematically why it occurs with sigmoid activation
in deep networks.
b) Instead of a fully connected feedforward network, if we were to consider a random graph model where
hidden units were arbitrarily dropped with probability 0.5, would applying the chain rule change? If
so, how?
6.2 Implementation (15 points)
Implement a flexible MLP framework:
1 class MLP :
2 def __init__ ( self , layer_sizes , activations ,
3 loss =  cross_entropy , learning_rate =0.01 ,
4 optimizer =  sgd , batch_size =32 ,
5 weight_init =  xavier , regularization = None ,
6 lambda_reg =0.01) :
7 """
8 layer_sizes : list of layer dimensions [ input , hidden1 , ... , output ]
9 activations : list of activation functions for each layer
10 loss :  mse  or  cross_entropy
11 optimizer :  sgd ,  momentum ,  adam
12 weight_init :  random ,  xavier , he
13 regularization : None , l1 , l2
14 """
15 pass
16
17 def forward ( self , X):
18 """ Forward pass , return activations at each layer """
19 pass
20
21 def backward ( self , X , y):
22 """ Backward pass , compute gradients """
23 pass
24
5


25 def fit ( self , X_train , y_train , X_val , y_val , epochs =100) :
26 """
27 Train with early stopping based on validation loss
28 Returns : training history
29 """
30 pass
Implement the following components separately for modularity:
 Activation functions (sigmoid, tanh, ReLU, leaky ReLU) and their derivatives
 Loss functions (MSE) and their derivatives
 Optimizers (SGD, momentum, Adam, Nesterov-Accelerated GD, AdaGrad, RMSProp, Muon)
 Weight initialization schemes (Xavier-Glorot, He)
6.3 Experiments: Architecture Ablation (15 points)
Using the dataset:
6.3.1 Depth Ablation
a) Fix hidden layer width at 64 neurons and vary depth:
 1 hidden layer: [input, 64, output]
 2 hidden layers: [input, 64, 64, output]
 3 hidden layers: [input, 64, 64, 64, output]
 4 hidden layers: [input, 64, 64, 64, 64, output]
b) For each architecture:
 Plot train/validation loss and accuracy curves
 Report final test accuracy
 Analyze training time per epoch
 Check for overfitting (train-val gap)
c) Create a summary plot: Test accuracy vs. depth
6.3.2 Width Ablation
1. Fix depth at 2 hidden layers and vary width in the follow ranges:
(a) [input dimension, 8, 8, output dimension]
(b) [input dimension, 16, 16, output dimension]
(c) [input dimension, 32, 32, output dimension]
(d) [input dimension, 64, 64, output dimension]
(e) [input dimension, 128, 128, output dimension]
(f) [input dimension, 256, 256, output dimension]
2. Plot test accuracy vs. number of parameters (log scale)
3. Discuss the width vs. depth performance relationship
6


6.3.3 Activation Function Comparison
a) Fix architecture at [input, 64, 64, output] and compare:
 Sigmoid in all layers
 Tanh in all layers
 ReLU in hidden layers, sigmoid in output
 Leaky ReLU in hidden layers, sigmoid in output
b) For each configuration:
 Plot loss curves
 Report gradient statistics (mean, std) during training
 Analyze dead neurons (for ReLU variants)
c) Verify vanishing gradients experimentally: measure gradient magnitude at each layer for sigmoid vs.
ReLU networks
6.4 Loss Function Analysis (10 points)
a) Come up 3 different loss functions to solve the problem and compare the results obtained using these
3 different loss functions. Discuss which loss is more suitable and why.
6.5 Optimizer Comparison (10 points)
Using the best architecture from above:
a) Use the following PyTorch optimizers and compare:
 SGD
 SGD with momentum ( = 0.9)
 Adam ( = 0.9,  = 0.999,  = 108)
1 2
 AdaGrad
 Nesterov accelerated GD
 RMSProp
 Muon
b) For each optimizer:
 Plot convergence curves (use same learning rate initially, then tune separately)
 Report time to reach 90% of best validation accuracy
 Analyze update directions (plot weight update magnitudes)
c) Learning rate sensitivity: for each optimizer, plot final accuracy vs. learning rate
7


6.6 Regularization (5 points)
a) Add L1 and L2 regularization to your MLP
b) Compare:
 No regularization
 L1 with   {0.001, 0.01, 0.1}
 L2 with   {0.001, 0.01, 0.1}
c) Plot train vs. validation accuracy for each
d) For L1: plot weight sparsity (fraction of weights < 103) vs.
6.7 Success Analysis (5 points)
a) Visualize learned features: extract hidden layer activations and project to 2D using PCA. Color by
class label.
b) Compare MLP performance vs. Adaline. Create a comparison table.
Deliverables: Code (mlp.py, activations.py, optimizers.py), extensive plots, and detailed analysis.
7 Neural Network Features for Kernel Methods (Advanced)
This section explores using neural networks as feature extractors for kernel-based methods.
7.1 Theory (10 points)
a) Review kernel methods: Define a valid kernel function. State Mercers theorem.
b) Explain how a neural networks hidden layer can be viewed as a learned feature representation (x).
How does this relate to the kernel trick?
7.2 Implementation (20 points)
7.2.1 Neural Network as Feature Extractor
a) Consider the best MLP that you have learnt so far
b) Extract features from the penultimate layer (before output):  (x) = h(L1)(x)
NN
c) Visualize NN features using t-SNE plot
d) Implement a kernel SVM using these neural features:
1 # Define kernel using NN features
2 def neural_kernel ( x_i , x_j , feature_extractor ):
3 """
4 k( x_i , x_j ) = < phi_NN ( x_i ) , phi_NN ( x_j ) >
5 """
6 phi_i = feature_extractor ( x_i )
7 phi_j = feature_extractor ( x_j )
8 return np . dot ( phi_i , phi_j )
e) Train kernel Support Vector Regression (SVR) with neural kernel on Phase 4 data
8


7.2.2 Comparison with Standard Kernels
Implement and compare the following kernels for SVR:
a) Linear kernel: k(x , x ) = xT x
i j i j
b) Polynomial kernel: k(x , x ) = (xT x + c)d (try d  {2, 3})
i j i j
c) RBF (Gaussian) kernel: k(x , x ) = exp(x  x 2) (try   {0.01, 0.1, 1.0})
i j i j
d) Neural kernel: k(x , x ) =  (x )T  (x )
i j NN i NN j
For each kernel:
 Report test accuracy
 Plot decision boundaries (project to 2D)
 Compute and visualize the kernel matrix K = k(x , x )
ij i j
9
