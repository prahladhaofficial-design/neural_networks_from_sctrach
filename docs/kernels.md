# Kernels


Kernels and Neural Network
Representations
Kernel methods:
a) Kernel Methods, Valid Kernels, and Mercers The-
orem
Technical focus: Review kernel methods: Define a valid kernel function. State Mercers theo-
rem.
Analysis:
1. Review of Kernel Methods:
Kernel methods are a class of algorithms for pattern analysis and machine learning (the
most famous being the Support Vector Machine). The core idea is to enable linear learn-
ing algorithms to learn non-linear decision boundaries or functions. This is achieved
by mapping the original input data x  X into a high-dimensional (sometimes infinite-
dimensional) feature space V via a non-linear mapping (x). Instead of explicitly com-
puting the coordinates in this high-dimensional space, kernel methods use the kernel
trick, calculating the inner product between the mapped data points directly using a
kernel function: k(x, x) = (x), (x) .
V
2. Definition of a Valid Kernel Function:
A function k : X  X  R is considered a valid kernel function (or a Mercer kernel) if
there exists a vector space V with an inner product ,  and a feature map  : X  V
V
such that for all x, x  X :
k(x, x) = (x), (x)
V
Practically, a kernel is valid if it is symmetric (k(x, x) = k(x, x)) and positive semi-
definite (PSD). The PSD condition means that for any finite set of points {x , x , . . . , x }
1 2 n
X , the n  n Gram matrix K with entries K = k(x , x ) is positive semi-definite. That
ij i j
is, for any vector c  Rn, we have cT Kc  0.
3. Mercers Theorem:
Mercers Theorem provides the formal mathematical justification for when a function can
be written as an inner product in a Hilbert space.
Statement: Let X be a compact subset of Rn, and let k : X  X  R be a continuous,
symmetric function. Then, k(x, y) can be represented as an absolutely and uniformly
convergent series:

(cid:88)
k(x, y) =   (x) (y)
i i i
i=1
1


where   0 are the non-negative eigenvalues and   L (X ) are the corresponding
i i 2
orthonormal eigenfunctions, if and only if k satisfies Mercers condition:
(cid:90) (cid:90)
k(x, y)c(x)c(y) dx dy  0
X X
for all square-integrable functions c  L (X ).
2
b) Neural Networks and the Kernel Trick
Technical focus: Explain how a neural networks hidden layer can be viewed as a learned feature
representation (x). How does this relate to the kernel trick?
Analysis:
1. Hidden Layers as Learned Feature Representations (x):
A standard feedforward neural network can be functionally decomposed into two parts:
a feature extractor and a linear predictor. Consider an L-layer neural network for a
classification or regression task. The layers 1 through L  1 consist of a series of linear
transformations followed by non-linear activation functions. We can bundle all these in-
termediate transformations into a single, highly parameterized non-linear function (x; ),
where  represents all the weights and biases of the hidden layers.
The output of the penultimate layer (the last hidden layer) is exactly this feature
vector: h = (x; ). The final output layer, y = W h + b, is simply a standard linear
model (like linear regression or logistic regression) operating on the new representation
h. Therefore, the neural networks hidden layers act as a feature map (x) that transforms
the raw input data into a new geometrical space where the classes are linearly separable
or the target variable is linearly predictable.
2. Relationship to the Kernel Trick:
Both neural networks and kernel methods aim to solve non-linear problems by mapping
inputs into a feature space where a linear model can succeed. However, their approaches
are fundamentally opposite:
 Kernel Trick (Fixed, Implicit Mapping): In kernel methods (like SVMs), the
feature mapping (x) is implicitly defined by a human-chosen kernel function (e.g.,
Polynomial, Radial Basis Function). We never directly compute (x); instead, we
compute the similarity k(x, y) in the high-dimensional space. The feature space is
mathematically rich (potentially infinite-dimensional) but strictly fixed and agnostic
to the training data distribution.
 Neural Networks (Learned, Explicit Mapping): In deep learning, the map-
ping (x; ) is explicitly computed. Instead of relying on a pre-defined mathematical
kernel, the network learns the mapping itself from the data via backpropagation.
The similarity between two inputs in a neural network can be viewed as an adaptive
kernel: k (x, y) = (x; )T (y; ).

Summary: The kernel trick relies on a priori, fixed, mathematically elegant pro-
jections into infinite-dimensional spaces. In contrast, neural networks act as adaptive
kernel machines that learn finite-dimensional, task-specific feature representations di-
rectly optimized from the data.
2
