# Universal Approximation Theorem


Universal Approximation Theorem
Universal Approximation Theorem
(a) Formal Statement of the UAT
Technical focus: State a formal version of the Universal Approximation Theorem for a feedforward
neural network with a single hidden layer and a sigmoid (or any non-polynomial continuous) ac-
tivation function. Clearly specify: the class of target functions being approximated, assumptions
on the domain, the notion of approximation and the norm used.
Analysis:
Let  : R  R be a non-constant, bounded, and continuous activation function (such as the
sigmoid function).
Let N be the class of functions representing single-hidden-layer feedforward neural networks
with a linear output layer, defined as:
(cid:40) (cid:12) (cid:41)
(cid:88) N (cid:12)
N = F (x) = v (wT x + b ) (cid:12) N  N, v , b  R, w  Rn
i i i (cid:12) i i i
(cid:12)
i=1
The Universal Approximation Theorem formally states that N is dense in C(K).
Specifically:
 Assumptions on the domain:
The domain K must be a compact subset of Rn (e.g., a closed and bounded interval like
[0, 1]n).
 Class of target functions:
The target functions f belong to C(K), which is the space of all continuous functions
defined on the compact domain K.
 Notion of approximation and norm:
1


The approximation is in the uniform sense. The norm used is the supremum norm
(or uniform norm / L norm), defined as

f  F  = sup |f (x)  F (x)|

xK
Formally:
For any continuous function f  C(K) and any error tolerance  > 0, there exists a neural
network F  N such that
sup |F (x)  f (x)| <
xK
(b) Proof Outline and Intuition
Technical focus: Outline the main idea behind a proof of the Universal Approximation Theorem.
A full formal proof is not required. Your answer should briefly explain the intuition or the key
theoretical argument (e.g., constructive approximation using basis-like functions, or a connection
to classical results from approximation theory such as the StoneWeierstrass theorem).
Analysis:
There are two primary ways to understand the proof of the Universal Approximation Theo-
rem: an abstract functional analysis approach and a constructive approach.
1. Constructive Approximation Intuition (Basis-like functions):
The most intuitive approach is constructive.
Imagine an activation function like a sigmoid. If we take the difference of two heavily
weighted, shifted sigmoids, we can approximate a rectangular step function (or a bump func-
tion) in one dimension.
By combining multiple step functions from different neurons, we can create a piecewise
constant approximation of any continuous function, much like a Riemann sum.
In multidimensional spaces, overlapping step functions from different directions can construct
isolated towers (or localized bumps) over small regions.
By adjusting the heights of these towers using the output weights v , we can uniformly
i
approximate any continuous surface to arbitrary precision over a compact domain.
2. Functional Analysis Theoretical Argument (Cybenko / Hornik):
Theoretical proofs often rely on classical approximation theory:
2


 Cybenkos Proof (1989) uses the Hahn-Banach Theorem and the Riesz Representa-
tion Theorem.
The core argument is a proof by contradiction: if the set of neural networks N is not dense
in C(K), there must exist a non-zero signed measure that annihilates all functions in N .
Cybenko showed that a continuous sigmoidal activation function is discriminatory, mean-
ing the only measure that can annihilate it is the zero measure, thus leading to a contra-
diction.
 Horniks Proof (1989) relies on the Stone-Weierstrass Theorem.
By defining a specific algebra of functions constructed from neural network units, one
can show that if the activation function is continuous and non-polynomial, the networks
function space can universally approximate continuous functions.
(c) Practical Limitations of the UAT
Technical focus: Discuss two important limitations of the Universal Approximation Theorem from
a practical machine learning perspective. For each limitation, explain why it is relevant when
designing or training neural networks.
Analysis:
While theoretically profound, the Universal Approximation Theorem has significant practical
limitations:
1. The bound on network size (Curse of Dimensionality):
Explanation:
The theorem is an existence theorem. It guarantees that a network exists to approximate
any function, but it places no upper bound on the number of hidden units (N ) required.
For highly complex or high-dimensional functions, the required number of hidden neurons
often grows exponentially with the input dimension n.
Relevance:
In practice, designing an exponentially wide network is computationally impossible.
We have finite memory and compute limits. Consequently, simply relying on a single massive
hidden layer is incredibly inefficient compared to using deeper, narrower networks which can often
represent the same complex functions with far fewer parameters.
2. Trainability and Optimization:
Explanation:
3


The theorem only guarantees that optimal weights (w , b , v ) exist ; it says absolutely nothing
i i i
about how to find them.
Finding these weights constitutes a highly non-convex optimization problem.
Relevance:
When training models using local optimization methods like Stochastic Gradient Descent
(SGD), the network might easily get stuck in bad local minima, saddle points, or suffer from
vanishing/exploding gradients.
Even if a perfect approximation configuration exists in the weight space, our training algo-
rithms are not guaranteed to converge to it.
(d) UAT with ReLU and Increased Depth
Technical focus: Briefly comment on how the Universal Approximation property changes when using
ReLU activations and allowing network depth to increase while keeping the width relatively small.
Analysis:
When transitioning from shallow, arbitrarily wide networks to deep, bounded-width net-
works (specifically using ReLU activations), the theoretical landscape shifts from width-based
universal approximation to depth-based universal approximation.
 Bounded Width, Arbitrary Depth:
It has been proven (e.g., Lu et al., 2017) that a deep neural network with ReLU activations
can universally approximate any continuous function on a compact domain if its width is
strictly strictly greater than the input dimension n (specifically, a width of at least n + 1
or n + 4 depending on the exact theoretical construction).
 Role of Depth vs. Width:
Instead of relying on an infinite number of neurons in a single layer to construct basis-
like functions, a bounded-width network uses its depth to recursively fold, partition, and
compose the input space.
Each layer combines linear transformations with the ReLU non-linearity to create an ex-
ponentially growing number of linear regions.
 Representational Efficiency:
Deep networks can be drastically more parameter-efficient.
Functions that would require an exponentially large width in a single-layer network can
4


often be approximated by deep ReLU networks with only a polynomial number of param-
eters.
This theoretically justifies the modern deep learning paradigm: depth is often more pow-
erful than width for extracting hierarchical abstractions.
5
