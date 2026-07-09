# Adalineanalysis


5.1 ADALINE (Adaptive Linear Neuron) (1960) - THEORY
Technical focus:
(a) Discuss the ADALINE work introduced by Widrow & Hoff (1960) and provide the ADA-
LINE learning algorithm.
(b) Show that ADALINE has a strong convergence guarantee to the global least-squares solu-
tion, and is independent of data separability.
Solution
(a) ADALINE and its Learning Algorithm
ADALINE (Adaptive Linear Neuron), introduced by Widrow and Hoff (1960), is a single-layer
linear neural network model. Its key contribution was the Least Mean Squares (LMS) learning
rule, which minimizes a continuous Mean Squared Error (MSE) objective using gradient descent.
Model
Given input vector:
x = [x , x , . . . , x ]T
1 2 d
Weight vector:
w = [w , w , . . . , w ]T
1 2 d
Bias: b
Net input:
z = wT x + b
Output (linear activation):
y = z
Objective Function
1
J (w) = (t  y)2
2
For N samples:
N
1 (cid:88)
J (w) = (t  wT x )2
i i
2
i=1
1


Derivation of Learning Rule
For one sample:
1
J (w) = (t  wT x)2
2
Gradient:
J (w) = (t  wT x)x
Let error:
e = t  y
Thus,
J (w) = ex
Using gradient descent:
w  w  J (w)
w  w + ex
Bias update:
b  b + e
ADALINE Learning Rule (WidrowHoff Rule):
w  w + ex
b  b + e
where e = t  y
(b) Convergence to Global Least-Squares Solution
Let the dataset in matrix form be:
X  RNd, w  Rd, t  RN
Model output:
y = Xw
2


Cost Function
1
J (w) = t  Xw2
2
Expand:
1
J (w) = (t  Xw)T (t  Xw)
2
Gradient and Hessian
J (w) = XT Xw  XT t
2J (w) = XT X
Since XT X is positive semidefinite, the cost function is convex. If X has full column rank, it is
strictly convex.
Global Least-Squares Solution
Set gradient to zero:
XT Xw = XT t
w = (XT X)1XT t
Convergence Condition
Batch gradient descent update:
w = w  (XT Xw  XT t)
k+1 k k
Gradient descent converges if:
2
0 <  <

max
where  is the largest eigenvalue of XT X.
max
Under this condition:
w  w
k
3


Independence from Data Separability
ADALINE minimizes squared error:
min t  Xw2
w
Even if the data are not linearly separable:
 The least-squares solution exists.
 The objective remains convex.
 Gradient descent converges.
Therefore, convergence does not depend on linear separability.
Final Conclusion
ADALINE minimizes a convex quadratic MSE objective. Because the objective is convex,
gradient descent converges to the unique global least-squares solution for a sufficiently small
learning rate. This convergence guarantee is independent of whether the data are linearly
separable.
4
