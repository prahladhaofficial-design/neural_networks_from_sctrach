# Gradient Descent, Softmax, and Hinge-Loss Analysis


One Step of Gradient Descent in a Deep Neural Network:
Softmax and Multiclass Hinge Loss Analysis
March 10, 2026
Project Focus
3. One Step of Gradient Descent in a Deep Neural Network
Consider the following feedforward neural network with three affine layers (four layers including
the input), ReLU activations in the hidden layers, and two output classes.
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
There are two classes, and the true class label is y = 1 (using 1-based indexing). The learning
rate is  = 0.1. Assume the derivative of the ReLU function is
(cid:40)
1, t > 0
ReLU(t) =
0, t  0
Part A: Softmax with Cross-Entropy Loss
The softmax function is defined as q =
esi
and the cross-entropy loss for a single example
i (cid:80) esj
j
is L =  log q
CE y
(a) Perform the forward pass and compute z(1), a(1), z(2), a(2), s, and the softmax probabilities
q.
(b) Compute the cross-entropy loss L .
CE
1


(c) Compute the gradient of the loss with respect to the logits s.
(d) Using backpropagation, compute the gradients with respect to all weights and biases:
 ,  ,  ,  ,  ,  .
W (3) b(3) W (2) b(2) W (1) b(1)
(e) Perform one gradient descent update step for all parameters and report the updated
values.
Part B: Multiclass Hinge Loss
(cid:80)
The multiclass hinge loss is defined as L = max(0, s  s + 1)
hinge j=y j y
(a) Compute the hinge loss for the given example.
(b) Compute the gradient of the hinge loss with respect to the logits s.
(c) Using backpropagation, compute the gradients with respect to all weights and biases.
(d) Perform one gradient descent update step for all parameters and report the updated
values.
Solution
Part A: Softmax with Cross-Entropy Loss
(a) Perform the forward pass and compute z(1), a(1), z(2), a(2), s, and the softmax prob-
abilities q.
1. Layer 1:
(cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21)
1 0 1 0 1
z(1) = W (1)x + b(1) = + =
0 1 1 0 1
(cid:20) (cid:21) (cid:20) (cid:21)
max(0, 1) 1
a(1) = ReLU(z(1)) = =
max(0, 1) 1
2. Layer 2:
(cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21)
1 1 1 0 1(1) + (1)(1) 0
z(2) = W (2)a(1) + b(2) = + = =
1 1 1 0 1(1) + 1(1) 2
(cid:20) (cid:21) (cid:20) (cid:21)
max(0, 0) 0
a(2) = ReLU(z(2)) = =
max(0, 2) 2
3. Layer 3 (Logits):
(cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21)
1 0 0 0 0
s = W (3)a(2) + b(3) = + =
0 1 2 0 2
So, s = 0 and s = 2.
1 2
4. Softmax probabilities:
es1 1
q = =  0.1192
1 es1 + es2 1 + e2
es2 e2
q = =  0.8808
2 es1 + es2 1 + e2
2


(cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21)
1 1 0 0
z(1) = , a(1) = , z(2) = , a(2) =
1 1 2 2
(cid:34) (cid:35)
(cid:20) (cid:21) 1 (cid:20) (cid:21)
0 0.1192
s = , q = 1+e2
2 e2 0.8808
1+e2
(b) Compute the cross-entropy loss L .
CE
The true class is y = 1. The cross-entropy loss is given by:
L =  log q =  log q
CE y 1
(cid:18) (cid:19)
1
=  log = log(1 + e2)
1 + e2
Using e  2.718, e2  7.389: L = log(8.389)  2.1269
CE
L = log(1 + e2)  2.1269
CE
(c) Compute the gradient of the loss with respect to the logits s.
(cid:16) (cid:17)
Derivation: Let L
CE
=  log q
y
= s
y
+ log (cid:80)
j
esj . Applying the chain rule to derive the
gradient with respect to the i-th logit s :
i

  L s C
i
E =   s
i
s y + log (cid:88)
j
esj  =  iy + (cid:80)
j
1 esj    s
i
 (cid:88)
j
esj
esi
=  + = q
iy (cid:80) esj i iy
j
where  is 1 if i = y, 0 otherwise.
iy
For y = 1:
L 1 1  (1 + e2) e2
CE
= q  1 =  1 = =  0.1192  1 = 0.8808
s 1 1 + e2 1 + e2 1 + e2
1
L e2
CE
= q  0 =  0.8808
s 2 1 + e2
2
(cid:20) (cid:21)
q  1
Thus,  L = 1 . Let p = e2  0.8808.
s q 1+e2
2
(cid:34) (cid:35)
e2 (cid:20)
0.8808
(cid:21)
 L = 1+e2
s e2 0.8808
1+e2
(d) Using backpropagation, compute the gradients with respect to all weights and
biases.
Derivation of formulas using the Multivariate Chain Rule: For an affine layer k, z(k) = W (k)a(k1)+
b(k). Using matrix calculus, the differential is dz(k) = dW (k)a(k1) + W (k)da(k1) + db(k). From
this, the local gradients are:
 L = ( L)(a(k1))T
W (k) z(k)
 L =  L
b(k) z(k)
 L = (W (k))T  L
a(k1) z(k)
3


For the ReLU activation a(k) = ReLU(z(k)), the gradient is applied element-wise:
 L =  L  ReLU(z(k))
z(k) a(k)
Calculations:
(cid:20) (cid:21)
0.8808
Layer 3: s = z(3), so  L =  L  .
z(3) s 0.8808
(cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21)
 L = ( L)(a(2))T = p (cid:2) 0 2 (cid:3) = 0 2p  0 1.7616
W (3) s p 0 2p 0 1.7616
(cid:20) (cid:21) (cid:20) (cid:21)
p 0.8808
 L =  L =
b(3) s p 0.8808
(cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21)
1 0 p p 0.8808
 L = (W (3))T  L = =
a(2) s 0 1 p p 0.8808
(cid:20) (cid:21)
0
Layer 2: We know z(2) = . From the given ReLU derivative, ReLU(0) = 0 and ReLU(2) =
2
(cid:20) (cid:21)
0
1. Thus, ReLU(z(2)) = .
1
(cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21)
p 0 0 0
 L =  L  ReLU(z(2)) =  =
z(2) a(2) p 1 p 0.8808
(cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21)
 L = ( L)(a(1))T = 0 (cid:2) 1 1 (cid:3) = 0 0  0 0
W (2) z(2) p p p 0.8808 0.8808
(cid:20) (cid:21) (cid:20) (cid:21)
0 0
 L =  L =
b(2) z(2) p 0.8808
(cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21)
1 1 0 0(1) + p(1) p 0.8808
 L = (W (2))T  L = = =
a(1) z(2) 1 1 p 0(1) + p(1) p 0.8808
(cid:20) (cid:21) (cid:20) (cid:21)
1 1
Layer 1: We know z(1) = , so ReLU(z(1)) = .
1 1
(cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21)
p 1 p 0.8808
 L =  L  ReLU(z(1)) =  =
z(1) a(1) p 1 p 0.8808
(cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21)
 L = ( L)xT = p (cid:2) 1 1 (cid:3) = p p  0.8808 0.8808
W (1) z(1) p p p 0.8808 0.8808
(cid:20) (cid:21) (cid:20) (cid:21)
p 0.8808
 L =  L =
b(1) z(1) p 0.8808
(cid:20) (cid:21) (cid:20) (cid:21)
0 1.7616 0.8808
  ,
W (3) 0 1.7616 b(3) 0.8808
(cid:20) (cid:21) (cid:20) (cid:21)
0 0 0
  ,
W (2) 0.8808 0.8808 b(2) 0.8808
(cid:20) (cid:21) (cid:20) (cid:21)
0.8808 0.8808 0.8808
  ,
W (1) 0.8808 0.8808 b(1) 0.8808
(e) Perform one gradient descent update step for all parameters and report the
updated values.
4


The gradient descent update rule is  =    L, with  = 0.1.
new old
(cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21)
1 0 0 1.7616 1 0 0 0.1762 1 0.1762
W (3) =  0.1 =  =
new 0 1 0 1.7616 0 1 0 0.1762 0 0.8238
(cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21)
0 0.8808 0.0881
b(3) =  0.1 =
new 0 0.8808 0.0881
(cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21)
1 1 0 0 1 1 0 0 1 1
W (2) =  0.1 =  =
new 1 1 0.8808 0.8808 1 1 0.0881 0.0881 0.9119 0.9119
(cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21)
0 0 0
b(2) =  0.1 =
new 0 0.8808 0.0881
(cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21)
1 0 0.8808 0.8808 1 0 0.0881 0.0881 0.9119 0.0881
W (1) =  0.1 =  =
new 0 1 0.8808 0.8808 0 1 0.0881 0.0881 0.0881 0.9119
(cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21)
0 0.8808 0.0881
b(1) =  0.1 =
new 0 0.8808 0.0881
(cid:20) (cid:21) (cid:20) (cid:21)
1 0.1762 0.0881
W (3) = , b(3) =
new 0 0.8238 new 0.0881
(cid:20) (cid:21) (cid:20) (cid:21)
1 1 0
W (2) = , b(2) =
new 0.9119 0.9119 new 0.0881
(cid:20) (cid:21) (cid:20) (cid:21)
0.9119 0.0881 0.0881
W (1) = , b(1) =
new 0.0881 0.9119 new 0.0881
Part B: Multiclass Hinge Loss
(a) Compute the hinge loss for the given example.
The true class is y = 1. The logits are s = 0 and s = 2. The multiclass hinge loss formulation
1 2
is:
(cid:88)
L = max(0, s  s + 1)
hinge j y
j=y
For j = 2:
L = max(0, s  s + 1)
hinge 2 1
= max(0, 2  0 + 1)
= max(0, 3) = 3
L = 3
hinge
(b) Compute the gradient of the hinge loss with respect to the logits s.
(cid:80)
Derivation: Let L = max(0, s  s + 1). The partial derivative with respect to an
hinge j=y j y
incorrect class logit s (i = y) is:
i
(cid:40)
L  1 if s  s + 1 > 0
hinge i y
= max(0, s  s + 1) =
i y
s i s i 0 otherwise
The partial derivative with respect to the correct class logit s is the negative sum of the above
y
active margins:
(cid:40)
L hinge (cid:88)  (cid:88) 1 if s j  s y + 1 > 0
= max(0, s  s + 1) =
j y
s y s y 0 otherwise
j=y j=y
5


Applying this to our case (y = 1): Since s  s + 1 = 2  0 + 1 = 3 > 0:
2 1
L
hinge
= 1
s
2
L
hinge
= 1
s
1
(cid:20) (cid:21)
1
 L =
s 1
(c) Using backpropagation, compute the gradients with respect to all weights and
biases.
We use the exact same chain rule equations as derived in Part A(d). The only difference is
(cid:20) (cid:21)
1
the new upstream gradient  L = . All forward pass activations and matrices remain
s 1
unchanged.
Layer 3:
(cid:20) (cid:21) (cid:20) (cid:21)
 L = ( L)(a(2))T = 1 (cid:2) 0 2 (cid:3) = 0 2
W (3) s 1 0 2
(cid:20) (cid:21)
1
 L =  L =
b(3) s 1
(cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21)
1 0 1 1
 L = (W (3))T  L = =
a(2) s 0 1 1 1
(cid:20) (cid:21)
0
Layer 2: ReLU(z(2)) = .
1
(cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21)
1 0 0
 L =  L  ReLU(z(2)) =  =
z(2) a(2) 1 1 1
(cid:20) (cid:21) (cid:20) (cid:21)
 L = ( L)(a(1))T = 0 (cid:2) 1 1 (cid:3) = 0 0
W (2) z(2) 1 1 1
(cid:20) (cid:21)
0
 L =  L =
b(2) z(2) 1
(cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21)
1 1 0 1
 L = (W (2))T  L = =
a(1) z(2) 1 1 1 1
(cid:20) (cid:21)
1
Layer 1: ReLU(z(1)) = .
1
(cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21)
1 1 1
 L =  L  ReLU(z(1)) =  =
z(1) a(1) 1 1 1
(cid:20) (cid:21) (cid:20) (cid:21)
 L = ( L)xT = 1 (cid:2) 1 1 (cid:3) = 1 1
W (1) z(1) 1 1 1
(cid:20) (cid:21)
1
 L =  L =
b(1) z(1) 1
6


(cid:20) (cid:21) (cid:20) (cid:21)
0 2 1
 = ,  =
W (3) 0 2 b(3) 1
(cid:20) (cid:21) (cid:20) (cid:21)
0 0 0
 = ,  =
W (2) 1 1 b(2) 1
(cid:20) (cid:21) (cid:20) (cid:21)
1 1 1
 = ,  =
W (1) 1 1 b(1) 1
(d) Perform one gradient descent update step for all parameters and report the
updated values.
Using the update rule  =   0.1 L:
new old
(cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21)
1 0 0 2 1 0 0 0.2 1 0.2
W (3) =  0.1 =  =
new 0 1 0 2 0 1 0 0.2 0 0.8
(cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21)
0 1 0.1
b(3) =  0.1 =
new 0 1 0.1
(cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21)
1 1 0 0 1 1 0 0 1 1
W (2) =  0.1 =  =
new 1 1 1 1 1 1 0.1 0.1 0.9 0.9
(cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21)
0 0 0
b(2) =  0.1 =
new 0 1 0.1
(cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21)
1 0 1 1 1 0 0.1 0.1 0.9 0.1
W (1) =  0.1 =  =
new 0 1 1 1 0 1 0.1 0.1 0.1 0.9
(cid:20) (cid:21) (cid:20) (cid:21) (cid:20) (cid:21)
0 1 0.1
b(1) =  0.1 =
new 0 1 0.1
(cid:20) (cid:21) (cid:20) (cid:21)
1 0.2 0.1
W (3) = , b(3) =
new 0 0.8 new 0.1
(cid:20) (cid:21) (cid:20) (cid:21)
1 1 0
W (2) = , b(2) =
new 0.9 0.9 new 0.1
(cid:20) (cid:21) (cid:20) (cid:21)
0.9 0.1 0.1
W (1) = , b(1) =
new 0.1 0.9 new 0.1
7
