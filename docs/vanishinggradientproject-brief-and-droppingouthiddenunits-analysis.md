# Vanishinggradientproject Brief And Droppingouthiddenunits Analysis


6.1 - MLP - vanishing gradient problem and dropping out
hidden units
Technical focus:
(a) Explain the vanishing gradient problem. Show mathematically why it occurs with sigmoid
activation in deep networks.
(b) Instead of a fully connected feedforward network, if we were to consider a random graph
model where hidden units were arbitrarily dropped with probability 0.5, would applying the
chain rule change? If so, how?
Solution
(a) Vanishing Gradient Problem
The vanishing gradient problem occurs in deep neural networks when gradients become ex-
tremely small as they are propagated backward through many layers. As a result, early layers
learn very slowly or stop learning altogether.
Backpropagation in Deep Networks
Consider a deep network with L layers.
Let:
a(l) = (z(l))
z(l) = W (l)a(l1)
Using backpropagation, the gradient of the loss J with respect to weights in layer l is:
J
= (l)(a(l1))T
W (l)
where
(cid:16) (cid:17)
(l) = W (l+1)T (l+1)  (z(l))
Thus, the gradient at early layers involves repeated multiplication:
(cid:32) L (cid:33)
(cid:89)
(1) = W (k)T diag((z(k1))) (L)
k=2
Hence, gradients involve a product of many derivatives.
1


Sigmoid Derivative
The sigmoid activation function is:
1
(x) =
1 + ex
Its derivative is:
(x) = (x)(1  (x))
Since 0 < (x) < 1, we have:
1
0 < (x)
4
The maximum value 1 occurs at x = 0.
4
Why Vanishing Occurs
During backpropagation, gradients multiply many sigmoid derivatives:
L
(cid:89)
(z(l))
l=1
Since each term satisfies:
1
(z(l))
4
For a deep network:
(cid:18)
1
(cid:19)L
 0 as L
4
Therefore,
Gradients shrink exponentially with depth
This causes the vanishing gradient problem.
(b) Effect of Randomly Dropping Hidden Units (Probability 0.5)
Suppose each hidden unit is independently dropped with probability 0.5. This resembles a
random graph or dropout-like structure.
Let m(l) be a binary mask:
(l)
m  Bernoulli(0.5)
i
2


The forward pass becomes:
a(l) = m(l)  (z(l))
Does the Chain Rule Change?
The chain rule itself does not change. It still applies as:
J J y
=
x y x
However, the computational graph is modified.
Since:
a(l) = m(l)  (z(l))
the gradient becomes:
(cid:16) (cid:17)
(l) = W (l+1)T (l+1)  (z(l))  m(l)
(l)
Thus, units that were dropped (m = 0) receive zero gradient.
i
Conclusion for (b)
The chain rule does not change; only the computational graph .changes.
Dropping units multiplies gradients by a binary mask, effectively blocking gradient flow through
removed nodes, but the fundamental differentiation rule remains the same.
Final Conclusion
(a) Vanishing gradients occur because sigmoid derivatives are bounded above by 1 , causing
4
exponential decay of gradients across layers.
(b) Randomly dropping hidden units modifies the computational graph but does not alter the
chain rule itself. Gradients are simply masked where units are removed.
3
