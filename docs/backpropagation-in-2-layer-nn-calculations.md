# Backpropagation In 2 Layer Nn Calculations


Derivation of learning parameter equations of
Backpropagation for a 2-Layer Neural Network
1. Network Architecture
Input layer: n nodes
1
Hidden layer: n nodes
2
Output layer: 1 node
Activation function: tanh()
Loss function: Mean Squared Error (MSE)
Parameters
W  Rn2n1, b  Rn21
1 1
W  R1n2, b  R
2 2
2. Forward Propagation
z = W x + b
1 1 1
a = tanh(z )
1 1
z = W a + b
2 2 1 2
y = tanh(z )
2
3. Loss Function
For one sample:
1
L = (y  y)2
2
For m samples:
m
1 (cid:88)
L = (y  y )2
i i
2m
i=1
1


4. Backpropagation
Derivative of tanh
d
tanh(z) = 1  tanh2(z)
dz
Output Layer Error
L
 = = (y  y)(1  y2)
2
z
2
Gradients for Output Layer
L
=  aT
W 2 1
2
L
=
2
b
2
Hidden Layer Error
L
 = = (W T  )  (1  a2)
1 z 2 2 1
1
Gradients for Hidden Layer
L
=  xT
1
W
1
L
=
1
b
1
5. Parameter Updates (Gradient Descent)
L
W  W
2 2
W
2
L
b  b
2 2
b
2
L
W  W
1 1
W
1
L
b  b
1 1
b
1
where  is the learning rate.
2


6. Xavier Initialization
To prevent vanishing/exploding gradients, weights are initialized as:
(cid:18) (cid:19)
1
W  N 0,
1
n
1
(cid:18) (cid:19)
1
W  N 0,
2
n
2
Biases are initialized as:
b = 0, b = 0
1 2
This keeps the variance of activations approximately constant across layers.
7. Final Compact Form
 = (y  y)(1  y2)
2
 = (W T  )  (1  a2)
1 2 2 1
L
=  aT
W 2 1
2
L
=  xT
1
W
1
3
