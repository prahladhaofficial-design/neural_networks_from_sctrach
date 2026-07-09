import numpy as np
import time

from core.activations import ACTIVATIONS
from core.losses import LOSSES
from core.optimizers import *
from core.weight_initializations import WEIGHT_INITIALIZERS


class MLP:

    def __init__(self , layer_sizes , activations ,
                 loss='mse' , learning_rate=0.01,
                 optimizer='sgd' , batch_size=32 ,
                 weight_init='xavier' , regularization=None ,
                 lambda_reg=0.01):

        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("Number of activations must equal number of layers-1")

        if weight_init not in WEIGHT_INITIALIZERS:
            raise ValueError("Invalid weight initialization")

        self.layer_sizes = layer_sizes
        self.activations = activations
        self.lr = learning_rate
        self.batch_size = batch_size
        self.regularization = regularization
        self.lambda_reg = lambda_reg

        self.loss , self.loss_deriv = LOSSES[loss]

        if optimizer == "sgd":
            self.optimizer = SGD()
        elif optimizer == "momentum":
            self.optimizer = Momentum()
        elif optimizer == "adam":
            self.optimizer = Adam()
        elif optimizer == "adagrad":
            self.optimizer = AdaGrad()
        elif optimizer == "nesterov":
            self.optimizer = Nesterov()
        elif optimizer == "rmsprop":
            self.optimizer = RMSProp()
        elif optimizer == "muon":
            self.optimizer = Muon()
        else:
            raise ValueError("Unsupported Optimizer")

        init_fn = WEIGHT_INITIALIZERS[weight_init]

        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes)-1):

            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]

            W = init_fn(in_dim , out_dim)
            b = np.zeros((1 , out_dim))

            self.weights.append(W)
            self.biases.append(b)

    # -------------------------------------------------

    def forward(self , X):

        activations = [X]
        self.zs = []

        A = X

        for i in range(len(self.weights)):

            W = self.weights[i]
            b = self.biases[i]

            Z = A @ W + b
            self.zs.append(Z)

            act_fn , _ = ACTIVATIONS[self.activations[i]]

            A = act_fn(Z)

            activations.append(A)

        self.activations_cache = activations

        return activations

    # -------------------------------------------------

    def backward(self , X , y):

        activations = self.forward(X)

        grads_W = []
        grads_b = []

        self.grad_stats = []
        self.grad_magnitudes = []

        m = X.shape[0]

        delta = self.loss_deriv(y , activations[-1])

        for i in reversed(range(len(self.weights))):

            act_fn , act_deriv = ACTIVATIONS[self.activations[i]]

            delta = delta * act_deriv(self.zs[i])

            dW = (activations[i].T @ delta) / m
            db = np.sum(delta , axis=0 , keepdims=True) / m

            if self.regularization == "l2":
                dW += self.lambda_reg * self.weights[i]

            if self.regularization == "l1":
                dW += self.lambda_reg * np.sign(self.weights[i])

            grads_W.insert(0 , dW)
            grads_b.insert(0 , db)

            self.grad_stats.append({
                "mean": np.mean(dW),
                "std": np.std(dW)
            })

            self.grad_magnitudes.append(np.linalg.norm(dW))

            delta = delta @ self.weights[i].T

        return grads_W , grads_b

    # -------------------------------------------------

    def predict(self , X):
        return self.forward(X)[-1]

    # -------------------------------------------------

    def evaluate(self , X , y):

        preds = self.predict(X)
        return self.loss(y , preds)

    # -------------------------------------------------

    def count_parameters(self):

        total = 0

        for W , b in zip(self.weights , self.biases):

            total += W.size
            total += b.size

        return total

    # -------------------------------------------------

    def dead_neurons(self , X):

        activations = self.forward(X)

        dead_counts = []

        for A in activations[1:-1]:

            dead = np.sum(np.all(A <= 0 , axis=0))
            dead_counts.append(dead)

        return dead_counts

    # -------------------------------------------------

    def weight_sparsity(self):

        total_weights = 0
        small_weights = 0

        for W in self.weights:

            total_weights += W.size

            small_weights += np.sum(np.abs(W) < 1e-3)

        sparsity = small_weights / total_weights

        return sparsity

    # -------------------------------------------------

    def extract_features(self, X, layer_index=1):
        
        """
        Extract activations from a hidden layer
        
        layer_index:
            0 = input
            1 = first hidden layer
            2 = second hidden layer
            ...
        """

        activations = self.forward(X)

        if layer_index >= len(activations):
            raise ValueError("Layer index out of range")

        return activations[layer_index]


    def fit(self , X_train , y_train ,
            X_val=None , y_val=None ,
            epochs=100 , use_validation=False):

        history = {
            "train_loss": [],
            "epoch_time": [],
            "update_magnitude": []
        }

        if use_validation:
            history["val_loss"] = []

        n = X_train.shape[0]

        for epoch in range(epochs):

            start = time.time()

            epoch_update_mag = 0

            indices = np.random.permutation(n)

            X_train = X_train[indices]
            y_train = y_train[indices]

            for i in range(0 , n , self.batch_size):

                X_batch = X_train[i:i+self.batch_size]
                y_batch = y_train[i:i+self.batch_size]

                grads_W , grads_b = self.backward(X_batch , y_batch)

                params = self.weights + self.biases
                grads = grads_W + grads_b

                old_params = [p.copy() for p in params]

                self.optimizer.update(params , grads , self.lr)

                new_params = self.weights + self.biases

                for old , new in zip(old_params , new_params):
                    epoch_update_mag += np.linalg.norm(new - old)

            train_loss = self.evaluate(X_train , y_train)

            history["update_magnitude"].append(epoch_update_mag)

            if use_validation:

                val_loss = self.evaluate(X_val , y_val)
                history["val_loss"].append(val_loss)

            history["train_loss"].append(train_loss)

            epoch_time = time.time() - start
            history["epoch_time"].append(epoch_time)

            if use_validation:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"Time: {epoch_time:.3f}s")
            else:
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Time: {epoch_time:.3f}s")

        return history
