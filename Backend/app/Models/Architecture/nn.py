"""
Variable-depth NeuralNetwork for binary DTI classification.

Supports configurable hidden layer sizes and inverted dropout.
He initialisation for all layers; ReLU for hidden, sigmoid for output.
"""

import numpy as np


# --- Activations ---

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)


# --- Loss ---

def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# ---------------------------------------------------------------------------

class NeuralNetwork:
    """
    Variable-depth network for binary DTI classification.

    Args:
        input_size_drug    (int):       drug fingerprint dimension
        input_size_protein (int):       protein feature dimension
        hidden_sizes       (list[int]): neuron counts per hidden layer, e.g. [64, 32]
        output_size        (int):       number of output neurons (1 for binary)
        learning_rate      (float):     SGD step size
        dropout_rate       (float):     inverted dropout keep probability (0.0 = off)
    """

    def __init__(self, input_size_drug, input_size_protein,
                 hidden_sizes, output_size, learning_rate=0.01,
                 dropout_rate=0.0):

        self.input_size   = input_size_drug + input_size_protein
        self.lr           = learning_rate
        self.dropout_rate = dropout_rate
        self.training     = True   # toggled to False during predict()

        layer_sizes = [self.input_size] + list(hidden_sizes) + [output_size]
        self.weights = []
        self.biases  = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            W = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(W)
            self.biases.append(b)

        # Forward-pass cache (populated each call)
        self.zs          = []   # pre-activation values
        self.activations = []   # activations[0] = input X; activations[-1] = y_hat
        self.masks       = []   # inverted dropout masks (None where not applied)

    # -----------------------------------------------------------------------

    def forward(self, X_drug, X_protein):
        X = np.concatenate((X_drug, X_protein), axis=1)
        self.zs          = []
        self.activations = [X]
        self.masks       = []

        n_hidden = len(self.weights) - 1   # all layers except output

        for i in range(n_hidden):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            a = relu(z)

            if self.training and self.dropout_rate > 0.0:
                # Inverted dropout: scale at train time so predict() needs no adjustment
                mask = (np.random.rand(*a.shape) > self.dropout_rate) / (1.0 - self.dropout_rate)
                a = a * mask
            else:
                mask = None

            self.zs.append(z)
            self.activations.append(a)
            self.masks.append(mask)

        # Output layer (sigmoid, no dropout)
        z_out = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        y_hat = sigmoid(z_out)
        self.zs.append(z_out)
        self.activations.append(y_hat)
        self.masks.append(None)

        return y_hat

    def backward(self, y):
        n_samples = y.shape[0]
        n_layers  = len(self.weights)

        dWs = [None] * n_layers
        dbs = [None] * n_layers

        # Output layer: BCE + sigmoid simplification → delta = y_hat - y
        delta   = self.activations[-1] - y
        dWs[-1] = np.dot(self.activations[-2].T, delta) / n_samples
        dbs[-1] = np.sum(delta, axis=0, keepdims=True)  / n_samples

        # Hidden layers (back to front)
        for i in range(n_layers - 2, -1, -1):
            grad = np.dot(delta, self.weights[i + 1].T)   # w.r.t. dropped activation
            if self.masks[i] is not None:
                grad = grad * self.masks[i]                # through inverted dropout
            delta   = grad * relu_derivative(self.zs[i])  # through ReLU
            dWs[i]  = np.dot(self.activations[i].T, delta) / n_samples
            dbs[i]  = np.sum(delta, axis=0, keepdims=True)  / n_samples

        return dWs, dbs

    def update_weights(self, dWs, dbs):
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * dWs[i]
            self.biases[i]  -= self.lr * dbs[i]

    def predict(self, X_drug, X_protein):
        self.training = False
        out = self.forward(X_drug, X_protein)
        self.training = True
        return out
