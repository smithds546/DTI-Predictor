import numpy as np


# --- Activations ---

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

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
    Single hidden-layer network for binary DTI classification.

    Args:
        init_method (str):        'small' (0.01) or 'he' (sqrt(2/fan_in))
        hidden_activation (str):  'sigmoid' or 'relu'
    """

    def __init__(self, input_size_drug, input_size_protein,
                 hidden_size, output_size, learning_rate=0.01,
                 init_method='small', hidden_activation='sigmoid'):

        self.input_size        = input_size_drug + input_size_protein
        self.learning_rate     = learning_rate
        self.hidden_activation = hidden_activation

        if init_method == 'he':
            self.W1 = np.random.randn(self.input_size, hidden_size) * np.sqrt(2.0 / self.input_size)
            self.W2 = np.random.randn(hidden_size, output_size)     * np.sqrt(2.0 / hidden_size)
        else:  # 'small'
            self.W1 = np.random.randn(self.input_size, hidden_size) * 0.01
            self.W2 = np.random.randn(hidden_size, output_size)     * 0.01

        self.b1 = np.zeros((1, hidden_size))
        self.b2 = np.zeros((1, output_size))

        self.X_concat = None
        self.z1 = None;  self.a1 = None
        self.z2 = None;  self.y_hat = None

    def forward(self, X_drug, X_protein):
        self.X_concat = np.concatenate((X_drug, X_protein), axis=1)

        self.z1 = np.dot(self.X_concat, self.W1) + self.b1
        self.a1 = relu(self.z1) if self.hidden_activation == 'relu' else sigmoid(self.z1)

        self.z2    = np.dot(self.a1, self.W2) + self.b2
        self.y_hat = sigmoid(self.z2)   # output always sigmoid
        return self.y_hat

    def backward(self, y):
        n_samples = y.shape[0]

        # BCE + sigmoid output: dL/dz2 = y_hat - y
        delta_output = self.y_hat - y
        dW2 = np.dot(self.a1.T, delta_output)
        db2 = np.sum(delta_output, axis=0, keepdims=True)

        hidden_grad = relu_derivative(self.z1) if self.hidden_activation == 'relu' \
                      else sigmoid_derivative(self.z1)
        delta_hidden = np.dot(delta_output, self.W2.T) * hidden_grad
        dW1 = np.dot(self.X_concat.T, delta_hidden)
        db1 = np.sum(delta_hidden, axis=0, keepdims=True)

        dW1 /= n_samples;  db1 /= n_samples
        dW2 /= n_samples;  db2 /= n_samples
        return dW1, db1, dW2, db2

    def update_weights(self, dW1, db1, dW2, db2):
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def predict(self, X_drug, X_protein):
        return self.forward(X_drug, X_protein)
