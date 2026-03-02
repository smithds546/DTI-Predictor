import numpy as np


# --- Activation Functions and their Derivatives ---

def relu(x):
    """
    ReLU activation function: f(x) = max(0, x)
    Used in hidden layers to avoid vanishing gradients.
    """
    return np.maximum(0, x)


def relu_derivative(x):
    """
    Derivative of ReLU: f'(x) = 1 if x > 0 else 0
    """
    return (x > 0).astype(float)


def sigmoid(x):
    """
    Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    Used only on the output layer to produce a probability in (0, 1).
    """
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


# --- Loss Function ---

def binary_cross_entropy(y_true, y_pred):
    """
    Binary Cross-Entropy loss: L = -(1/N) * sum(y*log(p) + (1-y)*log(1-p))
    Predictions are clipped to avoid log(0).
    """
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


class NeuralNetwork:
    """
    Two-hidden-layer neural network for binary DTI classification.

    Architecture:
    [Drug | Protein] -> Concatenate -> Hidden1 (ReLU) -> Hidden2 (ReLU) -> Output (Sigmoid)

    Design choices vs. basic.py:
      - ReLU in hidden layers avoids vanishing gradients from stacked sigmoids.
      - A second hidden layer increases representational capacity.
      - BCE loss is appropriate for binary classification (vs. MSE in the baseline).
      - He initialisation (sqrt(2/fan_in)) suits ReLU units.
      - BCE + sigmoid output simplifies the output delta to (y_hat - y).
    """

    def __init__(self, input_size_drug, input_size_protein,
                 hidden_size_1, hidden_size_2, output_size, learning_rate=0.01):
        """
        Args:
            input_size_drug (int):    Number of drug features.
            input_size_protein (int): Number of protein features.
            hidden_size_1 (int):      Neurons in the first hidden layer.
            hidden_size_2 (int):      Neurons in the second hidden layer.
            output_size (int):        Number of output neurons (1 for binary).
            learning_rate (float):    Step size for gradient descent.
        """
        self.input_size = input_size_drug + input_size_protein
        self.learning_rate = learning_rate

        # He initialisation: scale = sqrt(2 / fan_in), suitable for ReLU
        self.W1 = np.random.randn(self.input_size, hidden_size_1) * np.sqrt(2.0 / self.input_size)
        self.b1 = np.zeros((1, hidden_size_1))

        self.W2 = np.random.randn(hidden_size_1, hidden_size_2) * np.sqrt(2.0 / hidden_size_1)
        self.b2 = np.zeros((1, hidden_size_2))

        self.W3 = np.random.randn(hidden_size_2, output_size) * np.sqrt(2.0 / hidden_size_2)
        self.b3 = np.zeros((1, output_size))

        # Cache intermediate values for backprop
        self.X_concat = None
        self.z1 = None;  self.a1 = None   # hidden layer 1
        self.z2 = None;  self.a2 = None   # hidden layer 2
        self.z3 = None;  self.y_hat = None # output layer

    def forward(self, X_drug, X_protein):
        """
        Forward propagation.

        Args:
            X_drug (np.ndarray):    Shape (n_samples, input_size_drug).
            X_protein (np.ndarray): Shape (n_samples, input_size_protein).

        Returns:
            np.ndarray: Predicted probabilities, shape (n_samples, 1).
        """
        self.X_concat = np.concatenate((X_drug, X_protein), axis=1)

        # Hidden layer 1 — ReLU
        self.z1 = np.dot(self.X_concat, self.W1) + self.b1
        self.a1 = relu(self.z1)

        # Hidden layer 2 — ReLU
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = relu(self.z2)

        # Output layer — Sigmoid (probability)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.y_hat = sigmoid(self.z3)

        return self.y_hat

    def backward(self, y):
        """
        Backpropagation. Computes gradients for all three weight matrices.

        BCE + sigmoid output simplifies the output delta:
            dBCE/dz3 = y_hat - y  (sigmoid derivative cancels out)

        ReLU hidden layers use relu_derivative(z) as the local gradient.

        Args:
            y (np.ndarray): True labels, shape (n_samples, 1).

        Returns:
            Tuple of gradients: (dW1, db1, dW2, db2, dW3, db3)
        """
        n_samples = y.shape[0]

        # --- Output layer ---
        # dBCE/dz3 = y_hat - y  (BCE + sigmoid simplification)
        delta3 = self.y_hat - y

        dW3 = np.dot(self.a2.T, delta3)
        db3 = np.sum(delta3, axis=0, keepdims=True)

        # --- Hidden layer 2 ---
        delta2 = np.dot(delta3, self.W3.T) * relu_derivative(self.z2)

        dW2 = np.dot(self.a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)

        # --- Hidden layer 1 ---
        delta1 = np.dot(delta2, self.W2.T) * relu_derivative(self.z1)

        dW1 = np.dot(self.X_concat.T, delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)

        # Normalise by batch size
        dW1 /= n_samples;  db1 /= n_samples
        dW2 /= n_samples;  db2 /= n_samples
        dW3 /= n_samples;  db3 /= n_samples

        return dW1, db1, dW2, db2, dW3, db3

    def update_weights(self, dW1, db1, dW2, db2, dW3, db3):
        """Gradient descent weight update."""
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3

    def train(self, X_drug_train, X_protein_train, y_train, epochs):
        """
        Full-batch gradient descent training loop.

        Args:
            X_drug_train (np.ndarray):    Drug training features.
            X_protein_train (np.ndarray): Protein training features.
            y_train (np.ndarray):         Binary labels, shape (n_samples, 1).
            epochs (int):                 Maximum number of training iterations.
        """
        print(f"Starting training for {epochs} epochs...")
        for i in range(epochs):
            y_hat = self.forward(X_drug_train, X_protein_train)
            loss  = binary_cross_entropy(y_train, y_hat)

            dW1, db1, dW2, db2, dW3, db3 = self.backward(y_train)
            self.update_weights(dW1, db1, dW2, db2, dW3, db3)

            if (i + 1) % 100 == 0:
                print(f"Epoch {i + 1}/{epochs}  BCE Loss: {loss:.6f}")

        print("Training complete.")

    def predict(self, X_drug, X_protein):
        """Returns sigmoid probabilities for new data."""
        return self.forward(X_drug, X_protein)
