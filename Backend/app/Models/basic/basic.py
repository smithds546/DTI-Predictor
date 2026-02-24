import numpy as np


# --- Activation Function and its Derivative ---

def sigmoid(x):
    """
    Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    """
    # Clip values to prevent overflow in exp
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Derivative of the sigmoid function: f'(x) = f(x) * (1 - f(x))
    """
    s = sigmoid(x)
    return s * (1 - s)


# --- Loss Function and its Derivative ---

def mean_squared_error(y_true, y_pred):
    """
    Mean Squared Error loss: L = (1/N) * sum((y_true - y_pred)^2)
    """
    return np.mean(np.power(y_true - y_pred, 2))


def mean_squared_error_derivative(y_true, y_pred):
    """
    Derivative of MSE loss with respect to y_pred: dL/dy_pred = 2 * (y_pred - y_true) / N
    """
    return 2 * (y_pred - y_true) / y_true.size


class NeuralNetwork:
    """
    A simple neural network with one hidden layer.

    Architecture:
    [Input 1 (Drug), Input 2 (Protein)] -> Concatenate -> Hidden Layer (sigmoid) -> Output Layer (sigmoid)
    """

    def __init__(self, input_size_drug, input_size_protein, hidden_size, output_size, learning_rate=0.1):
        """
        Initializes the network's weights and biases.

        Args:
            input_size_drug (int): Number of features for the drug input.
            input_size_protein (int): Number of features for the protein input.
            hidden_size (int): Number of neurons in the hidden layer.
            output_size (int): Number of neurons in the output layer.
            learning_rate (float): Step size for gradient descent.
        """
        self.input_size = input_size_drug + input_size_protein
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights with small random values to break symmetry
        self.W1 = np.random.randn(self.input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

        # Variables to store intermediate values
        self.X_concat = None  # To store the concatenated input
        self.z1 = None  # Weighted input to hidden layer
        self.a1 = None  # Activation of hidden layer
        self.z2 = None  # Weighted input to output layer
        self.y_hat = None  # Activation of output layer (prediction)

    def forward(self, X_drug, X_protein):
        """
        Performs the forward propagation pass.

        Args:
            X_drug (np.array): Input drug data of shape (n_samples, input_size_drug).
            X_protein (np.array): Input protein data of shape (n_samples, input_size_protein).

        Returns:
            np.array: The network's prediction of shape (n_samples, output_size).
        """
        # 1. Concatenate Inputs along the feature axis (axis=1)
        self.X_concat = np.concatenate((X_drug, X_protein), axis=1)

        # 2. From Input to Hidden Layer
        # Weighted sum: Z1 = X_concat * W1 + b1
        self.z1 = np.dot(self.X_concat, self.W1) + self.b1
        # Activation: A1 = sigmoid(Z1)
        self.a1 = sigmoid(self.z1)

        # 3. From Hidden to Output Layer
        # Weighted sum: Z2 = A1 * W2 + b2
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        # Activation (prediction): y_hat = sigmoid(Z2)
        self.y_hat = sigmoid(self.z2)

        return self.y_hat

    def backward(self, y):
        """
        Performs the backward propagation pass (backpropagation)
        and calculates the gradients for weights and biases.

        Args:
            y (np.array): True target values of shape (n_samples, output_size).
        """
        n_samples = y.shape[0]

        # --- 1. Calculate Gradients for Output Layer (W2, b2) ---

        # Error at the output layer
        # We use (y_hat - y) * sigmoid_derivative(z2)
        delta_output = (self.y_hat - y) * sigmoid_derivative(self.z2)

        # Gradient for W2: dE/dW2 = A1.T * delta_output
        dW2 = np.dot(self.a1.T, delta_output)

        # Gradient for b2: dE/db2 = sum(delta_output)
        db2 = np.sum(delta_output, axis=0, keepdims=True)

        # --- 2. Calculate Gradients for Hidden Layer (W1, b1) ---

        # Propagate error to the hidden layer
        delta_hidden = np.dot(delta_output, self.W2.T) * sigmoid_derivative(self.z1)

        # Gradient for W1: dE/dW1 = X_concat.T * delta_hidden
        dW1 = np.dot(self.X_concat.T, delta_hidden)

        # Gradient for b1: dE/db1 = sum(delta_hidden)
        db1 = np.sum(delta_hidden, axis=0, keepdims=True)

        # Normalize gradients by batch size
        dW1 /= n_samples
        db1 /= n_samples
        dW2 /= n_samples
        db2 /= n_samples

        return dW1, db1, dW2, db2

    def update_weights(self, dW1, db1, dW2, db2):
        """
        Updates the network's weights and biases using gradient descent.
        """
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, X_drug_train, X_protein_train, y_train, epochs):
        """
        Trains the neural network using forward and backward propagation.

        Args:
            X_drug_train (np.array): Input drug training data.
            X_protein_train (np.array): Input protein training data.
            y_train (np.array): Target training data.
            epochs (int): Number of training iterations.
        """
        print(f"Starting training for {epochs} epochs...")
        for i in range(epochs):
            # 1. Forward Pass
            y_hat = self.forward(X_drug_train, X_protein_train)

            # 2. Calculate Loss (for monitoring)
            loss = mean_squared_error(y_train, y_hat)

            # 3. Backward Pass (Calculate Gradients)
            # We no longer pass X, as the concatenated X is stored in self.X_concat
            dW1, db1, dW2, db2 = self.backward(y_train)

            # 4. Update Weights
            self.update_weights(dW1, db1, dW2, db2)

            # Print loss every 1000 epochs
            if (i + 1) % 1000 == 0:
                print(f"Epoch {i + 1}/{epochs}, Loss: {loss:.6f}")

        print("Training complete.")

    def predict(self, X_drug, X_protein):
        """
        Makes a prediction on new data.
        """
        return self.forward(X_drug, X_protein)