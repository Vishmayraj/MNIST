import numpy as np


# ─── Activations ────────────────────────────────────────────────────────────

def relu(Z):
    return np.maximum(0, Z)

def relu_deriv(Z):
    return (Z > 0).astype(float)

def softmax(Z):
    # Subtract max for numerical stability
    shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


# ─── Loss ────────────────────────────────────────────────────────────────────

def cross_entropy_loss(Y_hat, Y):
    """
    Y_hat : predicted probabilities  (m, 10)
    Y     : one-hot ground-truth      (m, 10)
    """
    m = Y.shape[0]
    return -np.sum(Y * np.log(Y_hat + 1e-8)) / m


# ─── Weight init ─────────────────────────────────────────────────────────────

def init_params(input_dim=784, hidden_dim=128, output_dim=10, seed=42):
    rng = np.random.default_rng(seed)
    # He initialisation — works well with ReLU
    W1 = rng.standard_normal((input_dim, hidden_dim)) * np.sqrt(2 / input_dim)
    b1 = np.zeros((1, hidden_dim))
    W2 = rng.standard_normal((hidden_dim, output_dim)) * np.sqrt(2 / hidden_dim)
    b2 = np.zeros((1, output_dim))
    return W1, b1, W2, b2


# ─── Forward pass ────────────────────────────────────────────────────────────

def forward(X, W1, b1, W2, b2):
    """
    Returns every intermediate value we'll need for backprop.
    X  : (m, 784)
    """
    Z1 = X @ W1 + b1        # (m, 128)  — linear
    A1 = relu(Z1)           # (m, 128)  — non-linear
    Z2 = A1 @ W2 + b2       # (m, 10)   — linear
    A2 = softmax(Z2)        # (m, 10)   — probabilities
    return Z1, A1, Z2, A2


# ─── Backward pass ───────────────────────────────────────────────────────────

def backward(X, Y, Z1, A1, A2, W2):
    """
    Chain rule from loss → W1/b1, W2/b2.
    """
    m = X.shape[0]

    # Output layer gradient (softmax + cross-entropy simplifies beautifully)
    dZ2 = A2 - Y                             # (m, 10)
    dW2 = (A1.T @ dZ2) / m                   # (128, 10)
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    # Hidden layer gradient
    dA1 = dZ2 @ W2.T                         # (m, 128)
    dZ1 = dA1 * relu_deriv(Z1)               # (m, 128)  — gate dead neurons
    dW1 = (X.T @ dZ1) / m                    # (784, 128)
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2


# ─── SGD update ──────────────────────────────────────────────────────────────

def sgd_update(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2
    return W1, b1, W2, b2


# ─── Predict ─────────────────────────────────────────────────────────────────

def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward(X, W1, b1, W2, b2)
    return np.argmax(A2, axis=1)
