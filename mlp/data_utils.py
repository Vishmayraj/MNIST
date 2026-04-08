import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


def one_hot(y, num_classes=10):
    m = y.size
    o = np.zeros((m, num_classes))
    o[np.arange(m), y] = 1
    return o


def load_mnist(test_size=0.2, seed=42):
    """
    Downloads MNIST via sklearn (cached after first run).
    Returns normalised numpy arrays with one-hot labels.
    """
    print("Loading MNIST (this may take a moment the first time)...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

    # Here X is the grayscale value (0 to 1) of each pixel (in total 784 pixel per image)
    # And Y is the vector encoding of label
    X = mnist.data.astype(np.float32) / 255.0   # (70000, 784)
    y = mnist.target.astype(int)                 # (70000,)
    Y = one_hot(y)                               # (70000, 10)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=test_size, random_state=seed
    )

    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test
