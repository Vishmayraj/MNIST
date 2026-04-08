"""
Save and load model weights using numpy's .npz format.
No Keras, no pickle — just raw arrays.

Usage
-----
    save_weights("weights/epoch_50.npz", W1, b1, W2, b2)
    W1, b1, W2, b2 = load_weights("weights/epoch_50.npz")
"""

import numpy as np
from pathlib import Path


def save_weights(path: str, W1, b1, W2, b2):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, W1=W1, b1=b1, W2=W2, b2=b2)
    print(f"Weights saved → {path}")


def load_weights(path: str):
    data = np.load(path)
    W1, b1, W2, b2 = data["W1"], data["b1"], data["W2"], data["b2"]
    print(f"Weights loaded ← {path}")
    return W1, b1, W2, b2
