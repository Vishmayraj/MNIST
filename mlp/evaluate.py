"""
Load saved weights and evaluate on the test set.

  python evaluate.py                          # uses weights/final.npz
  python evaluate.py --weights weights/epoch_30.npz
"""

import argparse
import numpy as np

from data_utils import load_mnist
from model import predict
from weights_io import load_weights


def evaluate(weights_path="weights/final.npz"):
    _, X_test, _, y_test = load_mnist()
    W1, b1, W2, b2 = load_weights(weights_path)

    y_pred = predict(X_test, W1, b1, W2, b2)
    y_true = np.argmax(y_test, axis=1)

    acc = np.mean(y_pred == y_true)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")

    # Per-class breakdown
    print("\nPer-class accuracy:")
    for cls in range(10):
        mask = y_true == cls
        cls_acc = np.mean(y_pred[mask] == y_true[mask])
        print(f"  Digit {cls}: {cls_acc * 100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="weights/final.npz")
    args = parser.parse_args()
    evaluate(args.weights)
