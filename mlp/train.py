"""
Training entry point.

  python train.py              # default 50 epochs
  python train.py --epochs 100 --lr 0.05 --batch_size 256
"""

import argparse
import numpy as np

from data_utils import load_mnist
from model import init_params, forward, backward, sgd_update, cross_entropy_loss, predict
from weights_io import save_weights


def get_batches(X, Y, batch_size):
    m = X.shape[0]
    indices = np.random.permutation(m)
    for start in range(0, m, batch_size):
        idx = indices[start : start + batch_size]
        yield X[idx], Y[idx]


def train(epochs=50, lr=0.1, batch_size=256, save_every=10):
    X_train, X_test, y_train, y_test = load_mnist()
    W1, b1, W2, b2 = init_params()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0

        for X_batch, Y_batch in get_batches(X_train, y_train, batch_size):
            Z1, A1, Z2, A2 = forward(X_batch, W1, b1, W2, b2)
            loss = cross_entropy_loss(A2, Y_batch)
            dW1, db1, dW2, db2 = backward(X_batch, Y_batch, Z1, A1, A2, W2)
            W1, b1, W2, b2 = sgd_update(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)
            epoch_loss += loss
            num_batches += 1

        avg_loss = epoch_loss / num_batches

        if epoch % 5 == 0 or epoch == 1:
            y_pred = predict(X_test, W1, b1, W2, b2)
            y_true = np.argmax(y_test, axis=1)
            acc = np.mean(y_pred == y_true)
            print(f"Epoch {epoch:>3}/{epochs}  |  Loss: {avg_loss:.4f}  |  Val Acc: {acc:.4f}")

        if epoch % save_every == 0:
            save_weights(f"weights/epoch_{epoch}.npz", W1, b1, W2, b2)

    # Always save final weights
    save_weights("weights/final.npz", W1, b1, W2, b2)
    print("\nTraining complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--lr",         type=float, default=0.1)
    parser.add_argument("--batch_size", type=int,   default=256)
    parser.add_argument("--save_every", type=int,   default=10)
    args = parser.parse_args()

    train(args.epochs, args.lr, args.batch_size, args.save_every)
