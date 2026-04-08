
# 🧠 MNIST MLP — From Scratch (Deep Dive)

> A complete, from-first-principles implementation of a Multi-Layer Perceptron trained on MNIST using only NumPy.
> No PyTorch. No TensorFlow. Just math, logic, and full control.

---

# 📌 Table of Contents

1. Introduction
2. What This Project Teaches You
3. Dataset Overview (MNIST)
4. High-Level Architecture
5. Data Pipeline (data_utils.py)
6. One-Hot Encoding Explained
7. Model Architecture (model.py Overview)
8. Forward Pass — Step-by-Step
9. Activation Functions (ReLU & Softmax)
10. Loss Function — Cross Entropy
11. Backward Pass — Intuition + Math
12. Gradient Flow & Chain Rule
13. Training Loop (train.py Deep Dive)
14. Mini-Batch Gradient Descent
15. Optimization (SGD)
16. Evaluation Pipeline (evaluate.py)
17. Accuracy Metrics
18. Weight Saving & Loading
19. Numerical Stability Tricks
20. Common Pitfalls & Debugging
21. Performance Analysis
22. Extending This Project
23. What the Model is Actually Learning
24. Why Matrix Multiplication Works
25. Deriving the Key Gradient: dZ2 = A2 - Y
26. Chain Rule — Full Perspective
27. Why Gradients Are Averaged (÷ m)
28. Why ReLU Works So Well
29. Why Softmax Only at Output Layer
30. Why Cross-Entropy Beats MSE
31. Learning Rate — Deep Understanding
32. Batch Size Trade-offs
33. Why Training Improves Over Time
34. Decision Boundaries
35. Why MLP is Limited for Images
36. Overfitting
37. Regularization (Concept)
38. Debugging Tips
39. Why Your Model Got ~97%
40. Advanced Improvements
41. From This to Real ML Systems
Final Conclusion

---

# 🚀 1. Introduction

This project implements a **fully connected neural network (MLP)** from scratch to classify handwritten digits (0–9) from the MNIST dataset.

Unlike frameworks like TensorFlow or PyTorch, this implementation:

* Forces you to understand **every mathematical operation**
* Gives full control over:

  * Forward pass
  * Backward pass
  * Gradient updates
* Builds **real ML intuition**, not API memorization

---

# 🎯 2. What This Project Teaches You

By the end, you will understand:

* Matrix-based neural networks
* Forward propagation
* Backpropagation (chain rule in action)
* Activation functions
* Loss functions
* Gradient descent
* Mini-batch training
* Model evaluation

---

# 🧾 3. Dataset Overview (MNIST)

* 70,000 grayscale images
* Each image: **28 × 28 pixels**
* Flattened into **784-dimensional vectors**

```text
28 × 28 → flatten → 784
```

---

## Data Split

| Dataset | Samples |
| ------- | ------- |
| Train   | 56,000  |
| Test    | 14,000  |

---

## Data Representation

### Input (X)

```python
Shape: (m, 784)
```

Each row = one image
Each value = pixel intensity (0–255 → normalized to 0–1)

---

### Labels (y)

```python
Shape: (m,)
Values: 0–9
```

---

### One-hot Labels (Y)

```python
Shape: (m, 10)
```

Example:

```python
Digit 3 → [0,0,0,1,0,0,0,0,0,0]
```

---

# 🔄 4. High-Level Architecture

```text
Input Layer (784)
        ↓
Hidden Layer (128) — ReLU
        ↓
Output Layer (10) — Softmax
```

---

## Mathematical Representation

```text
Z1 = XW1 + b1
A1 = ReLU(Z1)
Z2 = A1W2 + b2
A2 = Softmax(Z2)
```

---

# 📦 5. Data Pipeline (`data_utils.py`)

---

## Loading Data

```python
mnist = fetch_openml('mnist_784')
```

---

## Normalization

```python
X = mnist.data / 255.0
```

### Why?

* Keeps values between **0 and 1**
* Improves training stability
* Prevents exploding gradients

---

## Train-Test Split

```python
train_test_split(X, Y)
```

---

## Final Shapes

```text
X_train → (56000, 784)
Y_train → (56000, 10)
```

---

# 🔥 6. One-Hot Encoding Explained

---

## Function

```python
def one_hot(y, num_classes=10):
    m = y.size
    o = np.zeros((m, num_classes))
    o[np.arange(m), y] = 1
    return o
```

---

## Key Idea

Convert scalar labels into vectors:

```text
2 → [0,0,1,0,0,0,0,0,0,0]
```

---

## Why Needed?

Because model outputs:

```text
[0.1, 0.2, 0.7, ...]
```

Shapes must match for loss computation.

---

# 🧠 7. Model Overview (`model.py`)

---

## Components

* Activation functions
* Loss function
* Weight initialization
* Forward pass
* Backward pass
* Parameter updates

---

# ⚙️ 8. Forward Pass — Detailed

---

## Step 1

```python
Z1 = X @ W1 + b1
```

* Linear transformation
* Shape: `(m, 128)`

---

## Step 2

```python
A1 = ReLU(Z1)
```

---

## Step 3

```python
Z2 = A1 @ W2 + b2
```

---

## Step 4

```python
A2 = Softmax(Z2)
```

---

## Output

```python
A2 → probabilities
Shape: (m, 10)
```

---

# ⚡ 9. Activation Functions

---

## ReLU

```python
f(x) = max(0, x)
```

### Properties

* Introduces non-linearity
* Efficient
* Sparse activation

---

## ReLU Derivative

```python
f'(x) = 1 if x > 0 else 0
```

---

## Softmax

```math
P(x_i) = \frac{e^{x_i}}{\sum e^{x_j}}
```

---

### Why Softmax?

* Converts scores → probabilities
* Ensures:

  * values ∈ [0,1]
  * sum = 1

---

### Numerical Stability Trick

```python
Z - max(Z)
```

Prevents overflow in exponentials.

---

# 📉 10. Loss Function — Cross Entropy

---

## Formula

```math
L = -\frac{1}{m} \sum Y \cdot \log(Ŷ)
```

---

## Intuition

Measures:

> “How wrong is the prediction?”

---

## Example

```python
Y_hat = [0.1, 0.7, 0.2]
Y     = [0, 1, 0]

Loss = -log(0.7)
```

---

## Key Insight

* Only correct class matters
* Punishes confident wrong predictions heavily

---

# 🔁 11. Backward Pass (Core Idea)

---

## Goal

Compute gradients:

```text
∂Loss/∂W1, ∂Loss/∂W2, etc.
```

---

## Key Identity

```python
dZ2 = A2 - Y
```

---

## Why Important?

Simplifies derivative of:

```text
Softmax + Cross Entropy
```

---

# 🔗 12. Chain Rule (Simplified)

---

## Flow

```text
Loss
 ↓
A2
 ↓
Z2
 ↓
A1
 ↓
Z1
 ↓
Weights
```

---

## Concept

Each layer contributes to error → propagate backwards.

---

# 🔄 13. Training Loop (`train.py`)

---

## Core Loop

```python
for batch:
    forward
    loss
    backward
    update
```

---

## Epoch

One full pass over dataset.

---

# 📦 14. Mini-Batch Gradient Descent

---

## Why Not Full Batch?

* Slow
* Memory heavy

---

## Why Mini-Batch?

* Faster updates
* Better generalization

---

## Example

```python
batch_size = 256
```

---

# ⚙️ 15. Optimization (SGD)

---

## Update Rule

```python
W = W - lr * dW
```

---

## Learning Rate

Controls step size.

* Too high → unstable
* Too low → slow

---

# 📊 16. Evaluation (`evaluate.py`)

---

## Prediction

```python
np.argmax(A2)
```

---

## Accuracy

```python
accuracy = correct / total
```

---

# 📈 17. Per-Class Accuracy

---

## Why?

Shows which digits are harder.

Example:

* Digit 1 → easy
* Digit 8 → harder

---

# 💾 18. Weight Storage

---

## Saving

```python
np.savez("weights.npz")
```

---

## Loading

```python
np.load("weights.npz")
```

---

## Contents

```text
W1, b1, W2, b2
```

---

# ⚠️ 19. Numerical Stability

---

## Tricks Used

* Softmax shifting
* log( + 1e-8 )

---

## Why?

Prevent:

* overflow
* log(0)

---

# 🧨 20. Common Pitfalls

---

### ❌ Not normalizing input

### ❌ Wrong shapes

### ❌ Using @ instead of *

### ❌ Forgetting softmax

### ❌ Incorrect gradient flow

---

# 📊 21. Performance

---

Expected:

```text
~92–95% (baseline)
~97% (tuned)
```

---

# 🚀 22. Extensions

---

## Improve Model

* Add more layers
* Try different activations

---

## Upgrade Architecture

* Convolutional Neural Networks (CNN)

---

## Regularization

* Dropout
* L2 penalty

---

## Hyperparameter Tuning

* Learning rate
* Batch size

---

# 🧠 Final Thought

This project is not about MNIST.

It’s about:

> Understanding how neural networks actually work under the hood.

---

**If you truly understand this file, you don’t “use ML” — you build it.**

# 🔬 MNIST MLP — Deep Dive (Part 2)

> This section goes beyond implementation into **mathematical intuition, derivations, optimization theory, and design reasoning**.

---

# 🧠 23. What the Model is *Actually Learning*

---

## Not Digits — Patterns

Your model is NOT learning:

```text
"This is a 3"
```

It is learning:

```text
"This combination of strokes resembles a 3"
```

---

## Layer-wise Interpretation

### Layer 1 (128 neurons)

Each neuron detects:

* edges
* curves
* stroke directions

---

### Layer 2 (Output)

Combines features:

```text
loop + vertical line → maybe 9  
curve + open top → maybe 3
```

---

## 🔥 Insight

> Neural networks learn **hierarchical representations**

---

# 🧮 24. Why Matrix Multiplication Works

---

## Core Operation

```python
Z1 = X @ W1 + b1
```

---

## What is happening?

Each neuron computes:

```math
z = x_1w_1 + x_2w_2 + ... + x_nw_n + b
```

---

## Vectorized Form

Instead of loops:

```python
for neuron:
    compute dot product
```

We do:

```python
X @ W1
```

---

## 🔥 Insight

> Matrix multiplication = many dot products at once

---

# ⚡ 25. Deriving the Key Gradient: `dZ2 = A2 - Y`

---

## Normally (complex):

You would compute:

```math
∂Loss/∂Z2 = ∂Loss/∂A2 × ∂A2/∂Z2
```

---

## But with:

* Softmax
* Cross-Entropy

---

## It simplifies to:

```math
dZ2 = A2 - Y
```

---

## 🔥 Why this matters

* Faster computation
* Cleaner code
* Stable gradients

---

## 🧠 Interpretation

```text
prediction - truth
```

---

# 🔗 26. Chain Rule — Full Perspective

---

## The full dependency chain:

```text
W1 → Z1 → A1 → Z2 → A2 → Loss
```

---

## Chain rule:

```math
∂Loss/∂W1 = ∂Loss/∂Z1 × ∂Z1/∂W1
```

---

## Backprop is just:

> Applying chain rule repeatedly, backwards

---

# ⚙️ 27. Why Gradients Are Averaged (÷ m)

---

```python
dW = (X.T @ dZ) / m
```

---

## Why divide by m?

Without it:

* gradients scale with batch size
* unstable updates

---

## With averaging:

* consistent updates
* independent of batch size

---

## 🔥 Insight

> Gradient = average contribution per sample

---

# 🔥 28. Why ReLU Works So Well

---

## Compared to Sigmoid

### Sigmoid:

```math
σ(x) = 1 / (1 + e^{-x})
```

Problems:

* vanishing gradients
* slow training

---

## ReLU advantages:

* simple
* fast
* avoids vanishing gradient (mostly)

---

## ⚠️ Limitation

Dead neurons:

```text
Z ≤ 0 → gradient = 0 forever
```

---

# ⚙️ 29. Why Softmax Only at Output Layer

---

## Hidden layers:

Need:

```text
non-linearity
```

---

## Output layer:

Need:

```text
probability distribution
```

---

## 🔥 Insight

> Softmax is for **interpretation**, not feature learning

---

# 📉 30. Why Cross-Entropy Beats MSE

---

## If using MSE:

```math
Loss = (Y - Ŷ)^2
```

---

## Problems:

* weak gradients
* slow learning

---

## Cross-Entropy:

```math
Loss = -log(p_correct)
```

---

## Advantages:

* strong gradients when wrong
* faster convergence

---

# ⚡ 31. Learning Rate — Deep Understanding

---

## Update Rule

```python
W = W - lr * gradient
```

---

## Effects

### Large LR

* fast learning
* risk of overshooting

---

### Small LR

* slow learning
* stable

---

## 🔥 Insight

> Learning rate controls how aggressively the model learns

---

# 📦 32. Batch Size Trade-offs

---

## Small batch

* noisy gradients
* better generalization

---

## Large batch

* stable gradients
* faster computation (GPU-friendly)

---

## Your choice: 256

Good balance.

---

# 🔥 33. Why Training Improves Over Time

---

Each iteration:

```text
wrong → adjust → less wrong → adjust → better
```

---

## Mathematically:

Gradient descent minimizes loss:

```math
minimize L(W)
```

---

# 🧠 34. Decision Boundaries

---

Your model creates regions like:

```text
this region → digit 3  
this region → digit 8
```

---

## With ReLU:

Boundaries are:

```text
piecewise linear
```

---

# 🔬 35. Why MLP is Limited for Images

---

## Problem

Flattening destroys spatial info:

```text
pixel (0,0) unrelated to pixel (0,1)
```

---

## Result

* loses structure
* less efficient

---

## Solution

CNNs:

* preserve spatial relationships
* use filters

---

# ⚠️ 36. Overfitting

---

## Definition

Model memorizes training data.

---

## Symptoms

* high train accuracy
* lower test accuracy

---

## Fixes

* more data
* regularization
* dropout

---

# 🔒 37. Regularization (Concept)

---

## L2 Regularization

```math
Loss = original + λ||W||^2
```

---

## Effect

* discourages large weights
* smoother model

---

# 🧪 38. Debugging Tips

---

### If loss = NaN

* check log(0)
* check overflow

---

### If accuracy stuck

* learning rate issue
* bad initialization

---

### If model not learning

* check gradients
* check shapes

---

# 📈 39. Why Your Model Got ~97%

---

Because:

* good initialization
* ReLU activation
* cross-entropy loss
* proper normalization

---

# 🚀 40. Advanced Improvements

---

## Add Layer

```text
784 → 128 → 64 → 10
```

---

## Try Activations

* Leaky ReLU
* GELU

---

## Add Dropout

Randomly disable neurons during training.

---

# 🧠 41. From This to Real ML Systems

---

This exact pipeline scales to:

* recommendation systems
* NLP models
* vision systems

---

## Only difference:

* bigger models
* more data
* better architectures

---

# 🏁 Final Conclusion

---

You now understand:

* How data flows
* How predictions are made
* How errors are computed
* How learning happens

---

## 🔥 Most Important Insight

> Neural networks are NOT magic.

They are:

```text
matrix multiplications + calculus + optimization
```

---

## 🚀 Where You Go Next

If you want to reach top-tier level:

1. Implement a CNN from scratch
2. Train on a custom dataset
3. Optimize performance
4. Build something real

---

> If you can build this from scratch, you’re no longer “learning ML” —
> you’re **doing ML**.


## Repo layout

```
mnist-mlp/
├── model.py          # forward, backward, activations, loss
├── data_utils.py     # MNIST loader + one-hot encoding
├── weights_io.py     # save / load .npz weight files
├── train.py          # training loop (mini-batch SGD)
├── evaluate.py       # load weights → test accuracy
├── requirements.txt
└── weights/          # .npz files saved here during training
```

## Quickstart

```bash
pip install -r requirements.txt

# Train (saves weights/final.npz + checkpoints every 10 epochs)
python train.py

# Custom run
python train.py --epochs 100 --lr 0.05 --batch_size 128

# Evaluate a checkpoint
python evaluate.py --weights weights/epoch_30.npz
```

## Expected accuracy
~92–95% on the test set after 50 epochs with default settings.

## Weight format
Weights are saved as `.npz` (numpy's zip archive).  
Keys: `W1`, `b1`, `W2`, `b2`.

```python
import numpy as np
data = np.load("weights/final.npz")
W1 = data["W1"]   # shape (784, 128)
```
