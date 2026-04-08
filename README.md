# 🧠 MNIST Study Repository

> A personal deep dive into the **MNIST handwritten digits dataset**, exploring neural networks from scratch — starting with a Multi-Layer Perceptron (MLP) and progressing toward Convolutional Neural Networks (CNNs).

---

## 📌 About This Repository

This repository documents my journey of **understanding machine learning fundamentals at a low level** — not by relying on high-level frameworks, but by building models from scratch and analyzing how they work internally.

The focus is on:

* Understanding **core ML concepts**
* Implementing models using **NumPy (no PyTorch / TensorFlow initially)**
* Developing strong intuition for:

  * Forward pass
  * Backpropagation
  * Optimization
  * Model design

---

## 📊 Dataset: MNIST

The MNIST dataset consists of:

* 70,000 grayscale images
* Each image: **28 × 28 pixels**
* 10 classes (digits 0–9)

It is a standard benchmark for learning and experimenting with classification models.

---

## 📂 Repository Structure

```text
.
├── mlp/   # Multi-Layer Perceptron (from scratch using NumPy)
├── cnn/   # Convolutional Neural Network (planned)
```

---

## 🔧 Current Work: MLP (Multi-Layer Perceptron)

Located in the `mlp/` directory.

This implementation includes:

* Fully connected neural network
* Forward propagation
* Backpropagation (manual gradient computation)
* Mini-batch gradient descent
* Weight saving/loading
* Evaluation metrics

### Goal

To fully understand how neural networks work **without abstraction layers**.

---

## 🚀 Upcoming Work: CNN (Convolutional Neural Network)

Planned in the `cnn/` directory.

This will involve:

* Convolutional layers
* Feature maps
* Pooling operations
* Improved performance on image data

### Goal

To explore **spatial feature learning** and understand why CNNs outperform MLPs on image tasks.

---

## 🎯 Purpose of This Project

This repository is not just about achieving high accuracy.

It is about:

> Building a deep, first-principles understanding of machine learning systems.

---

## 🧠 Philosophy

* No blind use of frameworks
* No black-box learning
* Every component is understood and justified

---

## 📈 Future Extensions

* Hyperparameter tuning
* Regularization techniques
* Visualization of learned features
* Applying models to custom datasets

---

## 🏁 Status

* ✅ MLP implemented and trained
* 🔄 CNN implementation in progress

---

> This repository represents a progression from **understanding → building → optimizing** machine learning models.
