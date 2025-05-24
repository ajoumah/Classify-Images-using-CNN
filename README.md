# 🖼️ Classify Images using CNN



---

## 🚀 Project Overview

This project demonstrates image classification using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. CNNs are a cornerstone of deep learning for computer vision tasks, enabling automatic extraction of hierarchical features from images for accurate categorization.

Here, we use the CIFAR-10 dataset — a widely-used benchmark containing 60,000 32x32 color images across 10 classes such as airplane, automobile, bird, cat, and dog.

---

## 🧠 Deep Learning Explanation

Convolutional Neural Networks (CNNs) mimic the human visual cortex by applying convolutional filters to detect edges, textures, and patterns at multiple spatial scales. Pooling layers reduce dimensionality while preserving important features, and fully connected dense layers map these extracted features to class probabilities.

Key components:
- **Conv2D layers:** Learn spatial filters.
- **MaxPooling2D:** Downsample feature maps to reduce computation.
- **Dropout:** Prevent overfitting by randomly disabling neurons during training.
- **Softmax output:** Converts outputs to probability distributions over classes.

This architecture balances depth and capacity to achieve good accuracy on CIFAR-10 without overfitting.

---

## ⚙️ Installation

```bash
pip install tensorflow keras numpy scikit-image matplotlib
