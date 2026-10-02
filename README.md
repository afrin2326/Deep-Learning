# Deep Learning Projects Portfolio

This repository contains a collection of deep learning projects that explore fundamental and advanced concepts in neural networks. The projects cover a range of topics, from the theoretical limits of simple classifiers to the practical application of regularization techniques for complex computer vision tasks.

---

## Table of Contents

1.  [The Perceptron: Foundations and Limitations](#1-the-perceptron-foundations-and-limitations)
    -   [Perceptron Demo: A Practical Application](#perceptron-demo-a-practical-application)
    -   [The Problem with the Perceptron: The XOR Gate](#the-problem-with-the-perceptron-the-xor-gate)
2.  [Fine-Grained Visual Classification with Regularization](#2-fine-grained-visual-classification-with-regularization)
    -   [Project Overview](#project-overview)
    -   [Methodology](#methodology)
    -   [Results and Analysis](#results-and-analysis)
    -   [Conclusion](#conclusion)
3.  [Repository Structure](#3-repository-structure)
4.  [How to Use](#4-how-to-use)

---

## 1. The Perceptron: Foundations and Limitations

This section explores the perceptron, the simplest type of artificial neural network, through two projects. One demonstrates its use in a real-world classification task, while the other highlights its fundamental limitation.

### Perceptron Demo: A Practical Application

This project demonstrates the application of a `scikit-learn` Perceptron to predict student placement based on features like CGPA and IQ.

-   **Notebook:** `perceptron-demo.ipynb`
-   **Dataset:** A simple CSV file (`placement-dataset.csv`) with features `cgpa`, `iq`, and a binary `placement` target.
-   **Goal:** To build a linear classifier that can distinguish between placed and not-placed students.
-   **Methodology:**
    1.  **Data Loading & Preprocessing:** The dataset was loaded using pandas, and rows with missing values were dropped to create a clean dataset for training.
    2.  **Exploratory Data Analysis (EDA):** A scatter plot was generated to visualize the distribution of the two classes based on CGPA and IQ. The plot showed that the data points were largely, but not perfectly, linearly separable.
    3.  **Model Training:** A `Perceptron` model from `scikit-learn` was trained on the cleaned data.
    4.  **Visualization:** The decision boundary learned by the perceptron was visualized using `mlxtend.plotting.plot_decision_regions`, showing how the model separates the feature space.

-   **Key Takeaway:** The perceptron can successfully find a linear boundary to separate classes that are mostly linearly separable, providing a simple yet effective solution for this type of problem.

---

### The Problem with the Perceptron: The XOR Gate

This project uses the classic XOR problem to illustrate the fundamental limitation of single-layer perceptrons: their inability to solve non-linearly separable problems.

-   **Notebooks:** `perceptron-trick.ipynb`, `problem-with-perceptron.ipynb`
-   **Goal:** To train a single-layer perceptron on the XOR (exclusive OR) logic gate.
-   **Methodology:**
    1.  **Dataset Creation:** The truth tables for AND, OR, and XOR gates were created.
    2.  **Visualization:** The data points for each gate were plotted.
        -   The **AND** and **OR** gate data points are linearly separable (a single straight line can separate the 0s from the 1s).
        -   The **XOR** gate data points are **not linearly separable**. No single straight line can separate the class '0' from class '1'.
    3.  **Model Training:** A single-layer perceptron was trained on the XOR dataset.
    4.  **Result:** The model fails to converge to a 100% accurate solution. When its decision boundary is visualized, it's clear that it cannot separate the classes, as any single line will misclassify at least one point.

-   **Key Takeaway:** Single-layer perceptrons are only capable of learning linearly separable patterns. The XOR problem is a canonical example of a non-linear problem, and it demonstrates the necessity of multi-layer networks (with non-linear activation functions) to solve more complex, real-world tasks.

---

## 2. Fine-Grained Visual Classification with Regularization

This project investigates the impact of different regularization techniques on a custom Convolutional Neural Network (CNN) for the challenging task of Fine-Grained Visual Classification (FGVC) using the CIFAR-100 dataset.

-   **Notebook:** `Fine-Grained Visual Classification with Regularization.ipynb`
-   **Dataset:** CIFAR-100, containing 100 fine-grained classes (e.g., different species of flowers, animals, vehicles).
-   **Goal:** To analyze how Dropout and Batch Normalization affect a model's ability to generalize and combat overfitting.

### Project Overview

A custom 4-layer CNN was designed and trained under three different configurations for 20 epochs each:

| Model                | Description          | Dropout | Batch Norm |
| -------------------- | -------------------- | ------- | ---------- |
| **Model A**          | Baseline             | ❌       | ❌          |
| **Model B**          | Dropout Only         | ✅ (0.5) | ❌          |
| **Model C**          | Full Regularization  | ✅ (0.5) | ✅          |

### Methodology

-   **Architecture:** A custom CNN with 4 convolutional blocks (channels: 32 → 64 → 128 → 256), each followed by ReLU and MaxPooling. Global Average Pooling was used before two fully connected layers.
-   **Data Augmentation:** Random horizontal flips, random rotations (±15°), and color jitter were applied to the training set.
-   **Optimizer:** AdamW with a learning rate of 0.001 and weight decay of 1e-4.
-   **Scheduler:** CosineAnnealingLR over 20 epochs.
-   **Loss Function:** CrossEntropyLoss.

### Results and Analysis

The models were evaluated on a held-out test set using Accuracy, Precision, Recall, and F1-Score.

#### Performance Metrics Comparison

| Model                | Test Accuracy | F1-Score (Macro) | Precision (Macro) | Recall (Macro) |
| -------------------- | ------------- | ---------------- | ----------------- | -------------- |
| Model_A_Baseline     | 0.4520        | 0.4504           | 0.4568            | 0.4520         |
| Model_B_Dropout      | 0.4551        | 0.4541           | 0.4557            | 0.4551         |
| **Model_C_FullReg**  | **0.4959**    | **0.4984**       | **0.5051**        | **0.4959**     |

#### Generalization Analysis

-   **Model A (Baseline):** Showed clear signs of overfitting. Training accuracy reached **88.27%**, while test accuracy was only **45.20%**, resulting in a significant generalization gap of **43.07%**.
-   **Model B (Dropout):** Dropout was effective in reducing overfitting. The gap between training accuracy (**66.60%**) and test accuracy (**45.51%**) was much smaller at **21.09%**.
-   **Model C (Full Regularization):** This model achieved the **best test accuracy (49.59%)**. Batch Normalization stabilized the training process, and Dropout prevented overfitting, allowing the model to reach a higher level of performance.

#### Class-wise Performance (Best Model - Model C)

-   **Easy Classes (High Accuracy):** road (0.81), plain (0.81), sunflower (0.78), orange (0.77), wardrobe (0.74)
-   **Challenging Classes (Low Accuracy):** woman (0.23), seal (0.24), otter (0.25), turtle (0.25), man (0.26)

These results highlight that visually similar fine-grained classes remain difficult to distinguish, even with strong regularization.

### Conclusion

This project confirms that regularization is essential for FGVC tasks.
-   **Batch Normalization** dramatically improves training stability and allows the model to converge faster and to a better solution.
-   **Dropout** is effective at preventing co-adaptation of neurons and reducing overfitting.
-   The **combination of both techniques** yields the most robust and accurate model, highlighting that different regularization methods can have a synergistic effect.

### Future Work

1.  **Advanced Regularization:** Stochastic Depth, Label Smoothing, MixUp, and CutMix.
2.  **Attention Mechanisms:** Self-attention, Channel Attention (SENet), and Spatial Attention.
3.  **Ensemble Methods:** Model averaging and Knowledge Distillation.
4.  **Transfer Learning:** Fine-tuning pre-trained models like ResNet and EfficientNet.

---


---

## 3. How to Use

To run these notebooks, you will need a Python environment with the following libraries installed.

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
