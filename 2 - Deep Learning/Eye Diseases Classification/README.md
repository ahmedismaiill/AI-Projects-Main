# 👁️ Eye Diseases Classification from Retinal Images

A deep learning project that classifies retinal fundus images into four categories of eye conditions using a **Custom CNN** and **EfficientNetB7 (transfer learning)**.

---

## 🩺 Problem Statement

Early diagnosis of eye diseases like **Cataract**, **Glaucoma**, and **Diabetic Retinopathy** is critical for vision preservation. This project aims to automate disease detection from retinal images using Convolutional Neural Networks.

---

## 📂 Dataset Overview

The dataset includes approximately **4200** high-resolution fundus images categorized into:

| Class                 | Images |
|-----------------------|--------|
| Cataract              | 1038   |
| Diabetic Retinopathy  | 1098   |
| Glaucoma              | 1007   |
| Normal                | 1074   |

You can collect datasets from:

- 🔗 [Eye Disease Retinal Images Dataset (Kaggle)](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)

### 📁 Directory Structure


Each folder should contain `.jpg` images for its respective disease class.

---

## 🧠 Models

### 🔧 1. Custom CNN

A 4-block CNN model built from scratch with ReLU activations, MaxPooling, and a fully connected head.

**Performance:**

Test Accuracy : 89%
F1-Score (Macro) : 0.89

Class-wise F1 Scores:

Glaucoma : 0.79

Normal : 0.85

Diabetic Retinopathy : 0.99

Cataract : 0.91


### ⚡ 2. EfficientNetB7 (Transfer Learning)

Uses ImageNet-pretrained EfficientNetB7 as the base model with added custom top layers.

**Performance:**

Test Accuracy : 89%
F1-Score (Macro) : 0.89

Class-wise F1 Scores:

Glaucoma : 0.83

Normal : 0.83

Diabetic Retinopathy : 0.95

Cataract : 0.94


---

## 📊 Training Details

- Image size: **224x224x3**
- Batch size: **32**
- Optimizer: **Adam**
- Loss Function: **Sparse Categorical Crossentropy**
- Epochs: **200**
- Callbacks:
  - EarlyStopping
  - ModelCheckpoint
  - ReduceLROnPlateau (for CNN)

---
