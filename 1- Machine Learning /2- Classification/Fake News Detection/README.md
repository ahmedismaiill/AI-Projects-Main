# 📰 Fake News Detection using XGBoost and TF-IDF

## 📖 Overview
This project aims to detect fake news articles using **Natural Language Processing (NLP)** and **XGBoost**.  
It leverages TF-IDF (Term Frequency–Inverse Document Frequency) to convert textual data into numerical features, followed by an **XGBoost classifier** to distinguish between **Fake** and **True** news articles.

The model achieves over **99% accuracy** on the test dataset, demonstrating strong generalization and effective handling of textual information.

---

## 📂 Dataset Description

The dataset contains labeled news articles divided into two separate files:

| File | Description | Samples |
|------|--------------|----------|
| `Fake.csv` | Fake news articles | 23,502 |
| `True.csv` | True (real) news articles | 21,417 |

### 📊 Dataset Columns
| Column | Description |
|--------|--------------|
| **Title** | Title of the news article |
| **Text** | Body text of the news article |
| **Subject** | Subject or topic of the article |
| **Date** | Publication date of the article |

### 📎 Dataset Link
You can download the dataset from the following link:  
👉 [Fake News Detection Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset/data) 

---

## ⚙️ Project Pipeline

1. **Data Loading and Exploration**  
   - Load and merge the `Fake.csv` and `True.csv` files.  
   - Assign labels (`Fake = 0`, `True = 1`).  

2. **Data Cleaning and Preparation**  
   - Remove duplicate records.  
   - Combine the `Title` and `Text` columns into a single feature.  
   - Handle missing values and preprocess text (lowercasing, punctuation removal, etc.).

3. **Feature Extraction (TF-IDF)**  
   - Transform textual data into numerical vectors using `TfidfVectorizer`.

4. **Model Training (XGBoost)**  
   - Train an XGBoost classifier on the TF-IDF features.  
   - Tune hyperparameters for optimal performance.

5. **Evaluation and Visualization**  
   - Evaluate model performance using accuracy, precision, recall, and F1-score.  
   - Plot confusion matrix and classification report.

---

## 📈 Results

| Metric | Training | Testing |
|--------|-----------|----------|
| **Accuracy** | 99.75% | 99.63% |
| **Precision** | 1.00 | 0.99 |
| **Recall** | 0.99 | 1.00 |
| **F1-Score** | 1.00 | 1.00 |

The model demonstrates excellent generalization, with minimal difference between training and testing performance.

---
