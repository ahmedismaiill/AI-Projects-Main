# 💬 SMS Spam Detection with TF-IDF | Achieved 98% Accuracy

A machine learning project that classifies SMS messages as **spam** or **ham** using TF-IDF vectorization and a classification model, achieving **98% accuracy**.

---

## 📖 About the Dataset

You can download the SMS Spam Collection dataset from Kaggle: [here](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).

### Context

The **SMS Spam Collection** is a set of SMS messages collected for research on SMS spam detection. It contains **5,574 English messages** labeled as **ham** (legitimate) or **spam**.

### Content

* Each line in the dataset represents one message.
* **v1**: Label (`ham` or `spam`)
* **v2**: Raw SMS text

The corpus is compiled from multiple sources:

1. **Grumbletext Web site** – 425 spam messages manually extracted from a UK forum where users reported spam.
2. **NUS SMS Corpus (NSC)** – 3,375 ham messages randomly chosen from a dataset of ~10,000 legitimate messages collected in Singapore.
3. **Caroline Tag's PhD Thesis** – 450 ham messages.
4. **SMS Spam Corpus v.0.1 Big** – 1,002 ham and 322 spam messages.

## ⚡ Features

* Preprocessing of SMS text
* TF-IDF vectorization for feature extraction
* Classification with machine learning (e.g., Logistic Regression, XGBoost, or your choice)
* **Accuracy achieved: 98%**

---

## 📂 File Structure

```
├── spam (2).csv                 # Original dataset
├── SMS Spam Detection with TF-IDF Achieved 98%.ipynb # Jupyter notebook with preprocessing, training, evaluation
├── README.md                # Project documentation
```
