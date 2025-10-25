# 🏦 Loan Approval Prediction Dataset

## 📄 Overview

This repository provides a **synthetic dataset** designed for **loan approval prediction** tasks using machine learning.
Inspired by the [original Kaggle Loan Approval Classification dataset](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data), this version has been **enhanced with additional financial risk variables** and **balanced using SMOTENC**, allowing for more robust experimentation and evaluation of classification models.

---

## 📁 Dataset Metadata

* **Records:** 45,000 samples
* **Features:** 13 columns *(after removing data-leakage variable)*
* **Target Variable:** `loan_status` (binary classification: `1 = approved`, `0 = rejected`)

| Column Name                  | Description                                                    | Type        |
| ---------------------------- | -------------------------------------------------------------- | ----------- |
| `person_age`                 | Age of the person                                              | Float       |
| `person_gender`              | Gender of the person                                           | Categorical |
| `person_education`           | Highest education level                                        | Categorical |
| `person_income`              | Annual income in USD                                           | Float       |
| `person_emp_exp`             | Years of employment experience                                 | Integer     |
| `person_home_ownership`      | Home ownership status (e.g., rent, own, mortgage)              | Categorical |
| `loan_amnt`                  | Loan amount requested                                          | Float       |
| `loan_intent`                | Purpose of the loan (e.g., personal, education, medical, etc.) | Categorical |
| `loan_int_rate`              | Interest rate on the loan                                      | Float       |
| `loan_percent_income`        | Loan amount as a percentage of the person’s income             | Float       |
| `cb_person_cred_hist_length` | Length of credit history in years                              | Float       |
| `credit_score`               | Credit score of the person                                     | Integer     |
| `loan_status`                | Loan approval status (`1 = approved`, `0 = rejected`)          | Integer     |

> ⚠️ **Note:**
> The column `previous_loan_defaults_on_file` was **removed** due to **data leakage**, as it directly correlated with the loan approval decision, leading to artificially inflated performance.

---

## 📊 Notebook Analysis

The accompanying Jupyter Notebook demonstrates:

* **Exploratory Data Analysis (EDA)** using **Plotly** for rich, interactive visualizations
* **Data preprocessing** and **balancing using SMOTE** to handle class imbalance
* **Feature engineering** and encoding of categorical variables
* **Model training** using **XGBoost**, optimized for high predictive accuracy
* **Model evaluation** using multiple metrics

📘 **Notebook:** [`Loan Approval Classification App 🚀 | (92%,MLflow)`](https://www.kaggle.com/code/ahmedismaiil/loan-approval-classification-app-92-mlflow)

---

## 🧠 Model Performance — XGBoost + SMOTE

| Metric                    | Value |
| ------------------------- | ----- |
| **Accuracy**              | 0.92  |
| **Precision (Class 0)**   | 0.92  |
| **Recall (Class 0)**      | 0.98  |
| **F1-score (Class 0)**    | 0.95  |
| **Precision (Class 1)**   | 0.88  |
| **Recall (Class 1)**      | 0.67  |
| **F1-score (Class 1)**    | 0.76  |
| **Macro Avg F1-score**    | 0.85  |
| **Weighted Avg F1-score** | 0.91  |

✅ **Final Accuracy:** **92%** using **XGBoost** on **SMOTE-balanced data**
📉 **Leakage Check:** Dataset verified after removing `previous_loan_defaults_on_file` to ensure fair model evaluation.

---

## 🏁 Summary

This project demonstrates a **realistic financial risk modeling workflow**, from **data preprocessing and visualization** to **model optimization and evaluation**, achieving **92% accuracy** with **XGBoost** on a **SMOTE-balanced**, **leakage-free dataset**.

---

<img width="1919" height="1018" alt="Screenshot 2025-10-25 210109" src="https://github.com/user-attachments/assets/e1f84f75-c261-4189-b84f-d86f2b0f43bf" />

<img width="1919" height="1016" alt="Screenshot 2025-10-25 210118" src="https://github.com/user-attachments/assets/2e9c295e-f50a-4390-90db-d36f080dc0a5" />

<img width="1918" height="923" alt="Screenshot 2025-10-25 210147" src="https://github.com/user-attachments/assets/4a6ba0f3-c334-478f-8c41-3249cb04688c" />

<img width="1919" height="1017" alt="Screenshot 2025-10-25 210208" src="https://github.com/user-attachments/assets/7bc590d4-0823-4a17-81eb-962521d111c0" />


