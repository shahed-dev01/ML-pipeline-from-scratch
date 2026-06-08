# Hand made Rural Healthcare Urgent Care Predictor 🏥🩺

An end-to-end Machine Learning pipeline designed to process messy, real-world healthcare data and predict whether a patient requires urgent medical attention.

## 📌 Project Overview
In real-world scenarios, healthcare data collected from rural camps is often incomplete and unstructured. This project demonstrates a robust Machine Learning architecture that automatically cleans missing values, encodes categorical features, and trains a predictive model without data leakage.

## 🚀 Key Features
* **Dynamic Data Preprocessing:** Utilizes `scikit-learn`'s `ColumnTransformer` to handle different data types simultaneously.
* **Missing Value Imputation:** Automatically fills missing numerical data with medians and categorical data with the most frequent values.
* **Advanced Encoding:** Implements both `OneHotEncoder` (for nominal data like zones) and `OrdinalEncoder` (for ranked data like income levels).
* **Pipeline Architecture:** Prevents data leakage by encapsulating preprocessing and model training within a single `Pipeline`.
* **Robust Evaluation:** Uses K-Fold Cross-Validation to ensure the model generalizes well and avoids the illusion of overfitting.

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (RandomForestClassifier, Pipelines, Preprocessing)

## 📂 Dataset Structure (Simulated)
The model processes a simulated dataset (`rural_health_data.csv`) containing the following features:
* `Age`, `BMI`, `Blood_Sugar` (Numerical)
* `Gender`, `Village_Zone` (Nominal Categorical)
* `Income_Level` (Ordinal Categorical)
* **Target:** `Needs_Urgent_Care` (Binary: 0 or 1)

## ⚙️ How to Run
1. Clone this repository to your local machine.
2. Ensure you have the required libraries installed:
   ```bash
   pip install pandas numpy scikit-learn
