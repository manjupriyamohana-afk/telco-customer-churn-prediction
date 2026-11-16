# Telecom Customer Churn Prediction & Explainability

An end-to-end machine learning project that identifies telecom subscribers at high risk of churning and explains the key drivers behind customer attrition using **XGBoost** and **SHAP**.

## Business Problem
Acquiring a new telecom customer costs significantly more than retaining an existing one. Because churners represent a minority of the customer base (~26.5%), standard models often miss high-risk users. This pipeline solves that by balancing training data with **SMOTE** and lowering the classification probability threshold from `0.50` to `0.35` to maximize churn recall.

## Key Results & Explainability
![SHAP Feature Importance](shap_churn_summary.png)

### Top Drivers of Customer Churn:
* **Contract Type:** Month-to-month subscribers show the highest likelihood of leaving compared to one- or two-year contracts.
* **Tenure:** Customers in their first 12 months (`0-1 Year`) are at peak risk; retention stabilizes sharply after two years.
* **Service Stickiness:** Customers with fewer add-on services (`Total_Services`) such as Tech Support and Online Security churn at much higher rates.

## Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning & Balancing:** Scikit-Learn, XGBoost, Imbalanced-Learn (SMOTE)
* **Explainability & Visualization:** SHAP, Matplotlib, Seaborn

## How to Run
1. Open `telco_customer_churn_prediction.ipynb` in Google Colab.
2. Download the [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle and upload `WA_Fn-UseC_-Telco-Customer-Churn.csv` to your Colab session.
3. Run all cells sequentially to train the model and generate the SHAP evaluation plots.
