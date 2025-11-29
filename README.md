Customer Churn Prediction — End-to-End ML Pipeline

Project Goal 
The objective of the project is **to predict customer churn** in a telecom company using machine learning.

With this prediction, one can take proactive efforts toward retaining those customers and reduce revenue loss.

This project demonstrates an entire ML workflow, from pre-processing the data, engineering features, to training, evaluating, and comparing models.


Dataset 
Dataset used: Telco Customer Churn (XLSX format)

Source: IBM sample dataset

Loaded using: pd.read_excel()

- Total entries: 7,043 customers

- Columns : 33 (demographics, account information, service usage, charges, and churn labels)

Target variable: `Churn Value` (1 = churned, 0 = stayed)



**Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
---
Tools & Libraries
- **Python**
- **Pandas** (data manipulation)
- **NumPy** (numerical operations)
- **Scikit-learn** (preprocessing, model training, metrics)

- **XGBoost** (advanced model)

- **Matplotlib / Seaborn** (visualizations)

- **Google Colab** (for running notebooks in the cloud)

---
Project Workflow 
1. **Data Loading**
Load the Telco Customer Churn dataset.
2. **Data Cleaning & Preprocessing **

- Convert `Total Charges` to numeric
• Drop unnecessary columns: `CustomerID`, `Lat Long`, `City`, `Churn Reason`, `Churn Label`
- Encode categorical variables
- Scale numeric features

- Train/Test Split
3. **Model Training**
Train three models:
Logistic Regression
- Random Forest Classifier

- Classifier XGBoost

4. **Model Evaluation**

Compare models on:

- Accuracy
Confusion Matrix
- Precision, Recall, F1-score
- ROC-AUC Score

5. **Comparison & Selection**

Compare various models in order to identify the best performing model.

--- Model Performance 
| Model                | Accuracy | ROC-AUC |
|----------------------|----------|---------| 
| Logistic Regression  | 0.9189   | 0.9755  | 
| Random Forest        | 0.9331   | 0.9788  | 
|XGBoost               |0.9260    |0.9827   |
> **Best model:** XGBoost (highest ROC-AUC) 
