---

```markdown
# 🏦 Loan Approval Prediction - Machine Learning Project

This project aims to build a classification model that predicts whether a loan application will be approved based on the applicant's demographic and financial information.

---

## 📌 Project Objective

Banks need a reliable way to determine the eligibility of loan applicants. By training predictive models on historical application data, we can automate and optimize the decision-making process, improving both efficiency and accuracy.

---

## 📂 Dataset Overview

The dataset contains 614 records of past loan applications with 13 features, including personal, income, credit, and property details.

### 📄 Feature Descriptions:

| Feature              | Description                               |
|----------------------|-------------------------------------------|
| `Loan_ID`            | Unique loan application ID                |
| `Gender`             | Applicant gender                          |
| `Married`            | Marital status                            |
| `Dependents`         | Number of dependents                      |
| `Education`          | Education level                           |
| `Self_Employed`      | Employment type                           |
| `ApplicantIncome`    | Monthly income of applicant               |
| `CoapplicantIncome`  | Monthly income of co-applicant            |
| `LoanAmount`         | Loan amount requested                     |
| `Loan_Amount_Term`   | Term of the loan (in months)              |
| `Credit_History`     | Credit history (1: good, 0: bad)          |
| `Property_Area`      | Urban / Semiurban / Rural                 |
| `Loan_Status`        | Target variable (Y = Approved, N = Denied)|

---

## 🔧 Technologies Used

- **Python 3.11**
- `pandas`, `numpy` – Data manipulation
- `matplotlib`, `seaborn` – Data visualization
- `scikit-learn` – Model building and evaluation
- `xgboost` – Boosting classifier
- `RandomizedSearchCV` – Hyperparameter tuning

---

## 🧹 Data Preprocessing

✔ Missing values handled:
- **Numerical columns**: filled with **mean**
- **Categorical columns**: filled with **mode**

✔ Feature Engineering:
- `Total_Income`: Sum of applicant and co-applicant income
- Log-transformed skewed features for normalization:
  - `ApplicantIncomeLog`, `LoanAmountLog`, `Total_Income_Log`, etc.

✔ Categorical encoding:
- `LabelEncoder` used for features like `Gender`, `Education`, etc.

✔ Dropped unnecessary columns:
- `Loan_ID`, original income and amount columns (replaced with transformed ones)

---

## 📈 Exploratory Data Analysis

- Visualized income, loan amount, and other distributions
- Plotted correlation heatmap to explore feature relationships
- Identified skewness and normalized key features using log transformations

---

## 🤖 Models Trained

| Model                | Type               |
|----------------------|--------------------| 
| Decision Tree        | Tree-based         |
| Random Forest        | Ensemble (Bagging) |
| Extra Trees          | Ensemble (Bagging) |
| XGBoost (optional)   | Ensemble (Boosting)|

---

## 🧪 Model Evaluation

Each model was evaluated using:
- **Test Set Accuracy**
- **5-Fold Cross-Validation Score**

### 📊 Results:

| Model                | Test Accuracy | Cross-Validation Score |
|----------------------|----------------|--------------------------|
| Logistic Regression  | 78.86%         | 80.95%                   |
| Random Forest        | 78.86%         | 77.85%                   |
| Extra Trees          | 73.98%         | 77.36%                   |
| Decision Tree        | 65.85%         | 70.20%                   |

> 🔍 Note: XGBoost training was skipped due to computational time limits, but it's recommended for future evaluation using `RandomizedSearchCV`.

---

## 📌 Observations & Insights

- **Logistic Regression** had the best overall generalization performance.
- **Random Forest** matched LR on test accuracy but slightly underperformed in cross-validation.
- **Extra Trees** showed consistent but slightly lower performance.
- **Decision Tree** underperformed due to overfitting on training data.

---

## 🚀 Future Improvements

✅ **Try Grid Search for exhaustive hyperparameter tuning**

✅ **Use more advanced ensemble models (LightGBM, CatBoost)**

✅ **Include SHAP or permutation importance to interpret model decisions**

✅ **Deploy the model via Flask or Streamlit for real-time predictions**

✅ **Create pipelines using `Pipeline` and `ColumnTransformer` from `sklearn` for cleaner preprocessing**

---

## 📁 Project Structure

```
Loan_Prediction_Project/
├── Loan Prediction Dataset.csv
├── loan_prediction.ipynb
├── README.md
└── models/
```
