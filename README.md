# 📉 Customer Churn Prediction

A production-grade machine learning pipeline to predict customer churn using XGBoost.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green)
![Scikit-Learn](https://img.shields.io/badge/ScikitLearn-1.3-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🎯 Project Overview

Built an end-to-end customer churn prediction system on the Telco dataset
(7,043 customers). Achieved **80%+ accuracy** and **0.85+ ROC-AUC** using
XGBoost with optimized hyperparameters.

**Key Results:**
- ✅ Accuracy  : 80%+
- ✅ ROC-AUC   : 0.85+
- ✅ Dataset   : 7,043 customers, 19 features
- ✅ Algorithm : XGBoost (Gradient Boosting)

---

## 📊 Results

| Metric     | Score  |
|------------|--------|
| Accuracy   | 80%+   |
| ROC-AUC    | 0.85+  |
| Precision  | 0.65+  |
| Recall     | 0.55+  |

---

## 🛠️ Tech Stack

- **Python 3.13**
- **XGBoost** — Gradient boosting model
- **Scikit-learn** — Preprocessing & evaluation
- **Pandas / NumPy** — Data manipulation
- **Matplotlib / Seaborn** — Visualizations
- **Jupyter Notebook** — Development environment

---

## 📁 Project Structure
customer-churn-prediction/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_xgboost_model.ipynb
│   └── 04_evaluation.ipynb
├── models/
│   ├── xgb_churn_model.pkl
│   └── scaler.pkl
├── results/
│   ├── churn_distribution.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── roc_curve.png
│   └── precision_recall.png
├── requirements.txt
└── README.md
---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/Chepuri25/customer-churn-prediction
cd customer-churn-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run notebooks in order
```bash
jupyter notebook
```
Navigate to `notebooks/` and run 01 → 02 → 03 → 04

---

## 💡 Key Insights

- **Contract type** is the strongest predictor of churn
- **Tenure** — longer customers are less likely to churn
- **Monthly charges** — higher charges correlate with churn
- **Internet service type** significantly impacts churn rate

---

## 📞 Contact

**Pravallika Chepuri**
- 📧 Email: pchepur1@asu.edu
- 💼 LinkedIn: https://www.linkedin.com/in/pravallika-chepuri
- 🐙 GitHub: https://github.com/Chepuri25

---
**Built with ❤️ by Pravallika Chepuri | Arizona State University**
