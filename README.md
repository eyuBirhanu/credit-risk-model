# 🏦 Credit Risk Scoring Model (Bati Bank)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.0-009688?logo=fastapi&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3.0-F7931E?logo=scikit-learn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.8.0-0194E2?logo=mlflow&logoColor=white)
![Code Style](https://img.shields.io/badge/Code%20Style-Flake8-black)

## 📌 Project Overview
**Bati Bank** is partnering with an eCommerce platform to introduce a **Buy-Now-Pay-Later (BNPL)** service. This project implements a **Credit Scoring Model** to estimate the likelihood of a loan default.

> **Objective:** Categorize users into **High Risk** (Bad) and **Low Risk** (Good) groups to optimize loan approvals and minimize financial loss.

---

## 💼 Business Understanding & Compliance

### 🏛️ Basel II Capital Accord
This project adheres to the **Internal Ratings-Based (IRB)** approach:
- **Risk Measurement:** We calculate the **Probability of Default (PD)**.
- **Auditability:** The model must be interpretable, not a "black box".
- **Strategy:** Prefer interpretable models (Logistic Regression + WoE) or use SHAP values for complex models (XGBoost).

### 📊 The Proxy Variable Strategy (RFM)
Since the dataset lacks historical default labels, we engineer a **proxy variable** using **RFM Analysis**:

| Component | Definition | Assumption |
| :--- | :--- | :--- |
| **Recency (R)** | Days since last transaction | **Low R** = Active/Engaged |
| **Frequency (F)** | Total number of transactions | **High F** = Committed User |
| **Monetary (M)** | Total spend amount | **High M** = High Value |

> **Classification Logic:** High F, High M, and Low R users are "Good" (Low Risk). Inactive or low-value users are "Bad" (High Risk).

---

## 🤖 Model Strategy

We evaluate two distinct approaches to balance accuracy and interpretability:

| Approach | Pros | Cons |
| :--- | :--- | :--- |
| **Logistic Regression (WoE)** | ✅ Highly interpretable<br>✅ Standard in banking<br>✅ Easy regulatory compliance | ❌ May miss complex, non-linear patterns |
| **Gradient Boosting (XGBoost/LGBM)** | ✅ High predictive accuracy<br>✅ Handles non-linear data well | ❌ "Black Box" nature<br>❌ Requires SHAP for explainability |

---

## 📁 Project Structure

```text
credit-risk-model/
├── .github/workflows/   # 🚀 CI/CD pipeline
├── data/                # 💾 Raw and Processed Data
├── notebooks/           # 📓 EDA and Prototyping
├── src/                 # 🛠️ Production Source Code
│   ├── api/             #    └── FastAPI implementation
│   ├── data_processing.py
│   └── train.py
├── tests/               # 🧪 Unit Tests
├── Dockerfile           # 🐳 Containerization
└── requirements.txt     # 📦 Dependencies
```

---

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8+
- Git

### 1. Clone the Repository
```bash
git clone <repo_url>
cd credit-risk-model
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Run Exploratory Data Analysis (EDA)
Launch the Jupyter Notebook to explore the dataset and RFM analysis:
```bash
jupyter notebook notebooks/eda.ipynb
```

### Start the Prediction API
Run the FastAPI server locally:
```bash
uvicorn src.api.main:app --reload
```
> The API will be available at `http://127.0.0.1:8000`. API docs at `/docs`.