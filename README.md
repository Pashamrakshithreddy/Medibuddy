# 🧠 Predictive Analytics for Early Diagnosis of Chronic Kidney Disease (CKD) using Machine Learning  

![CKD](https://img.shields.io/badge/ML-Predictive%20Analytics-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

---

## 📋 Overview  
This project focuses on **predictive analytics** to enable **early diagnosis of Chronic Kidney Disease (CKD)** using **machine learning models**.  
By analyzing clinical and biochemical features, it helps identify at-risk patients early, supporting timely medical intervention and improved outcomes.

---

## 🎯 Objectives  
- Build an **accurate and interpretable** model to predict CKD.  
- Preprocess and clean real-world healthcare data.  
- Compare multiple ML algorithms for best performance.  
- Provide **explainability** using SHAP.  
- Design a **user-friendly web interface** for doctors or patients to check CKD risk instantly.

---

## ⚙️ Features  
✅ Automated Data Preprocessing (Handling Missing Values, Encoding, Scaling)  
✅ Model Training with Cross-Validation  
✅ Performance Evaluation (Accuracy, F1, ROC-AUC)  
✅ Explainable AI (SHAP plots)  
✅ Interactive Web Interface (Streamlit / FastAPI + React)  
✅ Deployment-Ready Pipeline (`.joblib` format)

---

## 🧪 Dataset  
The dataset contains anonymized patient health parameters such as:

| Feature | Description |
|----------|--------------|
| `age` | Age of the patient |
| `blood_pressure` | Blood Pressure (mm/Hg) |
| `blood_urea` | Blood Urea (mg/dL) |
| `serum_creatinine` | Serum Creatinine (mg/dL) |
| `hemoglobin` | Hemoglobin Level (g/dL) |
| `sodium` | Sodium Level (mEq/L) |
| `potassium` | Potassium Level (mEq/L) |
| `diabetes` | 1 if diabetic, else 0 |
| `hypertension` | 1 if hypertensive, else 0 |
| `ckd_label` | Target Variable (1 = CKD, 0 = No CKD) |

> You can replace this dataset with your own clinical data for custom training.

---

## 🧠 Machine Learning Workflow  

1. **Data Loading & Cleaning**  
2. **Exploratory Data Analysis (EDA)**  
3. **Feature Engineering & Scaling**  
4. **Model Training (Logistic Regression, Random Forest, XGBoost)**  
5. **Model Evaluation (Confusion Matrix, ROC-AUC, Precision-Recall)**  
6. **Explainability using SHAP**  
7. **Deployment as a Web App**

---

## 📊 Results  
- **Best Model:** XGBoost Classifier  
- **Accuracy:** 97%  
- **ROC-AUC:** 0.98  
- **Precision:** 96%  
- **Recall (Sensitivity):** 98%  
- **F1-Score:** 0.97  

---

## 💻 Tech Stack  
| Category | Technologies |
|-----------|---------------|
| **Programming Language** | Python 3.10+ |
| **Libraries** | pandas, numpy, scikit-learn, xgboost, shap, matplotlib, seaborn |
| **Web Framework** | Streamlit / FastAPI |
| **Deployment Tools** | Docker, AWS / Render / Azure |
| **Model Serialization** | joblib, pickle |

---

## 🧩 Installation & Usage  
### 
🔹 Step 1: Clone the repository  

git clone https://github.com/pashamrakshithreddy/Medibudgit](https://github.com/Pashamrakshithreddy/Medibuddy.git
cd Medibuddy.py
 🔹Step 2: Install dependencies
 
pip install -r requirements.txt

🔹 Step 3: Run the model or web app
# To run backend API
python app.py

# To run UI using Streamlit
streamlit run app.py
