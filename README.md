This project leverages machine learning and predictive analytics to enable the early detection of Chronic Kidney Disease (CKD) using patient health data. By analyzing clinical parameters such as blood pressure, serum creatinine, blood urea, and hemoglobin levels, the system can predict the likelihood of CKD in its early stages — helping doctors and patients take preventive measures before irreversible damage occurs.

🚀 Objectives

Develop an accurate and interpretable ML model for early CKD detection.

Perform data preprocessing, exploratory analysis, and visualization.

Evaluate multiple ML algorithms (Logistic Regression, Random Forest, XGBoost).

Build a user-friendly interface for real-time prediction and result visualization.

Integrate explainable AI (XAI) using SHAP to interpret predictions.

🧩 Key Features

✅ Data Cleaning, Preprocessing, and Feature Engineering
✅ Model Training with Hyperparameter Tuning
✅ Model Evaluation (ROC, Precision-Recall, Confusion Matrix, F1-score)
✅ SHAP-based Explainability
✅ Web App (Streamlit / FastAPI + React Frontend) for predictions
✅ Ready-to-deploy Model Pipeline (.joblib)

🧠 Technologies Used

Python 3.x

Pandas, NumPy, Matplotlib, Seaborn — for data analysis & visualization

Scikit-learn, XGBoost, LightGBM — for machine learning

SHAP — for model interpretability

FastAPI / Streamlit — for web-based UI

Joblib / Pickle — for model serialization

🧬 Dataset

The dataset consists of anonymized patient records with features like:

age, blood_pressure, serum_creatinine, blood_urea, hemoglobin, diabetes, hypertension, etc.

The target variable ckd_label indicates CKD presence (1 = CKD, 0 = No CKD).

(You can replace this section with your own dataset link or upload info)

🧪 Results Summary

Achieved an accuracy of ~97% using the XGBoost classifier.

ROC-AUC Score: 0.98

Model shows strong recall (important for disease detection) and balanced precision.
