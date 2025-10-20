import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# --- 1. CONFIGURATION AND DATA LOADING ---
# ASSUMPTION: The file 'kidney_disease.csv' is in the same folder as this script.
DATA_PATH = "kidney_disease.csv"

# Column names are manually defined as the dataset lacks a header
COLUMNS = [
    'id', 'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr',
    'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm',
    'cad', 'appet', 'pe', 'ane', 'classification'
]

print("1. Loading dataset from local file...")
try:
    # Load data locally
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS, na_values=['?'])
    df = df.drop('id', axis=1) # Drop ID column as it is not a feature
except FileNotFoundError:
    print(f"Error: File '{DATA_PATH}' not found. Please ensure the CSV file is downloaded and placed in the same folder as ckd_predictor.py.")
    exit()
except Exception as e:
    print(f"Error loading data. Error: {e}")
    exit()


# --- 2. DATA CLEANING AND PREPROCESSING ---

print("2. Cleaning and converting data types...")

# Separate features into Numerical and Categorical lists
numerical_cols = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
target_col = 'classification'


# *** FIX: Force all numerical columns to numeric type to catch hidden strings/corrupt data ***
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# Fix inconsistent string values (Common issue in this dataset)
df['classification'] = df['classification'].replace(
    {'ckd\t': 'ckd', 'notckd': 'notckd'}
)
df['dm'] = df['dm'].replace({'\tno': 'no', '\tyes': 'yes', ' yes': 'yes'})
df['cad'] = df['cad'].replace({'\tno': 'no'})
df['pcv'] = df['pcv'].replace({'\t?': np.nan})
df['wc'] = df['wc'].replace({'\t?': np.nan, '\t6200': '6200', '\t8400': '8400'})
df['rc'] = df['rc'].replace({'\t?': np.nan})

# Convert columns that should be numeric but are currently objects (due to '?' or '\t' values)
# NOTE: This block is now less critical due to the explicit loop above, but kept for robustness.
cols_to_convert = ['pcv', 'wc', 'rc']
for col in cols_to_convert:
    df[col] = pd.to_numeric(df[col], errors='coerce')


# Label Encode the target variable ('ckd' -> 0, 'notckd' -> 1)
le = LabelEncoder()
df[target_col] = le.fit_transform(df[target_col])


# --- 3. HANDLING MISSING VALUES (Imputation) ---

# Imputer for Numerical Features (using Median)
# This step now receives clean numerical data (mostly floats/NaNs)
num_imputer = SimpleImputer(strategy='median')
df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

# Imputer for Categorical Features (using Most Frequent/Mode)
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])


# --- 4. FEATURE ENCODING & SCALING ---

print("3. Encoding and scaling features...")

# One-Hot Encode categorical features
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separate features (X) and target (y)
X = df.drop(target_col, axis=1)
y = df[target_col]

# Scale the numerical features (important for most ML algorithms)
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X)


# --- 5. MODEL TRAINING ---

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 # REMOVED: stratify=y
)

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)

print("4. Training Random Forest Classifier...")
# FIX APPLIED HERE: Use X_train and y_train for training
model.fit(X_train, y_train)


# --- 6. MODEL EVALUATION ---

y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)


print("\n--- 5. MODEL PERFORMANCE EVALUATION ---")
print(f"Model Used: Random Forest Classifier")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nConfusion Matrix (Rows=Actual, Columns=Predicted):")
print(conf_matrix)
print("\nClassification Report:")
print(report)

print("\n(Interpretation: 'ckd' is class 0, 'notckd' is class 1)")

# --- 7. FEATURE IMPORTANCE (Explainability) ---

print("\n--- 6. FEATURE IMPORTANCE ---")
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
top_10_features = feature_importances.nlargest(10)
print("Top 10 Most Important Features for Prediction:")
print(top_10_features.to_string())


# --- 8. SAVING MODEL ARTIFACTS ---
joblib.dump(model, 'ckd_random_forest_model.pkl')
joblib.dump(scaler, 'ckd_scaler.pkl')
joblib.dump(list(X.columns), 'model_columns.pkl') # Save the column order

print("\nModel, Scaler, and Column List saved successfully.")
