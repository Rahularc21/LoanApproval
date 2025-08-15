import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Starting model training...")

# Load and clean data
df = pd.read_csv('Copy of loan_approval_ - loan_approval_impure.csv.csv')
print(f"Data loaded: {df.shape}")

# Clean data
df = df[~df['LoanApproved'].isnull()]
df = df[df['LoanApproved'].isin([0, 1])]

# Handle missing values
for col in ['ApplicantIncome', 'LoanAmount', 'CreditScore']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

# Create feature
df['IncomePerLoan'] = df['ApplicantIncome'] / (df['LoanAmount'].fillna(0) + 1)

# Encode categorical variables
le_edu = LabelEncoder()
le_self = LabelEncoder()
df['Education_encoded'] = le_edu.fit_transform(df['Education'].fillna('Graduate'))
df['SelfEmployed_encoded'] = le_self.fit_transform(df['SelfEmployed'].fillna('No'))

# Prepare features
feature_cols = ['ApplicantIncome', 'LoanAmount', 'CreditScore', 'IncomePerLoan', 'Education_encoded', 'SelfEmployed_encoded']
X = df[feature_cols]
y = df['LoanApproved'].astype(int)

print(f"Features prepared: {X.shape}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
print("Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_acc = accuracy_score(y_test, lr_pred)
print(f"LR Accuracy: {lr_acc:.4f}")

# Train Decision Tree
print("Training Decision Tree...")
dt_model = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_leaf=3)
dt_model.fit(X_train, y_train)  # No scaling needed for DT
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
print(f"DT Accuracy: {dt_acc:.4f}")

# Save models
print("Saving models...")
joblib.dump(lr_model, 'loan_approval_lr_cloud.pkl')
joblib.dump(dt_model, 'loan_approval_dt_cloud.pkl')
joblib.dump(scaler, 'loan_approval_scaler_cloud.pkl')

# Save encoders
joblib.dump(le_edu, 'education_encoder.pkl')
joblib.dump(le_self, 'selfemployed_encoder.pkl')

print("✅ Models saved successfully!")
print(f"Files created:")
print("- loan_approval_lr_cloud.pkl")
print("- loan_approval_dt_cloud.pkl") 
print("- loan_approval_scaler_cloud.pkl")
print("- education_encoder.pkl")
print("- selfemployed_encoder.pkl")

