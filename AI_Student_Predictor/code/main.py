# main.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# Load dataset
data = pd.read_csv('C:/OneDrive/Desktop/CODE/AI_Student_Predictor/dataset/student_performance.csv')


# Create 'pass/fail' column based on scores
# You can modify this logic as per your requirement
data['pass/fail'] = np.where((data['math score'] >= 50) & (data['reading score'] >= 50) & (data['writing score'] >= 50), 1, 0)

# Preprocessing
label_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

for col in label_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Feature and Target
X = data.drop('pass/fail', axis=1)  # Assuming target column is 'pass/fail'
y = data['pass/fail']

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# Model 2: Support Vector Machine
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

# Ensure 'code' directory exists
os.makedirs('code', exist_ok=True)

# Save better model
if accuracy_score(y_test, y_pred_rf) > accuracy_score(y_test, y_pred_svm):
    joblib.dump(rf, 'code/best_model.pkl')
else:
    joblib.dump(svm, 'code/best_model.pkl')

# Save scaler
joblib.dump(scaler, 'code/scaler.pkl')


