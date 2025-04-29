# -E23CSEU2071_AI-ML-Project
Title: AI-Based Student Performance Predictor Subtitle: Predicting student pass/fail status using machine learning

#E23CSEU2071_AI/ML-Project
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
data = pd.read_csv('dataset/student_performance.csv')
label_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
for col in label_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
X = data.drop('pass/fail', axis=1)
y = data['pass/fail']
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
if accuracy_score(y_test, y_pred_rf) > accuracy_score(y_test, y_pred_svm):
    joblib.dump(rf, 'code/best_model.pkl')
else:
    joblib.dump(svm, 'code/best_model.pkl')
joblib.dump(scaler, 'code/scaler.pkl')
