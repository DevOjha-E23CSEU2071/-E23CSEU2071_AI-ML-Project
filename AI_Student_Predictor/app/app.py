from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model and scaler
model = joblib.load('../code/best_model.pkl')
scaler = joblib.load('../code/scaler.pkl')

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            int(request.form['gender']),
            int(request.form['race']),
            int(request.form['education']),
            int(request.form['lunch']),
            int(request.form['prep']),
            float(request.form['math']),
            float(request.form['reading']),
            float(request.form['writing'])
        ]
        scaled_features = scaler.transform([features])
        prediction = model.predict(scaled_features)[0]
        return render_template('index.html', prediction=f'Student will {"PASS" if prediction==1 else "FAIL"}')
    except Exception as e:
        return render_template('index.html', prediction=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
