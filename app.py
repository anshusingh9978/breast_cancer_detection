from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# 🚀 This will now show About page first
@app.route('/')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = ''
    if request.method == 'POST':
        try:
            feature_names = [
                'radius_mean', 'texture_mean', 'perimeter_mean',
                'area_mean', 'smoothness_mean', 'compactness_mean'
            ]
            features = [float(request.form[feat]) for feat in feature_names]
            scaled = scaler.transform([features])
            prediction = model.predict(scaled)[0]
            result = 'Malignant (Cancer)' if prediction == 1 else 'Benign (No Cancer)'
        except Exception as e:
            result = f"Error: {str(e)}"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
