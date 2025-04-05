from flask import Flask, render_template, request
import numpy as np
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load(r"D:\STET-details\msc-\uma maheswari\uma\Predictive-maintenance-for-healthcare-equipment-main\equipment_failure_model1.pkl")

# Define home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        usage_hours = float(request.form['usage_hours'])
        temperature = float(request.form['temperature'])
        vibration_level = float(request.form['vibration_level'])
        pressure_level = float(request.form['pressure_level'])
        last_maintenance = float(request.form['last_maintenance'])

        # Create NumPy array
        input_data = np.array([[usage_hours, temperature, vibration_level, pressure_level, last_maintenance]])

        # Make prediction
        prediction = model.predict(input_data)[0]

        # Convert prediction to message
        result = " Equipment Failure Expected! " if prediction == 1 else " Equipment is Safe!"

        return render_template('index.html', prediction=result)
    
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
