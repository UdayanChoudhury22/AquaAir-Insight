from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('air_quality.joblib')

app = Flask(__name__, template_folder='templates')


@app.route('/')
def home():
    return render_template('idx.html')


@app.route('/predict', methods=['POST'])
def predict():
   
    so2 = float(request.form['so2'])
    no2 = float(request.form['no2'])
    rspm = float(request.form['rspm'])
    spm = float(request.form['spm'])

    
    input_data = np.array([[so2, no2, rspm, spm]])
    
    prediction = model.predict(input_data)

    prediction_value = round(prediction[0], 2)
    
    prediction_range = get_air_quality(prediction)

    return render_template('idx.html', prediction=prediction_value, prediction_range=prediction_range)

def get_air_quality(aqi_value):
    
    if aqi_value <= 50:
        return 'Good'
    elif aqi_value <= 100:
        return 'Moderate'
    elif aqi_value <= 150:
        return 'Unhealthy for Sensitive Groups'
    elif aqi_value <= 200:
        return 'Unhealthy'
    elif aqi_value <= 300:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'

if __name__ == '__main__':
    app.run(debug=True)
