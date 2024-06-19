# app.py

from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

model = joblib.load('model.pkl')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    features = [float(request.form['feature1']), 
                float(request.form['feature2']), 
                float(request.form['feature3']), 
                float(request.form['feature4']), 
                float(request.form['feature5']), 
                float(request.form['feature6']), 
                float(request.form['feature7']), 
                float(request.form['feature8']), 
                float(request.form['feature9'])]
    
    
    features_array = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features_array)
    
    # Determine the prediction result
    if prediction[0] == 1:
        result = 'Potable'
    else:
        result = 'Non-Potable'
    
    
    return render_template('index.html', prediction_text=f'The water quality is predicted as: {result}')


if __name__ == '__main__':
    app.run(debug=True)
