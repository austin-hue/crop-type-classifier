from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model (update the path if needed)
model = joblib.load('cropmodel.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Only use the specified features
    features = [
        float(request.form['Nitrogen']),
        float(request.form['Phosphorus']),
        float(request.form['Potassium']),
        float(request.form['Temperature']),
        float(request.form['Humidity']),
        float(request.form['pH']),
        float(request.form['Rainfall'])
    ]
    input_data = np.array([features])
    prediction = model.predict(input_data)
    return render_template('index.html', prediction_text=f'Predicted Crop Type: {prediction[0]}')

if __name__ == '__main__':
    app.run(debug=True)