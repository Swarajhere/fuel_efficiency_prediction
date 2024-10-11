from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load the trained model and scaler
model = joblib.load('saved_models/random_forest_model.pkl')
scaler_X = joblib.load('saved_models/scaler_X.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Receive input data in JSON format
    data = request.get_json()

    # Ensure data contains all required fields
    try:
        input_data = np.array([[data['cylinders'], data['displacement'], data['horsepower'],
                                data['weight'], data['acceleration'], data['model_year'],
                                data['origin']]])
    except KeyError as e:
        return jsonify({'error': f"Missing field: {str(e)}"}), 400

    # Debugging: Print original input data before scaling
    print(f"Original Input Data: {input_data}")

    # Scale the input data using the pre-trained scaler
    input_data_scaled = scaler_X.transform(input_data)

    # Debugging: Print scaled input data
    print(f"Scaled Input Data: {input_data_scaled}")

    # Make prediction
    prediction = model.predict(input_data_scaled)

    # Return the prediction as a JSON response
    return jsonify({'prediction': float(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
