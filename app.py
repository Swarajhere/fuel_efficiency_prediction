from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model and scalers
model = load_model('saved_models/mlp_fuel_efficiency_model.keras')
scaler_X = joblib.load('saved_models/scaler_X.pkl')
scaler_y = joblib.load('saved_models/scaler_y.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Prepare input features
    input_features = np.array([[data['cylinders'], data['displacement'],
                                 data['horsepower'], data['weight'],
                                 data['acceleration'], data['model_year'],
                                 data['origin']]])

    # Scale input features
    scaled_input = scaler_X.transform(input_features)
    print("Scaled Input:", scaled_input)  # Debugging line

    # Predict the scaled value
    scaled_prediction = model.predict(scaled_input)
    print("Scaled Prediction:", scaled_prediction)  # Debugging line

    # Inverse transform to get the original scale
    original_prediction = scaler_y.inverse_transform(scaled_prediction)

    # Convert prediction to a native float
    predicted_mpg = original_prediction[0][0].item()
    print("Predicted MPG:", predicted_mpg)  # Debugging line

    return jsonify({'predicted_mpg': predicted_mpg})



if __name__ == "__main__":
    app.run(debug=True)
