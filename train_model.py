import os
import pandas as pd
import numpy as np
import joblib  # For saving and loading the scalers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow import keras

# Create directories if they don't exist
os.makedirs('saved_models', exist_ok=True)


# Load and prepare the data
def load_data():
    """Loads and prepares the dataset for training."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']

    df = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)

    # Drop missing values
    df = df.dropna()

    # Map Origin to numerical values: 1 = USA, 2 = Europe, 3 = Asia
    df['Origin'] = df['Origin'].map({1: 1, 2: 2, 3: 3})

    X = df.drop('MPG', axis=1)
    y = df['MPG']

    return X, y


# Create and compile the model (MLP)
def create_mlp_model(input_shape):
    """Creates and compiles a fully connected neural network (MLP) model."""
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=input_shape),
        keras.layers.Dropout(0.2),  # Add dropout for regularization
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)  # Output layer
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# Train the model
def train_model():
    """Main function to train the MLP model."""
    X, y = load_data()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features (X) and target (y)
    scaler_X = StandardScaler()
    scaler_y = MinMaxScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1))

    # Create and train the model
    model = create_mlp_model(input_shape=(X_train_scaled.shape[1],))
    model.fit(X_train_scaled, y_train_scaled, epochs=100, validation_split=0.2)

    # Save the model and scalers
    model.save('saved_models/mlp_fuel_efficiency_model.keras')
    joblib.dump(scaler_X, 'saved_models/scaler_X.pkl')  # Save feature scaler
    joblib.dump(scaler_y, 'saved_models/scaler_y.pkl')  # Save target scaler


if __name__ == "__main__":
    train_model()
