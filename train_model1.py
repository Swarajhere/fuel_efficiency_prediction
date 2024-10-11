import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def load_data():
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'  # URL for the dataset
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']

    print("Loading data from URL...")

    try:
        df = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)
        df = df.dropna()  # Drop missing values
        df['Origin'] = df['Origin'].map({1: 1, 2: 2, 3: 3})  # Map Origin if needed

        print("Data loaded successfully. Number of rows:", df.shape[0])
        X = df.drop('MPG', axis=1)
        y = df['MPG']

        return X, y
    except Exception as e:
        print("Error loading data:", e)
        return None, None


def train_model():
    print("Starting model training...")
    X, y = load_data()

    if X is None or y is None:
        print("Training aborted due to data loading issues.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)

    model = RandomForestRegressor()
    model.fit(X_train_scaled, y_train)

    print("Model training completed.")

    # Save the model and scaler
    joblib.dump(model, 'saved_models/random_forest_model.pkl')
    joblib.dump(scaler_X, 'saved_models/scaler_X.pkl')
    print("Model and scaler saved successfully.")


if __name__ == "__main__":
    train_model()
