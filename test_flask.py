import requests

url = 'http://127.0.0.1:5000/predict'
data = {
    "Cylinders": 4,
    "Displacement": 2.5,
    "Horsepower": 130,
    "Weight": 1500,
    "Acceleration": 12.5,
    "Model_Year": 2018,
    "Origin": 1  # Origin as per the rule (1: USA, 2: Europe, 3: Asia)
}

response = requests.post(url, json=data)
print("Prediction response:", response.json())
