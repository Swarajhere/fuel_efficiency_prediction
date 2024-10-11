import React, { useState } from "react";
import "./App.css";

function App() {
  const [inputData, setInputData] = useState({
    cylinders: "",
    displacement: "",
    horsepower: "",
    weight: "",
    acceleration: "",
    model_year: "",
    origin: "",
  });

  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setInputData({
      ...inputData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      setError(null); // Clear any previous errors
      const response = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(inputData),
      });

      if (!response.ok) {
        throw new Error("Network response was not ok");
      }

      const data = await response.json();
      setPrediction(data.prediction); // Set the prediction result from backend
    } catch (error) {
      setError(error.message); // Capture and display any errors
    }
  };

  return (
    <div className="App">
      <h1>Fuel Efficiency Predictor</h1>

      {/* Form for user input */}
      <form onSubmit={handleSubmit}>
        <input
          type="number"
          name="cylinders"
          value={inputData.cylinders}
          onChange={handleChange}
          placeholder="Cylinders"
          required
        />
        <input
          type="number"
          name="displacement"
          value={inputData.displacement}
          onChange={handleChange}
          placeholder="Displacement"
          required
        />
        <input
          type="number"
          name="horsepower"
          value={inputData.horsepower}
          onChange={handleChange}
          placeholder="Horsepower"
          required
        />
        <input
          type="number"
          name="weight"
          value={inputData.weight}
          onChange={handleChange}
          placeholder="Weight"
          required
        />
        <input
          type="number"
          name="acceleration"
          value={inputData.acceleration}
          onChange={handleChange}
          placeholder="Acceleration"
          required
        />
        <input
          type="number"
          name="model_year"
          value={inputData.model_year}
          onChange={handleChange}
          placeholder="Model Year"
          required
        />
        <input
          type="number"
          name="origin"
          value={inputData.origin}
          onChange={handleChange}
          placeholder="Origin (1-USA, 2-Europe, 3-Asia)"
          required
        />
        <button type="submit">Predict</button>
      </form>

      {/* Display the prediction result */}
      {prediction !== null && (
        <div>
          <h3>Predicted Fuel Efficiency (MPG): {prediction.toFixed(2)}</h3>
        </div>
      )}

      {/* Display any errors */}
      {error && (
        <div>
          <p style={{ color: "red" }}>Error: {error}</p>
        </div>
      )}
    </div>
  );
}

export default App;
