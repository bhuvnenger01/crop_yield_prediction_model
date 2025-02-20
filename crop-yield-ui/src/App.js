import React, { useState } from 'react';
import axios from 'axios';

function App() {
    const [city, setCity] = useState('');
    const [soilHealth, setSoilHealth] = useState('');
    const [prediction, setPrediction] = useState(null);

    const getPrediction = async () => {
        try {
            const response = await axios.post('http://127.0.0.1:5000/predict', {
                city,
                soil_health: soilHealth
            });
            setPrediction(response.data.crop_yield);
        } catch (error) {
            console.error("Error fetching prediction", error);
        }
    };

    return (
        <div style={{ textAlign: "center", marginTop: "50px" }}>
            <h1>🌾 Crop Yield Prediction</h1>
            <input 
                type="text" 
                placeholder="Enter City" 
                value={city} 
                onChange={(e) => setCity(e.target.value)} 
            />
            <input 
                type="text" 
                placeholder="Enter Soil Health (1-10)" 
                value={soilHealth} 
                onChange={(e) => setSoilHealth(e.target.value)} 
            />
            <button onClick={getPrediction}>Predict</button>
            {prediction && <h2>Predicted Yield: {prediction} tons/hectare</h2>}
        </div>
    );
}

export default App;
