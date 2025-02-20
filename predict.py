# predict.py
import torch
import numpy as np
import pickle
import requests
from models.gcn import GCN
from models.embedding_predictor import EmbeddingPredictor
from utils.data_processing import preprocess_input
import torch.nn as nn

class LSTMYieldPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMYieldPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def fetch_weather(lat, lon, api_key='YOUR_OPENWEATHER_API_KEY'):
    url = f'http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric'
    try:
        response = requests.get(url).json()
        if response['cod'] != 200:
            raise ValueError(f"API Error: {response['message']}")
        return {
            'temperature': response['main']['temp'],
            'humidity': response['main']['humidity'],
            'rainfall': response.get('rain', {}).get('1h', 0)
        }
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return {'temperature': 25, 'humidity': 80, 'rainfall': 0}

def load_models(run_id):
    model = GCN(in_channels=7, hidden_channels=256, out_channels=22)
    model.load_state_dict(torch.load(f'models/gcn_run{run_id}.pth', weights_only=True))
    embedding_predictor = EmbeddingPredictor(7, 256, 22)
    embedding_predictor.load_state_dict(torch.load(f'models/embedding_predictor_run{run_id}.pth', weights_only=True))
    lstm = LSTMYieldPredictor(input_size=7, hidden_size=64, output_size=1)
    lstm.load_state_dict(torch.load(f'models/lstm_run{run_id}.pth', weights_only=True))
    with open(f'models/rf_classifier_run{run_id}.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('models/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    return model, embedding_predictor, rf, le, lstm

rabi_crops = ['chickpea', 'lentil', 'kidneybeans', 'wheat']
kharif_crops = ['rice', 'maize', 'cotton', 'pigeonpeas', 'jute']

def recommend_crop_and_yield(N, P, K, temperature, humidity, pH, rainfall, season='kharif', run_id=1):
    model, embedding_predictor, rf, le, lstm = load_models(run_id)
    input_data = preprocess_input(N, P, K, temperature, humidity, pH, rainfall)
    
    # Crop recommendation
    with torch.no_grad():
        predicted_embedding = embedding_predictor(input_data)
    probs = rf.predict_proba(predicted_embedding.numpy())[0]
    top_indices = np.argsort(probs)[-3:][::-1]
    top_crops = le.inverse_transform(top_indices)
    valid_crops = rabi_crops if season.lower() == 'rabi' else kharif_crops
    filtered_crops = [crop for crop in top_crops if crop in valid_crops]
    crops = filtered_crops[:3] if filtered_crops else top_crops[:1]
    
    # Yield prediction
    with torch.no_grad():
        lstm_input = input_data.unsqueeze(1)
        yield_pred = lstm(lstm_input).item()
    
    return crops, yield_pred

if __name__ == "__main__":
    api_key = '204365a64a6e01e8c3ee829aced1886b'  # Replace with your key
    lat, lon = 28.6139, 77.2090  # New Delhi
    weather = fetch_weather(lat, lon, api_key)
    
    sample_input = {
        'N': 90, 'P': 42, 'K': 43,
        'temperature': weather['temperature'],
        'humidity': weather['humidity'],
        'pH': 6.5,
        'rainfall': weather['rainfall']
    }
    print(f"Weather Data: Temp={weather['temperature']}°C, Humidity={weather['humidity']}%, Rainfall={weather['rainfall']}mm")
    for run in range(1, 3):
        print(f"\nRun {run} Predictions:")
        crops, yield_pred = recommend_crop_and_yield(**sample_input, season='kharif', run_id=run)
        print("Kharif Crops:", crops)
        print("Predicted Yield:", yield_pred)
        crops, yield_pred = recommend_crop_and_yield(**sample_input, season='rabi', run_id=run)
        print("Rabi Crops:", crops)
        print("Predicted Yield:", yield_pred)