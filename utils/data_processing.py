# utils/data_processing.py
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import LabelEncoder
import pickle

def load_and_prepare_data(file_path='data/crop_recommendation.csv'):
    df = pd.read_csv(file_path)
    features = df[['N', 'P', 'K', 'temperature', 'humidity', 'pH', 'rainfall']].values
    
    # Normalize features
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    features = (features - mean) / std
    
    # Save mean and std
    with open('models/data_stats.pkl', 'wb') as f:
        pickle.dump({'mean': mean, 'std': std}, f)
    
    # Create edges based on feature similarity
    distances = cdist(features, features)
    edge_index = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if distances[i, j] < 0.5:
                edge_index.append([i, j])
                edge_index.append([j, i])
    edge_index = np.array(edge_index).T
    
    x = torch.tensor(features, dtype=torch.float)
    le = LabelEncoder()
    labels = le.fit_transform(df['label'])
    y = torch.tensor(labels, dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    train_mask = np.zeros(len(df), dtype=bool)
    train_idx = np.random.choice(len(df), int(0.8 * len(df)), replace=False)
    train_mask[train_idx] = True
    test_mask = ~train_mask
    
    data = Data(x=x, edge_index=edge_index, y=y, 
                train_mask=torch.tensor(train_mask), test_mask=torch.tensor(test_mask))
    return data, le

def preprocess_input(N, P, K, temperature, humidity, pH, rainfall):
    with open('models/data_stats.pkl', 'rb') as f:
        stats = pickle.load(f)
        mean = stats['mean']
        std = stats['std']
    input_data = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
    input_data = (input_data - mean) / std
    return torch.tensor(input_data, dtype=torch.float)