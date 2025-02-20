# models/embedding_predictor.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddingPredictor(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(EmbeddingPredictor, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x