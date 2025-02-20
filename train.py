# train.py
import torch
from torch_geometric.nn import GATConv  # Updated import
from sklearn.ensemble import RandomForestClassifier
from models.gcn import GCN
from models.embedding_predictor import EmbeddingPredictor
from utils.data_processing import load_and_prepare_data
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np

# Load and prepare data
data, le = load_and_prepare_data()

class LSTMYieldPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMYieldPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_and_evaluate(run_id, retrain=False, prev_model=None, prev_lstm=None):
    if retrain and prev_model:
        model = prev_model
    else:
        model = GCN(in_channels=7, hidden_channels=256, out_channels=len(le.classes_))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()

    if retrain and prev_lstm:
        lstm = prev_lstm
    else:
        lstm = LSTMYieldPredictor(input_size=7, hidden_size=64, output_size=1)
    optimizer_lstm = torch.optim.Adam(lstm.parameters(), lr=0.005)
    criterion_lstm = nn.MSELoss()

    def train_gcn():
        model.train()
        optimizer.zero_grad()
        out = model.classify(data)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    yield_data = torch.tensor(np.random.uniform(0, 100, len(data.y)), dtype=torch.float).unsqueeze(1)

    def train_lstm():
        lstm.train()
        optimizer_lstm.zero_grad()
        x = data.x.unsqueeze(1)
        out = lstm(x)
        loss = criterion_lstm(out[data.train_mask], yield_data[data.train_mask])
        loss.backward()
        optimizer_lstm.step()
        return loss.item()

    print(f"\nRun {run_id} - Training GCN...")
    for epoch in range(500):
        loss = train_gcn()
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                _, pred = model.classify(data).max(dim=1)
                train_acc = (pred[data.train_mask] == data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
                test_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
                print(f'Epoch {epoch}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    print(f"\nRun {run_id} - Training LSTM Yield Predictor...")
    for epoch in range(500):
        loss = train_lstm()
        if epoch % 50 == 0:
            lstm.eval()
            with torch.no_grad():
                pred_yield = lstm(data.x.unsqueeze(1))
                mse = criterion_lstm(pred_yield[data.test_mask], yield_data[data.test_mask])
                print(f'Epoch {epoch}, MSE Loss: {loss:.4f}, Test MSE: {mse:.4f}')

    model.eval()
    with torch.no_grad():
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
        print(f'Run {run_id} - Final Test Accuracy: {test_acc:.4f}')

    with torch.no_grad():
        embeddings = model(data).detach()

    rf = RandomForestClassifier(n_estimators=200)
    rf.fit(embeddings[data.train_mask].numpy(), data.y[data.train_mask].numpy())

    embedding_predictor = EmbeddingPredictor(7, 256, model.out_channels)
    optimizer_ep = torch.optim.Adam(embedding_predictor.parameters(), lr=0.005)
    criterion_ep = nn.MSELoss()

    print(f"\nRun {run_id} - Training Embedding Predictor...")
    for epoch in range(500):
        optimizer_ep.zero_grad()
        predicted_embeddings = embedding_predictor(data.x)
        loss = criterion_ep(predicted_embeddings, embeddings)
        loss.backward()
        optimizer_ep.step()
        if epoch % 50 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

    torch.save(model.state_dict(), f'models/gcn_run{run_id}.pth')
    torch.save(embedding_predictor.state_dict(), f'models/embedding_predictor_run{run_id}.pth')
    torch.save(lstm.state_dict(), f'models/lstm_run{run_id}.pth')
    with open(f'models/rf_classifier_run{run_id}.pkl', 'wb') as f:
        pickle.dump(rf, f)
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    return model, lstm

print("Initial Training...")
model, lstm = train_and_evaluate(1)

print("\nRetraining with Simulated New Data...")
data.x = data.x + torch.tensor(np.random.normal(0, 0.1, data.x.shape), dtype=torch.float)
model, lstm = train_and_evaluate(2, retrain=True, prev_model=model, prev_lstm=lstm)