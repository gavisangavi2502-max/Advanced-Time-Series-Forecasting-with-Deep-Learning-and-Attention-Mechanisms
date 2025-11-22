
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from data_gen import generate_data
from preprocess import load_and_prepare
from model_lstm_attn import LSTMAttnModel
from model_transformer import TransformerModel
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = generate_data()
df.to_csv("data.csv", index=False)
df, scaled, sc = load_and_prepare("data.csv")

seq_len = 20
X, y = [], []
for i in range(len(scaled)-seq_len):
    X.append(scaled[i:i+seq_len])
    y.append(scaled[i+seq_len, 2])  # scaled y index

X, y = np.array(X), np.array(y)

train_end = int(0.7*len(X))
val_end = int(0.85*len(X))

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

def train_model(model):
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    train_loader = DataLoader(TensorDataset(
        torch.tensor(X_train).float(), torch.tensor(y_train).float().unsqueeze(1)), batch_size=32, shuffle=True)

    val_loader = DataLoader(TensorDataset(
        torch.tensor(X_val).float(), torch.tensor(y_val).float().unsqueeze(1)), batch_size=32)

    for epoch in range(15):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

    return model

# Train LSTM Attention
lstm = LSTMAttnModel(input_dim=X.shape[2])
lstm = train_model(lstm)
torch.save(lstm.state_dict(), "lstm_attn.pt")

# Train Transformer
trans = TransformerModel(input_dim=X.shape[2])
trans = train_model(trans)
torch.save(trans.state_dict(), "transformer.pt")

def evaluate(model):
    model.eval()
    X_t = torch.tensor(X_test).float()
    preds = model(X_t).detach().numpy().flatten()
    return preds

pred_lstm = evaluate(lstm)
pred_trans = evaluate(trans)

def metrics(p):
    mae = mean_absolute_error(y_test, p)
    rmse = mean_squared_error(y_test, p)**0.5
    mape = np.mean(np.abs((y_test - p) / y_test)) * 100
    return mae, rmse, mape

ml, rl, pl = metrics(pred_lstm)
mt, rt, pt = metrics(pred_trans)

with open("results.txt","w") as f:
    f.write(f"LSTM-Attention: MAE={ml}, RMSE={rl}, MAPE={pl}\n")
    f.write(f"Transformer: MAE={mt}, RMSE={rt}, MAPE={pt}\n")
