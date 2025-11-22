
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from data_gen import generate_data
from preprocess import load_data
from model_lstm_attn import LSTMAttnModel
from model_transformer import TransformerModel

df = generate_data()
df.to_csv("data.csv", index=False)

df, scaled, sc = load_data()
seq_len = 20
X, y = [], []
for i in range(len(scaled)-seq_len):
    X.append(scaled[i:i+seq_len])
    y.append(scaled[i+seq_len, -1])
X, y = np.array(X), np.array(y)

X_t = torch.tensor(X, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True)

model = LSTMAttnModel(input_dim=3)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

for epoch in range(5):
    for xb, yb in loader:
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()

torch.save(model.state_dict(), "lstm_attn.pt")
