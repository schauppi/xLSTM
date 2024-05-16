import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.mLSTM import mLSTM


def generate_sine_wave(seq_len, num_sequences, input_size):
    x = np.linspace(0, 2 * np.pi, seq_len)
    y = np.sin(x)
    y = np.repeat(y.reshape(1, seq_len, 1), num_sequences, axis=0)
    y = np.repeat(y, input_size, axis=2)
    return torch.tensor(y).float()


input_size = 5
hidden_size = 5
mem_dim = 5
seq_len = 100
num_sequences = 2

data = generate_sine_wave(
    seq_len=seq_len, num_sequences=num_sequences, input_size=input_size
)

# [batch_size, seq_len, hidden_size/features]

model = mLSTM(input_size=input_size, hidden_size=hidden_size, mem_dim=mem_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(200):
    init_states = model.init_hidden_params(num_sequences)

    optimizer.zero_grad()
    loss = 0
    for t in range(seq_len - 1):
        x = data[:, t]
        y_true = data[:, t + 1]
        y_pred, init_states = model(x, init_states)
        loss += criterion(y_pred, y_true)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, loss {loss.item()}")
