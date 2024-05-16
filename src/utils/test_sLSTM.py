import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.sLSTM import sLSTM
from src.utils.sine_wave import generate_sine_wave

input_size = 5
hidden_size = 10
mem_dim = 5
seq_len = 100
num_sequences = 1

data = generate_sine_wave(
    seq_len=seq_len, num_sequences=num_sequences, input_size=input_size
)

model = sLSTM(input_size=input_size, hidden_size=hidden_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(500):
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

test_output = []
states = model.init_hidden_params(num_sequences)
for t in range(seq_len):
    x = data[:, t]
    y_pred, states = model(x, states)
    test_output.append(y_pred.detach().numpy().ravel()[0])

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.title(f"sLSTM - Original vs Predicted Sine Wave, hidden_size={hidden_size}")
plt.plot(data[0, :, 0].numpy(), label="Original")
plt.plot(test_output, label="Predicted")
plt.legend()
plt.savefig(f"images/sLSTM_{hidden_size}.png")