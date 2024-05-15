import numpy as np
import torch
import torch.nn as nn

from src.mLSTM import mLSTM


def generate_sine_wave(seq_len, num_sequences):
    x = np.linspace(0, 2 * np.pi, seq_len)
    y = np.sin(x)
    return torch.tensor(y).float().view(-1, 1).repeat(1, num_sequences).unsqueeze(0)


input_size = 1
hidden_size = 5
mem_dim = 5
seq_len = 100
num_sequences = 2

data = generate_sine_wave(seq_len=seq_len, num_sequences=num_sequences)
model = mLSTM(input_size=input_size, hidden_size=hidden_size, mem_dim=mem_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterien = nn.MSELoss()

for epoch in range(200):
    states = (torch.zeros(mem_dim, mem_dim), torch.zeros(mem_dim, 1))
