# test_implementation.py

from src.utils.sine_wave import generate_sine_wave
from src.sLSTM import sLSTM

import torch
import torch.nn as nn

n_embed = 128
seq_len = 100
num_sequences = 1
batch_size = 32

# Generate sine wave data
data = generate_sine_wave(seq_len=seq_len, num_sequences=num_sequences)

# Expand data to match the embedding size and batch size
data = data.repeat(batch_size, 1, n_embed)

# Define the model, optimizer, and loss function
model = sLSTM(n_embed)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Initialize hidden state
h = torch.zeros(batch_size, n_embed)

for epoch in range(200):
    optimizer.zero_grad()
    total_loss = 0
    for t in range(seq_len - 1):
        x = data[
            :, t : t + 1, :
        ]  # Select timestep t with shape (batch_size, 1, n_embed)
        y_true = data[:, t + 1, :]  # Select the next timestep as target
        y_pred, h = model(x, h)  # Forward pass through the model
        h = (
            h.detach()
        )  # Detach the hidden state to prevent backprop through the entire sequence
        loss = criterion(y_pred, y_true)  # Compute loss for the current step
        total_loss += loss  # Accumulate the loss tensor

    total_loss.backward()  # Backpropagate once per epoch
    optimizer.step()  # Update model parameters
    print(f"Epoch {epoch}, Loss: {total_loss.item()}")
