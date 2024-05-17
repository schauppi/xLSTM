import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from src.sLSTMCell import sLSTMCell

# Generate a sine wave
num_points = 100
time = np.linspace(0, 4 * np.pi, num_points)
data = np.sin(time)
data = torch.tensor(data, dtype=torch.float32)

# Hyperparameters
input_size = 1
hidden_size = 10
output_size = 1
num_layers = 1
learning_rate = 0.01
num_epochs = 500

# Model, loss function, and optimizer
model = sLSTMCell(input_size, hidden_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):

    hidden = model.init_hidden(1)

    optimizer.zero_grad()
    loss = 0

    for i in range(num_points - 1):
        inputs = data[i].view(1, 1)
        targets = data[i + 1].view(1, 1)
        outputs, hidden = model(inputs, hidden)
        loss += criterion(outputs, targets)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

"""# Test the model
model.eval()
with torch.no_grad():
    predictions = []
    hidden = model.init_hidden(1)
    for i in range(num_points - 1):
        inputs = data[i].view(1, 1)
        outputs, hidden = model(inputs, hidden)
        predictions.append(outputs.item())

# Plot the actual and predicted sine wave
plt.plot(time[1:], data[1:], label="Actual")
plt.plot(time[1:], predictions, label="Predicted")
plt.legend()
plt.show()"""
