import torch
import numpy as np


def generate_sine_wave(seq_len, num_sequences, input_size):
    x = np.linspace(0, 2 * np.pi, seq_len)
    y = np.sin(x)
    y = np.repeat(y.reshape(1, seq_len, 1), num_sequences, axis=0)
    y = np.repeat(y, input_size, axis=2)
    return torch.tensor(y).float()
