# sine_wave.py

import torch
import numpy as np


def generate_sine_wave(seq_len, num_sequences):
    x = np.linspace(0, 2 * np.pi, seq_len)
    y = np.sin(x)
    return torch.tensor(y).float().view(1, seq_len, 1).repeat(num_sequences, 1, 1)
