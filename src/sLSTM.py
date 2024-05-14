import torch
import torch.nn as nn

torch.manual_seed(42)

# https://github.com/jadechip/nanoXLSTM/blob/master/model.py
# https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091


class sLSTM(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.n_embed = n_embed

        # Input gate
        self.W_i = nn.Linear(n_embed, n_embed)
        self.U_i = nn.Linear(n_embed, n_embed)

        # Forget gate
        self.W_f = nn.Linear(n_embed, n_embed)
        self.U_f = nn.Linear(n_embed, n_embed)

        # Cell gate
        self.W_c = nn.Linear(n_embed, n_embed)
        self.U_c = nn.Linear(n_embed, n_embed)

        # Output gate
        self.W_o = nn.Linear(n_embed, n_embed)
        self.U_o = nn.Linear(n_embed, n_embed)

        # Output projection
        self.output_projection = nn.Linear(n_embed, n_embed)

    def forward(self, x, h):
        batch_size, seq_len, _ = x.size()

        # Initialize cell state
        c = torch.zeros(batch_size, self.n_embed)

        for t in range(seq_len):
            x_t = x[:, t, :]

            # Linear transformations
            Z_i = self.W_i(x_t) + self.U_i(h)
            Z_f = self.W_f(x_t) + self.U_f(h)
            Z_c = self.W_c(x_t) + self.U_c(h)
            Z_o = self.W_o(x_t) + self.U_o(h)

            # Gates activations
            i = torch.exp(Z_i)
            f = torch.exp(Z_f)
            o = torch.sigmoid(Z_o)

            # Update cell state
            c = f * c + i * torch.tanh(Z_c)

            # Normalize cell state
            normalizer_state = f + i
            c = c / normalizer_state

            # Update hidden state
            h = o * torch.tanh(c)

        # Apply final projection
        hidden_state = self.output_projection(h)

        return hidden_state, c


# Example usage:
n_embed = 128
batch_size = 32
seq_len = 10
model = sLSTM(n_embed)
x = torch.randn(batch_size, seq_len, n_embed)
h = torch.zeros(batch_size, n_embed)
output, c = model(x, h)

print(output.shape)
