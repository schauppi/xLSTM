import torch
import torch.nn as nn

torch.manual_seed(42)


class mLSTM(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.n_embeb = n_embed

        # Query
        self.W_q = nn.Linear(n_embed, n_embed)
        # Key
        self.W_k = nn.Linear(n_embed, n_embed)
        # Value
        self.W_v = nn.Linear(n_embed, n_embed)
        # Forget
        self.W_f = nn.Linear(n_embed, n_embed)
        # Input
        self.W_i = nn.Linear(n_embed, n_embed)
        # Output projection
        self.output_projection = nn.Linear(n_embed, n_embed)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_q(x)

        f = torch.sigmoid(self.W_f(x))
        i = torch.exp(self.W_i(x))
