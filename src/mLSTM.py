import torch
import torch.nn as nn
import math


class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, mem_dim):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mem_dim = mem_dim

        # Query matrice and bias term
        self.Wq = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bq = nn.Parameter(torch.randn(hidden_size))
        # Key matrice and bias term
        self.Wk = nn.Parameter(torch.randn(mem_dim, input_size))
        self.bk = nn.Parameter(torch.randn(mem_dim))
        # Value matrice and bias term
        self.Wv = nn.Parameter(torch.randn(mem_dim, input_size))
        self.bv = nn.Parameter(torch.randn(mem_dim))
        # Input matrice and bias term
        self.Wi = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bi = nn.Parameter(torch.randn(hidden_size))
        # Forget matrice and bias term
        self.Wf = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bf = nn.Parameter(torch.randn(hidden_size))
        # Out matrice and bias term
        self.Wo = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bo = nn.Parameter(torch.randn(hidden_size))

    def init_hidden_params(self, batch_size):
        c_init = torch.zeros(batch_size, self.mem_dim, self.mem_dim)
        n_init = torch.zeros(batch_size, self.mem_dim)
        return (c_init, n_init)

    def forward(self, x, hidden_states):
        cp, np = hidden_states

        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        # Query input - equation (22)
        qt = torch.matmul(x, self.Wq.T) + self.bq
        # Key input - equation (23)
        kt = (1 / math.sqrt(self.mem_dim)) * (torch.matmul(x, self.Wk.T) + self.bk)
        # Value input - equation (24)
        vt = torch.matmul(x, self.Wv.T) + self.bv

        # Input gate - equation (25)
        it_tilde = torch.matmul(x, self.Wi.T) + self.bi
        it = torch.sigmoid(it_tilde)

        # Forget gate - equation (26)
        ft_tilde = torch.matmul(x, self.Wf.T) + self.bf
        ft = torch.sigmoid(ft_tilde)

        vt = vt.unsqueeze(2)
        kt = kt.unsqueeze(1)

        # Cell state - equation (19)
        C = ft.unsqueeze(2) * cp + it.unsqueeze(2) * torch.bmm(vt, kt)

        # Normalizer state - equation (20)
        n = ft * np + it * kt.squeeze(2)

        # Hidden state part 1 - equation (21)
        h_tilde = torch.matmul(C, qt.unsqueeze(2)).squeeze(2) / (
            torch.max(
                torch.abs(torch.matmul(n.unsqueeze(1), qt.unsqueeze(2))).squeeze(2),
                torch.tensor(1.0),
            )
        )

        # Output gate - equation (27)
        ot = torch.sigmoid(torch.matmul(x, self.Wo.T) + self.bo)

        # Hidden state part 2 - equation (21)
        ht = ot * h_tilde

        return ht, (C, n)
