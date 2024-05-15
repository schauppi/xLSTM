import torch
import torch.nn as nn
import math

torch.manual_seed(42)


class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, mem_dim):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mem_dim = mem_dim

        # Query
        self.Wq = nn.Parameter(torch.randn(hidden_size, input_size))
        self.bq = nn.Parameter(torch.randn(hidden_size, 1))
        # Key
        self.Wk = nn.Parameter(torch.randn(mem_dim, input_size))
        self.bk = nn.Parameter(torch.randn(mem_dim, 1))
        # Value
        self.Wv = nn.Parameter(torch.randn(mem_dim, input_size))
        self.bV = nn.Parameter(torch.randn(mem_dim, 1))
        # Input
        self.Wi = nn.Parameter(torch.randn(1, input_size))
        self.bi = nn.Parameter(torch.randn(1))
        # Forget
        self.Wf = nn.Parameter(torch.randn(1, input_size))
        self.bf = nn.Parameter(torch.randn(1))
        # Out
        self.Wo = nn.Parameter(torch.randn(1, input_size))
        self.bo = nn.Parameter(torch.randn(1))

    def forward(self, x, hidden_states):
        cp, np = hidden_states

        qt = torch.matmul(self.Wq, x) + self.bq
        kt = (1, math.sqrt(self.mem_dim)) * (torch.matmul(self.Wk, x) + self.bk)
        vt = torch.matmul(self.Wv, x) + self.bv

        # Input gate
        it_tilde = torch.matmul(self.Wi, x) + self.bi
        it = torch.exp(it_tilde)

        # Forget gate
        ft_tilde = torch.matmul(self.Wf, x) + self.bf
        # Test torch.sigmoid as well - stated in the paper formula 26
        ft = torch.exp(ft_tilde)

        # Remove dimension
        vt = vt.squeeze()
        kt = kt.squeeze()

        # cell state calculation using outer product
        C = ft * cp + it * torch.ger(vt, kt)
        # normalizer state
        n = ft * np + it * kt.unsqueeze(1)

        # hidden state - stated in paper formula 21
        h_tilde = torch.matmul(C, qt) / (
            torch.max(torch.abs(torch.matmul(n.T, qt)), torch.tensor(1.0))
        )
        # output gate
        ot = torch.sigmoid(torch.matmul(self.Wo, x) + self.bo)

        # Hidden state
        ht = ot * h_tilde

        return ht, (C, n)
