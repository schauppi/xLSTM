import torch
import torch.nn as nn
import math

torch.manual_seed(42)


class mLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.Wi = nn.Parameter(
            nn.init.xavier_uniform_(torch.randn(input_size, hidden_size)),
            requires_grad=True,
        )

        self.Wf = nn.Parameter(
            nn.init.xavier_uniform_(torch.randn(input_size, hidden_size)),
            requires_grad=True,
        )

        self.Wo = nn.Parameter(
            nn.init.xavier_uniform_(torch.randn(input_size, hidden_size)),
            requires_grad=True,
        )

        self.Wq = nn.Parameter(
            nn.init.xavier_uniform_(torch.randn(input_size, hidden_size)),
            requires_grad=True,
        )

        self.Wk = nn.Parameter(
            nn.init.xavier_uniform_(torch.randn(input_size, hidden_size)),
            requires_grad=True,
        )

        self.Wv = nn.Parameter(
            nn.init.xavier_uniform_(torch.randn(input_size, hidden_size)),
            requires_grad=True,
        )

        if self.bias:
            self.bi = nn.Parameter((torch.zeros(hidden_size)), requires_grad=True)
            self.bf = nn.Parameter((torch.zeros(hidden_size)), requires_grad=True)
            self.bo = nn.Parameter((torch.zeros(hidden_size)), requires_grad=True)
            self.bq = nn.Parameter((torch.zeros(hidden_size)), requires_grad=True)
            self.bk = nn.Parameter((torch.zeros(hidden_size)), requires_grad=True)
            self.bv = nn.Parameter((torch.zeros(hidden_size)), requires_grad=True)

    def init_hidden(self, batch_size):
        return (
            torch.zeros(batch_size, self.hidden_size, self.hidden_size),
            torch.zeros(batch_size, self.hidden_size),
            torch.zeros(batch_size, self.hidden_size),
        )

    def forward(self, x, hidden):

        # Unpack hidden states -> (batch_size, hidden_size)
        c, n, m = hidden

        # Input gate - equation (25) -> (batch_size, hidden_size)
        i_tilde = (
            torch.matmul(x, self.Wi) + self.bi
            if self.bias
            else torch.matmul(x, self.Wi)
        )

        # Forget gate - equation (26) -> (batch_size, hidden_size)
        f_tilde = (
            torch.matmul(x, self.Wf) + self.bf
            if self.bias
            else torch.matmul(x, self.Wf)
        )

        # Output gate - equation (27) -> (batch_size, hidden_size)
        o_tilde = (
            torch.matmul(x, self.Wo) + self.bo
            if self.bias
            else torch.matmul(x, self.Wo)
        )

        # Query input - equation (22) -> (batch_size, hidden_size)
        qt = (
            torch.matmul(x, self.Wq) + self.bq
            if self.bias
            else torch.matmul(x, self.Wq)
        )

        # Key input - equation (23) -> (batch_size, hidden_size)
        kt = (
            (torch.matmul(x, self.Wk) + self.bk)
            / torch.sqrt(torch.tensor(self.hidden_size))
            if self.bias
            else torch.matmul(x, self.Wk) / torch.sqrt(torch.tensor(self.hidden_size))
        )

        # Value input - equation (24) -> (batch_size, hidden_size)
        vt = (
            torch.matmul(x, self.Wv) + self.bv
            if self.bias
            else torch.matmul(x, self.Wv)
        )

        # Input gate - equation (25) -> (batch_size, hidden_size)
        it = torch.exp(i_tilde)

        # Forget gate - equation (26) -> (batch_size, hidden_size)
        ft = torch.sigmoid(f_tilde)

        # Output gate - equation (27) -> (batch_size, hidden_size)
        ot = torch.sigmoid(o_tilde)

        # Stabilization state - equation (15) -> (batch_size, hidden_size)
        mt = torch.max(torch.log(ft) + m, torch.log(it))

        # Input stabilization state - equation (16) -> (batch_size, hidden_size)
        ip = torch.exp(i_tilde - mt)

        # Cell state - equation (19) -> (batch_size, hidden_size)
        ct = ft.unsqueeze(-1) * c + torch.einsum("bi,bk->bik", vt, kt)

        # Normalizer state - equation (20) -> (batch_size, hidden_size)
        nt = ft * n + ip * kt

        # Normalization inner product
        normalize_inner = torch.diagonal(torch.matmul(nt, qt.T))

        # Hidden state part 1 - equation (21) -> (batch_size, hidden_size)
        h_tilde = torch.einsum("bkj,bj -> bk", ct, qt) / torch.max(
            torch.abs(normalize_inner), torch.ones_like(normalize_inner)
        ).view(-1, 1)

        # Hidden state part 2 - equation (21) -> (batch_size, hidden_size)
        ht = ot * h_tilde

        return ht, (ct, nt, mt)
