import torch
import torch.nn as nn


class sLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(sLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.W = nn.Parameter(
            nn.init.xavier_uniform_(
                torch.randn(input_size + hidden_size, 4 * hidden_size)
            ),
            requires_grad=True,
        )

        if self.bias:
            self.B = nn.Parameter((torch.zeros(4 * hidden_size)), requires_grad=True)

        self.linear = nn.Linear(hidden_size, 1)

    def init_hidden(self, batch_size):
        return (
            torch.zeros(batch_size, self.hidden_size),
            torch.zeros(batch_size, self.hidden_size),
            torch.ones(batch_size, self.hidden_size),
            torch.ones(batch_size, self.hidden_size),
        )

    def forward(self, x, hidden):
        # Unpack hidden states -> (batch_size, hidden_size)
        h, c, n, m = hidden

        # Linear transformation -> (batch_size, 4 * hidden_size)
        gates = torch.matmul(torch.cat((x, h), dim=1), self.W)

        if self.bias:
            gates += self.B

        # Split the concatenated gates into their respective components -> (batch_size, hidden_size)
        zt, it, ft, ot = torch.split(gates, self.hidden_size, dim=1)

        # Cell input - equation (11) -> (batch_size, hidden_size)
        zt = torch.tanh(zt)

        # Input gate - equation (12) -> (batch_size, hidden_size)
        it = torch.exp(it)

        # Forget gate - equation (13) -> (batch_size, hidden_size)
        ft = torch.sigmoid(ft)

        # Output gate - equation (14) -> (batch_size, hidden_size)
        ot = torch.sigmoid(ot)

        # Stabilization state - equation (15) -> (batch_size, hidden_size)
        mt = torch.max(torch.log(ft) + m, torch.log(it))

        # Input stabilization state - equation (16) -> (batch_size, hidden_size)
        ip = torch.exp(it - mt)

        # Cell state - equation (8) -> (batch_size, hidden_size)
        ct = ft * c + ip * zt

        # Normalization state - equation (9) -> (batch_size, hidden_size)
        nt = ft * n + ip

        # Hidden state - equation (10) -> (batch_size, hidden_size)
        ht = ot * (c / nt)

        # Map the hidden state to the output -> (batch_size, 1)
        output = self.linear(ht)

        # -> (batch_size, 1), (batch_size, hidden_size, batch_size, hidden_size, batch_size, hidden_size, batch_size, hidden_size)
        return output, (ht, ct, nt, mt)
