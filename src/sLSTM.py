import torch
import torch.nn as nn

torch.manual_seed(42)


class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Cell input (zt)
        self.Wz = nn.Linear(input_size, hidden_size)
        self.Rz = nn.Linear(hidden_size, hidden_size)
        self.bz = nn.Parameter(torch.zeros(hidden_size))

        # Input gate (it)
        self.Wi = nn.Linear(input_size, hidden_size)
        self.Ri = nn.Linear(hidden_size, hidden_size)
        self.bi = nn.Parameter(torch.zeros(hidden_size))

        # Forget gate (ft)
        self.Wf = nn.Linear(input_size, hidden_size)
        self.Rf = nn.Linear(hidden_size, hidden_size)
        self.bf = nn.Parameter(torch.zeros(hidden_size))

        # Output gate (ot)
        self.Wo = nn.Linear(input_size, hidden_size)
        self.Ro = nn.Linear(hidden_size, hidden_size)
        self.bo = nn.Parameter(torch.zeros(hidden_size))

        # Output projection layer
        self.output_projection = nn.Linear(hidden_size, input_size)

    def _initialize_weights(self):
        for param in self.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)

    def init_hidden_params(self, batch_size):
        h_init = torch.zeros(batch_size, self.hidden_size)
        c_init = torch.zeros(batch_size, self.hidden_size)
        n_init = torch.ones(
            batch_size, self.hidden_size
        )  # Set to ones to avoid division by zero
        return (h_init, c_init, n_init)

    def forward(self, x, hidden_states):
        hp, cp, np = hidden_states

        # Cell input - equation (11)
        zt = torch.tanh(self.Wz(x) + self.Rz(hp) + self.bz)

        # Input gate - equation (12)
        it = torch.exp(self.Wi(x) + self.Ri(hp) + self.bi)

        # Forget gate - equation (13)
        ft = torch.sigmoid(self.Wf(x) + self.Rf(hp) + self.bf)

        # Output gate - equation (14)
        ot = torch.sigmoid(self.Wo(x) + self.Ro(hp) + self.bo)

        # Cell state - equation (8)
        C = ft * cp + it * zt

        # Normalizer state - equation (9)
        n = ft * np + it

        # Hidden state - equation (10)
        h_tilde = C / (n + 1e-7)

        # Update hidden state - equation (10)
        h = ot * torch.tanh(h_tilde)

        # Project the hidden state to the input size
        y = self.output_projection(h)

        return y, (h, C, n)
