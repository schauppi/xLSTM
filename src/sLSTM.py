import torch
import torch.nn as nn

torch.manual_seed(42)

# https://github.com/jadechip/nanoXLSTM/blob/master/model.py
# https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091


class sLSTM_base(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.W = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.U = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x, hidden_states):
        batch_size, seq_len, _ = x.size()

        # Compute the input and recurrent pre-activations
        Z = self.W(x) + self.U(hidden_states)

        # Split into i, f, c, o gates
        i, f, c, o = Z.chunk(4, dim=-1)

        # Apply exponential activations for input and forget gates
        i = torch.exp(i)
        f = torch.exp(f)

        # Compute cell state and normalizer state
        cell_state = f * hidden_states + i * torch.tanh(c)
        normalizer_state = f + i

        # Stabilize and normalize cell state
        cell_state = cell_state / normalizer_state

        # Compute output gate and hidden state
        o = torch.sigmoid(o)
        hidden_state = o * torch.tanh(cell_state)

        # Apply output projection
        hidden_state = self.o_proj(hidden_state)

        return hidden_state


class sLSTM(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.n_embed = n_embed

        # Input
        self.W_i = nn.Linear(n_embed, n_embed)
        self.U_i = nn.Linear(n_embed, n_embed)

        # Forget
        self.W_f = nn.Linear(n_embed, n_embed)
        self.U_f = nn.Linear(n_embed, n_embed)

        # Cell
        self.W_c = nn.Linear(n_embed, n_embed)
        self.U_c = nn.Linear(n_embed, n_embed)

        # Output
        self.W_o = nn.Linear(n_embed, n_embed)
        self.U_o = nn.Linear(n_embed, n_embed)

        # Output projection
        self.output_projection = nn.Linear(n_embed, n_embed)

    def forward(self, x, h):
        batch_size, seq_len, n_embed = x.size()

        # Initialize cell state as zeros
        c = torch.zeros_like(h)  # Ensuring `c` matches `h` in shape

        # Pre-activations
        Z_i = torch.zeros((batch_size, seq_len, self.n_embed))
        Z_f = torch.zeros((batch_size, seq_len, self.n_embed))
        Z_c = torch.zeros((batch_size, seq_len, self.n_embed))
        Z_o = torch.zeros((batch_size, seq_len, self.n_embed))

        for t in range(seq_len):
            x_t = x[:, t, :]
            h_t = h[:, t, :]  # Slice h to match the timestep
            Z_i[:, t, :] = self.W_i(x_t) + self.U_i(h_t)
            Z_f[:, t, :] = self.W_f(x_t) + self.U_f(h_t)
            Z_c[:, t, :] = self.W_c(x_t) + self.U_c(h_t)
            Z_o[:, t, :] = self.W_o(x_t) + self.U_o(h_t)

            # Update hidden and cell states
            i = torch.exp(Z_i[:, t, :])
            f = torch.exp(Z_f[:, t, :])
            c[:, t, :] = f * c[:, t, :] + i * torch.tanh(
                Z_c[:, t, :]
            )  # Update cell state
            o = torch.sigmoid(Z_o[:, t, :])
            h[:, t, :] = o * torch.tanh(c[:, t, :])  # Update hidden state

        # Apply final projection
        hidden_state = self.output_projection(h[:, -1, :])  # Use the last timestep

        return hidden_state, c


config = type(
    "config", (object,), {"n_embd": 10}
)  # Example config with embedding size 10
model_original = sLSTM_base(config)
model_rewritten = sLSTM(10)  # Assuming embedding size is directly passed

# Initialize input tensors
batch_size = 1
seq_len = 5
n_embd = config.n_embd
x = torch.randn(batch_size, seq_len, n_embd)
hidden_states = torch.randn(
    batch_size, seq_len, n_embd
)  # Adjust shape if your rewritten code uses (h, c)

output_original = model_original(x, hidden_states)
output_rewritten, cell_state_rewritten = model_rewritten(
    x, hidden_states
)  # Unpack the tuple to get hidden_state

print(output_original)
print(output_rewritten)

# Check if the outputs are close enough
print(
    "Outputs are close:", torch.allclose(output_original, output_rewritten, atol=1e-5)
)
