import torch
import torch.nn as nn


class VanillaRNNCell(nn.Module):
    """
    Single RNN time step.
    h_t = tanh(x_t @ W_xh + h_prev @ W_hh + b_h)
    y_t = h_t @ W_hy + b_y
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Weights for hidden state computation
        self.W_xh = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))

        # Weights for output computation
        self.W_hy = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.01)
        self.b_y = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        """
        Single RNN time step.

        Args:
            x_t: (B, D) — input at time t
            h_prev: (B, H) — hidden state from t-1

        Returns:
            h_t: (B, H) — new hidden state
            y_t: (B, O) — output logits
        """
        # Compute the new hidden state and output
        h_t = torch.tanh(x_t @ self.W_xh + h_prev @ self.W_hh + self.b_h)
        y_t = h_t @ self.W_hy + self.b_y
        return h_t, y_t


class VanillaRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = VanillaRNNCell(input_dim, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        """
        x: (B, T, D)
        h0: (B, H) or None
        returns:
            outputs: (B, T, O)
            h_t: (B, H) final hidden state
        """
        B, T, D = x.shape

        if h0 is None:
            h_t = torch.zeros(B, self.hidden_dim, device=x.device, dtype=x.dtype)
        else:
            h_t = h0

        outputs = []
        for t in range(T):
            x_t = x[:, t, :]  # (B, D)
            h_t, y_t = self.cell(x_t, h_t)
            outputs.append(y_t)  # each y_t is (B, O)

        outputs = torch.stack(outputs, dim=1)  # (B, T, O)
        return outputs, h_t
