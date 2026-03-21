import torch
import torch.nn as nn


class VanillaRNNCell(nn.Module):
    """
    Vanilla RNN cell (single time step).

    Equations:
        h_t = tanh(x_t @ W_xh + h_prev @ W_hh + b_h)
        y_t = h_t @ W_hy + b_y

    Shape cheat sheet:
        x_t:   (B, D)
        h_prev:(B, H)
        W_xh:  (D, H)
        W_hh:  (H, H)
        b_h:   (H,)
        W_hy:  (H, O)
        b_y:   (O,)
        h_t:   (B, H)
        y_t:   (B, O)
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Trainable parameters for hidden state update.
        self.W_xh = nn.Parameter(torch.randn(input_dim, hidden_dim) * 0.1)
        self.W_hh = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.1)
        self.b_h = nn.Parameter(torch.zeros(hidden_dim))

        # Trainable parameters for output projection.
        self.W_hy = nn.Parameter(torch.randn(hidden_dim, output_dim) * 0.1)
        self.b_y = nn.Parameter(torch.zeros(output_dim))

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor):
        """
        Args:
            x_t: (B, D)
            h_prev: (B, H)

        Returns:
            h_t: (B, H)
            y_t: (B, O)
        """
        h_t = torch.tanh(x_t @ self.W_xh + h_prev @ self.W_hh + self.b_h)
        y_t = h_t @ self.W_hy + self.b_y
        return h_t, y_t


class VanillaRNN(nn.Module):
    """
    Vanilla RNN over full sequence.

    Input:
        x: (B, T, D)

    Returns:
        outputs: (B, T, O)
        h_t: (B, H) final hidden state
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = VanillaRNNCell(input_dim, hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, h0: torch.Tensor | None = None):
        """
        Args:
            x: (B, T, D)
            h0: (B, H) initial hidden state, optional

        Returns:
            outputs: (B, T, O)
            h_t: (B, H)
        """
        batch_size, seq_len, _ = x.shape

        if h0 is None:
            h_t = torch.zeros(
                batch_size, self.hidden_dim, device=x.device, dtype=x.dtype
            )
        else:
            h_t = h0

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (B, D)
            h_t, y_t = self.cell(x_t, h_t)
            outputs.append(y_t)

        outputs = torch.stack(outputs, dim=1)  # (B, T, O)
        return outputs, h_t


if __name__ == "__main__":
    # Quick smoke test.
    B, T, D, H, out_dim = 2, 4, 3, 5, 2
    x = torch.randn(B, T, D)

    rnn = VanillaRNN(input_dim=D, hidden_dim=H, output_dim=out_dim)
    outputs, h_final = rnn(x)

    print("x shape:", x.shape)
    print("outputs shape:", outputs.shape)
    print("h_final shape:", h_final.shape)
