import torch
from rnn_cell import VanillaRNNCell, VanillaRNN


def test_cell_shapes():
    # B=batch, D=input_dim, H=hidden_dim, O=output_dim
    B, D, H, O = 4, 8, 16, 6

    x_t = torch.randn(B, D)  # single time step input
    h_prev = torch.randn(B, H)  # previous hidden state

    cell = VanillaRNNCell(input_dim=D, hidden_dim=H, output_dim=O)
    h_t, y_t = cell(x_t, h_prev)

    print("[Cell] x_t shape:", x_t.shape)
    print("[Cell] h_prev shape:", h_prev.shape)
    print("[Cell] h_t shape:", h_t.shape)
    print("[Cell] y_t shape:", y_t.shape)

    assert h_t.shape == (B, H)
    assert y_t.shape == (B, O)


def test_sequence_shapes():
    # B=batch, T=sequence length, D=input_dim, H=hidden_dim, O=output_dim
    B, T, D, H, O = 3, 7, 5, 11, 4

    x = torch.randn(B, T, D)

    model = VanillaRNN(input_dim=D, hidden_dim=H, output_dim=O)
    outputs, h_final = model(x)

    print("[Seq] x shape:", x.shape)
    print("[Seq] outputs shape:", outputs.shape)
    print("[Seq] h_final shape:", h_final.shape)

    assert outputs.shape == (B, T, O)
    assert h_final.shape == (B, H)


if __name__ == "__main__":
    test_cell_shapes()
    test_sequence_shapes()
    print("All shape checks passed.")
