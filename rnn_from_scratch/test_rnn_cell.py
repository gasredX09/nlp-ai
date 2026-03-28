import torch
from rnn_cell import VanillaRNNCell, VanillaRNN


def test_cell_shapes():
    # B=batch size, D=input dim, H=hidden dim, O=output dim
    B, D, H, O = 4, 8, 16, 6

    x_t = torch.randn(B, D)  # one time step input
    h_prev = torch.randn(B, H)  # previous hidden state

    cell = VanillaRNNCell(input_dim=D, hidden_dim=H, output_dim=O)
    h_t, y_t = cell(x_t, h_prev)

    print("Cell x_t: ", x_t.shape)  # (B, D)
    print("Cell h_prev: ", h_prev.shape)
    print("Cell h_t: ", h_t.shape)  # (B, H)
    print("Cell y_t: ", y_t.shape)  # (B, O)

    assert h_t.shape == (B, H), f"Expected hidden state shape (B, H), got {h_t.shape}"
    assert y_t.shape == (B, O), f"Expected output shape (B, O), got {y_t.shape}"
    print("Cell shape test passed!")


def test_sequence_shapes():
    # B=batch size, T=sequence length, D=input dim, H=hidden dim, O=output dim
    B, T, D, H, O = 3, 7, 5, 11, 4

    x = torch.randn(B, T, D)

    model = VanillaRNN(input_dim=D, hidden_dim=H, output_dim=O)
    outputs, h_final = model(x)

    print("Seq x: ", x.shape)  # (B, T, D)
    print("Seq outputs: ", outputs.shape)  # (B, T, O)
    print("Seq h_final: ", h_final.shape)  # (B, H)

    assert outputs.shape == (B, T, O), (
        f"Expected outputs shape (B, T, O), got {outputs.shape}"
    )
    assert h_final.shape == (B, H), (
        f"Expected final hidden state shape (B, H), got {h_final.shape}"
    )
    print("Sequence shape test passed!")


if __name__ == "__main__":
    test_cell_shapes()
    test_sequence_shapes()
    print("All shape checks passed!")
