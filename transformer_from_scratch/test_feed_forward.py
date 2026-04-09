import torch
from feed_forward import AddNorm, PositionwiseFeedForward
from utils import print_tensor_info, set_seed


def test_positionwise_feed_forward() -> None:
    print("\n=== Testing PositionwiseFeedForward ===")
    set_seed(42)

    batch_size = 2
    seq_len = 5
    d_model = 16
    d_ff = 64

    x = torch.randn(batch_size, seq_len, d_model)
    ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=0.1)
    output = ffn(x)

    print_tensor_info("input x", x)
    print_tensor_info("ffn output", output)

    print("\nExpected shape:")
    print("(batch_size, seq_len, d_model)")
    print(output.shape)


def test_add_norm() -> None:
    print("\n=== Testing AddNorm ===")
    set_seed(42)

    batch_size = 2
    seq_len = 5
    d_model = 16

    x = torch.randn(batch_size, seq_len, d_model)
    sublayer_output = torch.randn(batch_size, seq_len, d_model)

    add_norm = AddNorm(d_model=d_model, dropout=0.1)
    output = add_norm(x, sublayer_output)

    print_tensor_info("original x", x)
    print_tensor_info("sublayer_output", sublayer_output)
    print_tensor_info("add_norm output", output)

    print("\nExpected shape:")
    print("(batch_size, seq_len, d_model)")
    print(output.shape)


def test_tokenwise_behavior() -> None:
    print("\n=== Testing token-wise behavior of FFN ===")
    set_seed(42)

    batch_size = 1
    seq_len = 3
    d_model = 4
    d_ff = 8

    x = torch.randn(batch_size, seq_len, d_model)

    ffn = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=0.0)
    output = ffn(x)

    print("Input:")
    print(x)

    print("\nOutput:")
    print(output)

    print(
        "\nNotice that shape is preserved, and each token vector is transformed independently."
    )


if __name__ == "__main__":
    test_positionwise_feed_forward()
    test_add_norm()
    test_tokenwise_behavior()
