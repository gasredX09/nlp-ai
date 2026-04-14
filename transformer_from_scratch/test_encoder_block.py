import torch

from encoder_block import EncoderBlock
from utils import print_tensor_info, set_seed


def test_encoder_block_without_mask() -> None:
    print("\n=== Testing EncoderBlock without mask ===")
    set_seed(42)

    batch_size = 2
    seq_len = 5
    d_model = 16
    num_heads = 4
    d_ff = 64

    x = torch.randn(batch_size, seq_len, d_model)

    encoder_block = EncoderBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.1,
    )

    output, attention_weights = encoder_block(x)

    print_tensor_info("input x", x)
    print_tensor_info("output", output)
    print_tensor_info("attention_weights", attention_weights)

    print("\nExpected output shape:")
    print("(batch_size, seq_len, d_model)")
    print(output.shape)

    print("\nExpected attention weights shape:")
    print("(batch_size, num_heads, seq_len, seq_len)")
    print(attention_weights.shape)

    print("\nAttention row sums:")
    print(attention_weights.sum(dim=-1))


def test_encoder_block_with_padding_style_mask() -> None:
    print("\n=== Testing EncoderBlock with mask ===")
    set_seed(42)

    batch_size = 1
    seq_len = 4
    d_model = 8
    num_heads = 2
    d_ff = 32

    x = torch.randn(batch_size, seq_len, d_model)

    # Example mask: last position is blocked as a key for everyone
    mask = torch.tensor(
        [[[[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0]]]], dtype=torch.int64
    )

    encoder_block = EncoderBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.1,
    )

    output, attention_weights = encoder_block(x, mask=mask)

    print_tensor_info("input x", x)
    print_tensor_info("mask", mask)
    print_tensor_info("output", output)
    print_tensor_info("attention_weights", attention_weights)

    print("\nMask:")
    print(mask[0, 0])

    print("\nAttention weights for head 0:")
    print(attention_weights[0, 0])

    print("\nAttention weights for head 1:")
    print(attention_weights[0, 1])


if __name__ == "__main__":
    test_encoder_block_without_mask()
    test_encoder_block_with_padding_style_mask()
