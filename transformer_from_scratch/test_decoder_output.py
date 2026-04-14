import torch

from decoder_block import DecoderBlock
from utils import print_tensor_info, set_seed


def test_decoder_block_without_masks() -> None:
    print("\n=== Testing DecoderBlock without masks ===")
    set_seed(42)

    batch_size = 2
    target_seq_len = 5
    source_seq_len = 6
    d_model = 16
    num_heads = 4
    d_ff = 64

    x = torch.randn(batch_size, target_seq_len, d_model)
    encoder_output = torch.randn(batch_size, source_seq_len, d_model)

    decoder_block = DecoderBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.1,
    )

    output, self_attention_weights, cross_attention_weights = decoder_block(
        x=x,
        encoder_output=encoder_output,
    )

    print_tensor_info("decoder input x", x)
    print_tensor_info("encoder_output", encoder_output)
    print_tensor_info("decoder output", output)
    print_tensor_info("self_attention_weights", self_attention_weights)
    print_tensor_info("cross_attention_weights", cross_attention_weights)

    print("\nExpected output shape:")
    print("(batch_size, target_seq_len, d_model)")
    print(output.shape)

    print("\nExpected self-attention weights shape:")
    print("(batch_size, num_heads, target_seq_len, target_seq_len)")
    print(self_attention_weights.shape)

    print("\nExpected cross-attention weights shape:")
    print("(batch_size, num_heads, target_seq_len, source_seq_len)")
    print(cross_attention_weights.shape)


def test_decoder_block_with_causal_mask() -> None:
    print("\n=== Testing DecoderBlock with causal self-attention mask ===")
    set_seed(42)

    batch_size = 1
    target_seq_len = 4
    source_seq_len = 5
    d_model = 8
    num_heads = 2
    d_ff = 32

    x = torch.randn(batch_size, target_seq_len, d_model)
    encoder_output = torch.randn(batch_size, source_seq_len, d_model)

    self_attention_mask = (
        torch.tril(torch.ones(target_seq_len, target_seq_len, dtype=torch.int64))
        .unsqueeze(0)
        .unsqueeze(0)
    )

    decoder_block = DecoderBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.1,
    )

    output, self_attention_weights, cross_attention_weights = decoder_block(
        x=x,
        encoder_output=encoder_output,
        self_attention_mask=self_attention_mask,
    )

    print_tensor_info("decoder input x", x)
    print_tensor_info("encoder_output", encoder_output)
    print_tensor_info("self_attention_mask", self_attention_mask)
    print_tensor_info("decoder output", output)
    print_tensor_info("self_attention_weights", self_attention_weights)
    print_tensor_info("cross_attention_weights", cross_attention_weights)

    print("\nCausal self-attention mask:")
    print(self_attention_mask[0, 0])

    print("\nSelf-attention weights for head 0:")
    print(self_attention_weights[0, 0])

    print("\nCross-attention weights for head 0:")
    print(cross_attention_weights[0, 0])


def test_decoder_block_with_cross_attention_mask() -> None:
    print("\n=== Testing DecoderBlock with cross-attention mask ===")
    set_seed(42)

    batch_size = 1
    target_seq_len = 3
    source_seq_len = 4
    d_model = 8
    num_heads = 2
    d_ff = 32

    x = torch.randn(batch_size, target_seq_len, d_model)
    encoder_output = torch.randn(batch_size, source_seq_len, d_model)

    self_attention_mask = (
        torch.tril(torch.ones(target_seq_len, target_seq_len, dtype=torch.int64))
        .unsqueeze(0)
        .unsqueeze(0)
    )

    # block the last source position from being attended to
    cross_attention_mask = torch.tensor(
        [[[[1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0]]]], dtype=torch.int64
    )

    decoder_block = DecoderBlock(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        dropout=0.1,
    )

    output, self_attention_weights, cross_attention_weights = decoder_block(
        x=x,
        encoder_output=encoder_output,
        self_attention_mask=self_attention_mask,
        cross_attention_mask=cross_attention_mask,
    )

    print_tensor_info("decoder input x", x)
    print_tensor_info("encoder_output", encoder_output)
    print_tensor_info("cross_attention_mask", cross_attention_mask)
    print_tensor_info("decoder output", output)
    print_tensor_info("self_attention_weights", self_attention_weights)
    print_tensor_info("cross_attention_weights", cross_attention_weights)

    print("\nCross-attention mask:")
    print(cross_attention_mask[0, 0])

    print("\nCross-attention weights for head 0:")
    print(cross_attention_weights[0, 0])


if __name__ == "__main__":
    test_decoder_block_without_masks()
    test_decoder_block_with_causal_mask()
    test_decoder_block_with_cross_attention_mask()
