import torch

from attention import ScaledDotProductAttention, MultiHeadAttention
from utils import set_seed, print_tensor_info


def test_scaled_dot_product_attention() -> None:
    print("\n=== Testing ScaledDotProductAttention ===")
    set_seed(42)

    batch_size = 2
    num_heads = 1
    query_len = 3
    key_len = 4
    head_dim = 8

    query = torch.randn(batch_size, num_heads, query_len, head_dim)
    key = torch.randn(batch_size, num_heads, key_len, head_dim)
    value = torch.randn(batch_size, num_heads, key_len, head_dim)

    attention = ScaledDotProductAttention()
    output, attention_weights = attention(query, key, value)

    print_tensor_info("query", query)
    print_tensor_info("key", key)
    print_tensor_info("value", value)
    print_tensor_info("attention_weights", attention_weights)
    print_tensor_info("output", output)

    print("\nAttention row sums:")
    print(attention_weights.sum(dim=-1))


def test_multihead_attention() -> None:
    print("\n=== Testing MultiHeadAttention ===")
    set_seed(42)

    batch_size = 2
    seq_len = 5
    d_model = 16
    num_heads = 4

    x = torch.randn(batch_size, seq_len, d_model)

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    output, attention_weights = mha(x, x, x)

    print_tensor_info("input x", x)
    print_tensor_info("output", output)
    print_tensor_info("attention_weights", attention_weights)

    print("\nExpected output shape:")
    print("(batch_size, seq_len, d_model)")
    print(output.shape)

    print("\nExpected attention weights shape:")
    print("(batch_size, num_heads, seq_len, seq_len)")
    print(attention_weights.shape)

    print("\nAttention row sums per head:")
    print(attention_weights.sum(dim=-1))


def test_multi_head_attention_with_mask() -> None:
    print("\n=== Testing MultiHeadAttention with mask ===")
    set_seed(42)

    batch_size = 1
    seq_len = 4
    d_model = 8
    num_heads = 2

    x = torch.randn(batch_size, seq_len, d_model)

    # Simple lower-triangular causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.int64))
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    output, attention_weights = mha(x, x, x, mask=mask)

    print_tensor_info("input x", x)
    print_tensor_info("mask", mask)
    print_tensor_info("output", output)
    print_tensor_info("attention_weights", attention_weights)

    print("\nCausal mask:")
    print(mask[0, 0])

    print("\nAttention weights for head 0:")
    print(attention_weights[0, 0])

    print("\nAttention weights for head 1:")
    print(attention_weights[0, 1])


if __name__ == "__main__":
    test_scaled_dot_product_attention()
    test_multihead_attention()
    test_multi_head_attention_with_mask()
