import torch

from attention import ScaledDotProductAttention
from utils import set_seed, print_tensor_info


def main() -> None:
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

    print("\nAttention Weights shape should be:")
    print("(batch_size, num_heads, query_len, key_len)")
    print(attention_weights.shape)

    print("\nOutput shape should be:")
    print("(batch_size, num_heads, query_len, head_dim)")
    print(output.shape)

    print("\nCheck that attention weights sum to 1 across key positions:")
    row_sums = attention_weights.sum(dim=-1)
    print(row_sums)

    print("\n--- Masked attention test ---")

    mask = torch.tensor(
        [[[[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]]], dtype=torch.int64
    )

    mask = mask.repeat(batch_size, 1, 1, 1)
    masked_output, masked_attention_weights = attention(query, key, value, mask=mask)

    print_tensor_info("mask", mask)
    print_tensor_info("masked_attention_weights", masked_attention_weights)
    print_tensor_info("masked_output", masked_output)

    print("\nMasked attention weights:")
    print(masked_attention_weights)

    print("\nRow sums after masking:")
    print(masked_attention_weights.sum(dim=-1))

    # Experiments
    print(attention_weights[0, 0])
    print(masked_attention_weights[0, 0])


if __name__ == "__main__":
    main()
