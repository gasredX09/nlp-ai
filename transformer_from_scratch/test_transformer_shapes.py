import torch

from config import TransformerConfig
from embeddings import TokenEmbedding, PositionalEncoding
from utils import print_tensor_info, set_seed


def main() -> None:
    set_seed(42)

    config = TransformerConfig(
        vocab_size=50, d_model=128, num_heads=8, d_ff=512, max_seq_len=16, dropout=0.1
    )

    batch_size = 2
    seq_len = 10

    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    token_embedding = TokenEmbedding(config.vocab_size, config.d_model)
    positional_encoding = PositionalEncoding(
        d_model=config.d_model, max_seq_len=config.max_seq_len, dropout=config.dropout
    )

    embedded = token_embedding(x)
    encoded = positional_encoding(embedded)

    print_tensor_info("input token ids", x)
    print_tensor_info("embedded", embedded)
    print_tensor_info("positions encoded", encoded)

    print("\nFirst token embedding sample:")
    print(embedded[0, 0, :8])

    print("\nFirst token after positional encoding sample:")
    print(encoded[0, 0, :8])


if __name__ == "__main__":
    main()
