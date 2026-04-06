# Transformer Notes

## Step 1
Implemented:
- config
- token embeddings
- sinusoidal positional encoding

Key understanding:
- attention alone does not encode order
- positional information must be injected
- embeddings and positional encodings must have the same dimension
- output shape after embedding and positional encoding is:
  (batch_size, seq_len, d_model)

## Step 2
Implemented:
- scaled dot-product attention
- optional masking

Key understanding:
- raw attention scores come from Q @ K^T
- scores are scaled by sqrt(d_k)
- softmax is applied over key positions
- attention output is a weighted sum of V
- output shape is (batch_size, num_heads, query_len, head_dim)
- attention weights shape is (batch_size, num_heads, query_len, key_len)