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