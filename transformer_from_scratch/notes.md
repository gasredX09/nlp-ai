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

## Step 3
Implemented:
- multi-head attention
- learned Q, K, V projections
- split/combine head logic
- causal masking test

Key understanding:
- input hidden states are projected into Q, K, V
- d_model is split across num_heads
- each head runs attention independently
- head outputs are concatenated back into d_model
- self-attention uses the same tensor as query, key, and value
- causal masks prevent tokens from attending to future positions

## Step 4
Implemented:
- position-wise feed-forward network
- Add & Norm utility with residual connection and layer normalization

Key understanding:
- FFN acts independently on each token position
- FFN expands from d_model to d_ff and projects back to d_model
- attention mixes information across tokens
- FFN transforms information within each token
- residual connections help preserve information and improve optimization
- layer normalization stabilizes token representations