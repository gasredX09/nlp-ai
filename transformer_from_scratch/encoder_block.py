from typing import Optional, Tuple

import torch
import torch.nn as nn

from attention import MultiHeadAttention
from feed_forward import AddNorm, PositionwiseFeedForward


class EncoderBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.add_norm1 = AddNorm(d_model=d_model, dropout=dropout)

        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model, d_ff=d_ff, dropout=dropout
        )
        self.add_norm2 = AddNorm(d_model=d_model, dropout=dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x:    (batch_size, seq_len, d_model)
        mask: broadcastable to (batch_size, num_heads, seq_len, seq_len)

        returns:
            output:            (batch_size, seq_len, d_model)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        attention_output, attention_weights = self.self_attention(x, x, x, mask=mask)
        x = self.add_norm1(x, attention_output)

        ff_output = self.feed_forward(x)
        x = self.add_norm2(x, ff_output)

        return x, attention_weights
