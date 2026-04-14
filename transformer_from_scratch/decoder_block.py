from typing import Tuple, Optional

import torch
import torch.nn as nn

from attention import MultiHeadAttention
from feed_forward import AddNorm, PositionwiseFeedForward


class DecoderBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.add_norm1 = AddNorm(d_model=d_model, dropout=dropout)

        self.cross_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.add_norm2 = AddNorm(d_model=d_model, dropout=dropout)

        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model, d_ff=d_ff, dropout=dropout
        )
        self.add_norm3 = AddNorm(d_model=d_model, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x:                 (batch_size, target_seq_len, d_model)
        encoder_output:    (batch_size, source_seq_len, d_model)

        self_attention_mask:
            broadcastable to
            (batch_size, num_heads, target_seq_len, target_seq_len)

        cross_attention_mask:
            broadcastable to
            (batch_size, num_heads, target_seq_len, source_seq_len)

        returns:
            output:                 (batch_size, target_seq_len, d_model)
            self_attention_weights: (batch_size, num_heads, target_seq_len, target_seq_len)
            cross_attention_weights:(batch_size, num_heads, target_seq_len, source_seq_len)
        """
        self_attention_output, self_attention_weights = self.self_attention(
            x, x, x, mask=self_attention_mask
        )
        x = self.add_norm1(x, self_attention_output)

        cross_attention_output, cross_attention_weights = self.cross_attention(
            x, encoder_output, encoder_output, mask=cross_attention_mask
        )
        x = self.add_norm2(x, cross_attention_output)

        ff_output = self.feed_forward(x)
        x = self.add_norm3(x, ff_output)

        return x, self_attention_weights, cross_attention_weights
