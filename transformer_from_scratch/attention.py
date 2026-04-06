import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        query: (batch_size, num_heads, query_len, head_dim)
        key:   (batch_size, num_heads, key_len, head_dim)
        value: (batch_size, num_heads, key_len, head_dim)
        mask:  broadcastable to (batch_size, num_heads, query_len, key_len)

        returns:
            output:            (batch_size, num_heads, query_len, head_dim)
            attention_weights: (batch_size, num_heads, query_len, key_len)
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) // math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)

        return output, attention_weights
