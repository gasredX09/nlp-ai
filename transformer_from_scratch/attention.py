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
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, d_model)
        returns: (batch_size, num_heads, seq_len, head_dim)
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        x = x.transpose(1, 2)
        return x

    def _combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, num_heads, seq_len, head_dim)
        returns: (batch_size, seq_len, d_model)
        """
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, seq_len, self.d_model)
        return x

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        query: (batch_size, query_len, d_model)
        key:   (batch_size, key_len, d_model)
        value: (batch_size, key_len, d_model)
        mask:  broadcastable to (batch_size, num_heads, query_len, key_len)

        returns:
            output:            (batch_size, query_len, d_model)
            attention_weights: (batch_size, num_heads, query_len, key_len)
        """
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        attention_output, attention_weights = self.attention(q, k, v, mask=mask)
        combined = self._combine_heads(attention_output)
        output = self.out_proj(combined)

        return output, attention_weights
