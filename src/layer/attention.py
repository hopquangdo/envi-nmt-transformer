import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """(B, T, d_model) -> (B, num_heads, T, d_k)"""
        B, T, _ = x.size()
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None):
        """
        q: (B, T_q, d_model)
        k: (B, T_k, d_model)  — T_k có thể khác T_q trong cross-attention
        v: (B, T_k, d_model)
        mask: broadcastable to (B, num_heads, T_q, T_k)
        """
        B, T_q, _ = q.size()

        q = self._split_heads(self.q_linear(q))   # (B, H, T_q, d_k)
        k = self._split_heads(self.k_linear(k))   # (B, H, T_k, d_k)
        v = self._split_heads(self.v_linear(v))   # (B, H, T_k, d_k)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, T_q, T_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        attn = self.dropout(attn_weights)
        out = attn @ v  # (B, H, T_q, d_k)

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.out(out)
