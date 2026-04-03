import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Khởi tạo lớp Multi-Head Attention.

        Args:
            d_model (int): Số chiều của vector nhúng (embedding dimension). Ví dụ: 512.
            num_heads (int): Số lượng attention head song song. Phải là ước của d_model. Ví dụ: 8.
            dropout (float): Xác suất dropout áp dụng lên attention weights. Mặc định: 0.1.

        Note:
            d_k = d_model // num_heads = chiều của mỗi head. Ví dụ: 512 // 8 = 64.
        """
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
        """
        Chia nhỏ vector d_model thành nhiều head (đầu) song song.

        Input Demo:
            x: Tensor shape (B, T, d_model) -> (32, 50, 512)
        Output Demo:
            return: Tensor shape (B, num_heads, T, d_k) -> (32, 8, 50, 64)
        """
        B, T, _ = x.size()
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, q, k, v, mask=None, return_attn: bool = False):
        """
        Hàm xử lý Multi-Head Attention.

        Args:
            q (Tensor): Tensor Query shape (B, T_q, d_model). Ví dụ: (32, 20, 512).
            k (Tensor): Tensor Key shape (B, T_k, d_model). Ví dụ: (32, 20, 512).
            v (Tensor): Tensor Value shape (B, T_k, d_model). Ví dụ: (32, 20, 512).
            mask (Tensor | None): Mask nhị phân để che padding hoặc future tokens.
                - Padding mask (Encoder): shape (B, 1, 1, T_k).
                - Causal mask (Decoder): shape (B, 1, T_q, T_k).
                Vị trí có giá trị 0 sẽ bị gán -inf trước softmax.
            return_attn (bool): Nếu True, trả về thêm attention weight matrix. Mặc định: False.

        Returns:
            Tensor: Output shape (B, T_q, d_model). Ví dụ: (32, 20, 512).
            Tensor (tuỳ chọn): Attention weights shape (B, H, T_q, T_k) khi return_attn=True.
        """
        B, T_q, _ = q.size()

        q = self._split_heads(self.q_linear(q))  # (B, H, T_q, d_k)
        k = self._split_heads(self.k_linear(k))  # (B, H, T_k, d_k)
        v = self._split_heads(self.v_linear(v))  # (B, H, T_v, d_k)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, T_q, T_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)  # (B, H, T_q, T_k)
        attn = self.dropout(attn_weights)
        out = attn @ v  # (B, H, T_q, d_k)

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        out = self.out(out)

        if return_attn:
            return out, attn_weights  # attn_weights: (B, H, T_q, T_k)
        return out
