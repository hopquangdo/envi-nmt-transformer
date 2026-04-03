import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Cộng thêm mã hóa vị trí vào vector nhúng để mô hình biết thứ tự các từ.
        
        Input Demo:
            x: Tensor shape (B, T, d_model) -> (32, 20, 512)
        Output Demo:
            return: Tensor shape (B, T, d_model) -> (32, 20, 512)
        """
        T = x.size(1)
        x = x + self.pe[:, :T]
        return self.dropout(x)