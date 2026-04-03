import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        """
        Khởi tạo và tính sẵn bảng mã hóa vị trí (Sinusoidal Positional Encoding).

        Công thức:
            PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

        Bảng PE được tính 1 lần và lưu vào buffer (không phải tham số huấn luyện)
        để tránh tính lại mỗi lần forward.

        Args:
            d_model (int): Số chiều của vector nhúng. Ví dụ: 512.
            max_len (int): Độ dài chuỗi tối đa được hỗ trợ. Mặc định: 512.
            dropout (float): Tỉ lệ dropout áp dụng sau khi cộng PE. Mặc định: 0.1.
        """
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

        Vì Transformer không có tính tuần tự tự nhiên (khác RNN), PE cung cấp
        thông tin vị trí của từng token trong chuỗi thông qua sóng sin/cos.

        Args:
            x (Tensor): Tensor vector nhúng đã scale, shape (B, T, d_model). Ví dụ: (32, 20, 512).

        Returns:
            Tensor: Tensor sau khi cộng PE và áp dụng dropout, shape (B, T, d_model).
        """
        T = x.size(1)
        x = x + self.pe[:, :T]
        return self.dropout(x)