import math
from torch import nn


from src.layer.positional_encoding import PositionalEncoding
from src.layer.attention import MultiHeadAttention
from src.layer.feed_forward import FeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        # 1. Cơ chế Self-Attention: Giúp mô hình bắt được quan hệ giữa các từ trong cùng một câu
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        # 2. Mạng nơ-ron truyền thẳng (Feed-Forward Network)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        """
        Xử lý qua 1 lớp Encoder.
        
        Input Demo:
            x: Tensor shape (B, T, d_model) -> (32, 20, 512)
            src_mask: Tensor (B, 1, 1, T) -> (32, 1, 1, 20)
        Output Demo:
            return: Tensor shape (B, T, d_model) -> (32, 20, 512)
        """
        # Pre-LN (Layer Normalization TRƯỚC khi đưa vào Sub-layer) - Giúp mô hình huấn luyện ổn định hơn so với paper ban đầu
        normed = self.norm1(x)
        # Self-Attention với residual connection (x + attn(normed)). Query, Key, Value đều đến từ x.
        x = x + self.dropout(self.attn(normed, normed, normed, src_mask))
        
        # Đi qua khối Feed Forward cũng với cơ chế kết nối thặng dư (residual connection)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads,
                 d_ff=2048, dropout=0.1, max_len=512, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pe = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        """
        Xử lý qua toàn bộ bộ Encoder (gồm N lớp xếp chồng).
        
        Input Demo:
            x: Tensor IDs (B, T) -> (32, 20)
            src_mask: (B, 1, 1, T)
        Output Demo:
            return: Tensor ngữ cảnh (B, T, d_model) -> (32, 20, 512)
        """
        # Bước 1: Chuyển các ID từ thành vector (Embedding) và scale nó lên
        x = self.embed(x) * math.sqrt(self.d_model)
        # Bước 2: Cộng thêm mã hóa vị trí (Positional Encoding) vì tự bản thân Transformer không biết thứ tự từ vựng
        x = self.pe(x)
        
        # Bước 3: Đưa qua các lớp EncoderLayer (thường là 6 lớp xếp chồng)
        for layer in self.layers:
            x = layer(x, src_mask)
            
        # Chuẩn hóa đầu ra cuối cùng
        return self.norm(x)
