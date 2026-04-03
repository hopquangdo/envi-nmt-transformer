import math
from torch import nn

from src.layer.positional_encoding import PositionalEncoding
from src.layer.attention import MultiHeadAttention
from src.layer.feed_forward import FeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        # 1. Masked Self-Attention: Từ phía ngữ đích chỉ được phép nhìn các từ trước nó
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # 2. Cross-Attention: Lấy thông tin từ câu nguồn (Encoder Output) để dịch
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # 3. Mạng Feed-Forward (tương tự Encoder)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None, return_attn: bool = False):
        """
        Xử lý qua 1 lớp Decoder.
        
        Input Demo:
            x: Tensor đích hiện tại (B, T_tgt, d_model) -> (32, 15, 512)
            enc_out: Output từ Encoder (B, T_src, d_model) -> (32, 20, 512)
            tgt_mask: (B, 1, T_tgt, T_tgt)
            src_mask: (B, 1, 1, T_src)
        Output Demo:
            return: Tensor đặc trưng (B, T_tgt, d_model) -> (32, 15, 512)
        """
        # Pre-LN: norm TRƯỚC sublayer — ổn định hơn với AMP/FP16
        normed = self.norm1(x)
        # 1. Masked Self-Attention (Query, Key, Value đều là từ đích `x`)
        # Do có tgt_mask nên các từ không thể "nhìn trộm" từ tương lai để chép bài
        x = x + self.dropout(self.self_attn(normed, normed, normed, tgt_mask))

        normed = self.norm2(x)
        if return_attn:
            # 2. Cross-Attention (Query từ `x` (đích), Key/Value từ `enc_out` (nguồn))
            # Bước này giúp Decoder xem xét xem cần tập trung dịch từ tiếng Anh nào tiếp theo
            cross_out, cross_attn_w = self.cross_attn(normed, enc_out, enc_out, src_mask, return_attn=True)
            x = x + self.dropout(cross_out)
        else:
            x = x + self.dropout(self.cross_attn(normed, enc_out, enc_out, src_mask))

        # 3. Khối truyền thẳng Feed Forward cuối cùng
        x = x + self.dropout(self.ff(self.norm3(x)))

        if return_attn:
            return x, cross_attn_w  # cross_attn_w: (B, H, T_tgt, T_src)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads,
                 d_ff=2048, dropout=0.1, max_len=512, pad_idx=0, weight_tying=True):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pe = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        # Layer đưa vecto ẩn cuối cùng về danh sách các độ dự đoán cho toàn bộ từ vựng đích
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Trọng số chia sẻ theo paper: Attention Is All You Need
        # Chia sẻ tham số nhúng và Linear dự đoán để tái sử dụng lại không gian ngữ nghĩa, giảm bớt tham số model
        if weight_tying:
            self.fc_out.weight = self.embed.weight

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None, return_attn: bool = False):
        """
        Xử lý qua toàn bộ bộ Decoder (gồm N lớp xếp chồng).
        
        Input Demo:
            x: Tensor IDs đích (B, T_tgt) -> (32, 15)
            enc_out: Output từ Encoder (B, T_src, d_model) -> (32, 20, 512)
        Output Demo:
            return: Logits (B, T_tgt, vocab_size) -> (32, 15, 20000)
        """
        # Bước 1 & 2 giống như ở Encoder: Nhúng token và cộng thêm vị trí
        x = self.embed(x) * math.sqrt(self.d_model)
        x = self.pe(x)

        # Bước 3
        all_cross_attns = []
        for layer in self.layers:
            if return_attn:
                x, cross_attn_w = layer(x, enc_out, src_mask, tgt_mask, return_attn=True)
                all_cross_attns.append(cross_attn_w)
            else:
                x = layer(x, enc_out, src_mask, tgt_mask)

        x = self.norm(x)
        
        # Bước 4: Chuyển đổi trạng thái ẩn (d_model) ở lớp cuối cùng về số chiều là số từ (vocab_size)
        logits = self.fc_out(x)  # (B, T_tgt, vocab_size)

        if return_attn:
            return logits, all_cross_attns
        return logits
