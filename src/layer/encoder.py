import math
from torch import nn


from src.layer.positional_encoding import PositionalEncoding
from src.layer.attention import MultiHeadAttention
from src.layer.feed_forward import FeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Pre-LN: norm TRƯỚC sublayer — ổn định hơn với AMP/FP16
        normed = self.norm1(x)
        x = x + self.dropout(self.attn(normed, normed, normed, src_mask))
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
        x = self.embed(x) * math.sqrt(self.d_model)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)
