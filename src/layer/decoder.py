import math
from torch import nn

from src.layer.positional_encoding import PositionalEncoding
from src.layer.attention import MultiHeadAttention
from src.layer.feed_forward import FeedForward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        """
        x:        (B, T_tgt, d_model)
        enc_out:  (B, T_src, d_model)
        tgt_mask: causal mask (B, 1, T_tgt, T_tgt)
        src_mask: padding mask cho encoder output (B, 1, 1, T_src)
        """
        # Pre-LN: norm TRƯỚC sublayer — ổn định hơn với AMP/FP16
        normed = self.norm1(x)
        x = x + self.dropout(self.self_attn(normed, normed, normed, tgt_mask))

        normed = self.norm2(x)
        x = x + self.dropout(self.cross_attn(normed, enc_out, enc_out, src_mask))

        x = x + self.dropout(self.ff(self.norm3(x)))
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
        self.fc_out = nn.Linear(d_model, vocab_size)

        # Trọng số chia sẻ theo paper: Attention Is All You Need
        if weight_tying:
            self.fc_out.weight = self.embed.weight

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        x = self.embed(x) * math.sqrt(self.d_model)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        x = self.norm(x)
        return self.fc_out(x)  # (B, T_tgt, vocab_size)
