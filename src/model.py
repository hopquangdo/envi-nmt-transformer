from torch import nn

from src.layer.decoder import Decoder
from src.layer.encoder import Encoder
from src.utils.mask import create_src_mask, create_tgt_mask


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, num_layers=6,
                 num_heads=8, d_ff=2048, dropout=0.1, max_len=512,
                 pad_idx=0, weight_tying=True):
        super().__init__()
        self.pad_idx = pad_idx

        self.encoder = Encoder(
            vocab_size=src_vocab,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
            pad_idx=pad_idx,
        )
        self.decoder = Decoder(
            vocab_size=tgt_vocab,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
            pad_idx=pad_idx,
            weight_tying=weight_tying,
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier uniform init cho tất cả linear layers."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt):
        """
        src: (B, T_src) — token ids của source (English)
        tgt: (B, T_tgt) — token ids của target (Vietnamese), teacher-forced
        Returns:
            logits: (B, T_tgt, tgt_vocab_size)
        """
        src_mask = create_src_mask(src, self.pad_idx)   # (B, 1, 1, T_src)
        tgt_mask = create_tgt_mask(tgt, self.pad_idx)   # (B, 1, T_tgt, T_tgt)

        enc_out = self.encoder(src, src_mask)
        logits = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return logits