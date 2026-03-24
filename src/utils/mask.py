"""Utility functions để tạo mask cho Transformer."""
import torch


def create_src_mask(src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Tạo padding mask cho encoder.

    Args:
        src:     (B, T_src) — token ids
        pad_idx: index của <pad> token

    Returns:
        mask: (B, 1, 1, T_src) — 1 = vị trí hợp lệ, 0 = padding
    """
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def create_tgt_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Tạo combined mask cho decoder:
      - Padding mask (tránh attend vào <pad>)
      - Causal mask (tránh attend vào tương lai)

    Args:
        tgt:     (B, T_tgt) — token ids
        pad_idx: index của <pad> token

    Returns:
        mask: (B, 1, T_tgt, T_tgt) — 1 = cho phép, 0 = chặn
    """
    B, T = tgt.size()

    # Causal (look-ahead) mask: lower triangular
    causal_mask = torch.ones(T, T, device=tgt.device).tril().bool()   # (T, T)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)               # (1, 1, T, T)

    # Padding mask
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)              # (B, 1, 1, T)

    return causal_mask & pad_mask  # (B, 1, T, T)
