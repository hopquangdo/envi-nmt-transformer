"""Utility functions để tạo mask cho Transformer."""
import torch


def create_src_mask(src: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Tạo padding mask cho bộ Encoder để bỏ qua các ký tự <pad>.

    Input Demo:
        src: Tensor shape (B, T_src) -> ví dụ: torch.tensor([[1, 2, 3, 0, 0]]) (Batch=1, SeqLen=5)
        pad_idx: 0

    Output Demo:
        return: Tensor shape (B, 1, 1, T_src) -> torch.tensor([[[[True, True, True, False, False]]]])
    """
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)


def create_tgt_mask(tgt: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
    """
    Tạo mask kết hợp cho bộ Decoder:
      1. Padding mask: Tránh nhìn vào các ký tự <pad>.
      2. Causal mask (Look-ahead mask): Tránh nhìn vào các từ ở "tương lai".

    Input Demo:
        tgt: Tensor shape (B, T_tgt) -> ví dụ: torch.tensor([[5, 6, 0]])
        pad_idx: 0

    Output Demo:
        return: Tensor shape (B, 1, T_tgt, T_tgt)
        Giải thích: Kết quả là ma trận tam giác dưới để mỗi từ chỉ nhìn được chính nó và các từ trước đó.
    """
    B, T = tgt.size()

    # Causal (look-ahead) mask: lower triangular
    causal_mask = torch.ones(T, T, device=tgt.device).tril().bool()   # (T, T)
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)               # (1, 1, T, T)

    # Padding mask
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)              # (B, 1, 1, T)

    return causal_mask & pad_mask  # (B, 1, T, T)
