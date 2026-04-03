import torch
import sentencepiece as spm

from src.model import Transformer
from src.utils import load_config


def load_model(
    checkpoint_path: str,
    tokenizer_path: str,
    device: torch.device = None
):
    """
    Tải mô hình Transformer và bộ Tokenizer từ checkpoint đã train.

    Hỗ trợ 2 định dạng checkpoint:
        1. Dict đầy đủ: {'model': state_dict, 'optimizer': ..., 'epoch': ...}
        2. state_dict thuần túy (chỉ lưu trọng số mô hình)

    Args:
        checkpoint_path (str): Đường dẫn đến file checkpoint (.pt). Ví dụ: 'checkpoints/best_model.pt'.
        tokenizer_path (str): Đường dẫn đến file SentencePiece model (.model).
            Ví dụ: 'models/spm_tokenizer.model'.
        device (torch.device | None): Thiết bị chạy. Nếu None, tự động chọn CUDA
            nếu có sẵn, ngược lại sử dụng CPU.

    Returns:
        tuple[Transformer, SentencePieceProcessor, torch.device]:
            - model: Mô hình Transformer đã load trọng số, được đặt sang eval() mode.
            - tokenizer: Bộ SentencePiece tokenizer đã tải.
            - device: Thiết bị đang chạy thực tế.

    Raises:
        FileNotFoundError: Gán ra nếu checkpoint_path hoặc tokenizer_path không tồn tại.
        RuntimeError: Nếu state_dict không khớp với kiến trúc mô hình trong config.
    """
    config = load_config("configs/config.yaml")
    model_config = config["model"]

    # ===== device =====
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== tokenizer =====
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)

    vocab_size = tokenizer.vocab_size()
    pad_id = tokenizer.pad_id()

    # ===== model =====
    model = Transformer(
        src_vocab=vocab_size,
        tgt_vocab=vocab_size,
        d_model=model_config["d_model"],
        num_layers=model_config["num_layers"],
        num_heads=model_config["n_heads"],
        d_ff=model_config["dim_ff"],
        dropout=0.0,
        pad_idx=pad_id,
    ).to(device)

    # ===== load checkpoint =====
    ckpt = torch.load(checkpoint_path, map_location=device)

    # hỗ trợ 2 kiểu save
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)

    model.eval()

    print("=" * 50)
    print(f"✅ Model loaded from: {checkpoint_path}")
    print(f"Vocab size: {vocab_size}")
    print(f"Device: {device}")
    print("=" * 50)

    return model, tokenizer, device