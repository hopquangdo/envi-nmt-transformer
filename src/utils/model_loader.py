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

    Input Demo:
        checkpoint_path: 'checkpoints/best.pt'
        tokenizer_path: 'data/tokenizer/en_vi.model'
    Output Demo:
        return: (Transformer, SentencePieceProcessor, device)
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