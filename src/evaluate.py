import argparse
import torch
import pandas as pd
from tqdm import tqdm
from sacrebleu.metrics import BLEU

from src.utils import load_config
from src.utils.model_loader import load_model
from src.utils.translate import Translator
from src.data.dataset import get_dataloader


def evaluate_bleu(
    model,
    val_loader,
    tokenizer,
    device,
    max_len: int = 128,
    beam_size: int = 4,
    max_val_batches: int | None = None,
    method: str = "beam",
    show_samples: bool = False,
    desc="Translating"
) -> float:
    """
    Dịch toàn bộ (hoặc một phần) tập validation/test bằng model,
    trả về BLEU score.

    Input Demo:
        model: Transformer model đã train.
        val_loader: DataLoader chứa tập dữ liệu validation.
        tokenizer: SentencePieceProcessor.
        beam_size: 4 (Số lượng chùm tìm kiếm).
    Output Demo:
        return: float (ví dụ: 35.5 - Điểm số BLEU càng cao càng tốt).
    """
    translator = Translator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_len=max_len,
    )

    hyps, refs, srcs_text = [], [], []
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc=desc, leave=False)):
            if max_val_batches is not None and batch_idx >= max_val_batches:
                break

            # Decode tgt tensor → text tham chiếu
            for i in range(batch["src"].shape[0]):
                src_ids = batch["src"][i].tolist()
                bos, eos, pad = tokenizer.bos_id(), tokenizer.eos_id(), tokenizer.pad_id()
                
                src_core = [t for t in src_ids if t not in (bos, eos, pad)]
                src_text = tokenizer.decode(src_core)

                tgt_ids = batch["tgt"][i].tolist()
                tgt_core = [t for t in tgt_ids if t not in (bos, eos, pad)]
                ref_text = tokenizer.decode(tgt_core)

                # TODO: in the future translator can be batched, but now it's single
                hyp = translator.translate(src_text, method=method, beam_size=beam_size)
                
                hyps.append(hyp)
                refs.append(ref_text)
                
                if show_samples:
                    srcs_text.append(src_text)

    bleu = BLEU(tokenize="13a")
    score = bleu.corpus_score(hyps, [refs])
    
    if show_samples:
        print("\nSample predictions:")
        for i in range(min(3, len(hyps))):
            print(f"\nSRC: {srcs_text[i]}")
            print(f"REF: {refs[i]}")
            print(f"HYP: {hyps[i]}")

    return float(score.score)


def evaluate(config_path: str, max_samples=None):
    """
    Hàm main điều khiển quá trình đánh giá BLEU từ file cấu hình.

    Input Demo:
        config_path: 'configs/config.yaml'
        max_samples: 100 (Để test nhanh thay vì chạy hết tập test).
    Output Demo:
        return: score (float) - Điểm BLEU cuối cùng.
    """
    config   = load_config(config_path)
    inf_cfg  = config["inference"]
    ckpt_cfg = config["checkpoint"]
    tok_cfg  = config["tokenizer"]
    data_cfg = config["dataset"]
    train_cfg = config["training"]

    model, tokenizer, device = load_model(
        checkpoint_path=ckpt_cfg["best_model"],
        tokenizer_path=tok_cfg["model_path"],
    )

    method      = inf_cfg.get("method", "beam")
    beam_size   = inf_cfg.get("beam_size", 5)
    max_val_batches = None

    if max_samples is not None:
        batch_size = 1
        max_val_batches = max_samples
    else:
        batch_size = train_cfg.get("val_batch_size", 64)
        n_samples = inf_cfg.get("max_samples")
        if n_samples:
            max_val_batches = max(1, n_samples // batch_size)

    print("=" * 60)
    print(f"Evaluating from config: {config_path}")
    print(f"Method: {method} | Beam size: {beam_size}")
    print("=" * 60)

    _, test_loader = get_dataloader(
        data_sources=data_cfg["test_csv"],
        spm_model_path=tok_cfg["model_path"],
        batch_size=batch_size,
        pad_id=tokenizer.pad_id(),
        shuffle=False,
        num_workers=2,
        min_len=data_cfg.get("min_len", 1),
        max_len=data_cfg.get("max_len", 128),
        max_len_ratio=data_cfg.get("max_len_ratio", 9.0),
    )

    score = evaluate_bleu(
        model=model,
        val_loader=test_loader,
        tokenizer=tokenizer,
        device=device,
        max_len=inf_cfg["max_len"],
        beam_size=beam_size,
        max_val_batches=max_val_batches,
        method=method,
        show_samples=True,
        desc="Translating"
    )

    print("\n" + "=" * 60)
    print(f"BLEU: {score:.2f}")
    print("=" * 60)

    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NMT model (BLEU)")
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Đường dẫn file config YAML",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Giới hạn số câu để debug nhanh (override config)",
    )
    args = parser.parse_args()
    evaluate(config_path=args.config, max_samples=args.max_samples)