from __future__ import annotations

import torch
from torch.utils.data import DataLoader
import sentencepiece as spm
import os
import sys
import csv
import argparse
from typing import Optional

from src.utils import load_config

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.scheduler import NoamScheduler
from src.model import Transformer
from src.data.dataset import get_dataloader
from src.evaluate import evaluate_bleu


# ─────────────────────────────────────────────────────────────
# Main train
# ─────────────────────────────────────────────────────────────

def train(config_path: str, resume: str = "", init_model: str = ""):
    """
    Hàm chính thực hiện quá trình huấn luyện (Training Loop).

    Input Demo:
        config_path: 'configs/config.yaml' (Đường dẫn file cấu hình).
        resume: 'checkpoints/latest.pt' (Tiếp tục train từ checkpoint đầy đủ).
        init_model: 'checkpoints/best.pt' (Chỉ lấy trọng số model, bắt đầu epoch 0).
    """
    config      = load_config(config_path)
    model_cfg   = config["model"]
    train_cfg   = config["training"]
    data_cfg    = config["dataset"]
    ckpt_cfg    = config["checkpoint"]
    tok_cfg     = config["tokenizer"]
    inf_cfg     = config.get("inference", {})

    # ===== Seed & Device =====
    torch.manual_seed(train_cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===== Tokenizer =====
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tok_cfg["model_path"])
    vocab_size = tokenizer.vocab_size()
    pad_idx    = tokenizer.pad_id()
    print(f"Vocab size: {vocab_size} | pad_id: {pad_idx}")

    # ===== Train Dataset =====
    train_dataset, train_loader = get_dataloader(
        data_sources=data_cfg["train_sources"],
        spm_model_path=tok_cfg["model_path"],
        batch_size=train_cfg["batch_size"],
        pad_id=pad_idx,
        shuffle=True,
        num_workers=4,
        min_len=data_cfg.get("min_len", 1),
        max_len=data_cfg.get("max_len", 128),
        max_len_ratio=data_cfg.get("max_len_ratio", 9.0),
    )

    val_dataset, val_loader = get_dataloader(
        data_sources=data_cfg.get("val_csv"),
        spm_model_path=tok_cfg["model_path"],
        batch_size=train_cfg.get("val_batch_size", train_cfg["batch_size"]),
        pad_id=pad_idx,
        shuffle=False,
        num_workers=2,
        min_len=data_cfg.get("min_len", 1),
        max_len=data_cfg.get("max_len", 128),
        max_len_ratio=data_cfg.get("max_len_ratio", 9.0),
    )
    print(f"Train: {len(train_dataset):,} | Val: {len(val_dataset):,} cặp câu")

    # ===== Model =====
    model = Transformer(
        src_vocab=vocab_size,
        tgt_vocab=vocab_size,
        d_model=model_cfg["d_model"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["n_heads"],
        d_ff=model_cfg["dim_ff"],
        dropout=model_cfg["dropout"],
        pad_idx=pad_idx,
    ).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ===== Optimizer & Scheduler =====
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(
        optimizer,
        d_model=model_cfg["d_model"],
        warmup_steps=train_cfg["warmup_steps"],
    )

    # ===== Loss =====
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=pad_idx,
        label_smoothing=train_cfg.get("label_smoothing", 0.1),
    )
    scaler = torch.amp.GradScaler("cuda")

    # ===== Resume or Init =====
    start_epoch  = 0
    start_batch  = 0
    best_bleu    = -1.0

    resume_path = resume or ckpt_cfg.get("resume", "")
    init_path   = init_model or ckpt_cfg.get("init_model", "")

    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch  = ckpt.get("epoch", 0)
        start_batch  = ckpt.get("batch", 0)
        best_bleu    = ckpt.get("best_bleu", -1.0)
        scheduler._step = ckpt.get("scheduler_step", 0)
        print(f"Resumed Full State: epoch={start_epoch}, batch={start_batch}, best_bleu={best_bleu:.2f}")
    elif init_path and os.path.isfile(init_path):
        ckpt = torch.load(init_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Initialized Model Weights ONLY from: {init_path} (Starting fresh run)")

    # ===== Setup dirs =====
    save_dir   = ckpt_cfg["save_dir"]
    latest_pt  = ckpt_cfg["latest"]
    best_pt    = ckpt_cfg["best_model"]
    loss_csv   = train_cfg["loss_csv"]

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(loss_csv), exist_ok=True)

    ckpt_interval   = train_cfg.get("ckpt_interval", 10000)
    max_checkpoints = train_cfg.get("max_checkpoints", 3)
    grad_clip       = train_cfg.get("grad_clip", 0.5)
    patience        = train_cfg.get("patience", 3)
    epochs          = train_cfg["epochs"]

    # Giới hạn số batch validation để kịp trong 1 epoch lớn
    max_val_batches = train_cfg.get("max_val_batches", None)  # None = toàn bộ
    beam_size       = inf_cfg.get("beam_size", 4)
    inf_max_len     = inf_cfg.get("max_len", data_cfg.get("max_len", 128))

    checkpoint_paths  = []
    patience_counter  = 0

    # ===== CSV logger =====
    file_exists = os.path.isfile(loss_csv) and os.path.getsize(loss_csv) > 0
    mode = "a" if (resume_path or file_exists) else "w"
    csv_file = open(loss_csv, mode=mode, newline="", encoding="utf-8")
    writer   = csv.DictWriter(
        csv_file,
        fieldnames=["epoch", "batch", "loss", "lr", "val_bleu"],
    )
    if mode == "w":
        writer.writeheader()

    print("=" * 60)
    print(f"Epochs: {epochs} | Batch: {train_cfg['batch_size']} | Warmup: {train_cfg['warmup_steps']}")
    print(f"Early stop: patience={patience} epoch(s) theo val BLEU ↑")
    print(f"Val batches/epoch: {'all' if max_val_batches is None else max_val_batches}")
    print(f"Beam size: {beam_size}")
    print("=" * 60)

    total_batches = len(train_loader)

    for epoch in range(start_epoch, epochs):
        # ── Train ──
        model.train()
        total_loss  = 0.0
        num_batches = 0

        for i, batch in enumerate(train_loader):
            if epoch == start_epoch and i < start_batch:
                continue

            src          = batch["src"].to(device, non_blocking=True)
            tgt_input    = batch["tgt"][:, :-1].to(device, non_blocking=True)
            tgt_expected = batch["tgt"][:, 1:].to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                logits = model(src, tgt_input)
                loss   = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_expected.reshape(-1),
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scheduler.step()
            scaler.step(optimizer)
            scaler.update()

            total_loss  += loss.item()
            num_batches += 1

            writer.writerow({
                "epoch":    epoch + 1,
                "batch":    i,
                "loss":     round(loss.item(), 6),
                "lr":       f"{scheduler.current_lr:.8f}",
                "val_bleu": "",
            })
            csv_file.flush()

            if i % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} | Batch {i} / {total_batches} "
                    f"| Loss: {loss.item():.4f} | LR: {scheduler.current_lr:.6f}",
                    flush=True,
                )

            # Checkpoint định kỳ
            if i % ckpt_interval == 0 and i > 0:
                path = os.path.join(save_dir, f"model_e{epoch + 1}_b{i}.pt")
                state = {
                    "model":          model.state_dict(),
                    "optimizer":      optimizer.state_dict(),
                    "epoch":          epoch,
                    "batch":          i,
                    "best_bleu":      best_bleu,
                    "scheduler_step": scheduler._step,
                }
                torch.save(state, path)
                torch.save(state, latest_pt)
                print(f"[Checkpoint] Saved: {path}")
                checkpoint_paths.append(path)
                if len(checkpoint_paths) > max_checkpoints:
                    old = checkpoint_paths.pop(0)
                    if os.path.exists(old):
                        os.remove(old)

        avg_loss = total_loss / max(num_batches, 1)

        print(f"\nEpoch {epoch + 1} DONE | Avg Train Loss: {avg_loss:.4f} | LR: {scheduler.current_lr:.6f}")

        # ── Validate BLEU ──
        print(f"[Val] Đang đánh giá BLEU trên tập validation ...")
        val_bleu = evaluate_bleu(
            model=model,
            val_loader=val_loader,
            tokenizer=tokenizer,
            device=device,
            max_len=inf_max_len,
            beam_size=beam_size,
            max_val_batches=max_val_batches,
            show_samples=False,
            desc=f"Val BLEU (Epoch {epoch+1})"
        )
        print(f"[Val] BLEU = {val_bleu:.2f} (best so far: {best_bleu:.2f})")

        # Ghi BLEU vào dòng cuối cùng của epoch vào CSV
        writer.writerow({
            "epoch":    epoch + 1,
            "batch":    "epoch_end",
            "loss":     round(avg_loss, 6),
            "lr":       f"{scheduler.current_lr:.8f}",
            "val_bleu": round(val_bleu, 4),
        })
        csv_file.flush()

        # ── Early stopping theo BLEU ──
        if val_bleu > best_bleu:
            best_bleu       = val_bleu
            patience_counter = 0
            torch.save({
                "model":          model.state_dict(),
                "optimizer":      optimizer.state_dict(),
                "epoch":          epoch + 1,
                "batch":          0,
                "best_bleu":      best_bleu,
                "scheduler_step": scheduler._step,
            }, best_pt)
            print(f"[Best] BLEU improved → {best_bleu:.2f} | Saved: {best_pt}")
        else:
            patience_counter += 1
            print(f"[Early Stop] {patience_counter}/{patience} — best BLEU: {best_bleu:.2f}")
            if patience_counter >= patience:
                print(f"[Early Stop] Dừng tại epoch {epoch + 1}.")
                break

        # Đặt lại start_batch sau epoch đầu tiên khi resume
        start_batch = 0

        # Chuyển model về train mode cho epoch tiếp theo
        model.train()

    csv_file.close()
    print(f"\nDone! Best val BLEU: {best_bleu:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NMT model")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", default="", help="Resume full state (model, opt, batch)")
    parser.add_argument("--init_model", default="", help="Initialize only model weights from checkpoint")
    args = parser.parse_args()
    train(config_path=args.config, resume=args.resume, init_model=args.init_model)
