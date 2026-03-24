"""
download_data.py — Tải opus100 (en-vi) và PhoMT từ HuggingFace,
                   xuất ra 3 split: train.csv / dev.csv / test.csv

Chạy từ gốc project:
    python scripts/download_data.py

Yêu cầu:
    pip install datasets pandas tqdm
"""

import os
import sys
import pandas as pd
from pathlib import Path

try:
    from datasets import load_dataset as hf_load
except ImportError:
    raise ImportError("Chưa cài 'datasets'. Chạy: pip install datasets")


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def save_split(rows: list, out_path: Path):
    df = pd.DataFrame(rows, columns=["en", "vi"])
    df = df.dropna()
    df.to_csv(out_path, index=False, encoding="utf-8")
    size_mb = out_path.stat().st_size / 1e6
    print(f"  ✅  {out_path.name:12s} → {len(df):>9,} cặp  ({size_mb:.1f} MB)")


def _extract_rows(split_ds, source_name: str) -> list:
    """Tự phát hiện schema và trích xuất (en, vi)."""
    first = split_ds[0]
    if "translation" in first:
        return [
            {"en": ex["translation"]["en"], "vi": ex["translation"]["vi"]}
            for ex in split_ds
        ]
    elif "en" in first and "vi" in first:
        return [{"en": ex["en"], "vi": ex["vi"]} for ex in split_ds]
    else:
        raise ValueError(
            f"[{source_name}] Không nhận ra schema: {list(first.keys())}"
        )


# ─────────────────────────────────────────────────────────────
# 1. opus100  (Helsinki-NLP/opus-100, config "en-vi")
# ─────────────────────────────────────────────────────────────

def download_opus100(out_dir: Path):
    print("\n" + "=" * 55)
    print("📥  Tải Helsinki-NLP/opus-100  (en-vi) ...")
    print("=" * 55)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = hf_load("Helsinki-NLP/opus-100", "en-vi")

    print(f"  Splits có: {list(ds.keys())}")
    split_map = {
        "train":      "train.csv",
        "validation": "dev.csv",
        "test":       "test.csv",
    }
    for hf_split, fname in split_map.items():
        if hf_split not in ds:
            print(f"  ⚠️  Split '{hf_split}' không tồn tại — bỏ qua.")
            continue
        rows = _extract_rows(ds[hf_split], "opus100")
        save_split(rows, out_dir / fname)


# ─────────────────────────────────────────────────────────────
# 2. PhoMT  (vinai/PhoMT)
# ─────────────────────────────────────────────────────────────

def download_phomt(out_dir: Path):
    print("\n" + "=" * 55)
    print("📥  Tải vinai/PhoMT  (en-vi) ...")
    print("=" * 55)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        ds = hf_load("tuandunghcmut/PhoMT")
    except Exception as e:
        # PhoMT v2 dùng tên khác trên HF
        print(f"  ⚠️  vinai/PhoMT lỗi: {e}")
        print("  → Thử mt_eng_vietnamese ...")
        ds = hf_load("mt_eng_vietnamese", "iwslt2015-en-vi")

    print(f"  Splits có: {list(ds.keys())}")
    split_map = {
        "train":      "train.csv",
        "dev": "dev.csv",
        "test":       "test.csv",
    }
    for hf_split, fname in split_map.items():
        if hf_split not in ds:
            print(f"  ⚠️  Split '{hf_split}' không tồn tại — bỏ qua.")
            continue
        rows = _extract_rows(ds[hf_split], "PhoMT")
        save_split(rows, out_dir / fname)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent / "data"

    # download_opus100(base / "opus100")
    download_phomt(base / "PhoMT")

    print("\n✅  Hoàn tất! Cấu trúc data/:")
    for p in sorted(base.rglob("*.csv")):
        rel = p.relative_to(base)
        size_mb = p.stat().st_size / 1e6
        print(f"   data/{rel}   ({size_mb:.1f} MB)")
