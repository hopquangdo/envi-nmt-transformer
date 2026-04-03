import os
import argparse
import sentencepiece as spm

from src.data.loader import load_dataset
from src.utils import load_config

def train_spm(
    data_paths: list = None,
    prefix: str = "models/spm_tokenizer",
    vocab_size: int = 32000,
    min_len: int = 1,
    max_len: int = 200,
    max_len_ratio: float = 9.0,
    config_path: str = None,
):
    """
    Huấn luyện bộ Tokenizer SentencePiece (BPE) cho song ngữ Anh-Việt.

    Input Demo:
        data_paths: ['dataset/train.csv'] (Danh sách các file dữ liệu csv)
        vocab_size: 32000 (Số lượng từ vựng tối đa)
        prefix: 'models/my_tokenizer' (Tên file lưu model)
    Output Demo:
        (Tạo ra 2 file: .model và .vocab tại đường dẫn prefix)
    """
    # Load params từ config nếu có
    if config_path is not None:
        cfg        = load_config(config_path)
        data_paths = data_paths or cfg["dataset"]["train_sources"]
        vocab_size = cfg["tokenizer"].get("vocab_size", vocab_size)
        prefix     = cfg["tokenizer"].get("model_path", prefix).replace(".model", "")
        min_len    = cfg["dataset"].get("min_len", min_len)
        max_len    = cfg["dataset"].get("max_len", max_len)
        max_len_ratio = cfg["dataset"].get("max_len_ratio", max_len_ratio)

    if data_paths is None:
        data_paths = ["dataset/PhoMT/train.csv", "dataset/opus100/train.csv"]

    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    corpus_path = os.path.join(os.path.dirname(prefix), "temp_corpus.txt")

    # Dùng loader dùng chung: clean + filter cả PhoMT & opus100
    df = load_dataset(
        data_paths,
        min_len=min_len,
        max_len=max_len,
        max_len_ratio=max_len_ratio,
        shuffle=True
    )

    print(f"Ghi corpus: {len(df) * 2:,} dòng → {corpus_path}")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for en, vi in zip(df["en"], df["vi"]):
            f.write(en + "\n")
            f.write(vi + "\n")

    print(f"Training SentencePiece vocab_size={vocab_size}...")
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        pad_id=0,   unk_id=1,   bos_id=2,   eos_id=3,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<s>", eos_piece="</s>",
        character_coverage=1.0,       
        split_digits=True,
        add_dummy_prefix=True,
        remove_extra_whitespaces=True,
        normalization_rule_name="nmt_nfkc",
        shuffle_input_sentence=True,
    )

    os.remove(corpus_path)
    print(f"Tokenizer saved: {prefix}.model  &  {prefix}.vocab")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SentencePiece tokenizer")
    parser.add_argument(
        "--config", default="configs/config.yaml",
        help="Đường dẫn file config YAML",
    )
    args = parser.parse_args()
    train_spm(config_path=args.config)
