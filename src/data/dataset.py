import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm

from .loader import load_dataset


class TranslationDataset(Dataset):
    """
    Dataset chuẩn bị cặp câu (Anh-Việt) đã được token hóa.

    Input Demo:
        data_path: ['dataset/train.csv']
        spm_model_path: 'models/spm.model'
    Output Demo:
        __getitem__(idx): dict {
            "src": Tensor IDs (SeqLen_src) -> [2, 15, 6, 3]
            "tgt": Tensor IDs (SeqLen_tgt) -> [2, 8, 12, 3]
        }
    """
    def __init__(self, data_path, spm_model_path, min_len=1, max_len=512, max_len_ratio=9.0):
        self.df = load_dataset(
            data_path,
            min_len=min_len,
            max_len=max_len,
            max_len_ratio=max_len_ratio
        )

        self.spm = spm.SentencePieceProcessor()
        self.spm.load(spm_model_path)
        self.max_len = max_len

        self.bos_id = self.spm.bos_id()
        self.eos_id = self.spm.eos_id()
        self.pad_id = self.spm.pad_id()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        en_text = str(self.df.loc[idx, 'en'])
        vi_text = str(self.df.loc[idx, 'vi'])

        en_ids = self.spm.encode(en_text)
        vi_ids = self.spm.encode(vi_text)

        # Add BOS and EOS tokens, truncate to max_len
        src = [self.bos_id] + en_ids[:self.max_len - 2] + [self.eos_id]
        tgt = [self.bos_id] + vi_ids[:self.max_len - 2] + [self.eos_id]

        return {
            "src": torch.tensor(src, dtype=torch.long),
            "tgt": torch.tensor(tgt, dtype=torch.long)
        }


def collate_fn(batch, pad_id=0):
    """
    Hàm gộp các mẫu đơn lẻ thành một Batch và thực hiện Padding.

    Input Demo:
        batch: list các dict từ __getitem__ -> [{"src": T1, "tgt": T2}, ...]
    Output Demo:
        return: dict {
            "src": Tensor Padded (Batch, Max_SeqLen_src)
            "tgt": Tensor Padded (Batch, Max_SeqLen_tgt)
        }
    """
    src_batch = [item["src"] for item in batch]
    tgt_batch = [item["tgt"] for item in batch]

    # Pad sequences
    src_padded = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=pad_id)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=pad_id)

    return {
        "src": src_padded,
        "tgt": tgt_padded
    }


def get_dataloader(
    data_sources,
    spm_model_path,
    batch_size,
    pad_id,
    shuffle=True,
    num_workers=4,
    min_len=1,
    max_len=128,
    max_len_ratio=9.0,
):
    """
    Tạo nhanh bộ dữ liệu và trình tải dữ liệu (DataLoader).

    Input Demo:
        batch_size: 32
    Output Demo:
        return: (TranslationDataset, DataLoader)
    """
    dataset = TranslationDataset(
        data_sources,
        spm_model_path,
        min_len=min_len,
        max_len=max_len,
        max_len_ratio=max_len_ratio,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=bool(num_workers > 0),
        collate_fn=lambda x: collate_fn(x, pad_id=pad_id),
    )
    return dataset, loader

