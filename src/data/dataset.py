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
        """
        Trả về tổng số cặp câu (En-Vi) trong dataset.

        Returns:
            int: Số lượng mẫu dữ liệu.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Lấy một cặp câu đã được token hoá tại vị trí idx.

        Thêm token <bos> ở đầu và <eos> ở cuối mỗi chuỗi,
        sau đó cắt ngắn để không vượt quá max_len (tính cả BOS và EOS).

        Args:
            idx (int): Chỉ số của mẫu cần lấy.

        Returns:
            dict: {
                "src": Tensor ID câu tiếng Anh (SeqLen_src,), ví dụ: [2, 15, 6, 3],
                "tgt": Tensor ID câu tiếng Việt (SeqLen_tgt,), ví dụ: [2, 8, 12, 3].
            }
        """
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
    Tạo TranslationDataset và DataLoader từ các file CSV.

    Args:
        data_sources (list[str] | str): Đường dẫn tới một hoặc nhiều file CSV chứa cột 'en' và 'vi'.
        spm_model_path (str): Đường dẫn tới file SentencePiece model (.model).
        batch_size (int): Số cặp câu trong mỗi batch khi train. Ví dụ: 32.
        pad_id (int): ID của token <pad> dùng để padding các batch.
        shuffle (bool): Xáo trộn dữ liệu trước mỗi epoch. Mặc định: True.
        num_workers (int): Số worker song song để tải dữ liệu. Mặc định: 4.
        min_len (int): Độ dài từ tối thiểu (word-level) của mỗi câu. Mặc định: 1.
        max_len (int): Độ dài token tối đa sau khi encode (tính cả BOS/EOS). Mặc định: 128.
        max_len_ratio (float): Tỉ lệ chiều dài tối đa giữa câu dài nhất và ngắn nhất
            trong một cặp câu (lọc câu lệch quá nhiều). Mặc định: 9.0.

    Returns:
        tuple[TranslationDataset, DataLoader]: Dataset và DataLoader đã được cấu hình.
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

