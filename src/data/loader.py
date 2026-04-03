"""
loader.py — Load & tiền xử lý dữ liệu En↔Vi chuẩn NMT
=========================================================
Dùng chung cho dataset.py và tokenizer.py.

Chuẩn tham khảo:
  - fairseq / OpenNMT dataset preprocessing conventions
  - Helsinki-NLP opus-mt preprocessing
  - VLSP / PhoNLP corpus guidelines

>>> from src.data.loader import load_dataset
>>> df = load_dataset(["dataset/PhoMT/train.csv", "dataset/opus100/train.csv"])
"""

import re
import unicodedata
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────
# Regex patterns — biên soạn 1 lần, tái dùng toàn bộ module
# ─────────────────────────────────────────────────────────────
_RE_HTML        = re.compile(r"<[^>]+>")
_RE_URL         = re.compile(r"https?://\S+|www\.\S+")
_RE_MULTI_WS    = re.compile(r"\s+")
# Ký tự điều khiển & non-printable
_RE_CONTROL     = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# Subtitle / screenplay noise
_RE_BRACKET     = re.compile(r"\[.*?\]")          # [ Muttering ], [ALL SHOUTING]
_RE_PAREN       = re.compile(r"\(.*?\)")           # (laughing), (Chavez sighs)
_RE_DASH_LEAD   = re.compile(r"^\s*[-–—]+\s*")    # "- " / "– " đầu câu subtitle
_RE_MUSIC       = re.compile(r"[♪♫♩♬]+")          # ký hiệu nhạc
_RE_PUNCT_RUN   = re.compile(r"[-_.~…]{3,}")      # ----, ....., ~~~~, ……
_RE_ELLIPSIS    = re.compile(r"^[.…\s]+$")         # câu chỉ có dấu chấm lửng
_RE_SPEAKER     = re.compile(r"^[A-ZÀÁẠẢÃÂẤẦẨẪẬ\s]{2,}:\s*")  # "CHAVEZ: ", "NGƯỜI KỂ: "
_RE_ANNOUNCER   = re.compile(r"^>>\s*")            # ">> Hello" kiểu announcer
_RE_REPEAT_TOK  = re.compile(r"(\b\w+\b)(\s+\1){4,}")  # "ha ha ha ha ha" (lặp 5+ lần)



# ─────────────────────────────────────────────────────────────
# 1. LÀM SẠCH MỘT CÂU
# ─────────────────────────────────────────────────────────────

def normalize_unicode(text: str) -> str:
    """
    NFC normalize — chuẩn hoá tổ hợp dấu tiếng Việt.

    Input Demo:
        text: 'ạ' (với dấu nặng tách rời)
    Output Demo:
        return: 'ạ' (với dấu nặng được gộp chuẩn)
    """
    return unicodedata.normalize("NFC", text)


def clean_text(text: str) -> str:
    """
    Pipeline làm sạch câu (xoá HTML, URL, icon, dấu ngoặc subtitle...).

    Input Demo:
        text: 'Hello [Music] world!'
    Output Demo:
        return: 'Hello world!'
    """
    if not isinstance(text, str):
        return ""
    text = normalize_unicode(text)
    text = _RE_CONTROL.sub("", text)
    text = _RE_HTML.sub(" ", text)
    text = _RE_URL.sub(" ", text)
    text = _RE_BRACKET.sub(" ", text)      # [ Muttering ] → bỏ
    text = _RE_PAREN.sub(" ", text)        # (laughing)    → bỏ
    text = _RE_MUSIC.sub(" ", text)        # ♪             → bỏ
    text = _RE_PUNCT_RUN.sub(" ", text)    # ----          → bỏ
    text = _RE_SPEAKER.sub("", text)       # "CHAVEZ: "    → bỏ
    text = _RE_ANNOUNCER.sub("", text)     # ">> Hello"    → "Hello"
    text = _RE_DASH_LEAD.sub("", text)     # "- Hello"     → "Hello"
    text = _RE_REPEAT_TOK.sub(r"\1", text) # "ha ha ha ha ha" → "ha"
    text = _RE_MULTI_WS.sub(" ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────
# 2. LỌC CẶP CÂU
# ─────────────────────────────────────────────────────────────

def is_valid_pair(
    en: str,
    vi: str,
    min_len: int = 1,
    max_len: int = 200,
    max_len_ratio: float = 9.0,
) -> bool:
    """
    Lọc cặp câu Anh-Việt theo tiêu chuẩn tiền xử lý NMT (Moses/fairseq style).

    Một cặp hợp lệ phải vượt qua tất cả các tiêu chí sau:
        1. Không được rỗng (cả en lẫn vi).
        2. Độ dài (word-level) nằm trong khoảng [min_len, max_len].
        3. Tỉ lệ chiều dài (len_max / len_min) không vượt quá max_len_ratio.
        4. Hai câu không giống hệt nhau (en ≠ vi sau khi lowercase).
        5. Tỉ lệ ký tự chữ cái (alpha ratio) >= 40% ở cả hai câu (loại câu toàn số/ký hiệu).
        6. Không phải câu chỉ gồm dấu chấm lửng (ellipsis-only).

    Args:
        en (str): Câu tiếng Anh đã làm sạch.
        vi (str): Câu tiếng Việt đã làm sạch.
        min_len (int): Số từ tối thiểu. Mặc định: 1.
        max_len (int): Số từ tối đa. Mặc định: 200.
        max_len_ratio (float): Tỉ lệ chiều dài tối đa cho phép. Mặc định: 9.0.

    Returns:
        bool: True nếu cặp câu hợp lệ, False nếu cần loại bỏ.
    """
    if not en or not vi:
        return False

    en_len = len(en.split())
    vi_len = len(vi.split())

    if en_len < min_len or vi_len < min_len:
        return False
    if en_len > max_len or vi_len > max_len:
        return False

    ratio = max(en_len, vi_len) / max(min(en_len, vi_len), 1)
    if ratio > max_len_ratio:
        return False

    # Câu en và vi giống hệt nhau → không phải bản dịch
    if en.lower().strip() == vi.lower().strip():
        return False

    # Tỉ lệ chữ cái quá thấp → câu toàn số, ký hiệu, mã code
    def _alpha_ratio(s: str) -> float:
        letters = sum(c.isalpha() for c in s)
        return letters / max(len(s), 1)

    if _alpha_ratio(en) < 0.40 or _alpha_ratio(vi) < 0.40:
        return False

    # Câu chỉ toàn dấu chấm lửng
    if _RE_ELLIPSIS.match(en) or _RE_ELLIPSIS.match(vi):
        return False

    return True


# ─────────────────────────────────────────────────────────────
# 3. LOAD MỘT FILE CSV
# ─────────────────────────────────────────────────────────────

def _load_single(
    csv_path: str,
    min_len: int,
    max_len: int,
    max_len_ratio: float,
) -> pd.DataFrame:
    """
    Tải và làm sạch một file CSV dữ liệu song ngữ En-Vi.

    Hàm nội bộ, được gọi bởi load_dataset(). Thực hiện:
        1. Đọc CSV và chuẩn hoá tên cột về chữ thường.
        2. Kiểm tra sự tồn tại của cột 'en' và 'vi'.
        3. Áp dụng pipeline clean_text() cho từng câu.
        4. Lọc các cặp câu không hợp lệ qua is_valid_pair().

    Args:
        csv_path (str): Đường dẫn tới file CSV. File phải có cột 'en' và 'vi'.
        min_len (int): Số từ tối thiểu per câu để giữ lại.
        max_len (int): Số từ tối đa per câu để giữ lại.
        max_len_ratio (float): Tỉ lệ chiều dài tối đa giữa hai câu trong cặp.

    Returns:
        pd.DataFrame: DataFrame với cột ['en', 'vi'] đã làm sạch và lọc,
            với index được reset về 0.

    Raises:
        FileNotFoundError: Nếu file CSV không tồn tại tại csv_path.
        ValueError: Nếu file CSV không có cột 'en' hoặc 'vi'.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {csv_path}")

    df = pd.read_csv(csv_path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]

    if "en" not in df.columns or "vi" not in df.columns:
        raise ValueError(
            f"[{path.name}] Cần cột 'en' và 'vi'. "
            f"Hiện có: {df.columns.tolist()}"
        )

    n_raw = len(df)
    df = df[["en", "vi"]].dropna()

    # Làm sạch
    df["en"] = df["en"].map(clean_text)
    df["vi"] = df["vi"].map(clean_text)

    # Lọc độ dài / tỉ lệ
    mask = [
        is_valid_pair(en, vi, min_len, max_len, max_len_ratio)
        for en, vi in zip(df["en"], df["vi"])
    ]
    df = df[mask].reset_index(drop=True)

    print(f"  [{path.name}]  {n_raw:>9,} raw  →  {len(df):>9,} sạch  (loại {n_raw - len(df):,})")
    return df


# ─────────────────────────────────────────────────────────────
# 4. PUBLIC API — dùng bởi dataset.py và tokenizer.py
# ─────────────────────────────────────────────────────────────

def load_dataset(
    csv_paths: list[str] | str,
    min_len: int = 1,
    max_len: int = 200,
    max_len_ratio: float = 9.0,
    deduplicate: bool = True,
    shuffle: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Tải và xử lý sạch nhiều file CSV song ngữ En↔Vi, gộp thành một DataFrame.

    API công khai dùng bởi dataset.py và tokenizer.py.

    Args:
        csv_paths (list[str] | str): Một hoặc nhiều đường dẫn tới file CSV.
            Mỗi file phải có cột 'en' và 'vi'.
        min_len (int): Số từ tối thiểu per câu. Mặc định: 1.
        max_len (int): Số từ tối đa per câu. Mặc định: 200.
        max_len_ratio (float): Tỉ lệ chiều dài tối đa giữa hai câu. Mặc định: 9.0.
        deduplicate (bool): Loại bỏ các cặp câu trùng lặp giữa các file. Mặc định: True.
        shuffle (bool): Xáo trộn ngẫu nhiên kết quả cuối cùng. Mặc định: False.
        seed (int): Seed cho random shuffle để đảm bảo tái lập. Mặc định: 42.

    Returns:
        pd.DataFrame: DataFrame với cột ['en', 'vi'] đã làm sạch, lọc và tùy chọn dedup/shuffle.

    Example:
        >>> df = load_dataset(["dataset/PhoMT/train.csv", "dataset/opus100/train.csv"])
        >>> len(df)  # Số cặp câu hợp lệ sau khi xử lý
    """
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]

    print(f"load_corpus: {len(csv_paths)} file(s)")

    frames = [
        _load_single(p, min_len, max_len, max_len_ratio)
        for p in csv_paths
    ]
    df = pd.concat(frames, ignore_index=True)

    # Dedup cross-file
    n_before = len(df)
    if deduplicate:
        df = df.drop_duplicates(subset=["en", "vi"]).reset_index(drop=True)
        if len(df) < n_before:
            print(f"  Dedup: loại {n_before - len(df):,} cặp trùng")

    if shuffle:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        print("  Đã shuffle.")

    print(f"  Tổng: {len(df):,} cặp câu sạch")
    return df

