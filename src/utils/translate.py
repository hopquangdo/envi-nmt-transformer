import torch
from src.utils.decoding import greedy_decode, beam_search_decode
from src.utils import clean_tokens


class Translator:
    """
    Lớp hụ trợ dịch văn bản từ tiếng Anh sang tiếng Việt.

    Bao gồm 2 thuật toán giải mã: Greedy Search và Beam Search.
    Tự động chuyển model sang chế độ eval() khi khởi tạo.
    """

    def __init__(self, model, tokenizer, device, max_len=128):
        """
        Khởi tạo bộ dịch.

        Args:
            model (Transformer): Mô hình Transformer đã huấn luyện.
            tokenizer (SentencePieceProcessor): Bộ tokenizer đã tải.
            device (torch.device): Thiết bị chạy (CPU hoặc CUDA).
            max_len (int): Độ dài câu dịch tối đa (tính cả BOS/EOS). Mặc định: 128.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len

        self.bos_id = tokenizer.bos_id()
        self.eos_id = tokenizer.eos_id()
        self.pad_id = tokenizer.pad_id()

        self.model.eval()

    def encode(self, text):
        """
        Token hóa văn bản đầu vào và chạy qua Encoder để lấy biểu diễn ngữ cảnh.

        Args:
            text (str): Văn bản tiếng Anh đầu vào. Ví dụ: 'Hello world'.

        Returns:
            tuple[Tensor, Tensor]:
                - enc (Tensor): Biểu diễn ngữ cảnh từ Encoder, shape (1, T_src, d_model).
                - mask (Tensor): Padding mask, shape (1, 1, 1, T_src).
        """
        ids = self.tokenizer.encode(text)
        src = [self.bos_id] + ids[: self.max_len - 2] + [self.eos_id]

        src = torch.tensor([src], device=self.device)
        mask = (src != self.pad_id).unsqueeze(1).unsqueeze(2)

        enc = self.model.encoder(src, mask)
        return enc, mask

    def translate(self, text, method="beam", beam_size=5):
        """
        Dịch một câu văn bản hoàn chỉnh từ tiếng Anh sang tiếng Việt.

        Args:
            text (str): Văn bản tiếng Anh cần dịch. Ví dụ: 'Hello world'.
            method (str): Thuật toán giải mã. 'beam' (mặc định) hoặc 'greedy'.
            beam_size (int): Số chum tìm kiếm khi dùng Beam Search. Mặc định: 5.

        Returns:
            str: Câu đã dịch sang tiếng Việt. Ví dụ: 'Xin chào thế giới'.
        """
        enc, mask = self.encode(text)

        if method == "greedy":
            tokens = greedy_decode(
                self.model, enc, mask,
                self.bos_id, self.eos_id,
                self.max_len, self.device
            )
        else:
            tokens = beam_search_decode(
                self.model, enc, mask,
                self.bos_id, self.eos_id,
                self.max_len, beam_size,
                self.device
            )

        tokens = clean_tokens(tokens, self.bos_id, self.eos_id)
        return self.tokenizer.decode(tokens)
