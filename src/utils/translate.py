import torch
from src.utils.decoding import greedy_decode, beam_search_decode
from src.utils import clean_tokens


class Translator:
    def __init__(self, model, tokenizer, device, max_len=128):
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
        Chuẩn hóa văn bản đầu vào thành ID và thực hiện Encoder.

        Input Demo:
            text: 'Hello world'
        Output Demo:
            return: (enc, mask) -> enc: (1, T_src, d_model), mask: (1, 1, 1, T_src)
        """
        ids = self.tokenizer.encode(text)
        src = [self.bos_id] + ids[: self.max_len - 2] + [self.eos_id]

        src = torch.tensor([src], device=self.device)
        mask = (src != self.pad_id).unsqueeze(1).unsqueeze(2)

        enc = self.model.encoder(src, mask)
        return enc, mask

    def translate(self, text, method="beam", beam_size=5):
        """
        Dịch một câu văn bản hoàn chỉnh.

        Input Demo:
            text: 'Hello world'
            method: 'beam' hoặc 'greedy'
        Output Demo:
            return: 'Xin chào thế giới'
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
