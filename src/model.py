from torch import nn

from src.layer.decoder import Decoder
from src.layer.encoder import Encoder
from src.utils.mask import create_src_mask, create_tgt_mask


class Transformer(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, num_layers=6,
                 num_heads=8, d_ff=2048, dropout=0.1, max_len=512,
                 pad_idx=0, weight_tying=True):
        """
        Khởi tạo mô hình Transformer Encoder-Decoder đầy đủ cho bài toán dịch máy.

        Kiến trúc: Encoder (N lớp) → Decoder (N lớp) → Linear + Softmax (logits).
        Trọng số được khởi tạo bằng Xavier Uniform để tránh vanishing/exploding gradient.

        Args:
            src_vocab (int): Kích thước từ điển câu nguồn (tiếng Anh). Ví dụ: 32000.
            tgt_vocab (int): Kích thước từ điển câu đích (tiếng Việt). Ví dụ: 32000.
            d_model (int): Số chiều vector ẩn (embedding size). Mặc định: 512.
            num_layers (int): Số lớp Encoder và Decoder. Mặc định: 6.
            num_heads (int): Số head trong Multi-Head Attention. Mặc định: 8.
            d_ff (int): Chiều lớp ẩn trong Feed-Forward. Mặc định: 2048.
            dropout (float): Tỉ lệ dropout trong quá trình train. Mặc định: 0.1.
            max_len (int): Độ dài chuỗi tối đa cho Positional Encoding. Mặc định: 512.
            pad_idx (int): ID của token <pad>. Mặc định: 0.
            weight_tying (bool): Nếu True, chia sẻ trọng số Embedding và Linear đầu ra
                (giảm số tham số, tăng khả năng tổng quát). Mặc định: True.
        """
        super().__init__()
        self.pad_idx = pad_idx

        # Khởi tạo Encoder: Chịu trách nhiệm mã hóa câu tiếng Anh thành các vector ngữ nghĩa
        # d_model là kích thước của lớp nhúng (embedding size), đây là độ lớn của vetor biểu diễn từ
        self.encoder = Encoder(
            vocab_size=src_vocab,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
            pad_idx=pad_idx,
        )
        self.decoder = Decoder(
            vocab_size=tgt_vocab,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads, # Số lượng head (đầu) trong Multi-head Attention
            d_ff=d_ff,           # Kích thước của lớp ẩn trong mạng Feed Forward
            dropout=dropout,
            max_len=max_len,
            pad_idx=pad_idx,
            weight_tying=weight_tying,
        )

        # Khởi tạo trọng số mạng chuẩn hơn thay vì ngẫu nhiên mặc định
        self._init_weights()

    def _init_weights(self):
        """
        Khởi tạo trọng số tất cả lớp Linear bằng Xavier Uniform.

        Xavier Uniform giữ phương sai đầu ra gần bằng đầu vào, giúp tránh
        vanishing/exploding gradient khi mạng sau. Áp dụng cho mọi tensor
        có số chiều > 1 (bỏ qua bias và các vector 1D).
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt):
        """
        Luồng xử lý chính của mô hình Transformer (dùng trong training).

        Args:
            src (Tensor): Tensor ID câu nguồn (tiếng Anh), shape (B, T_src). Ví dụ: (32, 20).
            tgt (Tensor): Tensor ID câu đích (tiếng Việt) đã shift, shape (B, T_tgt). Ví dụ: (32, 15).
                Lưu ý: tgt là câu đích đã bỏ token cuối (tgt[:, :-1]) khi gọi từ train loop.

        Returns:
            Tensor: Logits dự đoán từ vựng, shape (B, T_tgt, tgt_vocab_size). Ví dụ: (32, 15, 32000).
        """
        # src_mask: Che đi các ký tự padding của câu tiếng Anh
        # tgt_mask: Che đi padding VÀ những từ ở tương lai của câu tiếng Việt (để mô hình không chép phạt)
        src_mask = create_src_mask(src, self.pad_idx)   # (B, 1, 1, T_src)
        tgt_mask = create_tgt_mask(tgt, self.pad_idx)   # (B, 1, T_tgt, T_tgt)

        # 1. Đưa câu nguồn (tiếng Anh) qua Encoder để lấy biểu diễn ngữ cảnh
        enc_out = self.encoder(src, src_mask)
        
        # 2. Đưa câu đích (tiếng Việt), và kết quả của Encoder vào Decoder 
        # để máy dịch từng từ một và trả về các logit (xác suất dự đoán từ vựng)
        logits = self.decoder(tgt, enc_out, src_mask, tgt_mask)
        return logits

    def forward_with_attn(self, src, tgt):
        """
        Tương tự forward() nhưng trả thêm cross-attention weights để trực quan hóa.

        Args:
            src (Tensor): Tensor ID câu nguồn, shape (B, T_src).
            tgt (Tensor): Tensor ID câu đích đã shift, shape (B, T_tgt).

        Returns:
            Tensor: Logits dự đoán, shape (B, T_tgt, tgt_vocab_size).
            list[Tensor]: Cross-attention weights của mỗi decoder layer,
                mỗi phần tử shape (B, H, T_tgt, T_src). Dùng để tạo attention heatmap.
        """
        src_mask = create_src_mask(src, self.pad_idx)
        tgt_mask = create_tgt_mask(tgt, self.pad_idx)

        enc_out = self.encoder(src, src_mask)
        logits, cross_attns = self.decoder(tgt, enc_out, src_mask, tgt_mask, return_attn=True)
        return logits, cross_attns