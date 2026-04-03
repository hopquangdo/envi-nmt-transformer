import math
from torch import nn


from src.layer.positional_encoding import PositionalEncoding
from src.layer.attention import MultiHeadAttention
from src.layer.feed_forward import FeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        """
        Khởi tạo một lớp Encoder đơn (Single Encoder Layer).

        Mỗi lớp gồm 2 sub-layer với cơ chế Pre-LN và Residual Connection:
            1. Multi-Head Self-Attention
            2. Position-wise Feed-Forward Network

        Args:
            d_model (int): Số chiều vector ẩn. Ví dụ: 512.
            num_heads (int): Số head trong Multi-Head Attention. Ví dụ: 8.
            d_ff (int): Chiều lớp ẩn trong Feed-Forward. Mặc định: 2048.
            dropout (float): Tỉ lệ dropout. Mặc định: 0.1.
        """
        super().__init__()
        # 1. Cơ chế Self-Attention: Giúp mô hình bắt được quan hệ giữa các từ trong cùng một câu
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        # 2. Mạng nơ-ron truyền thẳng (Feed-Forward Network)
        self.ff = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        """
        Xử lý qua 1 lớp Encoder với Pre-LN và Residual Connection.

        Args:
            x (Tensor): Tensor đầu vào shape (B, T, d_model). Ví dụ: (32, 20, 512).
            src_mask (Tensor | None): Padding mask shape (B, 1, 1, T) để bỏ qua các token <pad>.
                Ví dụ: (32, 1, 1, 20).

        Returns:
            Tensor: Tensor đầu ra shape (B, T, d_model). Ví dụ: (32, 20, 512).
        """
        # Pre-LN (Layer Normalization TRƯỚC khi đưa vào Sub-layer) - Giúp mô hình huấn luyện ổn định hơn so với paper ban đầu
        normed = self.norm1(x)
        # Self-Attention với residual connection (x + attn(normed)). Query, Key, Value đều đến từ x.
        x = x + self.dropout(self.attn(normed, normed, normed, src_mask))
        
        # Đi qua khối Feed Forward cũng với cơ chế kết nối thặng dư (residual connection)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads,
                 d_ff=2048, dropout=0.1, max_len=512, pad_idx=0):
        """
        Khởi tạo bộ Encoder gồm N lớp EncoderLayer xếp chồng.

        Luồng xử lý: Token IDs → Embedding → Scale → Positional Encoding → N × EncoderLayer → LayerNorm.

        Args:
            vocab_size (int): Kích thước từ điển nguồn (tiếng Anh). Ví dụ: 32000.
            d_model (int): Số chiều vector ẩn. Ví dụ: 512.
            num_layers (int): Số lớp EncoderLayer xếp chồng. Ví dụ: 6.
            num_heads (int): Số head trong Multi-Head Attention. Ví dụ: 8.
            d_ff (int): Chiều lớp ẩn Feed-Forward. Mặc định: 2048.
            dropout (float): Tỉ lệ dropout. Mặc định: 0.1.
            max_len (int): Độ dài chuỗi tối đa cho Positional Encoding. Mặc định: 512.
            pad_idx (int): ID của token <pad> trong từ điển. Mặc định: 0.
        """
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pe = PositionalEncoding(d_model, max_len, dropout)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, src_mask=None):
        """
        Xử lý qua toàn bộ bộ Encoder (gồm N lớp xếp chồng).

        Args:
            x (Tensor): Tensor ID câu nguồn, shape (B, T). Ví dụ: (32, 20).
            src_mask (Tensor | None): Padding mask shape (B, 1, 1, T). Ví dụ: (32, 1, 1, 20).

        Returns:
            Tensor: Tensor biểu diễn ngữ cảnh đã chuẩn hóa, shape (B, T, d_model). Ví dụ: (32, 20, 512).
        """
        # Bước 1: Chuyển các ID từ thành vector (Embedding) và scale nó lên
        x = self.embed(x) * math.sqrt(self.d_model)
        # Bước 2: Cộng thêm mã hóa vị trí (Positional Encoding) vì tự bản thân Transformer không biết thứ tự từ vựng
        x = self.pe(x)
        
        # Bước 3: Đưa qua các lớp EncoderLayer (thường là 6 lớp xếp chồng)
        for layer in self.layers:
            x = layer(x, src_mask)
            
        # Chuẩn hóa đầu ra cuối cùng
        return self.norm(x)
