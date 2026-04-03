from torch import nn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        """
        Khởi tạo mạng Feed-Forward (Position-wise FFN) hai lớp.

        Kiến trúc: Linear(d_model → d_ff) → ReLU → Dropout → Linear(d_ff → d_model) → Dropout.

        Args:
            d_model (int): Chiều của vector nhúng đầu vào và đầu ra. Ví dụ: 512.
            d_ff (int): Chiều của lớp ẩn bên trong (thường gấp 4 lần d_model). Mặc định: 2048.
            dropout (float): Tỉ lệ dropout sau mỗi lớp Linear. Mặc định: 0.1.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Khối mạng nơ-ron truyền thẳng (Point-wise Feed-Forward).

        Áp dụng cùng một phép biến đổi phi tuyến độc lập tại mỗi vị trí (position)
        trong chuỗi, không có sự tương tác giữa các vị trí.

        Args:
            x (Tensor): Tensor đầu vào shape (B, T, d_model). Ví dụ: (32, 20, 512).

        Returns:
            Tensor: Tensor đầu ra shape (B, T, d_model). Ví dụ: (32, 20, 512).
        """
        return self.net(x)
