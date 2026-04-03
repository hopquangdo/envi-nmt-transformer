from torch import nn


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
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
        
        Input Demo:
            x: Tensor shape (B, T, d_model) -> (32, 20, 512)
        Output Demo:
            return: Tensor shape (B, T, d_model) -> (32, 20, 512)
        """
        return self.net(x)
