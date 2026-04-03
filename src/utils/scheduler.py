class NoamScheduler:
    """
    Learning Rate Scheduler theo công thức Noam (từ paper 'Attention Is All You Need').

    Thuật toán:
        lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

    Trong đó:
        - Pha warmup: LR tăng dần tuyến tính từ 0 đến đỉnh trong warmup_steps bước.
        - Pha decay: LR giảm dần theo 1/sqrt(step) sau khi qua đỉnh.
    """

    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        """
        Khởi tạo Noam Learning Rate Scheduler.

        Args:
            optimizer: Optimizer của PyTorch (ví dụ: torch.optim.Adam).
                Luưu ý: optimizer nên được khởi tạo với lr=1.0 vì scheduler sẽ ghi đè hoàn toàn.
            d_model (int): Số chiều vector ẩn của Transformer. Ví dụ: 512.
                Dùng để scale đỉnh LR phù hợp với kích thước mô hình.
            warmup_steps (int): Số bước tăng dần (warmup). Mặc định: 4000.
        """
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step = 0

    def step(self):
        """
        Tăng bộ đếm bước và cập nhật Learning Rate theo công thức Noam.

        Công thức:
            scale = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
            lr    = scale

        Returns:
            float: Giá trị Learning Rate mới được áp dụng. Ví dụ: 0.0001.
        """
        self._step += 1
        scale = (self.d_model ** -0.5) * min(
            self._step ** -0.5,
            self._step * (self.warmup_steps ** -1.5)
        )
        for group in self.optimizer.param_groups:
            group["lr"] = scale
        return scale

    @property
    def current_lr(self):
        """
        Lấy giá trị Learning Rate hiện tại.

        Output Demo:
            return: float (ví dụ: 0.0001)
        """
        return self.optimizer.param_groups[0]["lr"]