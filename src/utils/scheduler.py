class NoamScheduler:
    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step = 0

    def step(self):
        """
        Cập nhật tốc độ học (Learning Rate) theo công thức Noam.

        Output Demo:
            return: float (Learning Rate mới, ví dụ: 0.0001).
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