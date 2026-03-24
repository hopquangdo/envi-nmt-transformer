class NoamScheduler:
    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step = 0

    def step(self):
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
        return self.optimizer.param_groups[0]["lr"]