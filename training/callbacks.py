def log_metrics(metrics: dict):
    parts = [f"{k}={v}" for k, v in metrics.items()]
    print(" ".join(parts))


class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.best = float("inf")
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best:
            self.best = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
