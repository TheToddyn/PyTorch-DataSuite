# utils.py
"""
Utilitários: EarlyStopping e logging.
"""
class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
    def __call__(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

def log_metrics(epoch, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1):
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Train F1={train_f1:.4f}, Val F1={val_f1:.4f}")
