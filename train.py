# train.py
"""
Função de treinamento eficiente com early stopping e logging.
"""
import yaml
import csv
import torch
import torch.optim as optim
import torch.nn as nn
from model import get_model
from dataset import get_dataloaders
from utils import EarlyStopping, log_metrics
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score

def train_model():
    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders(config)
    # Inferir input_dim/vocab_size para modelos tabular/texto
    if config["data_type"] == "tabular":
        sample, _ = next(iter(train_loader))
        input_dim = sample.shape[1]
        model = get_model(config["model"], config["num_classes"], config["data_type"])
        model.net[0] = nn.Linear(input_dim, 64)
    elif config["data_type"] == "text":
        sample, _ = next(iter(train_loader))
        vocab_size = int(sample.max().item()) + 1
        model = get_model(config["model"], config["num_classes"], config["data_type"])
        model.embedding = nn.Embedding(vocab_size, 32)
    else:
        model = get_model(config["model"], config["num_classes"], config["data_type"])
    if config["multi_gpu"] and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    early_stopping = EarlyStopping(patience=config["patience"])
    writer = SummaryWriter()
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    best_val_loss = float('inf')
    best_model_path = "best_model.pth"
    if config["log_csv"]:
        csv_file = open("metrics.csv", "w", newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "train_f1", "val_f1"])
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0
        train_preds, train_targets = [], []
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            train_loss += loss.item()
            train_preds.extend(outputs.argmax(1).cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average='weighted')
        val_loss = 0
        val_preds, val_targets = [], []
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_preds.extend(outputs.argmax(1).cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='weighted')
        log_metrics(epoch, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1)
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Acc/train', train_acc, epoch)
        writer.add_scalar('Acc/val', val_acc, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        scheduler.step(val_loss)
        if config["log_csv"]:
            csv_writer.writerow([epoch, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1])
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
        if early_stopping(val_loss):
            print("Early stopping!")
            break
    writer.close()
    if config["log_csv"]:
        csv_file.close()
    torch.save(model.state_dict(), "model.pth")
