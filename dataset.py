# dataset.py
"""
Dataset customizado para classificação de imagens.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
import csv

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        for label in os.listdir(root_dir):
            label_dir = os.path.join(root_dir, label)
            for fname in os.listdir(label_dir):
                self.samples.append((os.path.join(label_dir, fname), int(label)))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        from PIL import Image
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class TabularDataset(Dataset):
    def __init__(self, csv_path):
        self.data = []
        with open(csv_path) as f:
            reader = csv.reader(f)
            next(reader) # skip header
            for row in reader:
                features = [float(x) for x in row[:-1]]
                label = int(row[-1])
                self.data.append((features, label))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        features, label = self.data[idx]
        return torch.tensor(features, dtype=torch.float32), label

class TextDataset(Dataset):
    def __init__(self, txt_path, vocab=None):
        self.data = []
        with open(txt_path) as f:
            for line in f:
                text, label = line.strip().rsplit(',', 1)
                self.data.append((text, int(label)))
        self.vocab = vocab or self.build_vocab()
    def build_vocab(self):
        vocab = {}
        idx = 0
        for text, _ in self.data:
            for word in text.split():
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1
        return vocab
    def encode(self, text):
        return [self.vocab.get(word, 0) for word in text.split()]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoded = self.encode(text)
        return torch.tensor(encoded, dtype=torch.long), label

def get_dataloaders(config):
    batch_size = config["batch_size"]
    if config["data_type"] == "image":
        if config["augment"]:
            train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            train_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        train_ds = ImageDataset(f"{config['data_dir']}/train", train_transform)
        val_ds = ImageDataset(f"{config['data_dir']}/val", val_transform)
    elif config["data_type"] == "tabular":
        train_ds = TabularDataset(f"{config['data_dir']}/train.csv")
        val_ds = TabularDataset(f"{config['data_dir']}/val.csv")
    elif config["data_type"] == "text":
        train_ds = TextDataset(f"{config['data_dir']}/train.txt")
        val_ds = TextDataset(f"{config['data_dir']}/val.txt", vocab=train_ds.vocab)
    else:
        raise ValueError("Tipo de dado não suportado")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    return train_loader, val_loader
