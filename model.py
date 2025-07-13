# model.py
"""
Definição do modelo de IA (Exemplo: ResNet para classificação de imagens)
"""
import torch.nn as nn

def get_model(model_name, num_classes, data_type):
    if data_type == "image":
        if model_name == "resnet18":
            from torchvision import models
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == "efficientnet_b0":
            from torchvision import models
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif model_name == "vit_b_16":
            from torchvision import models
            model = models.vit_b_16(weights=None)
            model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        else:
            raise ValueError(f"Modelo {model_name} não suportado.")
    elif data_type == "tabular":
        class TabularNet(nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, num_classes)
                )
            def forward(self, x):
                return self.net(x)
        # input_dim será inferido no train.py
        model = TabularNet(input_dim=0, num_classes=num_classes) # input_dim será ajustado
    elif data_type == "text":
        class TextNet(nn.Module):
            def __init__(self, vocab_size, num_classes):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, 32)
                self.rnn = nn.GRU(32, 64, batch_first=True)
                self.fc = nn.Linear(64, num_classes)
            def forward(self, x):
                x = self.embedding(x)
                _, h = self.rnn(x)
                return self.fc(h.squeeze(0))
        # vocab_size será inferido no train.py
        model = TextNet(vocab_size=0, num_classes=num_classes) # vocab_size será ajustado
    else:
        raise ValueError(f"Tipo de dado {data_type} não suportado.")
    return model
