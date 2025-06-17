import torch.nn as nn


class DeepMLPClassifier(nn.Module):
    """Deeper MLP classifier for representation learning"""

    def __init__(self, input_dim: int, hidden_dims=None, output_dim: int = 14, dropout: float = 0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], output_dim))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)