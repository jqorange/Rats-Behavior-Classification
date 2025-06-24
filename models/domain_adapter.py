import torch.nn as nn

class DomainAdapter(nn.Module):
    """Simple two-layer adapter used before each encoder."""

    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x):
        return self.layers(x)