import torch.nn as nn

class DomainAdapter(nn.Module):
    """Simple two-layer adapter used before each encoder."""

    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 2 * d_model),
            nn.LayerNorm(2 * d_model),
            nn.ReLU(),
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x):
        return self.layers(x)