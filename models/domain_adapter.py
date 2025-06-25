import torch
import torch.nn as nn


def _build_align_block(d_model, dropout=0.1):
    return nn.Sequential(
        nn.Linear(d_model, d_model),
        nn.ReLU(),
        nn.LayerNorm(d_model),
        nn.Dropout(dropout),
        nn.Linear(d_model, d_model),
    )

class DomainAdapter(nn.Module):
    """Adapter placed before each encoder.

    It supports two modes:
    * ``aware`` - add a learnable session embedding to the projected features.
    * ``align`` - apply an extra alignment MLP shared across sessions.
    """

    def __init__(self, input_dim: int, d_model: int, num_sessions: int = 0,
                 dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, d_model)
        self.session_embed = (
            nn.Embedding(num_sessions, d_model) if num_sessions > 0 else None
        )
        self.align_block = _build_align_block(d_model, dropout)
        self.aware_block = _build_align_block(d_model, dropout)
        self.mode = "none"

    def set_mode(self, mode: str) -> None:
        assert mode in {"aware", "align", "none"}
        self.mode = mode

    def forward(self, x: torch.Tensor, session_idx: torch.Tensor | None = None) -> torch.Tensor:
        h = self.linear(x)

        if self.mode == "aware":
            emb = self.session_embed(session_idx).unsqueeze(1)
            h = h + emb
            h = self.aware_block(h)
        elif self.mode == "align":
            h = self.align_block(h)

        return h
