import torch
import torch.nn as nn

from .dilated_conv import DilatedConvEncoder


def check_nan(tensor: torch.Tensor, name: str) -> None:
    if torch.isnan(tensor).any():
        print(
            f"❌ NaN detected in {name} | mean={tensor.mean().item():.4f} std={tensor.std().item():.4f}"
        )
        raise ValueError(f"NaN found in {name}")
    if torch.isinf(tensor).any():
        print(
            f"❌ Inf detected in {name} | mean={tensor.mean().item():.4f} std={tensor.std().item():.4f}"
        )
        raise ValueError(f"Inf found in {name}")


class Encoder(nn.Module):
    """Single-modality temporal encoder with a reconstruction head."""

    def __init__(
        self,
        N_feat: int,
        d_model: int = 64,
        depth: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.input_proj = nn.Linear(N_feat, d_model)
        self.norm_in = nn.LayerNorm(d_model)

        self.tcn = DilatedConvEncoder(d_model, [d_model] * depth, kernel_size=3)
        self.dropout = nn.Dropout(dropout)
        self.norm_tcn = nn.LayerNorm(d_model)

        self.head_recon = nn.GRU(d_model, d_model, batch_first=True)
        self.norm_recon = nn.LayerNorm(d_model)
        self.recon_out = nn.Linear(d_model, N_feat)

    def forward(
        self,
        x: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an input sequence and predict its reconstruction.

        Args:
            x: [B, T, N_feat] input sequence.
            mask: Optional [B, T] boolean mask. ``True`` entries are kept,
                ``False`` entries will be zeroed before encoding.

        Returns:
            hidden: [B, T, D] encoded features.
            recon:  [B, T, N_feat] reconstruction logits of the input.
        """

        check_nan(x, "input x")

        h = self.input_proj(x)
        check_nan(h, "after input_proj")
        h = self.norm_in(h)

        if mask is not None:
            mask_exp = mask.unsqueeze(-1).expand_as(h)
            h = h.masked_fill(~mask_exp, 0.0)
            check_nan(h, "after masking")

        h_tcn = self.tcn(h.transpose(1, 2)).transpose(1, 2)
        check_nan(h_tcn, "after TCN")
        h = self.norm_tcn(h + self.dropout(h_tcn))
        check_nan(h, "after norm_tcn")

        recon_h, _ = self.head_recon(h)
        recon_h = self.norm_recon(recon_h)
        x_recon = self.recon_out(recon_h)
        check_nan(x_recon, "x_recon")

        return h, x_recon
