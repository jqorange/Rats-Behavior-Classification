import torch
import torch.nn as nn
import torch.nn.functional as F

class SamePadConv(nn.Module):
    """
    1D convolution with 'same' padding, supports dilation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        # Effective receptive field
        self.receptive_field = (kernel_size - 1) * dilation + 1
        # Compute padding so output length equals input length
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        # If receptive field is even, convolution with symmetric padding overshoots by 1
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x):
        # x: (B, C_in, T)
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, :-self.remove]
        return out

class ConvBlock(nn.Module):
    """
    A residual block of two dilated 1D conv layers with GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        # Project input to match output channels if needed or at final layer
        self.projector = (
            nn.Conv1d(in_channels, out_channels, 1)
            if (in_channels != out_channels or final) else None
        )

    def forward(self, x):
        # x: (B, C, T)
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual

class DilatedConvEncoder(nn.Module):
    """
    Stacks multiple dilated residual ConvBlocks to form a temporal encoder.
    channels: list of output channels for each block.
    """
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(channels):
            in_ch = in_channels if i == 0 else channels[i-1]
            is_final = (i == len(channels)-1)
            dilation = 2 ** i
            layers.append(
                ConvBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    final=is_final
                )
            )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, C_in, T)
        return self.net(x)
