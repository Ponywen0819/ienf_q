"""UNet binary segmentation model.

Architecture: encoder-decoder with skip connections.
  Input  : 2 channels (green channel + annotation)
  Output : 2 channels (background logit, foreground logit)
  Depth  : inc → down1..5 → up5..1 → outc
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive Conv2d → BatchNorm → ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    """MaxPool2d followed by DoubleConv (encoder block)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    """Transposed-conv upsampling + skip-connection concatenation + DoubleConv.

    Channel convention:
      - ConvTranspose2d: in_ch → in_ch // 2
      - torch.cat([skip, upsampled]): in_ch // 2 + in_ch // 2 = in_ch
      - DoubleConv: in_ch → out_ch
    Therefore skip must have in_ch // 2 channels.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Align spatial dimensions (handles odd input sizes)
        dh = x2.size(2) - x1.size(2)
        dw = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class UNet(nn.Module):
    """UNet for binary segmentation.

    Encoder: B → 2B → 4B → 8B → 16B → 32B  (B = base_channels)
    Decoder: 32B → 16B → 8B → 4B → 2B → B → out_channels

    Args:
        in_channels:   Number of input channels (default 2 = green + annotation).
        out_channels:  Number of output classes  (default 2 = BG / FG).
        base_channels: Feature-map width at the first encoder level (default 32).
                       Using 64 matches the original UNet paper; 32 halves memory.
    """

    def __init__(self, in_channels: int = 2, out_channels: int = 2, base_channels: int = 32):
        super().__init__()
        b = base_channels
        self.inc   = DoubleConv(in_channels, b)
        self.down1 = Down(b,      b * 2)
        self.down2 = Down(b * 2,  b * 4)
        self.down3 = Down(b * 4,  b * 8)
        self.down4 = Down(b * 8,  b * 16)
        self.down5 = Down(b * 16, b * 32)

        self.up5   = Up(b * 32, b * 16)
        self.up1   = Up(b * 16, b * 8)
        self.up2   = Up(b * 8,  b * 4)
        self.up3   = Up(b * 4,  b * 2)
        self.up4   = Up(b * 2,  b)
        self.outc  = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x = self.up5(x6, x5)
        x = self.up1(x,  x4)
        x = self.up2(x,  x3)
        x = self.up3(x,  x2)
        x = self.up4(x,  x1)
        return self.outc(x)
