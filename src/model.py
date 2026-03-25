import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Two conv layers with BatchNorm and ReLU activations.
    BatchNorm stabilises training and prevents the large-blob prediction
    instability observed in early training without it.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    """
    3-level U-Net with skip connections.

    Channel widths increased from 32/64/128 to 64/128/256 to give the model
    enough capacity to learn subtle JPEG artifact and boundary patterns in
    real COCO photographs at 256x256 resolution.

    Architecture:
        Encoder:     3 -> 64 (enc1) -> pool -> 128 (enc2) -> pool
        Bottleneck:  128 -> 256
        Decoder:     256 -> upsample -> cat(enc2) -> 128 (dec2)
                         -> upsample -> cat(enc1) ->  64 (dec1)
        Output:      64 -> 1  (raw logit map, same spatial size as input)

    Apply torch.sigmoid() to the output for inference probabilities.
    """
    def __init__(self):
        super().__init__()

        self.enc1 = ConvBlock(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(128, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        self.out = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out(d1)
