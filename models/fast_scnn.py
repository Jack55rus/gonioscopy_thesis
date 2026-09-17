import torch
from torch import nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
    ):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__(
            ConvBNReLU(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
            ),
            ConvBNReLU(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
            ),
        )


class FastSCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBNReLU(3, 32, stride=2),
            DepthwiseSeparableConv(32, 48, stride=2),
            DepthwiseSeparableConv(48, 64, stride=2),
            DepthwiseSeparableConv(64, 96, stride=2),
            DepthwiseSeparableConv(96, 128, stride=2),
        )

        self.classifier = nn.Sequential(
            DepthwiseSeparableConv(128, 128),
            DepthwiseSeparableConv(128, 128),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def forward(self, x):
        input_size = x.shape[-2:]
        x = self.encoder(x)
        x = self.classifier(x)

        return F.interpolate(
            x,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
