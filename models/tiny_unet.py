from torch import nn

from models.blocks import DoubleConv, UpBlock


class TinyUNet(nn.Module):
    def __init__(self, num_classes=3, base_channels=16):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.pool = nn.MaxPool2d(2)

        self.encoder1 = DoubleConv(3, c1)
        self.encoder2 = DoubleConv(c1, c2)
        self.encoder3 = DoubleConv(c2, c3)
        self.bottleneck = DoubleConv(c3, c4)

        self.decoder3 = UpBlock(c4, c3, c3)
        self.decoder2 = UpBlock(c3, c2, c2)
        self.decoder1 = UpBlock(c2, c1, c1)

        self.output = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))

        x = self.bottleneck(self.pool(e3))

        x = self.decoder3(x, e3)
        x = self.decoder2(x, e2)
        x = self.decoder1(x, e1)

        return self.output(x)
