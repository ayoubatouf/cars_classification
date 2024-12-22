import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(
        self, in_channels, out_channels, bottleneck_width, stride=1, downsample=None
    ):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, bottleneck_width, kernel_size=1, stride=stride, bias=False
        )
        self.bn1 = nn.BatchNorm2d(bottleneck_width)
        self.conv2 = nn.Conv2d(
            bottleneck_width,
            bottleneck_width,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(bottleneck_width)
        self.conv3 = nn.Conv2d(
            bottleneck_width, out_channels, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
