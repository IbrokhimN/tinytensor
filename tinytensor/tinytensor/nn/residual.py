import numpy as np
from tinytensor.nn.modules import Module
from tinytensor.nn.conv import Conv2d
from tinytensor.nn.batchnorm import BatchNorm2d
from tinytensor.nn.activations import ReLU


class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = BatchNorm2d(out_channels)
        self.conv2 = Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = BatchNorm2d(out_channels)

        self.relu = ReLU()

        if stride != 1 or in_channels != out_channels:
            self.downsample = Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out) 

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity

        #релу в конце
        out = self.relu(out)
        return out
