# готовые ResNet-архитектуры, собранные из кирпичей tinytensor.nn
# Каждая новая стадия удваивает число каналов и вдвое уменьшает картинку.

import numpy as np
from tinytensor.nn.modules import Module, Sequential
from tinytensor.nn.conv import Conv2d
from tinytensor.nn.batchnorm import BatchNorm2d
from tinytensor.nn.activations import ReLU
from tinytensor.nn.pooling import MaxPool2d, GlobalAvgPool2d
from tinytensor.nn.flatten import Flatten
from tinytensor.nn.linear import Linear
from tinytensor.nn.residual import ResidualBlock


def _make_stage(in_channels, out_channels, num_blocks, stride):
    blocks = []
    blocks.append(ResidualBlock(in_channels, out_channels, stride))
    for _ in range(num_blocks - 1):
        blocks.append(ResidualBlock(out_channels, out_channels, stride=1))
    return blocks


class ResNet(Module):
    def __init__(self, num_classes=10, in_channels=3, blocks_per_stage=(2, 2, 2, 2), small_input=True):
        # num_classes сколько классов на выходе
        # in_channels каналов во входной картинке (3 для RGB, 1 для MNIST)
        # blocks_per_stage сколько блоков в каждой из 4 стадий.
        #(2,2,2,2) = ResNet-18, (3,4,6,3) = ResNet-34
        # small_input True для мелких картинок типо сифара
        #лёгкий стем 3x3 без агрессивного уменьшения.
        # False классический стем 7x7 + maxpool
        super().__init__()

        layers = []

        #стем
        if small_input:
            layers.append(Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1))
            layers.append(BatchNorm2d(64))
            layers.append(ReLU())
        else:
            layers.append(Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3))
            layers.append(BatchNorm2d(64))
            layers.append(ReLU())
            layers.append(MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        #стейжс
        channels = [64, 128, 256, 512]
        in_ch = 64
        for i in range(4):
            out_ch = channels[i]
            stride = 1 if i == 0 else 2
            stage = _make_stage(in_ch, out_ch, blocks_per_stage[i], stride)
            layers.extend(stage)    
            in_ch = out_ch

        layers.append(GlobalAvgPool2d())

        layers.append(Flatten())
        layers.append(Linear(512, num_classes))

        self.net = Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# функции
def ResNet18(num_classes=10, in_channels=3, small_input=True):
    return ResNet(num_classes, in_channels, blocks_per_stage=(2, 2, 2, 2), small_input=small_input)


def ResNet34(num_classes=10, in_channels=3, small_input=True):
    return ResNet(num_classes, in_channels, blocks_per_stage=(3, 4, 6, 3), small_input=small_input)
