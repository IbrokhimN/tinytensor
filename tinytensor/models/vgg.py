from tinytensor.nn.modules import Module, Sequential
from tinytensor.nn.conv import Conv2d
from tinytensor.nn.pooling import MaxPool2d
from tinytensor.nn.flatten import Flatten
from tinytensor.nn.linear import Linear
from tinytensor.nn.activations import ReLU
from tinytensor.nn.batchnorm import BatchNorm2d


def _vgg_block(in_ch, out_ch, num_convs):
    layers = []
    for i in range(num_convs):
        c_in = in_ch if i == 0 else out_ch
        layers.append(Conv2d(c_in, out_ch, kernel_size=3, padding=1))  # padding=1 сохраняет размер
        layers.append(BatchNorm2d(out_ch))
        layers.append(ReLU())
    layers.append(MaxPool2d(kernel_size=2, stride=2))   # уменьшаем картинку вдвое
    return layers


class VGG(Module):
    def __init__(self, num_classes=10, in_channels=3, small_input=True):
        super().__init__()

        layers = []
        layers += _vgg_block(in_channels, 64, num_convs=2)   # 32 -> 16
        layers += _vgg_block(64, 128, num_convs=2)           # 16 -> 8
        layers += _vgg_block(128, 256, num_convs=3)          # 8 -> 4

        layers.append(Flatten())                             # 256*4*4 = 4096
        layers.append(Linear(256 * 4 * 4, 512))
        layers.append(ReLU())
        layers.append(Linear(512, num_classes))

        self.net = Sequential(*layers)

    def forward(self, x):
        return self.net(x)
