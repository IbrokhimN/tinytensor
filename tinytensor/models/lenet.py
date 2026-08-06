from tinytensor.nn.modules import Module, Sequential
from tinytensor.nn.conv import Conv2d
from tinytensor.nn.pooling import MaxPool2d
from tinytensor.nn.flatten import Flatten
from tinytensor.nn.linear import Linear
from tinytensor.nn.activations import ReLU


class LeNet(Module):
    def __init__(self, num_classes=10, in_channels=1):
        super().__init__()
        self.net = Sequential(
            Conv2d(in_channels, 6, kernel_size=5, padding=2),  # padding=2 сохраняет 28x28
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),                # 28 -> 14

            Conv2d(6, 16, kernel_size=5),                      # 14 -> 10 (без padding)
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),                # 10 -> 5

            Flatten(),                                         # 16*5*5 = 400
            Linear(16 * 5 * 5, 120),
            ReLU(),
            Linear(120, 84),
            ReLU(),
            Linear(84, num_classes),
        )

    def forward(self, x):
        return self.net(x)
