# LeNet-5 - классическая свёрточная сеть (Yann LeCun, 1998).
# самая простая CNN-архитектура, идеальна для MNIST.
# структура: 2 свёртки с пулингом -> 3 полносвязных слоя.
from tinytensor.nn.modules import Module, Sequential
from tinytensor.nn.conv import Conv2d
from tinytensor.nn.pooling import MaxPool2d
from tinytensor.nn.flatten import Flatten
from tinytensor.nn.linear import Linear
from tinytensor.nn.activations import ReLU


class LeNet(Module):
    def __init__(self, num_classes=10, in_channels=1):
        # in_channels=1 для ч/б (MNIST), 3 для цветных
        # рассчитано на картинки 28x28 (MNIST)
        super().__init__()
        self.net = Sequential(
            # блок 1: свёртка 6 каналов + пулинг. 28x28 -> 14x14
            Conv2d(in_channels, 6, kernel_size=5, padding=2),  # padding=2 сохраняет 28x28
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),                # 28 -> 14

            # блок 2: свёртка 16 каналов + пулинг. 14x14 -> 5x5
            Conv2d(6, 16, kernel_size=5),                      # 14 -> 10 (без padding)
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),                # 10 -> 5

            # классификатор: разворачиваем и 3 полносвязных слоя
            Flatten(),                                         # 16*5*5 = 400
            Linear(16 * 5 * 5, 120),
            ReLU(),
            Linear(120, 84),
            ReLU(),
            Linear(84, num_classes),
        )

    def forward(self, x):
        return self.net(x)
