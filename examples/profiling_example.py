# summary и benchmark: размер модели, latency, память.
# полезно для замеров на edge-девайсах (Raspberry Pi и т.п.).
import numpy as np
from tinytensor.models import LeNet
from tinytensor.nn import Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Flatten, Linear
from tinytensor.utils import summary, benchmark, count_params

# --- summary: структура и размер модели ---
model = LeNet(num_classes=10, in_channels=1)
summary(model, (1, 1, 28, 28))   # input_shape с батчем: (N, C, H, W)

# на своём Sequential summary покажет разбивку по слоям
my_cnn = Sequential(
    Conv2d(1, 16, 3, padding=1), BatchNorm2d(16), ReLU(), MaxPool2d(kernel_size=2),
    Conv2d(16, 32, 3, padding=1), BatchNorm2d(32), ReLU(), MaxPool2d(kernel_size=2),
    Flatten(), Linear(32 * 7 * 7, 10),
)
summary(my_cnn, (1, 1, 28, 28))

# --- benchmark: latency и память ---
x = np.random.randn(32, 1, 28, 28).astype(np.float32)   # один батч
res = benchmark(model, x, runs=100, warmup=10)

# res - обычный dict, можно достать цифры для таблицы/графика
print()
print(f"median latency: {res['latency_median_ms']:.2f} мс")
print(f"peak RAM:       {res['peak_ram_mb']:.2f} МБ")
print(f"параметров:     {res['params']:,}")

# сравнить две модели по размеру
print()
print(f"LeNet:  {count_params(model):,} параметров")
print(f"my_cnn: {count_params(my_cnn):,} параметров")
