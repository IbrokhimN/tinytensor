# ансамбль из трёх моделей на MNIST: голосуют по вероятностям (soft).
# обучаются по очереди, в конце график val-accuracy моделей vs честная точность ансамбля.
# для GPU оставь set_device("cuda"), для CPU убери.
from tinytensor.models import Ensemble, LeNet, ResNet
from tinytensor.nn import (Sequential, Conv2d, BatchNorm2d, ReLU,
                           MaxPool2d, Flatten, Linear, CrossEntropyLoss)
from tinytensor.optim import AdamW
from tinytensor.data import load_mnist
import tinytensor as tt

tt.set_device("cuda")

(x_train, y_train), (x_test, y_test) = load_mnist()

# третья модель — своя двухблочная CNN, по силе близкая к LeNet, но другой архитектуры
my_cnn = Sequential(
    Conv2d(1, 16, 3, padding=1), BatchNorm2d(16), ReLU(), MaxPool2d(kernel_size=2),   # 28 -> 14
    Conv2d(16, 32, 3, padding=1), BatchNorm2d(32), ReLU(), MaxPool2d(kernel_size=2),  # 14 -> 7
    Flatten(), Linear(32 * 7 * 7, 10),
)

ens = Ensemble([
    LeNet(num_classes=10, in_channels=1),
    ResNet(num_classes=10, in_channels=1, blocks_per_stage=(1, 1, 1, 1), small_input=True),
    my_cnn,
], voting_type="soft")

ens.compile(lambda p: AdamW(p, lr=1e-3), CrossEntropyLoss())

# validation_data обязателен, иначе не будет честной val_acc для графика
ens.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(x_test, y_test),
)

# честная точность ансамбля на тесте
acc = ens.evaluate(x_test, y_test)
print(f"точность ансамбля на тесте: {acc * 100:.2f}%")

# сохраняем весь ансамбль в один файл
ens.save("ensemble.tt")

# график: val-accuracy каждой модели по эпохам + линия ансамбля
ens.plot(x_test, y_test, save_path="ensemble_plot.png")
