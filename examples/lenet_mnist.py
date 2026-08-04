# ============================================================================
# Обучение LeNet на MNIST - пример "в 5 строк"
# ============================================================================
# LeNet - лёгкая сеть, реально обучается на CPU (в отличие от ResNet).
# Для GPU добавь строку: import tinytensor as tt; tt.set_device('cuda')
#
# Запуск: python lenet_mnist.py
# ============================================================================

from tinytensor.data import load_mnist
from tinytensor.models import LeNet
from tinytensor.optim import AdamW
from tinytensor.nn import CrossEntropyLoss

# 1. данные (load_mnist сам нормализует и добавляет канал -> [N,1,28,28])
(x_train, y_train), (x_test, y_test) = load_mnist()

# 2. модель: LeNet для 10 классов, 1 канал (ч/б)
model = LeNet(num_classes=10, in_channels=1)

# 3. компиляция + обучение (accuracy и прогресс-бар покажутся сами)
model.compile(lambda p: AdamW(p, lr=1e-3), CrossEntropyLoss())
model.fit(
    x_train, y_train,
    epochs=3,
    batch_size=64,
    validation_data=(x_test, y_test),
)
