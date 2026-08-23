# LeNet на MNIST без переобучения.
# patience - ранняя остановка, save_best - лучшие веса на диск и обратно в модель,
# в конце меряем точность на тесте (не train-accuracy из лога).
# Для GPU оставь set_device("cuda"), для CPU убери.
# Запуск: python lenet_mnist.py
from tinytensor.data import load_mnist
from tinytensor.models import LeNet
from tinytensor.optim import AdamW
from tinytensor.nn import CrossEntropyLoss
import tinytensor as tt

tt.set_device("cuda")

# load_mnist сам нормализует и добавляет канал -> [N,1,28,28]
(x_train, y_train), (x_test, y_test) = load_mnist()

model = LeNet(num_classes=10, in_channels=1)
model.compile(lambda p: AdamW(p, lr=1e-3), CrossEntropyLoss())

history = model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(x_test, y_test),
    patience=4,
    save_best="LeNet_MNIST.tt",
)

history.plot(save_path="plot.png")

# честная точность на тесте, модель уже держит лучшие веса
preds = model.predict(x_test)
test_acc = (preds == y_test).mean()
print(f"test accuracy: {test_acc * 100:.2f}%")
