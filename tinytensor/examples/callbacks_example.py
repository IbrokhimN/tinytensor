# колбэки в fit: чекпоинт, csv-лог, ранняя остановка, шедулер lr.
# передаются списком в callbacks=[...], дёргаются в конце каждой эпохи.
import numpy as np
from tinytensor.nn import Sequential, Conv2d, ReLU, MaxPool2d, Flatten, Linear, CrossEntropyLoss
from tinytensor.nn import ModelCheckpoint, CSVLogger, EarlyStop, LRScheduler
from tinytensor.optim import AdamW, StepLR
from tinytensor.data import load_mnist
import tinytensor as tt

tt.set_device("cuda")

(x_train, y_train), (x_test, y_test) = load_mnist()

model = Sequential(
    Conv2d(1, 16, 3, padding=1), ReLU(), MaxPool2d(kernel_size=2),
    Flatten(), Linear(16 * 14 * 14, 10),
)

opt = AdamW(model.parameters(), lr=1e-3)
model.compile(lambda p: opt, CrossEntropyLoss())

# шедулер для колбэка
sched = StepLR(opt, step_size=5, gamma=0.5)

# набор колбэков. можно комбинировать как хочешь
callbacks = [
    ModelCheckpoint("best_model.tt", monitor="val_loss"),   # сохраняет когда val_loss лучше
    CSVLogger("training_log.csv"),                          # пишет метрики в csv
    LRScheduler(sched),                                     # снижает lr по расписанию
    EarlyStop(monitor="val_loss", patience=3),              # стоп если 3 эпохи без улучшения
]

model.fit(
    x_train, y_train,
    epochs=30,
    batch_size=64,
    validation_data=(x_test, y_test),
    callbacks=callbacks,
)

# честная точность
acc = (model.predict(x_test) == y_test).mean()
print(f"test accuracy: {acc * 100:.2f}%")
